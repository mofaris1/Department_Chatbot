import sys
from flask import Flask, request, jsonify, send_from_directory, send_file
import pandas as pd
import sqlite3
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
import ollama
import traceback
import requests
import time
import re
from urllib.parse import quote_plus, unquote
from flask_cors import CORS
import fitz
import io
import asyncio
try:
    import edge_tts
    _EDGE_TTS_OK = True
except ImportError:
    _EDGE_TTS_OK = False
    print('⚠️  edge-tts not installed — run: pip install edge-tts')
from difflib import SequenceMatcher

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. Configuration
# ==========================================
OLLAMA_MODEL     = "deepseek-r1:8b"
EXCEL_FILE       = "project_data_V2.xlsx"
OLLAMA_HOST      = "http://localhost:11434"
THRESHOLD_DIRECT = 0.80
JUST_BASE_URL    = "https://www.just.edu.jo"
# Arabic TTS voice (edge-tts neural voices — install: pip install edge-tts)
TTS_VOICE_FEMALE = "ar-JO-SanaNeural"     # Jordan Arabic female (default)
TTS_VOICE_MALE   = "ar-JO-TaimANeural"    # Jordan Arabic male
TTS_VOICE        = TTS_VOICE_FEMALE

# Confidence thresholds for anti-hallucination
HIGH_CONFIDENCE_THRESHOLD = 0.75   # Return DB answer directly, no LLM
MEDIUM_CONFIDENCE_THRESHOLD = 0.40 # LLM with context needed
LOW_CONFIDENCE_FABRICATION_THRESHOLD = 0.40  # Block fabricated info below this

# ==========================================
# 2. Database
# ==========================================
conn   = sqlite3.connect("unanswered_questions.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS pending_questions
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   user_question TEXT, status TEXT)""")
conn.commit()

# ==========================================
# 3. PDF Knowledge Base
# ==========================================
# Put your PDF files in the "pdfs" folder next to app.py
# They will be auto-indexed on startup. No code changes needed.
PDF_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfs")
PDF_CHUNK_SIZE    = 400
PDF_CHUNK_OVERLAP = 80

_pdf_chunks: list[dict] = []
_pdf_index  = None
_pdf_ready  = False


def _extract_pdf_text(source: str) -> list[dict]:
    label = source if len(source) < 60 else source[-55:]
    try:
        if source.startswith("http://") or source.startswith("https://"):
            r = requests.get(source, timeout=20, headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
                ),
                "Referer": JUST_BASE_URL,
            })
            if r.status_code != 200:
                print(f"⚠️  PDF HTTP {r.status_code}: {label}")
                return []
            data = io.BytesIO(r.content)
        else:
            path = (os.path.join(os.path.dirname(os.path.abspath(__file__)), source)
                    if not os.path.isabs(source) else source)
            if not os.path.exists(path):
                print(f"⚠️  PDF not found: {path}")
                return []
            data = open(path, "rb")

        doc   = fitz.open(stream=data.read() if hasattr(data, "read") else data,
                          filetype="pdf")
        pages = []
        for i, page in enumerate(doc, 1):
            text = page.get_text().strip()
            if text:
                pages.append({"text": text, "source": source, "page": i})
        doc.close()
        print(f"✅ PDF loaded: {label} ({len(pages)} pages)")
        return pages
    except Exception as e:
        print(f"❌ PDF load error [{label}]: {e}")
        return []


def _is_garbled_line(line: str) -> bool:
    tokens = line.split()
    arabic = [t for t in tokens if re.search(r'[\u0600-\u06ff]', t)]
    if len(arabic) < 3:
        return False
    short   = sum(1 for t in arabic if len(t) <= 2)
    avg_len = sum(len(t) for t in arabic) / len(arabic)
    return avg_len < 3.2 or (short / len(arabic)) > 0.45


def _clean_page_text(text: str) -> str:
    cleaned = [ln for ln in text.splitlines() if not _is_garbled_line(ln)]
    return "\n".join(cleaned).strip()


def _chunk_pages(pages: list[dict]) -> list[dict]:
    chunks = []
    for p in pages:
        text  = _clean_page_text(p["text"])
        if not text:
            continue
        start = 0
        while start < len(text):
            end   = min(start + PDF_CHUNK_SIZE, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append({"text": chunk, "source": p["source"], "page": p["page"]})
            start += PDF_CHUNK_SIZE - PDF_CHUNK_OVERLAP
    return chunks


def _build_pdf_index(chunks: list[dict]):
    if not chunks:
        return None
    texts  = [c["text"] for c in chunks]
    emb    = sentence_model.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
    emb_np = emb.cpu().numpy().astype("float32")
    idx    = faiss.IndexFlatIP(emb_np.shape[1])
    idx.add(emb_np)
    return idx


def _extract_article_ref(text: str) -> str | None:
    patterns = [
        r'(المادة\s+\d+)', r'(الفقرة\s+[\dأ-ي]+)',
        r'(البند\s+[\dأ-يa-zA-Z]+)', r'(القسم\s+\d+)', r'(رقم\s+\d+)',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


def load_pdfs():
    global _pdf_chunks, _pdf_index, _pdf_ready
    os.makedirs(PDF_FOLDER, exist_ok=True)
    pdf_files = [
        os.path.join(PDF_FOLDER, f)
        for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        print(f"ℹ️  No PDFs found in {PDF_FOLDER}  — place .pdf files there to enable PDF search.")
        _pdf_ready = False
        return
    pages = []
    for src in pdf_files:
        pages.extend(_extract_pdf_text(src))
    _pdf_chunks = _chunk_pages(pages)
    _pdf_index  = _build_pdf_index(_pdf_chunks)
    _pdf_ready  = bool(_pdf_chunks)
    print(f"📄 PDF index ready: {len(_pdf_chunks)} chunks from {len(pdf_files)} file(s)")


def search_pdf(query: str, top_k: int = 4) -> list[dict]:
    if not _pdf_ready or _pdf_index is None:
        return []
    norm       = normalize(query)
    q_emb      = (sentence_model.encode([norm], normalize_embeddings=True, convert_to_tensor=True)
                  .cpu().numpy().astype("float32"))
    k          = min(top_k, len(_pdf_chunks))
    dists, ids = _pdf_index.search(q_emb, k)
    results    = []
    for rank in range(k):
        score = float(dists[0][rank])
        if score >= 0.35:
            chunk       = _pdf_chunks[ids[0][rank]]
            article_ref = _extract_article_ref(chunk["text"])
            results.append({"score": score, "text": chunk["text"],
                            "source": chunk["source"], "page": chunk["page"],
                            "article_ref": article_ref})
    return results


def format_pdf_source(source: str, page: int) -> str:
    if source.startswith("http"):
        decoded   = unquote(source)
        parts     = decoded.rstrip('/').split('/')
        filename  = parts[-1]
        folder    = parts[-2] if len(parts) >= 2 else ""
        label_raw = re.sub(r'^\d+\s*', '', folder).strip()
        if len(label_raw) < 3:
            label_raw = re.sub(r'\.pdf$', '', filename, flags=re.IGNORECASE).strip()
        if len(label_raw) > 50:
            label_raw = label_raw[:47] + "…"
        label = f"{label_raw} — صفحة {page}"
        return f"[{label}]({source}#page={page})"
    return f"{os.path.basename(source)} — صفحة {page}"


# ==========================================
# 4. Data & Index
# ==========================================
def load_excel_data():
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame({"Question": ["متى يبدأ تسجيل المواد؟"],
                           "Answer":   ["يبدأ التسجيل الأسبوع القادم."],
                           "Keywords": [""]})
        df.to_excel(EXCEL_FILE, index=False)
        return df
    xl     = pd.ExcelFile(EXCEL_FILE)
    frames = []
    for sheet in xl.sheet_names:
        raw   = pd.read_excel(xl, sheet_name=sheet)
        chunk = pd.DataFrame({
            "Question": raw.get("Question", pd.Series(dtype=str)),
            "Answer":   raw.get("Answer",   pd.Series(dtype=str)),
            "Keywords": raw.get("Keywords", pd.Series(dtype=str)),
        })
        frames.append(chunk)
    result = (pd.concat(frames, ignore_index=True)
                .dropna(subset=["Question", "Answer"]))
    result = result[result["Answer"].astype(str).str.strip() != ""].reset_index(drop=True)
    print(f"✅ Loaded {len(result)} rows from {len(xl.sheet_names)} sheet(s)")
    return result


df = load_excel_data()

print("⚙️  Loading embedding model …")
sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def get_embed_texts(dataframe):
    texts = []
    for _, row in dataframe.iterrows():
        q  = str(row["Question"])
        kw = str(row.get("Keywords", ""))
        texts.append(f"{q} {kw}".strip() if kw and kw.lower() != "nan" else q)
    return texts


def build_index(dataframe):
    emb    = sentence_model.encode(get_embed_texts(dataframe),
                                   normalize_embeddings=True,
                                   convert_to_tensor=True)
    emb_np = emb.cpu().numpy().astype("float32")
    idx    = faiss.IndexFlatIP(emb_np.shape[1])
    idx.add(emb_np)
    return idx


faiss_index = build_index(df)
print("✅ Ready.")
load_pdfs()


# ==========================================
# 5. Ollama Health
# ==========================================
_ollama_cache = {"ok": False, "ts": 0}


def is_ollama_available():
    now = time.time()
    if now - _ollama_cache["ts"] < 10:
        return _ollama_cache["ok"]
    for _ in range(2):
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
            if r.status_code == 200:
                _ollama_cache["ok"] = True
                _ollama_cache["ts"] = now
                return True
        except Exception:
            time.sleep(1)
    _ollama_cache["ok"] = False
    _ollama_cache["ts"] = now
    print("❌ Ollama غير متاح بعد محاولتين")
    return False


is_ollama_available()

# ==========================================
# 6. Text Normalisation
# ==========================================
DIALECT_MAP = {
    "شو": "ما",
    "وين": "أين",
    "اين": "أين",
    "كيفش": "كيف",
    "شلون": "كيف",
    "ليش": "لماذا",
    "هيك": "هكذا",
    "بدي": "أريد",
    "بقدر": "أستطيع",
    "لازم": "يجب",
    "اسجل": "تسجيل",
    "سجل": "تسجيل",
    "يسجل": "تسجيل",
    "للجامعه": "في الجامعة",
    "للجامعة": "في الجامعة",
    "ايش": "ما",
    "إيش": "ما",
    "وش": "ما",
    "زين": "جيد",
    "مو": "ليس",
    "احكيلي": "أخبرني",
    "حكيلي":  "أخبرني",
    "گلي":    "أخبرني",
    "قلي":    "أخبرني",
    "خبرني":  "أخبرني",
    "أخبرني": "أخبرني",
    "اخبرني": "أخبرني",
    "اعطيني": "أعطني",
    "عطيني":  "أعطني",
    "ايميل":  "إيميل",
    "الايميل": "الإيميل",
    "دكتوره":  "دكتور",
    "الدكتوره": "الدكتور",
    "بقدرش":   "هل يمكن",
    "اون لاين": "إلكتروني",
    "اونلاين":  "إلكتروني",
    "اية":      "أي",
    "لاين":     "إلكتروني",
    "اية لاين": "إلكتروني",
    "online":   "إلكتروني",
    "طيب":      "",
    "يعطيك":    "",
    "العافية":  "",
    "اهه":      "أريد مزيد من التفاصيل",
    "آه":       "أريد مزيد من التفاصيل",
    "تمام":     "",
    "ماشي":     "",
    "اوك":      "",
    "اوكي":     "",
    "ok":       "",
    "okay":     "",
}


def fix_common_typos_for_llm(query: str) -> str:
    q = query
    q = re.sub(r'اية\s*لاين', 'اون لاين', q)
    q = re.sub(r'بقدرش', 'هل يمكنني', q)
    return q


_PUNCT_RE = re.compile(r'[؟?،,\.!؛;:\-\(\)"\'«»\u200c\u200d]+')


def strip_punctuation(text: str) -> str:
    return _PUNCT_RE.sub(' ', text)


def normalize_arabic(text: str) -> str:
    text = strip_punctuation(text)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ى', 'ي', text)
    return re.sub(r'\s+', ' ', text).strip()


_PREFIXES = ["لل", "بال", "وال", "فال", "كال", "ال", "ل", "ب", "و", "ف", "ك"]


def strip_prefix(word: str) -> str:
    for pfx in _PREFIXES:
        if word.startswith(pfx) and len(word) > len(pfx) + 1:
            return word[len(pfx):]
    return word


def normalize(text: str) -> str:
    words  = text.split()
    mapped = [DIALECT_MAP.get(w, w) for w in words]
    joined = " ".join(w for w in mapped if w)
    return normalize_arabic(joined)


_RAW_STOP = {
    "ما", "هو", "هي", "في", "من", "على", "إلى", "عن", "مع", "هل", "كيف",
    "أين", "اين", "متى", "لماذا", "الجامعي", "الجامعية",
    "ال", "و", "يتم", "يمكن", "هذا", "هذه", "كان", "يكون",
}
STOP_WORDS = {normalize_arabic(w) for w in _RAW_STOP}
STOP_WORDS |= {strip_prefix(w) for w in list(STOP_WORDS)}

# ==========================================
# 6b. Fuzzy Name Matching
# ==========================================
_NAME_PAT = re.compile(
    r'(الدكتور|الأستاذ|الاستاذ|دكتور|استاذ|المهندس|مهندس|الدكتوره|الأستاذة)\s+'
    r'([\u0600-\u06ff]{2,}\s*[\u0600-\u06ff]*\s*[\u0600-\u06ff]*)',
    re.UNICODE
)

_known_names_cache: list[str] = []


def _build_known_names():
    """Extract all doctor/professor names from the DB questions."""
    global _known_names_cache
    names = []
    for _, row in df.iterrows():
        q = str(row["Question"])
        for title, name in _NAME_PAT.findall(q):
            full = f"{title} {name.strip()}"
            norm = normalize_arabic(full)
            if norm not in [normalize_arabic(n) for n in names]:
                names.append(full)
    _known_names_cache = names
    print(f"📛 Known names indexed: {len(names)}")


def fuzzy_name_match(query: str, threshold: float = 0.55) -> tuple[str | None, float]:
    """Return (best_matching_name, score) or (None, 0) if no good match."""
    if not _known_names_cache:
        return None, 0.0
    norm_q = normalize_arabic(query)
    best_score = 0.0
    best_name  = None
    for name in _known_names_cache:
        norm_name = normalize_arabic(name)
        # Partial ratio: check if query tokens appear in the name
        score = SequenceMatcher(None, norm_q, norm_name).ratio()
        # Boost if individual words match
        q_words = set(norm_q.split())
        n_words = set(norm_name.split())
        word_overlap_score = len(q_words & n_words) / max(len(q_words), 1)
        combined = 0.5 * score + 0.5 * word_overlap_score
        if combined > best_score:
            best_score = combined
            best_name  = name
    if best_score >= threshold:
        return best_name, best_score
    return None, 0.0


# Build known names cache after data is loaded
_build_known_names()


# ==========================================
# 7. Intent Detection
# ==========================================
META_PATTERNS = [
    r"مين\s*معي", r"من\s*(أنت|انت)", r"اسمك", r"انت\s*شو", r"شو\s*انت",
    r"كيف\s*(تشتغل|تعمل|بتشتغل|بتعمل)",
    r"شو\s*(الموديل|النموذج)", r"ما\s*(الموديل|النموذج)",
    r"ollama", r"\bllm\b", r"\bai\b",
    r"(موديل|نموذج).*(شغال|يعمل)",
    r"مين\s*(صمم|برمجك|عملك)",
]

CONVERSATIONAL_PATTERNS = [
    r"^(كيف\s*(حالك|الحال|أحوالك|حالكم))",
    r"^(شو|ايش|إيش)\s*(اخبارك|أخبارك)",
    r"(عن|عن\s*ايش|عن\s*إيش)\s*(سألتك|حكينا|تحدثنا|كنا)",
    r"(شو|ما|ايش)\s*(قلت|قلتلي|قلتيلي|كنا|كنت)\s*(نحكي|نتكلم|نقول)?",
    r"(ليش|لماذا)\s*(هيك|هكذا|جاوبت|قلت|رديت)",
    r"شو\s*قصدك", r"وضح\s*لي|وضحلي|شرحلي", r"مش\s*(فاهم|فهمت)",
    r"(كيف|شو|إيش|ايش|زي|مثل)\s*(ايش|شو|ما)?\s*(بتقدر|تقدر|بتعرف|تعرف|بتساعد|تساعد)",
    r"(بتقدر|تقدر|بتعرف|تعرف)\s*(تساعد|تعمل|تحكي|تخبر|تعطي)",
    r"(ايش|إيش|شو|ما|وش)\s*(خدماتك|تعمل|تقدر|تعرف|تساعد|تقدمه)",
    r"هل\s*أنت\s*(ذكاء|روبوت|بوت)",
    r"شكرا|شكراً|ممنون|يسلمو|يعطيك",
    r"(حلو|ممتاز|رائع|كويس|زين|منيح)\s*$",
    r"(غلط|غلطت|مش\s*صح|خطأ)\s*$",
]

ACADEMIC_KEYWORDS = [
    "تسجيل", "سجل", "اسجل", "يسجل", "مادة", "مواد", "ساعات", "فصل",
    "منهج", "دكتور", "دكتوره", "أستاذ", "استاذ", "مهندس",
    "قسم", "جامعة", "جامعه", "كلية", "كليه", "شعبة", "درجة", "علامة",
    "غياب", "امتحان", "مشروع", "تخرج", "خطة", "تأديب", "انتساب",
    "معدل", "إيميل", "ايميل", "مكتب", "نظام", "برنامج", "بحث",
    "وثيقة", "إجراء", "طلب", "استمارة", "شهادة", "توثيق", "رسوم",
    "مالية", "منحة", "محاضرة", "اختبار", "جدول",
    "إلكتروني", "اونلاين", "اون لاين", "online", "بقدرش", "التسجيل",
    "غش", "عقوبة", "عقوبات", "تأديب", "فصل", "رئيس", "رئيس القسم",
    "تدخين", "التدخين", "مخالفة", "انتهاك", "نظام الجامعة",
]

# ── Expansion intent: user wants MORE details on the previous answer ──
EXPANSION_PATTERNS = [
    r'^(اهه|آه|أه|ها)\s*$',
    r'(اعطيني|أعطني|عطيني|اريد|أريد)\s*(تفاصيل|معلومات|المزيد|اكثر|أكثر)',
    r'^(تفاصيل|تفصيل|المزيد|اكثر|أكثر)\s*$',
    r'(وضح|اشرح|شرح|شرحلي|وضحلي)\s*(لي|ليا|أكثر|المزيد)?',
    r'(اكمل|أكمل|كمّل|كمل|استمر)',
    r'(ايش|إيش|شو|وش|ما)\s*(يعني|معناها|قصدك|قصده)',
    r'مش\s*(فاهم|فهمت)\s*(الموضوع|هاد|هذا)?',
    r'^(طيب|ماشي|تمام|اوكي|اوك)\s+(اعطيني|وضحلي|شرحلي|أكثر|تفاصيل)',
]


def _matches(text: str, patterns: list) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


def is_expansion_request(query: str) -> bool:
    """True if the user is asking for more details about the previous answer."""
    q = query.strip()
    if _matches(q, EXPANSION_PATTERNS):
        return True
    # Very short queries with no academic keywords after mapping
    norm = normalize(q)
    if len(norm.split()) <= 2 and not any(kw in norm for kw in ACADEMIC_KEYWORDS):
        return True
    return False


def classify_intent(raw_query: str) -> str:
    norm_q = normalize(raw_query)
    if _matches(raw_query, META_PATTERNS) or _matches(norm_q, META_PATTERNS):
        return "meta"
    if _matches(raw_query, CONVERSATIONAL_PATTERNS) or _matches(norm_q, CONVERSATIONAL_PATTERNS):
        return "conversational"
    if (any(kw in raw_query for kw in ACADEMIC_KEYWORDS)
            or any(kw in norm_q for kw in ACADEMIC_KEYWORDS)):
        return "academic"
    if len(raw_query.split()) <= 2:
        return "conversational"
    return "unknown"


# ==========================================
# 8. FAISS Retrieval
# ==========================================
def word_overlap(q: str, stored: str) -> float:
    norm_q = normalize_arabic(q)
    norm_s = normalize_arabic(str(stored))
    qw = {strip_prefix(w) for w in norm_q.split()} - STOP_WORDS
    sw = {strip_prefix(w) for w in norm_s.split()} - STOP_WORDS
    if not qw or not sw:
        return 0.0
    return len(qw & sw) / len(qw | sw)


_NAME_TRIGGERS = {"الدكتور", "الاستاذ", "دكتور", "استاذ",
                  "الدكتوره", "الاستاذه", "المهندس", "مهندس",
                  "الدكتورة", "الاستاذة", "دكتورة"}


def name_overlap(query: str, stored_question: str) -> float:
    q_words = {w for w in normalize_arabic(query).split() if len(w) >= 3}
    s_words = {w for w in normalize_arabic(stored_question).split() if len(w) >= 3}
    if not q_words or not s_words:
        return 0.0
    matches = sum(1 for qw in q_words
                  if any(qw in sw or sw in qw for sw in s_words))
    return matches / len(q_words)


def retrieve_top_k(query: str, top_k: int = 5) -> list[dict]:
    norm          = normalize(query)
    is_name_query = any(w in norm for w in _NAME_TRIGGERS)
    sem_w  = 0.15 if is_name_query else 0.50
    olap_w = 0.85 if is_name_query else 0.50

    q_emb = (sentence_model.encode([norm], normalize_embeddings=True,
                                   convert_to_tensor=True)
             .cpu().numpy().astype("float32"))
    k_multiplier = 20 if is_name_query else 2
    k          = min(max(top_k * k_multiplier, 10), len(df))
    distances, indices = faiss_index.search(q_emb, k)

    scored = []
    for rank in range(k):
        sem      = float(distances[0][rank])
        ri       = indices[0][rank]
        stored_q = df.iloc[ri]["Question"]
        olap     = word_overlap(norm, stored_q)
        if is_name_query:
            name_score = name_overlap(norm, stored_q)
            olap = max(olap, name_score)
        score = sem_w * sem + olap_w * olap
        scored.append((score, ri))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"score": s, "question": str(df.iloc[ri]["Question"]),
         "answer": str(df.iloc[ri]["Answer"])}
        for s, ri in scored[:top_k]
    ]


# ==========================================
# 9. JUST Website Fallback
# ==========================================
_WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ar,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

JUST_PAGES = [
    ("صفحة أعضاء هيئة التدريس - قسم علم البيانات",
     "https://www.just.edu.jo/FacultiesAndDepartments/FacultyofComputer/Departments/DataScience/Pages/Staff.aspx"),
    ("قسم علم البيانات - جامعة العلوم والتكنولوجيا",
     "https://www.just.edu.jo/FacultiesAndDepartments/FacultyofComputer/Departments/DataScience/Pages/Home.aspx"),
    ("الموقع الرسمي لجامعة العلوم والتكنولوجيا الأردنية",
     "https://www.just.edu.jo"),
]


def search_just_website(query: str) -> list[dict]:
    clean_q = " ".join([w for w in query.split() if w not in {"طيب", "بتعرف", "يعني", "شو", "ايش", "هل", "من", "هو"}])
    print(f"🌐 Web search (Bing): {clean_q[:60]}")
    snippets = []
    try:
        session = requests.Session()
        session.headers.update(_WEB_HEADERS)
        bing_url = "https://www.bing.com/search?q=" + quote_plus(f"site:just.edu.jo {clean_q}")
        r = session.get(bing_url, timeout=10)
        
        if r.status_code == 200:
            blocks = re.split(r'<li class="b_algo', r.text)[1:]
            for b in blocks:
                url_m = re.search(r'href="([^"]+)"', b)
                snip_m = re.search(r'<div class="b_caption[^>]*>.*?<p[^>]*>(.*?)</p>', b, re.IGNORECASE | re.DOTALL)
                if not snip_m:
                    snip_m = re.search(r'<p[^>]*>(.*?)</p>', b, re.IGNORECASE | re.DOTALL)
                
                if url_m and snip_m:
                    url = url_m.group(1)
                    snip = re.sub(r'<[^>]+>', '', snip_m.group(1))
                    snip = re.sub(r'\s+', ' ', snip).strip()
                    if snip and len(snip) > 20 and not any(s["snippet"] == snip for s in snippets):
                        snippets.append({"snippet": snip, "url": url})
                        
    except Exception as e:
        print(f"⚠️ Bing search error: {e}")

    return snippets[:4]


# ==========================================
# 10. LLM Callers
# ==========================================
def clean_arabic_output(text: str) -> str:
    text = re.sub(
        r'\*?\*?ملاحظة\*?\*?[:\s]+[^\n]*(أتحدث|الفصح|لغت|محدودي|بطلاقة)[^\n]*\n?',
        '', text, flags=re.IGNORECASE
    )
    text = re.sub(
        r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\uff00-\uffef]+',
        '', text
    )
    text = re.sub(
        r'\*?\*?حول\s+(الدكتور|الأستاذ|المهندس)[^\n]*\n[^\n]*\n?',
        '', text, flags=re.IGNORECASE
    )
    # Remove <think ...> tags that deepseek-r1 sometimes emits
    text = re.sub(r'<think.*?>', '', text, flags=re.DOTALL)
    text = re.sub(r'</think\s*>', '', text, flags=re.DOTALL)
    lines   = text.split('\n')
    cleaned = []
    for line in lines:
        tokens = line.split()
        result = []
        for tok in tokens:
            core              = re.sub(r'^[\W_]+|[\W_]+$', '', tok, flags=re.UNICODE)
            has_arabic        = bool(re.search(r'[\u0600-\u06ff]', tok))
            has_latin         = bool(re.search(r'[a-zA-Z]', tok))
            has_email         = '@' in tok
            is_url            = tok.startswith(('http', 'www.'))
            is_pure_latin_noise = (
                bool(re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', core))
                and not has_arabic and not has_email and not is_url
                and len(core) >= 2
            )
            is_mixed_noise    = has_arabic and has_latin and not has_email and not is_url
            if not is_pure_latin_noise and not is_mixed_noise:
                result.append(tok)
        cleaned.append(' '.join(result))
    joined = re.sub(r'[ \t]+', ' ', '\n'.join(cleaned)).strip()
    # Final pass: remove isolated Latin fragments (1–6 letters, not inside emails/URLs/codes)
    # Keeps: A2, L3, Wi-Fi, VPN, PDF, emails, URLs
    # Removes: "ero", "office", leftover English words
    def _kill_latin_fragment(m):
        word = m.group(0)
        # Keep alphanumeric codes (contain digits), emails, URLs
        if re.search(r'\d', word): return word          # A2, L3, PH4
        if '@' in word or '://' in word: return word     # email / url
        # Keep common technical terms
        keep = {'pdf','vp n','vpn','wifi','wi','fi','toefl','ielts','ds','cs','it','ai','iot','url'}
        if word.lower() in keep: return word
        return ' '
    joined = re.sub(r'[a-zA-Z][a-zA-Z0-9\-\.]*', _kill_latin_fragment, joined)
    # Remove Arabic+Latin mixed tokens that slipped through (e.g. "وero", "مكتبoffice")
    joined = re.sub(
        r'(?<![\w@.])([\u0600-\u06ff]+[a-zA-Z]+|[a-zA-Z]+[\u0600-\u06ff]+)(?![\w@.])',
        ' ', joined
    )
    return re.sub(r'[ \t]+', ' ', joined).strip()


BOT_PERSONA = (
    "أنت مرشد أكاديمي ذكي اسمك 'المرشد الأكاديمي'، "
    "متخصص في قسم علم البيانات بجامعة العلوم والتكنولوجيا الأردنية.\n"
    "CRITICAL: Reply in Arabic ONLY. NEVER use English words. Use Arabic equivalents always.\n"
    "CRITICAL: لا تبدأ إجاباتك الأكاديمية والمباشرة بالتحيات الطويلة ('السلام عليكم ورحمة الله وبركاته' وغيرها)، يمكنك قول 'مرحباً' باختصار إذا لزم الأمر فقط، ثم قدم الإجابة المباشرة.\n"
    "CRITICAL: NEVER add any note about your Arabic language ability or fluency. "
    "Do NOT write 'ملاحظة' or any disclaimer. Just answer confidently.\n"
    "CRITICAL: أنت تعمل كمساعد بحث يعتمد بشكل أساسي على الموقع الرسمي للجامعة وقاعدة البيانات. "
    "إذا كان هناك تعارض، فالموقع الرسمي هو الأصح.\n"
    "إذا سُئلت عن شخص → استخرج إجابته من سياق الموقع الرسمي أو قاعدة البيانات (إيميل، مكتب، كلية). لا تختلق سيرة ذاتية من عندك أبداً.\n"
    "CRITICAL: في حال لم تجد الإجابة في السياق أبداً ولا تعرفها، اكتب كلمة 'مجهول' فقط بدون أي إضافات.\n"
    "CRITICAL: You MUST NOT invent, guess, or fabricate ANY information that is not explicitly stated in the provided context.\n"
    "CRITICAL: If you are not 100% certain about a fact (email, phone, office, name), write 'مجهول' instead of guessing.\n"
    "CRITICAL: Do NOT add any descriptive information about people (qualifications, research, education) unless it appears verbatim in the context.\n"
    "CRITICAL: When asked about a doctor/professor, ONLY report what is in the context: email, office, phone. NOTHING ELSE.\n"
    "عند الإجابة:\n"
    "• رتّب الإجابة في نقاط واضحة ومباشرة.\n"
    "• لا تختلق أي معلومة غير موجودة في السياق.\n"
    "• أجب بأسلوب ودي، واضح، ومنظم يركز على مساعدة الطالب.\n"
    "• تقع جامعة العلوم والتكنولوجيا الأردنية في مدينة إربد وليس في عمان.\n"
)


def _call_ollama(messages: list) -> str | None:
    if not is_ollama_available():
        return None
    try:
        resp = ollama.chat(model=OLLAMA_MODEL, messages=messages,
                           options={"temperature": 0.1})
        raw = (resp.message.content if hasattr(resp, "message")
               else resp["message"]["content"]).strip()
        return clean_arabic_output(raw)
    except Exception as e:
        print(f"🚨 Ollama error ({type(e).__name__}): {e}")
        _ollama_cache["ts"] = 0
        return None


def ollama_chat(query: str, history: list, db_context: str | None = None,
                is_followup: bool = False, expand_previous: bool = False) -> str | None:
    context_block = ""
    if db_context:
        if expand_previous:
            followup_rule = (
                "3. الطالب يطلب تفاصيل أكثر عن الإجابة السابقة. قدّم شرحاً موسّعاً وتفصيلياً\n"
                "   باستخدام كل المعلومات المتاحة في السياق. لا تكرر نفس الإجابة السابقة بحذافيرها.\n"
                "   أضف أمثلة، فقرات إضافية، أو تفسيرات توضيحية.\n"
            )
        elif is_followup:
            followup_rule = (
                "3. السؤال هو متابعة لإجابة سابقة. استخرج المعلومة المطلوبة مباشرةً من\n"
                "   قسم 'معلومات من إجابتك السابقة' وأجب في جملة واحدة فقط.\n"
            )
        else:
            followup_rule = (
                "3. إذا جاء السؤال بضمير (ايميله، مكتبه...) فاستخدم قسم الإجابة السابقة\n"
                "   لتحديد المقصود، ثم ابحث عنه في بقية السياق.\n"
            )
        # When web is the primary source, allow LLM to supplement with general JUST knowledge
        web_primary = db_context and db_context.startswith("[ 🌐")
        if web_primary and not expand_previous:
            strict_rule = (
                "4. يمكنك الإجابة من معلوماتك العامة عن الجامعة إذا تكاملت مع السياق.\n"
                "   لكن لا تخترع أرقاماً أو أسماءً أو تواريخ غير موجودة في السياق.\n"
            )
        else:
            strict_rule = "4. إذا لم تجد الإجابة في السياق اكتب مجهول فقط.\n"
        context_block = (
            "\n\n══ السياق الرسمي — يجب الاعتماد عليه ══\n"
            "قواعد صارمة:\n"
            "1. أجب فقط من هذا السياق أو معلوماتك العامة عن الجامعة.\n"
            "2. اذكر رقم المادة أو الفقرة إذا ظهر في السياق.\n"
            + followup_rule +
            strict_rule +
            "5. لا تضف 'حول الدكتور' أو أي سيرة ذاتية لم تكن في السياق.\n"
            "6. MUST NOT invent, guess, or fabricate ANY information not in the context.\n"
            "7. If not 100% certain about a fact, write 'مجهول' instead.\n"
            f"{db_context}\n"
            "══════════════════════════════════════\n"
        )

    system = (
        BOT_PERSONA
        + "يمكنك الإجابة على أي سؤال: أكاديمي أو اجتماعي أو عام.\n"
        + "• إذا سأل عن شخص → اذكر فقط إيميله ومكتبه من السياق ورقم هاتفه إن وجد. لا تكتب سيرة ذاتية.\n"
        + "• إذا جاء السؤال بضمير → أجب مباشرةً بالمعلومة من السياق.\n"
        + "• إذا كان السؤال أكاديمياً وليس في السياق إجابة → اكتب كلمة مجهول فقط.\n"
        + "لا تختلق أي معلومة غير موجودة في السياق."
        + context_block
    )

    msgs = [{"role": "system", "content": system}]
    for m in history[-10:]:
        if isinstance(m, dict):
            msgs.append(m)
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            msgs.append({"role": "user",      "content": str(m[0])})
            msgs.append({"role": "assistant", "content": str(m[1])})
    msgs.append({"role": "user", "content": query})
    return _call_ollama(msgs)


# ==========================================
# 11. Meta Answers
# ==========================================
def meta_answer(query: str) -> str | None:
    q = normalize(query).lower()
    if any(w in q for w in ["اسمك", "من انت", "مين انت", "انت شو", "شو انت", "مين معي"]):
        return (
            "إنا **المرشد الأكاديمي الذكي** 🎓\n"
            "مساعد آلي مخصص لقسم علم البيانات في جامعة العلوم والتكنولوجيا.\n"
            "أستطيع مساعدتك في أسئلتك الأكاديمية أو حتى مجرد الدردشة! 😊"
        )
    if any(w in q for w in ["موديل", "نموذج", "ollama", "llm"]):
        ok = is_ollama_available()
        return (
            f"**النموذج:** {OLLAMA_MODEL} عبر Ollama\n"
            f"**حالة Ollama:** {'✅ متصل' if ok else '❌ غير متصل'}\n"
            f"**نموذج التضمين:** paraphrase-multilingual-MiniLM-L12-v2\n"
            f"**قاعدة البيانات:** {len(df)} سؤال وجواب"
        )
    if re.search(r"كيف\s*(تشتغل|تعمل)", q):
        return (
            "أعمل بست طبقات 🔍:\n"
            "**1️⃣ FAISS:** أسترجع السياق ذو الصلة\n"
            "**2️⃣ إجابة مباشرة:** إذا كانت الثقة عالية (≥0.75) أُجيب من قاعدة البيانات مباشرة بدون LLM\n"
            "**3️⃣ Ollama (LLM):** أفهم السؤال وأصيغ الإجابة\n"
            "**4️⃣ RAG:** جمع المعلومات كلها بدقة\n"
            "**5️⃣ JUST Website:** أبحث في الموقع الرسمي إذا احتجت\n"
            "**6️⃣ طبقة التحقق:** أتأكد أن الإجابة مكتملة وصحيحة وأتحقق من الوقائع ضد قاعدة البيانات\n"
            "**7️⃣ Escalation:** أحوّل للقسم إذا السؤال جديد تماماً"
        )
    if re.search(r"مين\s*(صمم|برمجك|عملك)", q):
        return "تم تطويري كمشروع تخرج في قسم علم البيانات 🎓"
    return None


# ==========================================
# 12. Answer Validation
# ==========================================
def is_unknown_response(text: str) -> bool:
    t = clean_arabic_output(text.strip())
    
    if len(t) <= 15 and "مجهول" in t:
        return True
    
    unknown_phrases = [
        "لا أملك معلومات رسمية",
        "لا تتوفر لدي معلومات",
        "لا تتوفر معلومات في قاعدة",
        "غير مذكور في السياق",
        "ليس في السياق",
        "المعلومة المطلوبة غير موجودة"
    ]
    for phrase in unknown_phrases:
        if phrase in t and len(t) < 150:
            return True
            
    return False


def _is_hallucinated_name_answer(llm_ans: str, hits: list) -> bool:
    """Check if the LLM answer contains fabricated info when DB score is low."""
    if not hits or hits[0]["score"] >= LOW_CONFIDENCE_FABRICATION_THRESHOLD:
        return False
    fabrication_markers = [
        "تخصصه", "تخصصها", "درس في", "درست في",
        "حاصل على", "حاصلة على", "يدرّس", "تدرّس",
        "يدرس", "تدرس", "خبرة في", "خبرته",
        "أبحاثه", "أبحاثها", "منشوراته", "منشوراتها",
        "مؤهلاته", "مؤهلاتها", "ماجستير في", "دكتوراه في",
        "التحق بـ", "التحقت بـ", "عمل في", "عملت في",
    ]
    return any(m in llm_ans for m in fabrication_markers)


# ==========================================
# 13. Double-Check Layer + Answer Verification
# ==========================================
def detect_requested_info_type(query: str) -> str:
    q = normalize(query).lower()
    if re.search(r'(ايميل|إيميل|بريد|ميل|email)', q):
        return "email"
    if re.search(r'(مكتب|غرف|غرفه|office)', q):
        return "office"
    if re.search(r'(رقم|هاتف|تلفون|جوال|phone)', q):
        return "phone"
    if re.search(r'(مين هو|من هو|بتعرف|تعرف|اعرفني)', q):
        return "name_info"
    return "general"


def extract_info_from_text(text: str, info_type: str) -> str | None:
    if not text:
        return None
    if info_type == "email":
        match = re.search(r'[\w.\-+]+@[\w.\-]+\.\w{2,}', text)
        return match.group(0) if match else None
    if info_type == "office":
        match = re.search(
            r'(مكتب[^\n.،]{0,60}|رقم\s+\d+[^\n.،]{0,40}|office\s*\d+)',
            text, re.IGNORECASE
        )
        return match.group(0).strip() if match else None
    if info_type == "phone":
        match = re.search(r'(\+?\d[\d\s\-]{6,14}\d)', text)
        return match.group(0).strip() if match else None
    return None


def _scan_history_for_info(history: list, info_type: str) -> str | None:
    for msg in reversed(history[-12:]):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        found = extract_info_from_text(msg.get("content", ""), info_type)
        if found:
            return found
    return None


def _scan_db_for_info(hits: list, info_type: str) -> str | None:
    for hit in hits:
        found = extract_info_from_text(hit.get("answer", ""), info_type)
        if found:
            return found
    return None


def _build_followup_search_query(query: str, history: list) -> str:
    name_pat = re.compile(
        r'(الدكتور|الأستاذ|دكتور|استاذ|المهندس|مهندس)\s+[\u0600-\u06ff]{2,}\s*[\u0600-\u06ff]*',
        re.UNICODE
    )
    for msg in reversed(history[-10:]):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        match   = name_pat.search(content)
        if match:
            return f"{match.group(0)} {query}"
    return query


def _verify_llm_facts(llm_ans: str, hits: list, info_type: str) -> str | None:
    """Cross-check specific facts in LLM answer against DB hits.
    Returns corrected answer or None if no correction needed."""
    if info_type == "general" or not hits:
        return None
    
    # Extract the fact from the LLM answer
    llm_fact = extract_info_from_text(llm_ans, info_type)
    if not llm_fact:
        return None
    
    # Check if this fact exists in any DB hit
    for hit in hits:
        db_fact = extract_info_from_text(hit.get("answer", ""), info_type)
        if db_fact and db_fact == llm_fact:
            return None  # Fact matches, no correction needed
    
    # Fact doesn't match any DB hit — check if DB has a different value
    db_verified = _scan_db_for_info(hits, info_type)
    if db_verified and db_verified != llm_fact:
        print(f"🔴 Fact verification FAILED: LLM said '{llm_fact}', DB has '{db_verified}'")
        label = {"email": "الإيميل", "office": "المكتب", "phone": "الرقم"}.get(info_type, "المعلومة")
        return f"**{label}:** {db_verified}"
    
    # No DB match at all — the LLM fact is unverified
    if not db_verified:
        print(f"🔴 Fact verification FAILED: LLM said '{llm_fact}', not found in DB")
        return "مجهول"
    
    return None


def double_check_answer(
    query: str,
    llm_ans: str,
    hits: list,
    history: list,
    is_followup: bool,
    web_snippets: list[str] = None
) -> str:
    info_type = detect_requested_info_type(query)

    if is_unknown_response(llm_ans):
        return llm_ans

    # ── Answer Verification Layer ──
    # Cross-check specific facts in the LLM answer against DB hits
    if info_type != "general":
        verified = _verify_llm_facts(llm_ans, hits, info_type)
        if verified:
            print(f"✅ Answer verification replaced LLM output with verified fact")
            return verified

    if info_type != "general":
        already_has = extract_info_from_text(llm_ans, info_type)
        if already_has:
            return llm_ans

        print(f"🔎 Double-check: answer missing '{info_type}', scanning DB + history …")

        found_in_db = _scan_db_for_info(hits, info_type)
        if found_in_db:
            print(f"✅ Double-check recovered from DB: {found_in_db}")
            label = {"email": "الإيميل", "office": "المكتب", "phone": "الرقم"}.get(info_type, "المعلومة")
            return f"**{label}:** {found_in_db}"

        found_in_history = _scan_history_for_info(history, info_type)
        if found_in_history:
            print(f"✅ Double-check recovered from history: {found_in_history}")
            label = {"email": "الإيميل", "office": "المكتب", "phone": "الرقم"}.get(info_type, "المعلومة")
            return f"**{label}:** {found_in_history}"

        if web_snippets:
            for snippet in web_snippets:
                found_in_web = extract_info_from_text(snippet, info_type)
                if found_in_web:
                    print(f"✅ Double-check recovered from web: {found_in_web}")
                    label = {"email": "الإيميل", "office": "المكتب", "phone": "الرقم"}.get(info_type, "المعلومة")
                    return f"**{label}:** {found_in_web}"

        print(f"🔴 Double-check: '{info_type}' not found anywhere")
        return "مجهول"

    # ── Remove fabricated info about people when score is low ──
    if hits and hits[0]["score"] < LOW_CONFIDENCE_FABRICATION_THRESHOLD:
        fabrication_markers = [
            "تخصصه", "تخصصها", "حاصل على", "حاصلة على",
            "يدرّس", "تدرّس", "يدرس", "تدرس",
        ]
        if any(m in llm_ans for m in fabrication_markers):
            print(f"🔴 Low score ({hits[0]['score']:.2f}) + fabrication markers detected → replacing with مجهول")
            return "مجهول"

    cleaned = re.sub(
        r'\n?\*?\*?حول\s+(الدكتور|الأستاذ|المهندس)[^\n]*\n[^\n]{0,200}',
        '', llm_ans, flags=re.IGNORECASE
    ).strip()
    return cleaned if cleaned else llm_ans


# ==========================================
# 14. Main Bot Logic
# ==========================================
def _get_last_bot_answer(history: list) -> str | None:
    """Get the most recent assistant message from history."""
    for msg in reversed(history[-10:]):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content", "")
    return None


def _get_last_user_question(history: list) -> str | None:
    """Get the second-to-last user message (the one before the current)."""
    user_msgs = [m for m in history if isinstance(m, dict) and m.get("role") == "user"]
    if len(user_msgs) >= 2:
        return user_msgs[-2].get("content", "")
    elif user_msgs:
        return user_msgs[-1].get("content", "")
    return None


def ask_smart_bot(user_query: str, history: list) -> tuple[str, str]:
    """Returns (reply, confidence_level) where confidence_level is 'high', 'medium', or 'low'."""
    try:
        query  = user_query.strip()
        intent = classify_intent(query)
        print(f"🧠 Intent={intent!r} | Ollama={'✅' if is_ollama_available() else '❌'} | Q={query[:60]}")

        # ── 1. Meta ──────────────────────────────────────────────────
        if intent == "meta":
            ans = meta_answer(query)
            if ans:
                return ans, "high"
            llm = ollama_chat(query, history)
            return llm or meta_answer("من أنت"), "medium"

        # ── 2. Pure greetings ─────────────────────────────────────────
        PURE_GREETINGS = {"هلا", "مرحبا", "السلام عليكم", "يعطيك العافية",
                          "أهلا", "مرحباً", "صباح الخير", "مساء الخير",
                          "هلو", "هاي", "أهلاً", "أهلين"}
        if any(g in query for g in PURE_GREETINGS) and len(query.split()) <= 4:
            llm = ollama_chat(query, history)
            return llm or "أهلاً وسهلاً! 😊 كيف يمكنني مساعدتك اليوم؟", "high"

        # ── 2b. Expansion / "give me more details" ────────────────────
        if is_expansion_request(query) and history:
            last_answer   = _get_last_bot_answer(history)
            last_question = _get_last_user_question(history)
            if last_answer and last_question:
                print(f"🔁 Expansion detected — elaborating on: {last_question[:60]}")
                # Build rich expansion context
                expand_ctx = (
                    f"[ 💬 السؤال السابق للطالب ]\n{last_question}\n\n"
                    f"[ ✅ إجابتك السابقة — وسّعها وأضف تفاصيل ]\n{last_answer}\n\n"
                )
                # Also pull PDF for academic topics
                pdf_hits = search_pdf(last_question, top_k=4)
                if pdf_hits:
                    expand_ctx += "[ 📄 معلومات إضافية من الوثائق الرسمية ]\n"
                    for ph in pdf_hits:
                        prefix = f"[{ph['article_ref']}] " if ph.get("article_ref") else ""
                        expand_ctx += f"{prefix}{ph['text']}\n"
                llm_ans = ollama_chat(
                    f"أعطني تفاصيل أكثر وشرحاً موسّعاً عن: {last_question}",
                    history,
                    expand_ctx,
                    is_followup=False,
                    expand_previous=True
                )
                if llm_ans and not is_unknown_response(llm_ans):
                    return llm_ans, "medium"
                return last_answer, "medium"  # fall back to previous answer

        # ── 3. Retrieve FAISS context ─────────────────────────────────
        search_query = query
        # is_followup ONLY when pronouns reference a previous entity (not just short queries)
        FOLLOW_UP_WORDS = {"ايميله", "ايميلها", "إيميله", "إيميلها",
                           "مكتبه", "مكتبها", "رقمه", "رقمها",
                           "ايميلو", "مكتبو", "عنه", "عنها", "بتعرفه",
                           "شو عمله", "ايش عمله", "شو تخصصه", "ايش تخصصه"}
        is_followup = any(w in query for w in FOLLOW_UP_WORDS)

        # 3b. Enrich ONLY on true pronoun follow-ups to resolve person name from history
        if is_followup and history:
            enriched = _build_followup_search_query(query, history)
            if enriched != query:
                search_query = enriched
                print(f"🔗 Follow-up enriched: {search_query[:80]}")

        hits = retrieve_top_k(search_query, top_k=5)
        print("🔍 Top hits:")
        for h in hits:
            print(f"   score={h['score']:.3f} | {h['question'][:60]}")
        best_score = hits[0]["score"] if hits else -1
        best_ans   = hits[0]["answer"] if hits else None

        norm_q    = normalize(query)
        is_name_q = any(w in norm_q for w in _NAME_TRIGGERS)

        # ── 3c. Fuzzy name suggestion (typo correction) ───────────────
        if is_name_q and best_score < 0.40:
            fuzzy_match, fuzzy_score = fuzzy_name_match(query)
            if fuzzy_match and fuzzy_score >= 0.55:
                print(f"🔤 Fuzzy match: {fuzzy_match} (score={fuzzy_score:.2f})")
                # Re-search with corrected name
                corrected_hits = retrieve_top_k(fuzzy_match, top_k=5)
                corrected_score = corrected_hits[0]["score"] if corrected_hits else 0
                if corrected_score >= 0.45:
                    # Return info + suggest correction
                    corrected_answer = corrected_hits[0]["answer"]
                    return (
                        f"هل تقصد **{fuzzy_match}**؟ 🔍\n\n"
                        f"{corrected_answer}"
                    ), "medium"
                else:
                    # Just suggest without answer
                    return (
                        f"هل تقصد **{fuzzy_match}**؟ 🔍\n"
                        f"إذا نعم، اكتب الاسم بشكل كامل وسأساعدك بمعلوماته."
                    ), "low"

        # ── 3d. HIGH CONFIDENCE: Direct DB Answer (Anti-Hallucination) ──
        if hits and hits[0]["score"] >= HIGH_CONFIDENCE_THRESHOLD and intent == "academic":
            print(f"🟢 HIGH CONFIDENCE (score={hits[0]['score']:.3f}): Returning DB answer directly, no LLM needed")
            best = hits[0]
            source_label = "قاعدة البيانات"
            reply = f"{best['answer']}\n\n📄 المصدر: {source_label}"
            # Still do double-check for missing specific info
            reply = double_check_answer(query, reply, hits, history, is_followup, [])
            return reply, "high"

        # ── 4. Fetch Web Context ─────────────────────────────────
        web_results = []
        web_snippets = []
        if intent in ["academic", "unknown"] or is_name_q:
            # For low confidence, try web search first
            if best_score < MEDIUM_CONFIDENCE_THRESHOLD:
                print(f"🟡 LOW CONFIDENCE (score={best_score:.3f}): Trying web search first")
            web_results = search_just_website(query)
            web_snippets = [r["snippet"] for r in web_results]

        # ── 5. Build context ──────────────────────────────────────────
        ctx_parts        = []
        pdf_sources_used = []

        if web_snippets:
            ctx_parts.append("[ 🌐 معلومات من الموقع الرسمي للجامعة (المصدر الأساسي) ]")
            ctx_parts.extend(f"- {s}" for s in web_snippets)

        # 5a. Previous assistant answer for follow-up pronoun resolution
        if is_followup and history:
            last_bot = next(
                (m.get("content", "") for m in reversed(history[-10:])
                 if isinstance(m, dict) and m.get("role") == "assistant"),
                None
            )
            if last_bot:
                ctx_parts.append(
                    "[ 💬 معلومات من إجابتك السابقة — استخرج منها الإجابة المباشرة ]\n"
                    + last_bot
                )

        # 5b. Excel DB hits (Backup)
        _db_threshold = 0.35 if is_name_q else 0.45
        if hits and hits[0]["score"] >= _db_threshold:
            ctx_parts.append("[ ✅ من قاعدة البيانات (احتياطي) ]")
            ctx_parts.extend(
                f"س: {h['question']}\nج: {h['answer']}" for h in hits if h["score"] >= _db_threshold
            )

        # 5c. PDF fallback
        if intent == "academic" and not is_name_q and not is_followup:
            pdf_hits = search_pdf(search_query, top_k=4)
            if pdf_hits:
                ctx_parts.append("\n[ 📄 من الملفات والوثائق الرسمية ]")
                for ph in pdf_hits:
                    prefix = f"[{ph['article_ref']}] " if ph.get("article_ref") else ""
                    ctx_parts.append(f"{prefix}{ph['text']}")
                    src_str = format_pdf_source(ph["source"], ph["page"])
                    if ph.get("article_ref"):
                        src_str = f"{src_str} — {ph['article_ref']}"
                    if src_str not in pdf_sources_used:
                        pdf_sources_used.append(src_str)

        db_context = "\n".join(ctx_parts) if ctx_parts else None

        # ── 6. Call LLM (only for medium/low confidence) ───────────────
        confidence = "medium" if best_score >= MEDIUM_CONFIDENCE_THRESHOLD else "low"
        print(f"🤖 LLM | score={best_score:.2f} | intent={intent} | followup={is_followup} | confidence={confidence}")
        llm_query = fix_common_typos_for_llm(query)
        llm_ans = ollama_chat(llm_query, history, db_context, is_followup=is_followup)

        # ── 6b. Hallucination guard for name queries ──────────────────
        if llm_ans and is_name_q and _is_hallucinated_name_answer(llm_ans, hits):
            if not web_snippets:
                print("🔴 Hallucination detected — blocking LLM output")
                if best_ans and best_score >= 0.45:
                    return str(best_ans), "medium"
                return (
                    "لا تتوفر لدي معلومات كافية عن هذا الشخص. 🤔\n"
                    "جرّب ذكر الاسم الكامل أو تحقق من التهجئة."
                ), "low"
            else:
                print("⚠️ Hallucination guard triggered, but web snippets exist. Trusting LLM synthesis.")

        # ── 7. Ollama offline → best DB answer ───────────────────────
        if llm_ans is None:
            print("⚠️  Ollama offline — using direct DB answer")
            if best_ans and best_score >= 0.45:
                return str(best_ans), "medium"
            if intent == "conversational":
                return (
                    "أهلاً! 😊 يمكنني مساعدتك في:\n"
                    "• تسجيل المواد والجداول\n"
                    "• معلومات الدكاترة والإيميلات\n"
                    "• الأنظمة الجامعية ومتطلبات التخرج\n"
                    "فقط اسألني!"
                ), "low"
            return ("عذراً، النظام يعمل بوضع محدود الآن. "
                    "يرجى إعادة المحاولة أو صياغة سؤالك بشكل مختلف."), "low"

        # ── 6c. Double-check layer with verification ─────────────────
        print("🔎 Running double-check + verification layer …")
        llm_ans = double_check_answer(
            query, llm_ans, hits, history, is_followup, web_snippets
        )
        print(f"   After double-check: {llm_ans[:80]}")

        # ── 8. Unknown → LLM free answer → web fallback → escalate ──────
        if is_unknown_response(llm_ans):
            print("🔴 Answer is unknown — trying free LLM pass")

            # If web snippets exist, synthesise from them directly
            if web_snippets:
                web_context = "\n".join(f"- {s}" for s in web_snippets)
                direct_llm = ollama_chat(
                    query, history,
                    f"[ 🌐 معلومات من الموقع الرسمي ]\n{web_context}\n"
                    "أجب بناءً على ما سبق فقط، بدون إضافات.",
                    is_followup=False
                )
                if direct_llm and not is_unknown_response(direct_llm):
                    return direct_llm, "medium"
                return "وجدت المعلومات التالية من الموقع الرسمي:\n\n" + web_context, "low"

            # Last resort: let LLM answer from general JUST knowledge (no context restrictions)
            # This handles questions that are real but just not in the DB yet
            free_system = (
                BOT_PERSONA +
                "أجب على السؤال التالي من معلوماتك العامة عن جامعة العلوم والتكنولوجيا الأردنية. "
                "إذا لم تعرف الإجابة بشكل مؤكد، قل ذلك بوضوح ولا تخترع معلومات. "
                "يمكنك الإجابة العامة إذا كانت ذات صلة بالجامعة أو القسم."
            )
            free_msgs = [{"role": "system", "content": free_system}]
            for m in history[-6:]:
                if isinstance(m, dict):
                    free_msgs.append(m)
            free_msgs.append({"role": "user", "content": query})
            free_ans = _call_ollama(free_msgs)
            if free_ans and not is_unknown_response(free_ans):
                print("✅ Free LLM answered without context")
                return free_ans, "low"

            # Truly unanswerable → only escalate for very specific unknown academic questions
            if intent == "academic":
                cursor.execute(
                    "INSERT INTO pending_questions (user_question, status) VALUES (?, ?)",
                    (query, "Pending")
                )
                conn.commit()
                return (
                    "سؤالك وصلني ✅\n"
                    "لا أملك إجابة دقيقة الآن، سأحوّله لرئيس القسم للرد عليك.\n"
                    "هل يمكنك إضافة تفاصيل إضافية؟"
                ), "low"
            return (
                "لم أجد معلومات كافية عن هذا الموضوع 🤔\n"
                "هل يمكنك توضيح سؤالك؟"
            ), "low"

        # ── 9. Append PDF and Web sources ─────────────────────────────
        sources_text = ""
        if pdf_sources_used:
            sources_text += "\n".join(f"📄 {s}" for s in pdf_sources_used) + "\n"
        
        if web_results:
            sources_text += "\n".join(f"🔗 [الرابط]({r['url']})" for r in web_results[:2]) + "\n"
            
        if sources_text:
            return f"{llm_ans}\n\n**المصادر:**\n{sources_text.strip()}", confidence
        
        return llm_ans, confidence

    except Exception as e:
        print(f"🚨 Error: {e}")
        traceback.print_exc()
        return "أواجه مشكلة تقنية مؤقتة، يرجى المحاولة مجدداً.", "low"


# ==========================================
# 15. DB Helpers
# ==========================================
def submit_head_answer(q_id, answer):
    try:
        global df, faiss_index
        cursor.execute("SELECT user_question FROM pending_questions WHERE id=?", (q_id,))
        row = cursor.fetchone()
        if not row:
            return "❌ السؤال غير موجود", 404
        question = row[0]
        new_row  = pd.DataFrame({"Question": [question], "Answer": [answer], "Keywords": [""]})
        df       = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False)
        cursor.execute("UPDATE pending_questions SET status='Answered' WHERE id=?", (q_id,))
        conn.commit()
        faiss_index = build_index(df)
        _build_known_names()  # refresh name cache
        return "✅ تم حفظ الإجابة وتدريب البوت بنجاح!", 200
    except Exception as e:
        print(f"❌ submit_head_answer: {e}")
        return f"❌ خطأ: {e}", 400


def submit_bulk_answer(ids: list, answer: str):
    global df, faiss_index
    if not ids or not answer:
        return "❌ يرجى تحديد الأسئلة وكتابة الإجابة", 400
    new_rows  = []
    not_found = []
    for q_id in ids:
        cursor.execute(
            "SELECT user_question FROM pending_questions WHERE id=?", (q_id,))
        row = cursor.fetchone()
        if not row:
            not_found.append(q_id)
            continue
        new_rows.append({"Question": row[0], "Answer": answer, "Keywords": ""})
        cursor.execute(
            "UPDATE pending_questions SET status='Answered' WHERE id=?", (q_id,))
    if not new_rows:
        return "❌ لم يُعثر على أي من الأسئلة المحددة", 404
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)
    conn.commit()
    faiss_index = build_index(df)
    _build_known_names()  # refresh name cache
    saved   = len(new_rows)
    skipped = len(not_found)
    msg     = f"✅ تم حفظ الإجابة على {saved} سؤال وتدريب البوت بنجاح!"
    if skipped:
        msg += f" (لم يُعثر على {skipped} سؤال)"
    return msg, 200


# ==========================================
# 16. API Routes
# ==========================================
@app.route("/")
@app.route("/interface")
def serve_interface():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()
    for name in ("interface_v3.html", "interface_v2.html", "interface.html"):
        if os.path.exists(os.path.join(base_dir, name)):
            return send_from_directory(base_dir, name)
    return "<h2>No interface HTML found in the same folder as this script.</h2>", 404


@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.json
    reply, confidence = ask_smart_bot(
        data.get("query", ""), data.get("history", []))
    return jsonify({"reply": reply, "confidence": confidence})


@app.route("/api/pending", methods=["GET"])
def pending_api():
    df_p = pd.read_sql_query(
        "SELECT id, user_question FROM pending_questions WHERE status='Pending'", conn)
    return jsonify(df_p.to_dict(orient="records"))


@app.route("/api/answer", methods=["POST"])
def answer_api():
    data = request.json
    msg, code = submit_head_answer(data.get("id"), data.get("answer"))
    return jsonify({"success": code == 200, "message": msg}), code


@app.route("/api/answer-bulk", methods=["POST"])
def answer_bulk_api():
    data = request.json
    ids  = data.get("ids", [])
    ans  = data.get("answer", "").strip()
    if not ids:
        return jsonify({"success": False, "message": "❌ لم يتم تحديد أي أسئلة"}), 400
    if not ans:
        return jsonify({"success": False, "message": "❌ الإجابة فارغة"}), 400
    msg, code = submit_bulk_answer(ids, ans)
    return jsonify({"success": code == 200, "message": msg}), code


@app.route("/api/status", methods=["GET"])
def status_api():
    ok = is_ollama_available()
    return jsonify({
        "ollama_online": ok,
        "ollama_warning": "❌ Ollama offline — DB-only mode" if not ok else "✅ OK",
        "model":         OLLAMA_MODEL,
        "rows_loaded":   len(df),
    })


@app.route("/api/test-ollama", methods=["GET"])
def test_ollama():
    result = {"http_ok": False, "chat_ok": False, "model": OLLAMA_MODEL, "error": None}
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        result["http_ok"]          = r.status_code == 200
        result["available_models"] = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        result["error"] = f"HTTP: {e}"
    try:
        resp = ollama.chat(model=OLLAMA_MODEL,
                           messages=[{"role": "user", "content": "مرحبا"}],
                           options={"temperature": 0.1})
        text = (resp.message.content if hasattr(resp, "message")
                else resp["message"]["content"])
        result["chat_ok"]    = True
        result["test_reply"] = text[:100]
    except Exception as e:
        result["chat_error"] = f"{type(e).__name__}: {e}"
    return jsonify(result)



@app.route("/api/tts", methods=["POST"])
def api_tts():
    """Server-side Arabic TTS using edge-tts neural voices."""
    if not _EDGE_TTS_OK:
        return jsonify({"error": "edge-tts not installed"}), 503
    
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    voice_choice = data.get("voice", "female")
    rate_pct = int(data.get("rate", 0))   # -50 .. +50 percent
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    if len(text) > 4000:
        text = text[:4000]
    
    # Clean text for TTS
    clean = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    clean = re.sub(r'\*(.*?)\*', r'\1', clean)
    clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)
    clean = re.sub(r'<[^>]+>', '', clean)
    clean = re.sub(r'#{1,6}\s?', '', clean)
    clean = re.sub(r'[📄🌐🔗✅❌⚠️🤖🎓😊🔍💬•◆🔊⏸⏹]', '', clean)
    clean = re.sub(r'\|', '', clean)
    clean = re.sub(r'https?://\S+', '', clean)
    clean = re.sub(r'\s{2,}', ' ', clean).strip()
    
    if not clean:
        return jsonify({"error": "Empty text after cleaning"}), 400
    
    voice = TTS_VOICE_FEMALE if voice_choice == "female" else TTS_VOICE_MALE
    rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
    
    try:
        # Use asyncio to run edge_tts
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        async def _generate():
            communicate = edge_tts.Communicate(clean, voice, rate=rate_str)
            audio_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
            audio_buffer.seek(0)
            return audio_buffer
        
        audio_buffer = loop.run_until_complete(_generate())
        loop.close()
        
        if audio_buffer.getbuffer().nbytes == 0:
            return jsonify({"error": "No audio generated"}), 500
        
        return send_file(audio_buffer, mimetype="audio/mpeg", as_attachment=False,
                         download_name="tts.mp3")
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/tts/voices", methods=["GET"])
def tts_voices_api():
    """List available TTS voices."""
    voices = [
        {
            "id": "female",
            "name": "Sana (أنثى)",
            "voice": TTS_VOICE_FEMALE,
            "language": "ar-JO",
            "description": "صوت أنثى أردني - سانة"
        },
        {
            "id": "male",
            "name": "Taim (ذكر)",
            "voice": TTS_VOICE_MALE,
            "language": "ar-JO",
            "description": "صوت ذكر أردني - تيم"
        }
    ]
    return jsonify({
        "available": _EDGE_TTS_OK,
        "voices": voices,
        "default": "female"
    })


if __name__ == "__main__":
    app.run(port=5000, debug=False)
