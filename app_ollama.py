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
HIGH_CONFIDENCE_THRESHOLD = 0.60   # Return DB answer directly, no LLM (lowered from 0.75 for better recall)
MEDIUM_CONFIDENCE_THRESHOLD = 0.35 # LLM with context needed (lowered from 0.40)
LOW_CONFIDENCE_FABRICATION_THRESHOLD = 0.35  # Block fabricated info below this

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
    "مين": "من",  # Maps "مين" to "من" for normalization
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
    r'([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)',
    re.UNICODE
)

# Also extract bare names (without title) for better matching
_BARE_NAME_PAT = re.compile(
    r'(?:مين\s+)?([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})+)',
    re.UNICODE
)

_known_names_cache: list[str] = []


def _build_known_names():
    """Extract all doctor/professor names from the DB questions AND answers.
    Also extracts names from role-based answers like 'رئيس القسم: الدكتور فلان'."""
    global _known_names_cache
    names = []
    seen_norms = set()
    # Pattern for names after role labels in answers: "رئيس القسم: الدكتور فلان" or "رئيس قسم علم البيانات الدكتور فلان"
    _ROLE_NAME_PAT = re.compile(
        r'(?:رئيس|رئيسه|عميد|عميده)\s*(?:القسم|قسم)?\s*(?:علم\s*البيانات)?\s*:?\s*'
        r'(الدكتور|الأستاذ|الاستاذ|دكتور|استاذ|المهندس|مهندس|الدكتوره|الأستاذة)\s+'
        r'([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)',
        re.UNICODE
    )
    for _, row in df.iterrows():
        q = str(row["Question"])
        a = str(row["Answer"])
        # Extract from Question with titles
        for title, name in _NAME_PAT.findall(q):
            full = f"{title} {name.strip()}"
            norm = normalize_arabic(full)
            if norm not in seen_norms:
                names.append(full)
                seen_norms.add(norm)
        # Also extract from Answer with titles
        for title, name in _NAME_PAT.findall(a):
            full = f"{title} {name.strip()}"
            norm = normalize_arabic(full)
            if norm not in seen_norms:
                names.append(full)
                seen_norms.add(norm)
        # Extract names after role labels in answers
        for title, name in _ROLE_NAME_PAT.findall(a):
            full = f"{title} {name.strip()}"
            norm = normalize_arabic(full)
            if norm not in seen_norms:
                names.append(full)
                seen_norms.add(norm)
        # Extract bare names (the name part after title) for fuzzy matching
        for title, name in _NAME_PAT.findall(q):
            bare = name.strip()
            norm_bare = normalize_arabic(bare)
            if norm_bare not in seen_norms and len(bare.split()) >= 2:
                names.append(bare)
                seen_norms.add(norm_bare)
        for title, name in _NAME_PAT.findall(a):
            bare = name.strip()
            norm_bare = normalize_arabic(bare)
            if norm_bare not in seen_norms and len(bare.split()) >= 2:
                names.append(bare)
                seen_norms.add(norm_bare)
        for title, name in _ROLE_NAME_PAT.findall(a):
            bare = name.strip()
            norm_bare = normalize_arabic(bare)
            if norm_bare not in seen_norms and len(bare.split()) >= 2:
                names.append(bare)
                seen_norms.add(norm_bare)
    _known_names_cache = names
    print(f"📛 Known names indexed: {len(names)}")


def fuzzy_name_match(query: str, threshold: float = 0.45) -> tuple[str | None, float]:
    """Return (best_matching_name, score) or (None, 0) if no good match.
    Lowered default threshold to improve recall for bare-name queries."""
    if not _known_names_cache:
        return None, 0.0
    # Strip leading question words from the query for matching
    clean_q = re.sub(r'^(مين|من|هو|هي)\s+', '', query.strip(), flags=re.UNICODE)
    norm_q = normalize_arabic(clean_q)
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
        # Extra boost: if all query words are contained in the name
        if q_words <= n_words:
            combined = max(combined, 0.85)
        if combined > best_score:
            best_score = combined
            best_name  = name
    if best_score >= threshold:
        return best_name, best_score
    return None, 0.0


# Build known names cache after data is loaded
_build_known_names()


# ==========================================
# 6c. Person extraction & comprehensive info
# ==========================================
def _extract_person_from_query(query: str) -> str | None:
    """Extract a person's name from a query like 'مين قصي ابو عين' or 'احكيلي عن الدكتور فلان'."""
    q = query.strip()

    # Pattern 0: "مين رئيس القسم" / "مين رئيس قسم علم البيانات" — role-based query
    m = re.match(r'^مين\s+(رئيس|رئيسه|عميد|عميده)\s*(.*)', q, re.UNICODE)
    if m:
        role = m.group(1)
        rest = m.group(2).strip()
        # Return the role phrase for DB search (e.g., "رئيس القسم")
        if rest:
            return f"{role} {rest}"
        return role

    # Pattern 1: "مين فلان الفلاني" (who is X Y)
    m = re.match(r'^مين\s+([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)', q, re.UNICODE)
    if m:
        name_part = m.group(1).strip()
        # Don't return role phrases here — they're handled by Pattern 0
        if not re.match(r'^(رئيس|رئيسه|عميد|عميده)', name_part, re.UNICODE):
            return name_part

    # Pattern 2: "من هو/هي فلان" or "من فلان"
    m = re.match(r'^من\s+(?:هو\s+|هي\s+)?([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)', q, re.UNICODE)
    if m:
        return m.group(1).strip()

    # Pattern 3: "احكيلي/حكيلي/أخبرني عن الدكتور فلان"
    m = re.search(r'(?:احكيلي|حكيلي|أخبرني|اخبرني)\s+عن\s+(.*)', q, re.UNICODE)
    if m:
        after_an = m.group(1).strip()
        # Pronoun forms: "ه", "ها", "هو", "هي", "عنه", "عنها" → need history
        if after_an in ("ه", "ها", "هو", "هي", "عنه", "عنها"):
            pass  # fall through to history extraction below
        else:
            return after_an

    # Pattern 4: "احكيلي عنه" / "عنه" — pronoun, need history
    # Pattern 5: Name with title already in query
    m = re.search(r'(الدكتور|الأستاذ|الاستاذ|دكتور|استاذ|المهندس|مهندس)\s+([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)', q, re.UNICODE)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"

    # Pattern 6: Just a bare name (2+ Arabic words)
    m = re.match(r'^([\u0600-\u06ff]{2,}\s+[\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)', q, re.UNICODE)
    if m:
        return m.group(1).strip()

    return None


def _extract_person_from_history(history: list) -> str | None:
    """Try to find a person name in recent conversation history.
    Looks for titled names (الدكتور فلان) AND bare names (قصي ابو عين)."""
    # First try titled names (highest confidence)
    name_pat = re.compile(
        r'(الدكتور|الأستاذ|الاستاذ|دكتور|استاذ|المهندس|مهندس|الدكتوره|الأستاذة)\s+([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)',
        re.UNICODE
    )
    for msg in reversed(history[-10:]):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        for match in name_pat.finditer(content):
            return f"{match.group(1)} {match.group(2).strip()}"

    # Then try known bare names from DB cache
    if _known_names_cache:
        for msg in reversed(history[-10:]):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content", "")
            norm_content = normalize_arabic(content)
            for name in _known_names_cache:
                bare = re.sub(
                    r'^(الدكتور|الأستاذ|الاستاذ|دكتور|استاذ|المهندس|مهندس|الدكتوره|الأستاذة)\s+',
                    '', name
                ).strip()
                if bare and len(bare.split()) >= 2:
                    norm_bare = normalize_arabic(bare)
                    if norm_bare in norm_content:
                        return name  # Return the full name with title
    return None


def _collect_all_person_info(person_name: str) -> list[dict]:
    """Collect ALL database rows that mention this person.
    Uses fuzzy matching to find all relevant Q&A pairs.
    Also handles role-based queries like 'رئيس القسم'."""
    norm_name = normalize_arabic(person_name)
    bare_name = re.sub(
        r'^(الدكتور|الأستاذ|الاستاذ|دكتور|استاذ|المهندس|مهندس|الدكتوره|الأستاذة)\s+',
        '', person_name
    ).strip()
    norm_bare = normalize_arabic(bare_name)

    # Extract individual name words for matching (with and without ال prefix)
    name_words = set(norm_bare.split()) - {"من", "في", "عن", "علي", "مع", "هو", "هي"}
    # Also add stripped-prefix versions for flexible matching
    name_words_stripped = set()
    for w in list(name_words):
        stripped = strip_prefix(w)
        if stripped and len(stripped) >= 2:
            name_words_stripped.add(stripped)
    name_words |= name_words_stripped

    # For role-based queries like "رئيس القسم", keep important words
    is_role_query = any(w in norm_name for w in ["رئيس", "عميد"])
    if is_role_query:
        role_name_words = set(norm_name.split()) - {"من", "في", "عن", "علي", "مع", "هو", "هي"}
        # Add stripped versions of role words
        for w in list(role_name_words):
            stripped = strip_prefix(w)
            if stripped and len(stripped) >= 2:
                role_name_words.add(stripped)
        name_words = role_name_words

    hits = []
    seen_indices = set()

    for idx, row in df.iterrows():
        q = str(row["Question"])
        a = str(row["Answer"])
        norm_q = normalize_arabic(q)
        norm_a = normalize_arabic(a)

        # Pre-compute stripped versions of Q/A words for flexible matching
        q_words_set = set(norm_q.split())
        a_words_set = set(norm_a.split())
        q_words_stripped = {strip_prefix(w) for w in q_words_set if len(strip_prefix(w)) >= 2}
        a_words_stripped = {strip_prefix(w) for w in a_words_set if len(strip_prefix(w)) >= 2}

        # Check various matching strategies
        match_score = 0.0

        # 1. Full name/role in question or answer (exact substring)
        if norm_bare and len(norm_bare.split()) >= 2:
            if norm_bare in norm_q or norm_bare in norm_a:
                match_score = 1.0
        if is_role_query and norm_name:
            if norm_name in norm_q or norm_name in norm_a:
                match_score = 1.0
            # Also check without ال prefix
            norm_name_no_al = re.sub(r'\bال', '', norm_name)
            if norm_name_no_al in norm_q or norm_name_no_al in norm_a:
                match_score = 1.0

        # 2. Name words overlap (with stripped prefix matching)
        if name_words:
            overlap_q = len(name_words & (q_words_set | q_words_stripped)) / len(name_words) if name_words else 0
            overlap_a = len(name_words & (a_words_set | a_words_stripped)) / len(name_words) if name_words else 0
            word_overlap = max(overlap_q, overlap_a)
            match_score = max(match_score, word_overlap)

        # 3. Fuzzy match on the question and answer
        seq_ratio_q = SequenceMatcher(None, norm_name, norm_q).ratio()
        seq_ratio_a = SequenceMatcher(None, norm_name, norm_a).ratio()
        match_score = max(match_score, max(seq_ratio_q, seq_ratio_a) * 0.8)

        # 4. Check if the name (with title) appears
        if normalize_arabic(person_name) in norm_q or normalize_arabic(person_name) in norm_a:
            match_score = 1.0

        # 5. For role queries, also check if answer mentions the role + department
        if is_role_query:
            role_words = {"رئيس", "عميد"}
            for rw in role_words:
                if rw in norm_q or rw in norm_a:
                    # Boost score if the question/answer mentions the department
                    dept_words = {"قسم", "علم", "البيانات", "حاسوب", "معلومات"}
                    dept_match = any(dw in norm_q or dw in norm_a for dw in dept_words)
                    if dept_match:
                        match_score = max(match_score, 0.85)
                    else:
                        # Role mentioned but no department — moderate boost
                        match_score = max(match_score, 0.6)

        # 6. For bare name queries, check if ALL name parts appear in Q or A
        if not is_role_query and norm_bare and len(norm_bare.split()) >= 2:
            bare_parts = set(norm_bare.split())
            if bare_parts <= (q_words_set | q_words_stripped) or bare_parts <= (a_words_set | a_words_stripped):
                match_score = max(match_score, 0.9)

        # 7. Fuzzy match against known names in the cache
        if not is_role_query and _known_names_cache:
            for known in _known_names_cache:
                norm_known = normalize_arabic(known)
                if SequenceMatcher(None, norm_bare, norm_known).ratio() >= 0.7:
                    # Known name is similar — check if this Q/A mentions the known name
                    if norm_known in norm_q or norm_known in norm_a:
                        match_score = max(match_score, 0.85)
                    break

        if match_score >= 0.35 and idx not in seen_indices:
            seen_indices.add(idx)
            hits.append({
                "question": q,
                "answer": a,
                "score": match_score,
            })

    # Sort by score descending
    hits.sort(key=lambda x: x["score"], reverse=True)
    return hits


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
    "مين",  # "مين" = who, should be academic when followed by a name
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


# Patterns that specifically mean "tell me everything about" a person
_TELL_ABOUT_PATTERNS = [
    r'احكيلي\s+عن',
    r'حكيلي\s+عن',
    r'أخبرني\s+عن',
    r'اخبرني\s+عن',
    r'عرفه',
    r'عرفني',
    r'مين\s+هو\s+[\u0600-\u06ff]',
    r'مين\s+هي\s+[\u0600-\u06ff]',
    r'مين\s+(رئيس|رئيسه|عميد|عميده)',  # "مين رئيس القسم"
    r'^مين\s+[\u0600-\u06ff]{2,}\s+[\u0600-\u06ff]{2,}',  # "مين قصي ابو عين" — bare name who-query
]


def is_tell_about_request(query: str) -> bool:
    """True if user wants ALL info about a person (احكيلي عنه, مين هو فلان)."""
    q = query.strip()
    return _matches(q, _TELL_ABOUT_PATTERNS) or _matches(normalize(q), _TELL_ABOUT_PATTERNS)


def is_expansion_request(query: str) -> bool:
    """True if the user is asking for more details about the previous answer."""
    q = query.strip()
    # Do NOT treat "احكيلي عنه" as expansion — it's a "tell about" request
    if is_tell_about_request(q):
        return False
    if _matches(q, EXPANSION_PATTERNS):
        return True
    # Very short queries with no academic keywords after mapping
    norm = normalize(q)
    if len(norm.split()) <= 2 and not any(kw in norm for kw in ACADEMIC_KEYWORDS):
        return True
    return False


# Patterns that indicate the user is asking about a person by name (without title prefix)
_PERSON_QUERY_PATTERNS = [
    r'^مين\s+[\u0600-\u06ff]{2,}',       # "مين قصي" / "مين فلان"
    r'^من\s+[\u0600-\u06ff]{2,}',       # "من هو فلان"
    r'احكيلي\s+عن',                      # "احكيلي عنه" / "احكيلي عن الدكتور"
    r'حكيلي\s+عن',                       # "حكيلي عنه"
    r'أخبرني\s+عن',                      # "أخبرني عنه"
    r'اخبرني\s+عن',                      # "اخبرني عنه"
    r'عرفه\s*$',                         # "عرفه"
    r'عرفني\s*',                         # "عرفني"
    r'مين\s+هو',                         # "مين هو"
    r'مين\s+هي',                         # "مين هي"
]


def _is_person_query(raw_query: str) -> bool:
    """Check if the query is asking about a person, even without title prefix.
    Uses fuzzy matching to catch partial name matches."""
    q = raw_query.strip()
    norm_q = normalize(q)
    # Check patterns
    if _matches(q, _PERSON_QUERY_PATTERNS) or _matches(norm_q, _PERSON_QUERY_PATTERNS):
        return True
    # Check if query contains a known name (without title)
    if _known_names_cache:
        norm_q_clean = normalize_arabic(q)
        for name in _known_names_cache:
            bare_name = re.sub(r'^(الدكتور|الأستاذ|الاستاذ|دكتور|استاذ|المهندس|مهندس|الدكتوره|الأستاذة)\s+', '', name).strip()
            if bare_name and len(bare_name.split()) >= 2:
                norm_bare = normalize_arabic(bare_name)
                if norm_bare in norm_q_clean or norm_q_clean in norm_bare:
                    return True
                # Check word overlap with bare name
                q_words = set(norm_q_clean.split())
                n_words = set(norm_bare.split())
                if len(q_words & n_words) >= 2:
                    return True
        # Fuzzy name check as last resort
        fuzzy_match, fuzzy_score = fuzzy_name_match(q, threshold=0.50)
        if fuzzy_match:
            return True
    return False


def classify_intent(raw_query: str) -> str:
    norm_q = normalize(raw_query)
    if _matches(raw_query, META_PATTERNS) or _matches(norm_q, META_PATTERNS):
        return "meta"
    # Check person query BEFORE conversational (so "مين قصي" isn't labeled conversational)
    if _is_person_query(raw_query):
        return "academic"
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
                  "الدكتورة", "الاستاذة", "دكتورة",
                  "رئيس", "رئيسه", "عميد", "عميده"}


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
    ("بوابة الطالب الإلكترونية",
     "https://student.just.edu.jo"),
    ("كلية تكنولوجيا المعلومات وعلوم الحاسوب",
     "https://www.just.edu.jo/FacultiesAndDepartments/FacultyofComputer/Pages/Home.aspx"),
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
    """Clean LLM output: remove noise but PRESERVE URLs, emails, markdown links."""
    # 1. Remove disclaimers about Arabic ability
    text = re.sub(
        r'\*?\*?ملاحظة\*?\*?[:\s]+[^\n]*(أتحدث|الفصح|لغت|محدودي|بطلاقة)[^\n]*\n?',
        '', text, flags=re.IGNORECASE
    )
    # 2. Remove Chinese/CJK characters
    text = re.sub(
        r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\uff00-\uffef]+',
        '', text
    )
    # 3. Remove fabricated "about the doctor" sections
    text = re.sub(
        r'\*?\*?حول\s+(الدكتور|الأستاذ|المهندس)[^\n]*\n[^\n]*\n?',
        '', text, flags=re.IGNORECASE
    )
    # 4. Remove <think ...> tags that deepseek-r1 sometimes emits
    text = re.sub(r'<think.*?>', '', text, flags=re.DOTALL)
    text = re.sub(r'</think\s*>', '', text, flags=re.DOTALL)

    # ── CRITICAL: Protect URLs, emails, and markdown links BEFORE any cleaning ──
    _PROTECTED = []

    def _stash(m):
        """Replace a matched URL/email/link with a placeholder to protect it."""
        _PROTECTED.append(m.group(0))
        return f"\x00PROT{len(_PROTECTED) - 1}\x00"

    # Stash markdown links: [label](url)
    text = re.sub(r'\[[^\]]*\]\([^)]+\)', _stash, text)
    # Stash full URLs (http/https)
    text = re.sub(r'https?://[^\s)\]>"\']+', _stash, text)
    # Stash email addresses
    text = re.sub(r'[\w.\-+]+@[\w.\-]+\.\w{2,}', _stash, text)

    # 5. Now clean tokens — remove pure Latin noise (but NOT protected items)
    lines   = text.split('\n')
    cleaned = []
    for line in lines:
        tokens = line.split()
        result = []
        for tok in tokens:
            # Skip protected placeholders
            if '\x00PROT' in tok:
                result.append(tok)
                continue
            core       = re.sub(r'^[\W_]+|[\W_]+$', '', tok, flags=re.UNICODE)
            has_arabic = bool(re.search(r'[\u0600-\u06ff]', tok))
            has_latin  = bool(re.search(r'[a-zA-Z]', tok))
            has_email  = '@' in tok
            is_url     = tok.startswith(('http', 'www.'))
            is_pure_latin_noise = (
                bool(re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', core))
                and not has_arabic and not has_email and not is_url
                and len(core) >= 2
            )
            is_mixed_noise = has_arabic and has_latin and not has_email and not is_url
            if not is_pure_latin_noise and not is_mixed_noise:
                result.append(tok)
        cleaned.append(' '.join(result))
    joined = re.sub(r'[ \t]+', ' ', '\n'.join(cleaned)).strip()

    # 6. Remove isolated Latin fragments NOT inside placeholders
    #    (We must skip protected content this time)
    _KEEP_TERMS = {
        'pdf', 'vpn', 'wifi', 'wi', 'fi', 'toefl', 'ielts',
        'ds', 'cs', 'it', 'ai', 'iot', 'url', 'edu', 'jo', 'com',
        'org', 'net', 'ac', 'gov', 'http', 'https', 'www', 'just',
        'mail', 'web', 'home', 'staff', 'data', 'science', 'page',
        'aspx', 'html', 'php', 'index', 'login', 'portal', 'reg',
        'student', 'system', 'office', 'dept', 'fac', 'admin',
    }

    def _kill_latin_fragment(m):
        word = m.group(0)
        # Don't touch anything near a protected placeholder
        pos = m.start()
        if '\x00PROT' in joined[max(0, pos - 20):pos + len(word) + 20]:
            return word
        if re.search(r'\d', word):
            return word
        if '@' in word or '://' in word:
            return word
        if word.lower() in _KEEP_TERMS:
            return word
        return ' '

    joined = re.sub(r'[a-zA-Z][a-zA-Z0-9\-\.]*', _kill_latin_fragment, joined)

    # 7. Remove Arabic+Latin mixed noise (e.g. "وero", "مكتبoffice")
    #    But NOT if it contains a protected placeholder
    def _kill_mixed(m):
        if '\x00PROT' in m.group(0):
            return m.group(0)
        return ' '
    joined = re.sub(
        r'(?<![\w@.])([\u0600-\u06ff]+[a-zA-Z]+|[a-zA-Z]+[\u0600-\u06ff]+)(?![\w@.])',
        _kill_mixed, joined
    )

    # 8. Restore all protected URLs, emails, and markdown links
    for i, original in enumerate(_PROTECTED):
        joined = joined.replace(f"\x00PROT{i}\x00", original)

    return re.sub(r'[ \t]+', ' ', joined).strip()


BOT_PERSONA = (
    "أنت مرشد أكاديمي ذكي اسمه 'المرشد الأكاديمي'، "
    "متخصص في قسم علم البيانات بجامعة العلوم والتكنولوجيا الأردنية.\n"
    "CRITICAL: Reply in Arabic ONLY. NEVER use English words in your Arabic response. Use Arabic equivalents always.\n"
    "EXCEPTION: URLs and email addresses MUST be kept in their original Latin/English form. NEVER translate or remove them.\n"
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
    "\n"
    "═══ معلومات مهمة عن الجامعة — استخدمها عند الحاجة ═══\n"
    "- الموقع الرسمي لجامعة العلوم والتكنولوجيا الأردنية: https://www.just.edu.jo\n"
    "- بوابة الطالب الإلكترونية: https://student.just.edu.jo\n"
    "- صفحة أعضاء هيئة التدريس - قسم علم البيانات: "
    "https://www.just.edu.jo/FacultiesAndDepartments/FacultyofComputer/Departments/DataScience/Pages/Staff.aspx\n"
    "- صفحة قسم علم البيانات: "
    "https://www.just.edu.jo/FacultiesAndDepartments/FacultyofComputer/Departments/DataScience/Pages/Home.aspx\n"
    "═══════════════════════════════════════════════════\n"
    "\n"
    "CRITICAL: When you mention a website, ALWAYS include the FULL URL as a clickable markdown link.\n"
    "Example: [الموقع الرسمي للجامعة](https://www.just.edu.jo)\n"
    "NEVER write an empty link like [ ]( ) or leave the URL blank.\n"
    "CRITICAL: When you mention an email, ALWAYS include the FULL email address.\n"
    "Example: الإيميل: name@just.edu.jo\n"
    "NEVER write just '@' without the full email address.\n"
    "\n"
    "عند الإجابة:\n"
    "• رتّب الإجابة في نقاط واضحة ومباشرة.\n"
    "• لا تختلق أي معلومة غير موجودة في السياق.\n"
    "• أجب بأسلوب ودي، واضح، ومنظم يركز على مساعدة الطالب.\n"
    "• تقع جامعة العلوم والتكنولوجيا الأردنية في مدينة إربد وليس في عمان.\n"
    "• عند ذكر موقع إلكتروني، ضع الرابط الكامل بصيغة markdown قابل للنقر.\n"
    "• عند ذكر إيميل، اكتب العنوان كاملاً ولا تتركه فارغاً.\n"
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
            "3. عند ذكر رابط → اكتبه بصيغة markdown: [النص](https://الرابط) مع الرابط الكامل.\n"
            "4. عند ذكر إيميل → اكتب العنوان كاملاً مثل: name@just.edu.jo\n"
            "5. لا تترك رابط أو إيميل فارغاً أبداً. إذا لم تعرفه اكتب 'مجهول'.\n"
            + followup_rule +
            strict_rule +
            "8. لا تضف 'حول الدكتور' أو أي سيرة ذاتية لم تكن في السياق.\n"
            "9. MUST NOT invent, guess, or fabricate ANY information not in the context.\n"
            "10. If not 100% certain about a fact, write 'مجهول' instead.\n"
            f"{db_context}\n"
            "══════════════════════════════════════\n"
        )

    system = (
        BOT_PERSONA
        + "يمكنك الإجابة على أي سؤال: أكاديمي أو اجتماعي أو عام.\n"
        + "• إذا سأل عن شخص → اذكر فقط إيميله ومكتبه من السياق ورقم هاتفه إن وجد. لا تكتب سيرة ذاتية.\n"
        + "• إذا جاء السؤال بضمير → أجب مباشرةً بالمعلومة من السياق.\n"
        + "• إذا كان السؤال أكاديمياً وليس في السياق إجابة → اكتب كلمة مجهول فقط.\n"
        + "لا تختلق أي معلومة غير موجودة في السياق.\n"
        + "IMPORTANT: When including a website URL, format it as: [اسم الموقع](https://full-url.com)\n"
        + "IMPORTANT: When including an email, write the FULL address: name@just.edu.jo\n"
        + "IMPORTANT: NEVER leave a URL or email blank or write just '@'. If unknown, write 'مجهول'.\n"
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
    if re.search(r'(مين هو|من هو|بتعرف|تعرف|اعرفني|احكيلي عن|حكيلي عن|أخبرني عن|اخبرني عن|عرفني|عرفه|مين رئيس)', q):
        return "name_info"
    if re.search(r'(رئيس|رئيسه|عميد|عميده)', q):
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
# ==========================================
# 14b. Post-Processing: Fix Broken Links & Empty Emails
# ==========================================
# Known JUST URLs to inject when the LLM leaves a placeholder
_JUST_URL_MAP = {
    "الموقع الرسمي": "https://www.just.edu.jo",
    "الموقع الرسمي للجامعة": "https://www.just.edu.jo",
    "بوابة الطالب": "https://student.just.edu.jo",
    "بوابة الطالب الإلكترونية": "https://student.just.edu.jo",
    "صفحة التسجيل": "https://student.just.edu.jo",
    "نظام التسجيل": "https://student.just.edu.jo",
    "أعضاء هيئة التدريس": "https://www.just.edu.jo/FacultiesAndDepartments/FacultyofComputer/Departments/DataScience/Pages/Staff.aspx",
    "قسم علم البيانات": "https://www.just.edu.jo/FacultiesAndDepartments/FacultyofComputer/Departments/DataScience/Pages/Home.aspx",
}

# Registration-related keywords that should trigger URL injection
_REGISTRATION_KEYWORDS = [
    "تسجيل", "سجل", "اسجل", "يسجل", "التسجيل", "مواد", "مادة",
    "جدول", "إضافة", "حذف", "سحب", "drop", "add",
]


def _fix_broken_links(text: str) -> str:
    """Post-process: fix empty/broken markdown links and empty emails."""
    # Fix markdown links with empty URL: [text]( ) or [text](://) or [text]()
    def _fix_empty_link(m):
        label = m.group(1).strip()
        url = m.group(2).strip()
        # If URL is empty or broken, try to find the real URL
        if not url or url in ("://", ":// ", "/", " ", "://)"):
            # Search for a matching URL in our known map
            for key, real_url in _JUST_URL_MAP.items():
                if key in label or any(k in label for k in ["الموقع", "الجامعة", "الرسمي", "التسجيل", "البوابة"]):
                    return f"[{label}]({real_url})" if label else f"[{key}]({real_url})"
            # Check surrounding context for clues
            for key, real_url in _JUST_URL_MAP.items():
                if key in text[:m.start()] or key in text[m.end():]:
                    return f"[{label or key}]({real_url})"
            # Default to JUST homepage
            return f"[{label or 'الموقع الرسمي'}]({JUST_BASE_URL})"
        return m.group(0)

    text = re.sub(r'\[([^\]]*)\]\(([^)]*)\)', _fix_empty_link, text)

    # Fix empty emails: "الإيميل: @" or "البريد: @" → replace with مجهول
    text = re.sub(r'((?:الإيميل|البريد|ايميل|إيميل|Email)\s*:\s*)@(?!\S)', r'\1مجهول', text)
    # Also fix standalone "@" at end of line
    text = re.sub(r'@\s*$', 'مجهول', text, flags=re.MULTILINE)

    # Fix broken email patterns like "name@" without domain at end of line
    def _fix_broken_email(m):
        prefix = m.group(1)
        local = m.group(2)
        return f"{prefix}{local}@just.edu.jo" if local else f"{prefix}مجهول"

    return text


def _inject_just_url_for_query(text: str, query: str) -> str:
    """Inject JUST website URL when the query is about registration but no link was provided."""
    norm_q = normalize(query)
    is_reg_q = any(kw in norm_q for kw in _REGISTRATION_KEYWORDS)

    if not is_reg_q:
        return text

    # Check if text already contains a URL
    if re.search(r'https?://', text):
        return text

    # No URL found but it's a registration question → inject the student portal URL
    url_line = f"- [بوابة الطالب الإلكترونية](https://student.just.edu.jo)"
    just_line = f"- [الموقع الرسمي للجامعة](https://www.just.edu.jo)"

    # Add after the last line of the answer
    if "**المصادر:**" in text:
        # Insert before sources section
        text = text.replace("**المصادر:**", f"🔗 **روابط مفيدة:**\n{url_line}\n{just_line}\n\n**المصادر:**")
    else:
        text = text.rstrip() + f"\n\n🔗 **روابط مفيدة:**\n{url_line}\n{just_line}"

    return text


def _post_process_answer(text: str, query: str) -> str:
    """Final post-processing: fix broken links, inject URLs, ensure professional output."""
    text = _fix_broken_links(text)
    text = _inject_just_url_for_query(text, query)
    return text


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
                           "شو عمله", "ايش عمله", "شو تخصصه", "ايش تخصصه",
                           "احكيلي عنه", "حكيلي عنه"}
        is_followup = any(w in query for w in FOLLOW_UP_WORDS)

        # 3b. Enrich ONLY on true pronoun follow-ups to resolve person name from history
        if is_followup and history:
            enriched = _build_followup_search_query(query, history)
            if enriched != query:
                search_query = enriched
                print(f"🔗 Follow-up enriched: {search_query[:80]}")

        # ── 3e. "Tell me about" handling: collect ALL DB rows for a person ──
        if is_tell_about_request(query):
            # First, try to identify the person from the query or history
            # Handle pronoun cases like "احكيلي عنه" (tell me about him)
            person_name = _extract_person_from_query(query)
            if not person_name or person_name in ("عنه", "عنها", "هو", "هي", "ه", "ها"):
                person_name = _extract_person_from_history(history)
            # If still no name, try fuzzy match against known names
            if not person_name:
                fuzzy_match, fuzzy_score = fuzzy_name_match(query)
                if fuzzy_match:
                    person_name = fuzzy_match
            if person_name:
                print(f"👤 Tell-about request for: {person_name}")
                # Collect ALL matching rows from the DB
                all_person_hits = _collect_all_person_info(person_name)
                # If no hits found, try fuzzy name matching to find the right person
                if not all_person_hits:
                    fuzzy_match, fuzzy_score = fuzzy_name_match(person_name)
                    if fuzzy_match:
                        print(f"🔤 Tell-about fuzzy redirect: {person_name} → {fuzzy_match} (score={fuzzy_score:.2f})")
                        person_name = fuzzy_match
                        all_person_hits = _collect_all_person_info(person_name)
                if all_person_hits:
                    # Build a comprehensive answer with ALL info
                    combined_ctx = "[ 👤 معلومات شاملة عن الشخص من قاعدة البيانات ]\n"
                    for i, hit in enumerate(all_person_hits, 1):
                        combined_ctx += f"\n{i}. س: {hit['question']}\n   ج: {hit['answer']}\n"
                    # Also try PDF and web
                    pdf_hits = search_pdf(person_name, top_k=4)
                    if pdf_hits:
                        combined_ctx += "\n[ 📄 معلومات إضافية من الوثائق ]\n"
                        for ph in pdf_hits:
                            prefix = f"[{ph['article_ref']}] " if ph.get('article_ref') else ""
                            combined_ctx += f"{prefix}{ph['text']}\n"
                    web_snippets_tell = []
                    try:
                        web_results_tell = search_just_website(person_name)
                        web_snippets_tell = [r["snippet"] for r in web_results_tell]
                        if web_snippets_tell:
                            combined_ctx += "\n[ 🌐 معلومات من الموقع الرسمي ]\n"
                            for s in web_snippets_tell:
                                combined_ctx += f"- {s}\n"
                    except Exception:
                        pass
                    # Ask LLM to synthesize a comprehensive answer
                    llm_ans = ollama_chat(
                        f"أخبرني بكل المعلومات المتاحة عن: {person_name}",
                        history,
                        combined_ctx,
                        is_followup=False,
                        expand_previous=False
                    )
                    if llm_ans and not is_unknown_response(llm_ans):
                        return _post_process_answer(llm_ans, query), "medium"
                    # Fallback: just concatenate all DB answers
                    fallback = "\n".join(f"• {h['answer']}" for h in all_person_hits[:8])
                    return _post_process_answer(fallback, query), "medium"
                else:
                    # Person identified but no DB hits — try web search
                    print(f"👤 Person '{person_name}' found but no DB hits — trying web search")
                    try:
                        web_results_tell = search_just_website(person_name)
                        if web_results_tell:
                            web_ctx = "\n".join(f"- {r['snippet']}" for r in web_results_tell)
                            llm_ans = ollama_chat(
                                f"أخبرني بكل المعلومات المتاحة عن: {person_name}",
                                history,
                                f"[ 🌐 معلومات من الموقع الرسمي ]\n{web_ctx}",
                                is_followup=False
                            )
                            if llm_ans and not is_unknown_response(llm_ans):
                                return _post_process_answer(llm_ans, query), "medium"
                    except Exception:
                        pass

        hits = retrieve_top_k(search_query, top_k=5)
        print("🔍 Top hits:")
        for h in hits:
            print(f"   score={h['score']:.3f} | {h['question'][:60]}")
        best_score = hits[0]["score"] if hits else -1
        best_ans   = hits[0]["answer"] if hits else None

        norm_q    = normalize(query)
        is_name_q = any(w in norm_q for w in _NAME_TRIGGERS) or _is_person_query(query)

        # ── 3c. Fuzzy name suggestion (typo correction) ───────────────
        # Also try fuzzy match when user asks about a person without title prefix
        if is_name_q and best_score < 0.50:
            fuzzy_match, fuzzy_score = fuzzy_name_match(query)
            # Also try extracting just the name part (after مين)
            if not fuzzy_match or fuzzy_score < 0.55:
                extracted_name = _extract_person_from_query(query)
                if extracted_name:
                    fuzzy_match2, fuzzy_score2 = fuzzy_name_match(extracted_name)
                    if fuzzy_match2 and fuzzy_score2 > fuzzy_score:
                        fuzzy_match, fuzzy_score = fuzzy_match2, fuzzy_score2
            if fuzzy_match and fuzzy_score >= 0.45:
                print(f"🔤 Fuzzy match: {fuzzy_match} (score={fuzzy_score:.2f})")
                # Re-search with corrected name
                corrected_hits = retrieve_top_k(fuzzy_match, top_k=5)
                # Also collect ALL person info for better answer
                all_person_info = _collect_all_person_info(fuzzy_match)
                corrected_score = corrected_hits[0]["score"] if corrected_hits else 0
                if all_person_info and len(all_person_info) > 1:
                    # Multiple DB entries found — give comprehensive answer
                    combined_ctx = "[ 👤 معلومات شاملة عن الشخص من قاعدة البيانات ]\n"
                    for i, hit in enumerate(all_person_info, 1):
                        combined_ctx += f"\n{i}. س: {hit['question']}\n   ج: {hit['answer']}\n"
                    llm_ans = ollama_chat(
                        f"أخبرني بكل المعلومات المتاحة عن: {fuzzy_match}",
                        history,
                        combined_ctx,
                        is_followup=False,
                        expand_previous=False
                    )
                    if llm_ans and not is_unknown_response(llm_ans):
                        return (
                            f"هل تقصد **{fuzzy_match}**؟ 🔍\n\n"
                            f"{llm_ans}"
                        ), "medium"
                if corrected_score >= 0.35:
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
            # Inject JUST URLs for registration-related questions
            reply = _inject_just_url_for_query(reply, query)
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
            for i, s in enumerate(web_snippets):
                url = web_results[i]["url"] if i < len(web_results) else ""
                if url:
                    ctx_parts.append(f"- {s} (الرابط: {url})")
                else:
                    ctx_parts.append(f"- {s}")
            # Always include JUST main URLs for reference
            ctx_parts.append(f"روابط مهمة: الموقع الرسمي: {JUST_BASE_URL} | بوابة الطالب: https://student.just.edu.jo")

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
        # Determine confidence level more granularly
        if best_score >= HIGH_CONFIDENCE_THRESHOLD:
            confidence = "high"
        elif best_score >= MEDIUM_CONFIDENCE_THRESHOLD:
            confidence = "medium"
        else:
            confidence = "medium"  # Even low scores get medium if LLM can answer
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
    user_query = data.get("query", "")
    reply, confidence = ask_smart_bot(user_query, data.get("history", []))
    # Apply post-processing: fix broken links, inject URLs for registration queries
    reply = _post_process_answer(reply, user_query)
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
