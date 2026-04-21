import sys
from flask import Flask, request, jsonify, send_from_directory
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
import fitz   # PyMuPDF  — pip install pymupdf
import io

app = Flask(__name__)
CORS(app)

# ==========================================
# 1. Configuration
# ==========================================
# ── Model options ─────────────────────────────────────────────────
# llama3:latest → general-purpose, was used in app_ollama
# llama3.2:1b   → too small, poor Arabic, use only if RAM < 4GB
# llama3.1:8b   → good Arabic, needs ~8GB RAM
# aya:8b        → Arabic-focused, needs ~8GB RAM  ← RECOMMENDED (from app_ollama2)
# aya:35b       → best quality, needs ~24GB RAM
OLLAMA_MODEL = "aya:8b"         # ← change to "llama3:latest" if aya is not installed

# ── Excel file ────────────────────────────────────────────────────
# app_ollama  used: "project_data.xlsx"
# app_ollama2 used: "project_data_V2.xlsx"
EXCEL_FILE   = "project_data_V2.xlsx"  # ← change if needed

OLLAMA_HOST      = "http://localhost:11434"
THRESHOLD_DIRECT = 0.80   # skip LLM for extremely high-confidence name/email hits
JUST_BASE_URL    = "https://www.just.edu.jo"

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
# Add URLs (publicly accessible) OR local file paths.
# The JUST server blocks bots (403), so download those PDFs manually
# and add their local paths here.  Examples:
#   "regulations.pdf"                         ← file in same folder as this script
#   "C:/Users/user/Downloads/3.pdf"           ← absolute Windows path
#   "https://example.com/public.pdf"          ← publicly accessible URL

PDF_SOURCES = [
    # ← add your PDF paths/URLs here, e.g.:
    # "3.pdf",
    "https://www.just.edu.jo/aboutjust/RegulationsTemp/41%20%D9%86%D8%B8%D8%A7%D9%85%20%D8%AA%D8%A3%D8%AF%D9%8A%D8%A8%20%D8%A7%D9%84%D8%B7%D9%84%D8%A8%D8%A9/2.pdf"
]

# Smaller chunks = more precise retrieval (app_ollama2 improvement over app_ollama)
PDF_CHUNK_SIZE    = 400   # characters per chunk  (was 600 in app_ollama)
PDF_CHUNK_OVERLAP = 80    # overlap between chunks (was 100 in app_ollama)

_pdf_chunks: list[dict] = []   # [{text, source, page}]
_pdf_index  = None             # separate FAISS index for PDF chunks
_pdf_ready  = False


def _extract_pdf_text(source: str) -> list[dict]:
    """Load a PDF from URL or local path → list of {text, source, page} per page."""
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
    """
    Detect a single garbled OCR line.
    Signal: Arabic words are mostly 1-2 char fragments (split letters).
    Threshold: avg word length < 3.2 OR > 45% of words are ≤2 chars.
    Lines with fewer than 3 Arabic tokens are not judged (too short).
    """
    tokens  = line.split()
    arabic  = [t for t in tokens if re.search(r'[\u0600-\u06ff]', t)]
    if len(arabic) < 3:
        return False
    short   = sum(1 for t in arabic if len(t) <= 2)
    avg_len = sum(len(t) for t in arabic) / len(arabic)
    return avg_len < 3.2 or (short / len(arabic)) > 0.45


def _clean_page_text(text: str) -> str:
    """Strip garbled OCR lines from a page, keeping all clean lines intact."""
    cleaned = [ln for ln in text.splitlines() if not _is_garbled_line(ln)]
    return "\n".join(cleaned).strip()


def _chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Clean each page (remove garbled OCR lines), then split into
    overlapping chunks for FAISS retrieval.
    """
    chunks = []
    for p in pages:
        text = _clean_page_text(p["text"])
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
    """Encode all chunks and build a FAISS index."""
    if not chunks:
        return None
    texts  = [c["text"] for c in chunks]
    emb    = sentence_model.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
    emb_np = emb.cpu().numpy().astype("float32")
    idx    = faiss.IndexFlatIP(emb_np.shape[1])
    idx.add(emb_np)
    return idx


def _extract_article_ref(text: str) -> str | None:
    """
    Automatically extract article/paragraph/clause references from PDF text.
    Example outputs: "المادة 15", "الفقرة 3", "البند أ"
    (Feature from app_ollama2 — not present in app_ollama)
    """
    patterns = [
        r'(المادة\s+\d+)',
        r'(الفقرة\s+[\dأ-ي]+)',
        r'(البند\s+[\dأ-يa-zA-Z]+)',
        r'(القسم\s+\d+)',
        r'(رقم\s+\d+)',
    ]
    for pat in patterns:
        match = re.search(pat, text)
        if match:
            return match.group(1)
    return None


def load_pdfs():
    """Call once at startup — loads all configured PDFs into memory."""
    global _pdf_chunks, _pdf_index, _pdf_ready
    if not PDF_SOURCES:
        print("ℹ️  No PDF sources configured (PDF_SOURCES is empty).")
        return
    pages = []
    for src in PDF_SOURCES:
        pages.extend(_extract_pdf_text(src))
    _pdf_chunks = _chunk_pages(pages)
    _pdf_index  = _build_pdf_index(_pdf_chunks)
    _pdf_ready  = bool(_pdf_chunks)
    print(f"📄 PDF index ready: {len(_pdf_chunks)} chunks from {len(PDF_SOURCES)} source(s)")


def search_pdf(query: str, top_k: int = 4) -> list[dict]:
    """Return the top-k most relevant PDF chunks for a query, with article refs."""
    if not _pdf_ready or _pdf_index is None:
        return []
    norm  = normalize(query)
    q_emb = (sentence_model.encode([norm], normalize_embeddings=True, convert_to_tensor=True)
             .cpu().numpy().astype("float32"))
    k          = min(top_k, len(_pdf_chunks))
    dists, ids = _pdf_index.search(q_emb, k)
    results = []
    for rank in range(k):
        score = float(dists[0][rank])
        if score >= 0.35:
            chunk       = _pdf_chunks[ids[0][rank]]
            article_ref = _extract_article_ref(chunk["text"])
            results.append({
                "score":       score,
                "text":        chunk["text"],
                "source":      chunk["source"],
                "page":        chunk["page"],
                "article_ref": article_ref,
            })
    return results


def format_pdf_source(source: str, page: int) -> str:
    """Return a clickable Markdown link that opens the PDF directly at the right page."""
    if source.startswith("http"):
        decoded  = unquote(source)
        parts    = decoded.rstrip('/').split('/')
        filename = parts[-1]                          # e.g. "2.pdf"
        folder   = parts[-2] if len(parts) >= 2 else ""

        # Prefer the folder name (it's usually the meaningful document title)
        label_raw = re.sub(r'^\d+\s*', '', folder).strip()
        # Fall back to filename if folder name is too short or empty
        if len(label_raw) < 3:
            label_raw = re.sub(r'\.pdf$', '', filename, flags=re.IGNORECASE).strip()
        if len(label_raw) > 50:
            label_raw = label_raw[:47] + "…"

        label = f"{label_raw} — صفحة {page}"   # no 📄 here; sources block adds it
        url   = f"{source}#page={page}"
        return f"[{label}]({url})"
    # Local file — no hyperlink, just show name + page
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
    df = (pd.concat(frames, ignore_index=True)
            .dropna(subset=["Question", "Answer"]))
    df = df[df["Answer"].astype(str).str.strip() != ""].reset_index(drop=True)
    print(f"✅ Loaded {len(df)} rows from {len(xl.sheet_names)} sheet(s)")
    return df


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

# PDFs loaded AFTER sentence_model is ready (order matters)
load_pdfs()

# ==========================================
# 5. Ollama Health (cached, with retry)
# ==========================================
# app_ollama2 improvement: 10s cache (was 30s), 2 retries, 10s timeout (was 5s)
_ollama_cache = {"ok": False, "ts": 0}


def is_ollama_available():
    now = time.time()
    if now - _ollama_cache["ts"] < 10:
        return _ollama_cache["ok"]
    for attempt in range(2):
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
    "شو": "ما", "وين": "أين", "اين": "أين", "كيفش": "كيف",
    "شلون": "كيف", "ليش": "لماذا", "هيك": "هكذا",
    "بدي": "أريد", "بقدر": "أستطيع", "لازم": "يجب",
    "اسجل": "تسجيل", "سجل": "تسجيل", "يسجل": "تسجيل",
    "للجامعه": "في الجامعة", "للجامعة": "في الجامعة",
    "ايش": "ما", "إيش": "ما", "وش": "ما",
    "زين": "جيد", "مو": "ليس",
    # Tell-me noise verbs → drop entirely so they don't pollute name queries
    "احكيلي": "", "حكيلي": "", "گلي": "", "قلي": "",
    "خبرني": "", "أخبرني": "", "اخبرني": "",
    "اعطيني": "", "عطيني": "",
    # Spelling variants
    "ايميل": "إيميل", "الايميل": "الإيميل",
    "دكتوره": "دكتور", "الدكتوره": "الدكتور",
}

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
    # "بتقدر تساعدني / تعمل / تساعد" — help capability questions
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
]


def _matches(text: str, patterns: list) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)


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
                  "الدكتوره", "الاستاذه", "المهندس", "مهندس"}


def name_overlap(query: str, stored_question: str) -> float:
    """
    Dedicated name matching — searches for first/last name match separately.
    Uses partial matching since Arabic names can appear in different forms.
    (Feature from app_ollama2 — not present in app_ollama)
    """
    q_words = {w for w in normalize_arabic(query).split() if len(w) > 3}
    s_words = {w for w in normalize_arabic(stored_question).split() if len(w) > 3}
    if not q_words or not s_words:
        return 0.0
    matches = sum(
        1 for qw in q_words
        if any(qw in sw or sw in qw for sw in s_words)
    )
    return matches / len(q_words)


def retrieve_top_k(query: str, top_k: int = 5) -> list[dict]:
    norm          = normalize(query)
    is_name_query = any(w in norm for w in _NAME_TRIGGERS)
    # app_ollama2 improvement: stronger Jaccard weight for name queries (0.85 vs 0.80)
    sem_w  = 0.15 if is_name_query else 0.50
    olap_w = 0.85 if is_name_query else 0.50

    q_emb = (sentence_model.encode([norm], normalize_embeddings=True,
                                   convert_to_tensor=True)
             .cpu().numpy().astype("float32"))
    k = min(max(top_k * 2, 10), len(df))
    distances, indices = faiss_index.search(q_emb, k)

    scored = []
    for rank in range(k):
        sem      = float(distances[0][rank])
        ri       = indices[0][rank]
        stored_q = df.iloc[ri]["Question"]
        olap     = word_overlap(norm, stored_q)
        if is_name_query:
            name_score = name_overlap(norm, stored_q)
            olap = max(olap, name_score)   # take the better of the two
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
}

JUST_PAGES = [
    ("صفحة أعضاء هيئة التدريس - قسم علم البيانات",
     "https://www.just.edu.jo/FacultiesAndDepartments/FacultyofComputer/Departments/DataScience/Pages/Staff.aspx"),
    ("قسم علم البيانات - جامعة العلوم والتكنولوجيا",
     "https://www.just.edu.jo/FacultiesAndDepartments/FacultyofComputer/Departments/DataScience/Pages/Home.aspx"),
    ("الموقع الرسمي لجامعة العلوم والتكنولوجيا الأردنية",
     "https://www.just.edu.jo"),
]


def _fetch_text(url: str, timeout: int = 8) -> str | None:
    try:
        r = requests.get(url, headers=_WEB_HEADERS, timeout=timeout)
        if r.status_code != 200:
            return None
        text = re.sub(r'<[^>]+>', ' ', r.text)
        text = re.sub(r'&[a-z]+;', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()
    except Exception as e:
        print(f"⚠️ fetch: {e}")
        return None


def search_just_website(query: str) -> str | None:
    print(f"🌐 Searching JUST for: {query[:60]}")
    snippets     = []
    source_url   = None
    source_label = None

    q_words = [w for w in normalize(query).split()
               if len(w) > 3 and w not in STOP_WORDS]

    for label, url in JUST_PAGES:
        text = _fetch_text(url)
        if not text:
            continue
        found_snippet = None
        for qw in q_words:
            idx = text.find(qw)
            if idx != -1:
                found_snippet = text[max(0, idx - 100): idx + 500]
                break
        if found_snippet:
            snippets.append(found_snippet)
            if source_url is None:
                source_url   = url
                source_label = label
        elif not snippets:
            snippets.append(text[:1000])
            if source_url is None:
                source_url   = url
                source_label = label
        if len(snippets) >= 3:
            break

    # DuckDuckGo site search as extra context
    try:
        ddg_url = ("https://html.duckduckgo.com/html/?q="
                   + quote_plus(f"site:just.edu.jo {query}"))
        text = _fetch_text(ddg_url)
        if text:
            found_urls = re.findall(r'https?://(?:www\.)?just\.edu\.jo[^\s"\'<>]*', text)
            if found_urls and source_url is None:
                source_url   = found_urls[0]
                source_label = "نتائج البحث في موقع الجامعة"
            snippets.append(text[:1000])
    except Exception as e:
        print(f"⚠️ DDG: {e}")

    if not snippets:
        return None

    context = "\n---\n".join(snippets[:4])
    result  = _call_ollama([
        {"role": "system", "content": (
            "أنت مساعد أكاديمي. فيما يلي مقاطع من الموقع الرسمي لجامعة العلوم والتكنولوجيا الأردنية. "
            "استخرج الإجابة المباشرة على سؤال المستخدم من هذه المقاطع فقط. "
            "اكتب كلمة مجهول فقط إذا لم تجد الإجابة، لا شيء آخر.\n\n"
            f"المقاطع:\n{context}"
        )},
        {"role": "user", "content": query},
    ])

    if result and not is_unknown_response(result):
        src_line = (f"🔗 المصدر: [{source_label}]({source_url})"
                    if source_url else "🔗 المصدر: الموقع الرسمي للجامعة")
        return f"{result}\n\n{src_line}"
    return None

# ==========================================
# 10. LLM Callers
# ==========================================
def clean_arabic_output(text: str) -> str:
    """
    Remove garbage injected by llama3/aya:
    - CJK characters
    - Pure Latin noise (include, pad4, yes...)
    - Mixed Arabic+Latin tokens like مجeho (corrupted مجهول)
    - "ملاحظة: أنا لا أتحدث العربية الفصحاء" self-disclaimers
    """
    # ── Strip language-ability disclaimers from aya/llama ───────────────
    # Catches: "ملاحظة: أنا لا أتحدث العربية الفصحاء بطلاقة..."
    text = re.sub(
        r'\*?\*?ملاحظة\*?\*?[:\s]+[^\n]*(أتحدث|الفصح|لغت|محدودي|بطلاقة)[^\n]*\n?',
        '', text, flags=re.IGNORECASE
    )
    # ── Strip CJK garbage ────────────────────────────────────────────────
    text  = re.sub(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\uff00-\uffef]+', '', text)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        tokens = line.split()
        result = []
        for tok in tokens:
            core           = re.sub(r'^[\W_]+|[\W_]+$', '', tok, flags=re.UNICODE)
            has_arabic     = bool(re.search(r'[\u0600-\u06ff]', tok))
            has_latin      = bool(re.search(r'[a-zA-Z]', tok))
            has_email      = '@' in tok
            is_url         = tok.startswith(('http', 'www.'))
            is_pure_latin_noise = (
                bool(re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', core))
                and not has_arabic and not has_email and not is_url
                and len(core) >= 2
            )
            is_mixed_noise = has_arabic and has_latin and not has_email and not is_url
            if not is_pure_latin_noise and not is_mixed_noise:
                result.append(tok)
        cleaned.append(' '.join(result))
    return re.sub(r'[ \t]+', ' ', '\n'.join(cleaned)).strip()


BOT_PERSONA = (
    "أنت مرشد أكاديمي ذكي اسمك 'المرشد الأكاديمي'، "
    "متخصص في قسم علم البيانات بجامعة العلوم والتكنولوجيا الأردنية.\n"
    "CRITICAL: Reply in Arabic ONLY. NEVER use English words like 'include', 'data', 'yes', 'no'. "
    "Use Arabic equivalents always.\n"
    "CRITICAL: NEVER add any note about your Arabic language ability or fluency. "
    "Do NOT write 'ملاحظة' or any disclaimer about speaking فصحى. Just answer confidently.\n"
    "CRITICAL: أنت تعمل فقط بالمعلومات الموجودة في السياق المُعطى لك. "
    "لا تستخدم معرفتك العامة عن أشخاص حقيقيين (دكاترة، أساتذة، موظفين). "
    "إذا سُئلت عن شخص وليس في السياق سوى إيميله أو مكتبه → ذكر هذه المعلومات فقط ولا تختلق سيرة ذاتية.\n"
    "عند الإجابة:\n"
    "• رتّب الإجابة في نقاط واضحة إذا كانت تحتوي على أكثر من فكرة.\n"
    "• استخدم **عنوان** للفقرات إذا كانت الإجابة طويلة.\n"
    "• اذكر رقم المادة أو الفقرة أو البند بالضبط إذا كان موجوداً في السياق.\n"
    "• لا تختلق أي معلومة غير موجودة في السياق — لا أبحاث، لا منشورات، لا سيرة ذاتية.\n"
    "• أجب بأسلوب ودي، واضح، ومنظم — بدون أي ملاحظات جانبية عن قدراتك.\n"
)


def _call_ollama(messages: list) -> str | None:
    if not is_ollama_available():
        return None
    try:
        resp = ollama.chat(model=OLLAMA_MODEL, messages=messages,
                           options={"temperature": 0.3})
        if hasattr(resp, "message"):
            raw = resp.message.content.strip()
        else:
            raw = resp["message"]["content"].strip()
        return clean_arabic_output(raw)
    except Exception as e:
        print(f"🚨 Ollama error ({type(e).__name__}): {e}")
        _ollama_cache["ts"] = 0
        return None


def ollama_chat(query: str, history: list, db_context: str | None = None,
                is_followup: bool = False) -> str | None:
    """Returns the LLM answer or None if Ollama is unavailable."""
    context_block = ""
    if db_context:
        followup_rule = (
            "3. السؤال هو متابعة لإجابة سابقة. استخرج المعلومة المطلوبة مباشرةً من\n"
            "   قسم 'معلومات من إجابتك السابقة' وأجب في جملة واحدة فقط. لا تُعيد صياغة السؤال.\n"
        ) if is_followup else (
            "3. إذا جاء السؤال بضمير (ايميله، مكتبه، رقمه...) فاستخدم قسم الإجابة السابقة\n"
            "   لتحديد المقصود، ثم ابحث عنه في بقية السياق.\n"
        )
        context_block = (
            "\n\n══ السياق الرسمي — يجب الاعتماد عليه ══\n"
            "المعلومات التالية مستخرجة من قاعدة البيانات الأكاديمية والوثائق الرسمية.\n"
            "قواعد صارمة:\n"
            "1. أجب فقط من هذا السياق ولا تضف معلومات خارجه.\n"
            "2. اذكر رقم المادة أو الفقرة إذا ظهر في السياق.\n"
            + followup_rule +
            "4. إذا لم تجد الإجابة في السياق اكتب مجهول فقط.\n"
            f"{db_context}\n"
            "══════════════════════════════════════\n"
        )

    system = (
        BOT_PERSONA
        + "يمكنك الإجابة على أي سؤال: أكاديمي أو اجتماعي أو عام.\n"
        + "• إذا كان السؤال أكاديمياً وتوجد معلومة في السياق → استخدمها وأجب مباشرةً.\n"
        + "• إذا كان السؤال اجتماعياً أو عاماً → أجب بشكل ودي طبيعي بدون أي مصادر أو ملاحظات.\n"
        + "• إذا سأل عن شخص (دكتور/أستاذ) → أجب فقط بما هو موجود في السياق (إيميل، مكتب، تخصص).\n"
        + "  لا تكتب سيرة ذاتية، لا تعيد صياغة السؤال، لا تذكر أبحاثاً إلا إذا كانت في السياق.\n"
        + "• إذا جاء السؤال بضمير (ايميله/مكتبه/رقمه) → أجب مباشرةً بالمعلومة من السياق، لا تسرد أسئلة.\n"
        + "• إذا كان السؤال أكاديمياً وليس في السياق إجابة → اكتب كلمة مجهول فقط.\n"
        + "لا تختلق أي معلومة غير موجودة في السياق."
        + context_block
    )

    msgs = [{"role": "system", "content": system}]
    # Use last 10 turns so pronoun follow-ups always have enough history
    for m in history[-10:]:
        if isinstance(m, dict):
            msgs.append(m)
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            msgs.append({"role": "user",      "content": str(m[0])})
            msgs.append({"role": "assistant", "content": str(m[1])})
    msgs.append({"role": "user", "content": query})
    return _call_ollama(msgs)

# ==========================================
# 11. Meta Answers (no LLM needed)
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
            f"**حالة Ollama:** {'✅ متصل ويعمل' if ok else '❌ غير متصل — الإجابات من قاعدة البيانات فقط'}\n"
            f"**نموذج التضمين:** paraphrase-multilingual-MiniLM-L12-v2\n"
            f"**قاعدة البيانات:** {len(df)} سؤال وجواب"
        )
    if re.search(r"كيف\s*(تشتغل|تعمل)", q):
        return (
            "أعمل بأربع طبقات 🔍:\n"
            "**1️⃣ FAISS:** أسترجع السياق ذو الصلة\n"
            "**2️⃣ Ollama (LLM):** أفهم السؤال وأصيغ الإجابة\n"
            "**3️⃣ JUST Website:** أبحث في الموقع الرسمي إذا احتجت\n"
            "**4️⃣ Escalation:** أحوّل للقسم إذا السؤال جديد تماماً"
        )
    if re.search(r"مين\s*(صمم|برمجك|عملك)", q):
        return "تم تطويري كمشروع تخرج في قسم علم البيانات 🎓"
    return None


def is_unknown_response(text: str) -> bool:
    """Detect all variants of 'I don't know', including mixed Arabic+Latin corruptions."""
    t = clean_arabic_output(text.strip())
    if len(t) <= 6 and re.match(r'^مج', t):
        return True
    if len(t) <= 8 and t.startswith("مجه"):
        return True
    if len(t.split()) <= 1 and not re.search(r'[\u0660-\u0669\d]', t):
        meaningful = bool(re.search(r'(ساعة|ساعات|مادة|مواد|نعم|كلية|قسم)', t))
        if not meaningful and len(t) <= 8:
            return True
    return any(v in t for v in [
        "مجهول", "مجهم", "مجهو", "مجهل",
        "لا أعلم", "لا أعرف", "لا أملك",
        "غير متوفر", "غير موجود", "غير معروف",
        "لا توجد", "لا يوجد", "لم أجد",
        "ليس لدي معلومات", "لا تتوفر لديّ",
        "لا أستطيع الإجابة",
    ])

# ==========================================
# 12. Main Bot Logic
# ==========================================
def ask_smart_bot(user_query: str, history: list) -> str:
    """
    Flow:
      1. Meta  → instant rule-based answer
      2. Pure greetings → LLM only (no DB)
      3. Everything else:
         a. Retrieve top-5 from FAISS
         b. High-conf name queries → return DB directly (avoid hallucination)
         c. Build context: Excel (priority) then PDF only if Excel score < 0.45
         d. Call LLM with context
         e. LLM unavailable → return best DB answer directly
         f. LLM says مجهول → web fallback → escalate to dept. head
    """
    try:
        query  = user_query.strip()
        intent = classify_intent(query)
        print(f"🧠 Intent={intent!r} | Ollama={'✅' if is_ollama_available() else '❌'} | Q={query[:60]}")

        # ── 1. Meta ─────────────────────────────────────────────────
        if intent == "meta":
            ans = meta_answer(query)
            if ans:
                return ans
            llm = ollama_chat(query, history)
            return llm or meta_answer("من أنت")

        # ── 2. Pure greetings ────────────────────────────────────────
        PURE_GREETINGS = {"هلا", "مرحبا", "السلام عليكم", "يعطيك العافية",
                          "أهلا", "مرحباً", "صباح الخير", "مساء الخير",
                          "هلو", "هاي", "أهلاً", "أهلين"}
        if any(g in query for g in PURE_GREETINGS) and len(query.split()) <= 4:
            llm = ollama_chat(query, history)
            return llm or "أهلاً وسهلاً! 😊 كيف يمكنني مساعدتك اليوم؟"

        # ── 3. Retrieve FAISS context ────────────────────────────────
        search_query = query
        CONTEXT_WORDS  = {"طيب", "وين", "عن", "اين", "أين", "هيك"}
        FOLLOW_UP_WORDS = {"ايميله", "ايميلها", "إيميله", "إيميلها",
                           "مكتبه", "مكتبها", "رقمه", "رقمها"}
        is_followup = any(w in query for w in FOLLOW_UP_WORDS) or len(query.split()) <= 3
        if (any(w in query.split() for w in CONTEXT_WORDS) or is_followup) and history:
            context_parts = []
            for msg in reversed(history[-6:]):
                if isinstance(msg, dict) and msg.get("content", "") != query:
                    context_parts.append(msg.get("content", ""))
                if len(context_parts) >= 2:
                    break
            if context_parts:
                search_query = f"{query} {' '.join(context_parts)}"

        hits = retrieve_top_k(search_query, top_k=5)
        print("🔍 Top hits:")
        for h in hits:
            print(f"   score={h['score']:.3f} | {h['question'][:60]}")
        best_score = hits[0]["score"] if hits else -1
        best_ans   = hits[0]["answer"] if hits else None

        # ── 4. High-confidence name/email → skip LLM ─────────────────
        norm_q    = normalize(query)
        is_name_q = any(w in norm_q for w in _NAME_TRIGGERS)
        if best_score >= THRESHOLD_DIRECT and is_name_q:
            print(f"✅ Direct name-hit ({best_score:.2f})")
            return str(best_ans)

        # ── 5. Build context: Excel DB first, PDF only as fallback ────
        ctx_parts        = []
        excel_used       = False
        pdf_sources_used = []

        # 5a. Inject previous assistant answer for follow-up pronoun resolution.
        #     This lets the LLM know what "ايميله / مكتبه / رقمه" refers to.
        last_bot_msg = None
        if is_followup and history:
            last_bot_msg = next(
                (m.get("content", "") for m in reversed(history[-10:])
                 if isinstance(m, dict) and m.get("role") == "assistant"),
                None
            )
            if last_bot_msg:
                ctx_parts.append(
                    f"[ 💬 معلومات من إجابتك السابقة — استخرج منها الإجابة المباشرة ]\n"
                    f"{last_bot_msg}"
                )

        # 5b. Excel DB hits
        if hits and hits[0]["score"] >= 0.45:
            ctx_parts.append("[ ✅ من قاعدة البيانات — أولوية عالية ]")
            ctx_parts.extend(f"س: {h['question']}\nج: {h['answer']}" for h in hits)
            excel_used = True

        # 5c. PDF fallback — strict conditions:
        #   • Only for "academic" intent (not unknown/conversational/meta)
        #   • Only when Excel had no hit
        #   • Never for name/person queries (discipline-regs PDF has no staff info)
        #   • Never for follow-up pronoun queries (answer must come from history/Excel)
        if not excel_used and intent == "academic" and not is_name_q and not is_followup:
            pdf_hits = search_pdf(search_query, top_k=4)
            if pdf_hits:
                ctx_parts.append("\n[ من الملفات والوثائق الرسمية ]")
                for ph in pdf_hits:
                    prefix = f"[{ph['article_ref']}] " if ph.get("article_ref") else ""
                    ctx_parts.append(f"{prefix}{ph['text']}")
                    src_str = format_pdf_source(ph["source"], ph["page"])
                    if ph.get("article_ref"):
                        src_str = f"{src_str} — {ph['article_ref']}"
                    if src_str not in pdf_sources_used:
                        pdf_sources_used.append(src_str)

        db_context = "\n".join(ctx_parts) if ctx_parts else None

        # ── 6. Call LLM ──────────────────────────────────────────────
        print(f"🤖 LLM | score={best_score:.2f} | intent={intent} | followup={is_followup}")
        llm_ans = ollama_chat(query, history, db_context, is_followup=is_followup)

        # ── 7. Ollama offline → return best DB answer directly ───────
        if llm_ans is None:
            print("⚠️  Ollama offline — using direct DB answer")
            if best_ans and best_score >= 0.45:
                return str(best_ans)
            if intent == "conversational":
                return (
                    "أهلاً! 😊 يمكنني مساعدتك في:\n"
                    "• تسجيل المواد والجداول\n"
                    "• معلومات الدكاترة والإيميلات\n"
                    "• الأنظمة الجامعية ومتطلبات التخرج\n"
                    "فقط اسألني!"
                )
            return ("عذراً، النظام يعمل بوضع محدود الآن. "
                    "يرجى إعادة المحاولة أو صياغة سؤالك بشكل مختلف.")

        # ── 8. LLM returned unknown → web fallback then escalate ─────
        if is_unknown_response(llm_ans):
            print("🔴 LLM said مجهول → web fallback")
            web_ans = search_just_website(query)
            if web_ans:
                return web_ans

            if intent == "academic":
                cursor.execute(
                    "INSERT INTO pending_questions (user_question, status) VALUES (?, ?)",
                    (query, "Pending")
                )
                conn.commit()
                return (
                    "سؤالك وصلني ✅\n"
                    "لا أملك معلومات رسمية عن هذا الموضوع حالياً، "
                    "لذا سأحوّله لرئيس القسم للرد عليك قريباً.\n"
                    "هل يمكنك إضافة تفاصيل أكثر لمساعدتي في فهم سؤالك؟"
                )
            return (
                "لم أجد معلومات كافية عن هذا الموضوع 🤔\n"
                "هل يمكنك توضيح سؤالك أكثر؟ مثلاً: ذكر اسم الدكتور أو المادة أو الإجراء المحدد."
            )

        # ── 9. LLM answered — append PDF sources if used ─────────────
        if pdf_sources_used:
            sources_block = "\n".join(f"📄 {s}" for s in pdf_sources_used)
            return f"{llm_ans}\n\n**المصادر:**\n{sources_block}"
        return llm_ans

    except Exception as e:
        print(f"🚨 Error: {e}")
        traceback.print_exc()
        return "أواجه مشكلة تقنية مؤقتة، يرجى المحاولة مجدداً."

# ==========================================
# 13. DB Helpers
# ==========================================
def submit_head_answer(q_id, answer):
    """Answer a single pending question and retrain the bot."""
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
        return "✅ تم حفظ الإجابة وتدريب البوت بنجاح!", 200
    except Exception as e:
        print(f"❌ submit_head_answer: {e}")
        return f"❌ خطأ: {e}", 400


def submit_bulk_answer(ids: list, answer: str):
    """
    Answer multiple pending questions with the same answer.
    Rebuilds the FAISS index only once at the end.
    (Feature from app_ollama2 — not present in app_ollama)
    """
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
        question = row[0]
        new_rows.append({"Question": question, "Answer": answer, "Keywords": ""})
        cursor.execute(
            "UPDATE pending_questions SET status='Answered' WHERE id=?", (q_id,))

    if not new_rows:
        return "❌ لم يُعثر على أي من الأسئلة المحددة", 404

    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)
    conn.commit()
    faiss_index = build_index(df)

    saved   = len(new_rows)
    skipped = len(not_found)
    msg     = f"✅ تم حفظ الإجابة على {saved} سؤال وتدريب البوت بنجاح!"
    if skipped:
        msg += f" (لم يُعثر على {skipped} سؤال)"
    print(f"📦 Bulk answer: {saved} saved, {skipped} not found")
    return msg, 200

# ==========================================
# 14. API Routes
# ==========================================
@app.route("/")
@app.route("/interface")
def serve_interface():
    """Open http://localhost:5000 to get the chatbot UI directly."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ("interface_v3.html", "interface_v2.html", "interface.html"):
        if os.path.exists(os.path.join(base_dir, name)):
            return send_from_directory(base_dir, name)
    return "<h2>No interface HTML found in the same folder as this script.</h2>", 404


@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.json
    return jsonify({"reply": ask_smart_bot(
        data.get("query", ""), data.get("history", []))})


@app.route("/api/pending", methods=["GET"])
def pending_api():
    df_p = pd.read_sql_query(
        "SELECT id, user_question FROM pending_questions WHERE status='Pending'", conn)
    return jsonify(df_p.to_dict(orient="records"))


@app.route("/api/answer", methods=["POST"])
def answer_api():
    """Answer a single pending question (backward-compatible)."""
    data = request.json
    msg, code = submit_head_answer(data.get("id"), data.get("answer"))
    return jsonify({"success": code == 200, "message": msg}), code


@app.route("/api/answer-bulk", methods=["POST"])
def answer_bulk_api():
    """
    Answer multiple questions with the same answer in one request.
    Body: { "ids": [1, 3, 7], "answer": "الإجابة هنا" }
    """
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
        "ollama_warning": "❌ Ollama offline — LLM disabled, DB-only mode" if not ok else "✅ OK",
        "model":         OLLAMA_MODEL,
        "rows_loaded":   len(df),
    })


@app.route("/api/test-ollama", methods=["GET"])
def test_ollama():
    """Quick diagnostic — visit /api/test-ollama in your browser."""
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
        text = resp.message.content if hasattr(resp, "message") else resp["message"]["content"]
        result["chat_ok"]    = True
        result["test_reply"] = text[:100]
    except Exception as e:
        result["chat_error"] = f"{type(e).__name__}: {e}"
    return jsonify(result)


if __name__ == "__main__":
    app.run(port=5000, debug=False)