import sys
from flask import Flask, request, jsonify
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
from urllib.parse import quote_plus

app = Flask(__name__)

# ==========================================
# 1. Configuration
# ==========================================
# ── Model recommendation ──────────────────────────────────────────
# llama3.2:1b  → too small, poor Arabic, use only if RAM < 4GB
# llama3.1:8b  → good Arabic, needs ~8GB RAM  (recommended)
# aya:8b       → Arabic-focused, needs ~8GB RAM (best for Arabic)
# aya:35b      → best quality, needs ~24GB RAM
OLLAMA_MODEL = "llama3:latest"  # ← matches what you have installed (ollama list)

EXCEL_FILE  = "project_data.xlsx"
OLLAMA_HOST = "http://localhost:11434"

# Only skip LLM for extremely high-confidence name/email queries
THRESHOLD_DIRECT = 0.80

JUST_BASE_URL = "https://www.just.edu.jo"

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
# PDF Knowledge Base
# ==========================================
import fitz   # PyMuPDF  — pip install pymupdf
import io

# ── Configure your PDF sources here ──────────────────────────────────────────
# Add URLs (must be publicly accessible) OR local file paths.
# The JUST server blocks bots (403), so download those PDFs manually and add
# their local paths here. Example:
#   "regulations.pdf"                          ← file in same folder as server.py
#   "C:/Users/user/Downloads/gp/3.pdf"         ← absolute Windows path
#   "https://example.com/public.pdf"           ← publicly accessible URL

PDF_SOURCES = [
    # ← add your PDF paths/URLs here, e.g.:
    # "3.pdf",
    # "https://www.just.edu.jo/...pdf",
]

# Chunk size in characters — smaller = more precise retrieval
PDF_CHUNK_SIZE  = 600
PDF_CHUNK_OVERLAP = 100   # overlap between chunks so context isn't cut mid-sentence

# ─────────────────────────────────────────────────────────────────────────────
_pdf_chunks: list[dict] = []   # [{text, source, page}]
_pdf_index  = None             # separate FAISS index for PDF chunks
_pdf_ready  = False

def _extract_pdf_text(source: str) -> list[dict]:
    """
    Load a PDF from a URL or local path and return a list of
    {text, source, page} dicts — one per page.
    """
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
            # Local file — resolve relative paths from the script directory
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), source) \
                   if not os.path.isabs(source) else source
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

def _chunk_pages(pages: list[dict]) -> list[dict]:
    """Split page texts into overlapping chunks for finer retrieval."""
    chunks = []
    for p in pages:
        text = p["text"]
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
    """Return the top-k most relevant PDF chunks for a query."""
    if not _pdf_ready or _pdf_index is None:
        return []
    norm  = normalize(query)
    q_emb = (sentence_model.encode([norm], normalize_embeddings=True, convert_to_tensor=True)
             .cpu().numpy().astype("float32"))
    k         = min(top_k, len(_pdf_chunks))
    dists, ids = _pdf_index.search(q_emb, k)
    results = []
    for rank in range(k):
        score = float(dists[0][rank])
        if score >= 0.35:   # only include chunks with reasonable relevance
            chunk = _pdf_chunks[ids[0][rank]]
            results.append({
                "score":  score,
                "text":   chunk["text"],
                "source": chunk["source"],
                "page":   chunk["page"],
            })
    return results

def format_pdf_source(source: str, page: int) -> str:
    """Return a Markdown link for the PDF source."""
    if source.startswith("http"):
        return f"[الصفحة {page} — {source.split('/')[-1]}]({source}#page={page})"
    fname = os.path.basename(source)
    return f"ملف: {fname} (صفحة {page})"

# Load PDFs at startup
load_pdfs()

# ==========================================
# 3. Data & Index
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

# ==========================================
# 4. Ollama Health (cached 30s)
# ==========================================
_ollama_cache = {"ok": False, "ts": 0}

def is_ollama_available():
    now = time.time()
    if now - _ollama_cache["ts"] < 30:
        return _ollama_cache["ok"]
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        _ollama_cache["ok"] = r.status_code == 200
    except Exception:
        _ollama_cache["ok"] = False
    _ollama_cache["ts"] = now
    print(f"🔌 Ollama {'✅' if _ollama_cache['ok'] else '❌ OFFLINE — responses will be DB-only'}")
    return _ollama_cache["ok"]

is_ollama_available()

# ==========================================
# 5. Text Normalisation
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

_PREFIXES = ["لل","بال","وال","فال","كال","ال","ل","ب","و","ف","ك"]

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
    "ما","هو","هي","في","من","على","إلى","عن","مع","هل","كيف",
    "أين","اين","متى","لماذا","الجامعي","الجامعية",
    "ال","و","يتم","يمكن","هذا","هذه","كان","يكون",
}
STOP_WORDS = {normalize_arabic(w) for w in _RAW_STOP}
STOP_WORDS |= {strip_prefix(w) for w in list(STOP_WORDS)}

# ==========================================
# 6. Intent Detection
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
    r"(كيف|شو|إيش|ايش)\s*(بتقدر|تقدر|بتساعد|تساعد)\s*(تعمل|بتعمل)?$",
    r"(ايش|إيش|شو|ما)\s*خدماتك",
    r"هل\s*أنت\s*(ذكاء|روبوت|بوت)",
    r"شكرا|شكراً|ممنون|يسلمو|يعطيك",
    r"(حلو|ممتاز|رائع|كويس|زين|منيح)\s*$",
    r"(غلط|غلطت|مش\s*صح|خطأ)\s*$",
]

ACADEMIC_KEYWORDS = [
    "تسجيل","سجل","اسجل","يسجل","مادة","مواد","ساعات","فصل",
    "منهج","دكتور","دكتوره","أستاذ","استاذ","مهندس",
    "قسم","جامعة","جامعه","كلية","كليه","شعبة","درجة","علامة",
    "غياب","امتحان","مشروع","تخرج","خطة","تأديب","انتساب",
    "معدل","إيميل","ايميل","مكتب","نظام","برنامج","بحث",
    "وثيقة","إجراء","طلب","استمارة","شهادة","توثيق","رسوم",
    "مالية","منحة","محاضرة","اختبار","جدول",
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
# 7. FAISS Retrieval
# ==========================================
def word_overlap(q: str, stored: str) -> float:
    norm_q = normalize_arabic(q)
    norm_s = normalize_arabic(str(stored))
    qw = {strip_prefix(w) for w in norm_q.split()} - STOP_WORDS
    sw = {strip_prefix(w) for w in norm_s.split()} - STOP_WORDS
    if not qw or not sw:
        return 0.0
    return len(qw & sw) / len(qw | sw)

_NAME_TRIGGERS = {"الدكتور","الاستاذ","دكتور","استاذ",
                  "الدكتوره","الاستاذه","المهندس","مهندس"}

def retrieve_top_k(query: str, top_k: int = 5) -> list[dict]:
    norm = normalize(query)
    is_name_query = any(w in norm for w in _NAME_TRIGGERS)
    sem_w  = 0.20 if is_name_query else 0.50
    olap_w = 0.80 if is_name_query else 0.50

    q_emb = (sentence_model.encode([norm], normalize_embeddings=True,
                                   convert_to_tensor=True)
             .cpu().numpy().astype("float32"))
    k = min(max(top_k * 2, 10), len(df))
    distances, indices = faiss_index.search(q_emb, k)

    scored = []
    for rank in range(k):
        sem  = float(distances[0][rank])
        ri   = indices[0][rank]
        olap = word_overlap(norm, df.iloc[ri]["Question"])
        score = sem_w * sem + olap_w * olap
        scored.append((score, ri))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {"score": s, "question": str(df.iloc[ri]["Question"]),
         "answer": str(df.iloc[ri]["Answer"])}
        for s, ri in scored[:top_k]
    ]

# ==========================================
# 8. JUST Website Fallback
# ==========================================
_WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ar,en;q=0.9",
}

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

# Pages to search — ordered by relevance; first match wins for source attribution
JUST_PAGES = [
    ("صفحة أعضاء هيئة التدريس - قسم علم البيانات",
     "https://www.just.edu.jo/FacultiesAndDepartments/FacultyofComputer/Departments/DataScience/Pages/Staff.aspx"),
    ("قسم علم البيانات - جامعة العلوم والتكنولوجيا",
     "https://www.just.edu.jo/FacultiesAndDepartments/FacultyofComputer/Departments/DataScience/Pages/Home.aspx"),
    ("الموقع الرسمي لجامعة العلوم والتكنولوجيا الأردنية",
     "https://www.just.edu.jo"),
]

def search_just_website(query: str) -> str | None:
    print(f"🌐 Searching JUST for: {query[:60]}")
    snippets   = []   # (text_snippet, source_url, source_label)
    source_url   = None
    source_label = None

    # ── 1. Try each JUST page directly ────────────────────────────────────
    q_words = [w for w in normalize(query).split()
               if len(w) > 3 and w not in STOP_WORDS]

    for label, url in JUST_PAGES:
        text = _fetch_text(url)
        if not text:
            continue
        # Try to find a relevant snippet around the query keywords
        found_snippet = None
        for qw in q_words:
            idx = text.find(qw)
            if idx != -1:
                found_snippet = text[max(0, idx - 100): idx + 500]
                break
        if found_snippet:
            snippets.append(found_snippet)
            if source_url is None:          # first hit = primary source
                source_url   = url
                source_label = label
        elif not snippets:                  # fallback: use first 1000 chars
            snippets.append(text[:1000])
            if source_url is None:
                source_url   = url
                source_label = label

        if len(snippets) >= 3:
            break

    # ── 2. DuckDuckGo site search as extra context ─────────────────────────
    try:
        ddg_url = ("https://html.duckduckgo.com/html/?q="
                   + quote_plus(f"site:just.edu.jo {query}"))
        text = _fetch_text(ddg_url)
        if text:
            # Extract any just.edu.jo URLs found in the DDG results
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
    result = _call_ollama([
        {"role": "system", "content": (
            "أنت مساعد أكاديمي. فيما يلي مقاطع من الموقع الرسمي لجامعة العلوم والتكنولوجيا الأردنية. "
            "استخرج الإجابة المباشرة على سؤال المستخدم من هذه المقاطع فقط. "
            "اكتب كلمة مجهول فقط إذا لم تجد الإجابة، لا شيء آخر.\n\n"
            f"المقاطع:\n{context}"
        )},
        {"role": "user", "content": query},
    ])

    if result and not is_unknown_response(result):
        # Build attribution line with actual URL
        src_line = f"🔗 المصدر: [{source_label}]({source_url})" if source_url else "🔗 المصدر: الموقع الرسمي للجامعة"
        return f"{result}\n\n{src_line}"
    return None

# ==========================================
# 9. LLM Callers
# ==========================================

def clean_arabic_output(text: str) -> str:
    """
    Remove garbage injected by llama3:
    - CJK characters
    - Pure Latin noise (include, pad4, yes...)
    - Mixed Arabic+Latin tokens like مجeho (corrupted مجهول)
    """
    text = re.sub(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\uff00-\uffef]+', '', text)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        tokens = line.split()
        result = []
        for tok in tokens:
            core = re.sub(r'^[\W_]+|[\W_]+$', '', tok, flags=re.UNICODE)
            has_arabic = bool(re.search(r'[\u0600-\u06ff]', tok))
            has_latin  = bool(re.search(r'[a-zA-Z]', tok))
            has_email  = '@' in tok
            is_url     = tok.startswith(('http', 'www.'))
            is_pure_latin_noise = (
                bool(re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', core))
                and not has_arabic and not has_email and not is_url
                and len(core) >= 2
            )
            # Mixed Arabic+Latin (مجeho, مجeho.) = always garbage
            is_mixed_noise = has_arabic and has_latin and not has_email and not is_url
            if not is_pure_latin_noise and not is_mixed_noise:
                result.append(tok)
        cleaned.append(' '.join(result))
    return re.sub(r'[ \t]+', ' ', '\n'.join(cleaned)).strip()

BOT_PERSONA = (
    "أنت مرشد أكاديمي ذكي اسمك 'المرشد الأكاديمي'، "
    "متخصص في قسم علم البيانات بجامعة العلوم والتكنولوجيا الأردنية.\n"
    "CRITICAL: Reply in Arabic ONLY. NEVER use English words like 'include', 'data', 'yes', 'no'. "
    "Use Arabic equivalents always. أجب بأسلوب ودي، واضح، ومختصر.\n"
)

def _call_ollama(messages: list) -> str | None:
    if not is_ollama_available():
        return None
    try:
        resp = ollama.chat(model=OLLAMA_MODEL, messages=messages,
                           options={"temperature": 0.3})
        # Support both old (dict) and new (Pydantic object) ollama package
        if hasattr(resp, "message"):
            raw = resp.message.content.strip()
        else:
            raw = resp["message"]["content"].strip()
        return clean_arabic_output(raw)
    except Exception as e:
        print(f"🚨 Ollama error ({type(e).__name__}): {e}")
        _ollama_cache["ts"] = 0
        return None

def ollama_chat(query: str, history: list, db_context: str | None = None) -> str | None:
    """
    Returns the LLM answer or None if Ollama is unavailable.
    Caller is responsible for handling None with a DB fallback.
    """
    context_block = ""
    if db_context:
        context_block = (
            "\n\nلديك المعلومات الأكاديمية التالية من قاعدة بيانات القسم "
            "(استخدمها إذا كانت ذات صلة بالسؤال، وتجاهلها إذا كان السؤال عاماً):\n"
            f"{db_context}\n"
        )

    system = (
        BOT_PERSONA
        + "يمكنك الإجابة على أي سؤال: أكاديمي أو اجتماعي أو عام.\n"
        + "• إذا كان السؤال أكاديمياً وتوجد معلومة في السياق → استخدمها وأجب مباشرةً.\n"
        + "• إذا كان السؤال اجتماعياً أو عاماً (تحية، كيف حالك، امثله...) → أجب بشكل ودي طبيعي.\n"
        + "• إذا طلب المستخدم أمثلة → أعطه أمثلة من قائمة ما يمكنك مساعدته به.\n"
        + "• إذا كان السؤال أكاديمياً وليس في السياق إجابة → اكتب كلمة مجهول فقط، لا شيء آخر.\n"
        + "• إذا سألك عن محادثة سابقة → راجع تاريخ المحادثة وأجب.\n"
        + "لا تختلق أرقام غرف، إيميلات، أو معلومات غير موجودة في السياق."
        + context_block
    )

    msgs = [{"role": "system", "content": system}]
    for m in history[-8:]:
        if isinstance(m, dict):
            msgs.append(m)
        elif isinstance(m, (list, tuple)) and len(m) >= 2:
            msgs.append({"role": "user",      "content": str(m[0])})
            msgs.append({"role": "assistant", "content": str(m[1])})
    msgs.append({"role": "user", "content": query})

    return _call_ollama(msgs)   # None if Ollama is offline

# ==========================================
# 10. Meta Answers (no LLM needed)
# ==========================================
def meta_answer(query: str) -> str | None:
    q = normalize(query).lower()
    if any(w in q for w in ["اسمك","من انت","مين انت","انت شو","شو انت","مين معي"]):
        return (
            "إنا **المرشد الأكاديمي الذكي** 🎓\n"
            "مساعد آلي مخصص لقسم علم البيانات في جامعة العلوم والتكنولوجيا.\n"
            "أستطيع مساعدتك في أسئلتك الأكاديمية أو حتى مجرد الدردشة! 😊"
        )
    if any(w in q for w in ["موديل","نموذج","ollama","llm"]):
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
    """
    Detect all variants of 'I don't know' that llama3 may output.
    Includes mixed Arabic+Latin corruptions like مجeho.
    """
    # First clean the text to remove mixed-script garbage
    t = clean_arabic_output(text.strip())
    # After cleaning, مجeho → مج (pure Arabic chars left)
    # Short reply starting with مج is almost certainly a corrupted مجهول
    if len(t) <= 6 and re.match(r'^مج', t):
        return True
    # Standard Arabic unknown phrases
    if len(t) <= 8 and t.startswith("مجه"):
        return True
    # Very short reply with no useful info (1-2 words, no academic content)
    if len(t.split()) <= 1 and not re.search(r'[\u0660-\u0669\d]', t):
        # Single meaningless word
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
# 11. Main Bot Logic
# ==========================================
def ask_smart_bot(user_query: str, history: list) -> str:
    """
    Flow:
      1. Meta → instant rule-based
      2. Pure greetings → LLM only (no DB)
      3. Everything else:
         a. Retrieve top-5 from FAISS
         b. For high-conf name queries → return DB directly (avoid LLM hallucination)
         c. Call LLM with DB context
         d. If LLM unavailable → return best DB answer directly (NO canned fallback)
         e. If LLM says مجهول → try web → escalate
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
            return llm or meta_answer("من أنت")  # last resort

        # ── 2. Pure greetings ────────────────────────────────────────
        PURE_GREETINGS = {"هلا","مرحبا","السلام عليكم","يعطيك العافية",
                          "أهلا","مرحباً","صباح الخير","مساء الخير",
                          "هلو","هاي","أهلاً","أهلين"}
        if any(g in query for g in PURE_GREETINGS) and len(query.split()) <= 4:
            llm = ollama_chat(query, history)
            return llm or "أهلاً وسهلاً! 😊 كيف يمكنني مساعدتك اليوم؟"

        # ── 3. Retrieve FAISS context ────────────────────────────────
        search_query = query
        CONTEXT_WORDS = {"طيب","وين","عن","اين","أين","هيك"}
        FOLLOW_UP_WORDS = {"ايميله","ايميلها","إيميله","إيميلها","مكتبه","مكتبها","رقمه","رقمها"}
        is_followup = any(w in query for w in FOLLOW_UP_WORDS) or len(query.split()) <= 3
        if (any(w in query.split() for w in CONTEXT_WORDS) or is_followup) and history:
            # Collect context from last user + last assistant messages for better name resolution
            context_parts = []
            for msg in reversed(history[-6:]):
                if isinstance(msg, dict) and msg.get("content","") != query:
                    context_parts.append(msg.get("content",""))
                if len(context_parts) >= 2:
                    break
            if context_parts:
                search_query = f"{query} {' '.join(context_parts)}"

        hits       = retrieve_top_k(search_query, top_k=5)
        best_score = hits[0]["score"] if hits else -1
        best_ans   = hits[0]["answer"] if hits else None

        # ── 4. High-confidence name/email → skip LLM (no hallucination) ─
        norm_q = normalize(query)
        is_name_q = any(w in norm_q for w in _NAME_TRIGGERS)
        if best_score >= THRESHOLD_DIRECT and is_name_q:
            print(f"✅ Direct name-hit ({best_score:.2f})")
            return str(best_ans)

        # ── 5. Build context: Excel DB + PDF chunks ────────────────
        ctx_parts = []
        if hits:
            ctx_parts.append("[ من قاعدة البيانات ]")
            ctx_parts.extend(f"س: {h['question']}\nج: {h['answer']}" for h in hits)

        pdf_hits = search_pdf(search_query, top_k=4)
        pdf_sources_used = []
        if pdf_hits:
            ctx_parts.append("\n[ من الملفات والوثائق الرسمية ]")
            for ph in pdf_hits:
                ctx_parts.append(ph["text"])
                src_str = format_pdf_source(ph["source"], ph["page"])
                if src_str not in pdf_sources_used:
                    pdf_sources_used.append(src_str)

        db_context = "\n".join(ctx_parts) if ctx_parts else None

        # ── 6. Call LLM ──────────────────────────────────────────────
        print(f"🤖 LLM | score={best_score:.2f} | intent={intent}")
        llm_ans = ollama_chat(query, history, db_context)

        # ── 7. Ollama offline → return best DB answer directly ───────
        # FIX: NO more canned "أنا هنا لمساعدتك" when Ollama is down.
        # If we have a DB hit, return it. If not, give a helpful offline message.
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
            return (
                "عذراً، النظام يعمل بوضع محدود الآن. "
                "يرجى إعادة المحاولة أو صياغة سؤالك بشكل مختلف."
            )

        # ── 8. LLM returned unknown → web fallback then escalate ────
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
            # Unknown intent — ask for clarification
            return (
                "لم أجد معلومات كافية عن هذا الموضوع 🤔\n"
                "هل يمكنك توضيح سؤالك أكثر؟ مثلاً: ذكر اسم الدكتور أو المادة أو الإجراء المحدد."
            )

        # ── 9. LLM answered normally — append PDF sources if used ─────
        if pdf_sources_used:
            sources_block = "\n".join(f"📄 {s}" for s in pdf_sources_used)
            return f"{llm_ans}\n\n**المصادر:**\n{sources_block}"
        return llm_ans

    except Exception as e:
        print(f"🚨 Error: {e}")
        traceback.print_exc()
        return "أواجه مشكلة تقنية مؤقتة، يرجى المحاولة مجدداً."

# ==========================================
# 12. DB Helper
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
        return "✅ تم حفظ الإجابة وتدريب البوت بنجاح!", 200
    except Exception as e:
        print(f"❌ submit_head_answer: {e}")
        return f"❌ خطأ: {e}", 400

# ==========================================
# 13. API Routes
# ==========================================
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
    data = request.json
    msg, code = submit_head_answer(data.get("id"), data.get("answer"))
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
    """Quick diagnostic — visit /api/test-ollama in browser to see Ollama status."""
    import traceback as tb
    result = {"http_ok": False, "chat_ok": False, "model": OLLAMA_MODEL, "error": None}
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        result["http_ok"] = r.status_code == 200
        result["available_models"] = [m["name"] for m in r.json().get("models", [])]
    except Exception as e:
        result["error"] = f"HTTP: {e}"
    try:
        resp = ollama.chat(model=OLLAMA_MODEL,
                           messages=[{"role": "user", "content": "مرحبا"}],
                           options={"temperature": 0.1})
        text = resp.message.content if hasattr(resp, "message") else resp["message"]["content"]
        result["chat_ok"] = True
        result["test_reply"] = text[:100]
    except Exception as e:
        result["chat_error"] = f"{type(e).__name__}: {e}"
    return jsonify(result)

if __name__ == "__main__":
    app.run(port=5000, debug=False)