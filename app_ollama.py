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

app = Flask(__name__)

# ==========================================
# 1. Configuration
# ==========================================
OLLAMA_MODEL  = "llama3.2:1b"
EXCEL_FILE    = "project_data.xlsx"
OLLAMA_HOST   = "http://localhost:11434"

THRESHOLD_DIRECT = 0.75
THRESHOLD_MEDIUM = 0.50

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
    print(f"🔌 Ollama {'✅ Online' if _ollama_cache['ok'] else '❌ Offline'}")
    return _ollama_cache["ok"]

is_ollama_available()

# ==========================================
# 5. Text Normalisation
# ==========================================
DIALECT_MAP = {
    "شو": "ما", "وين": "أين", "اين": "أين", "كيفش": "كيف",
    "شلون": "كيف", "ليش": "لماذا", "هيك": "هكذا",
    "بدي": "أريد", "بقدر": "أستطيع", "لازم": "يجب",
    "هلأ": "الآن", "هلق": "الآن", "هسا": "الآن", "هساع": "الآن",
    "اسجل": "تسجيل", "للجامعه": "في الجامعة",
    "بدي اسجل": "كيفية التسجيل", "بقدر اسجل": "طريقة التسجيل",
    "شغال": "يعمل", "موديل": "نموذج", "بوت": "مساعد",
    "ايش": "ما", "إيش": "ما", "وش": "ما",
    "زين": "جيد", "مو": "ليس", "گلي": "أخبرني",
}

def normalize(text: str) -> str:
    words = text.split()
    return " ".join(DIALECT_MAP.get(w, w) for w in words)

# ==========================================
# 6. Intent Detection
# ==========================================
META_PATTERNS = [
    r"مين\s*معي", r"من\s*أنت", r"من\s*انت", r"اسمك", r"انت\s*شو", r"شو\s*انت",
    r"كيف\s*(تشتغل|تعمل|بتشتغل|بتعمل)",
    r"شو\s*(الموديل|النموذج)",
    r"ما\s*(الموديل|النموذج)",
    r"ollama", r"\bllm\b", r"\bai\b",
    r"(موديل|نموذج).*(شغال|يعمل)",
    r"مين\s*(صمم|برمجك|عملك)",
]

CONVERSATIONAL_PATTERNS = [
    r"ليش\s*(هيك|هكذا|جاوبت|قلت|رديت|تجاوب|كذا)",
    r"شو\s*قصدك",
    r"وضح\s*لي|وضحلي|شرحلي|شرح\s*لي",
    r"مش\s*(فاهم|فهمت)",
    r"(كيف|شو|إيش|ايش)\s*(بتقدر|تقدر|بتساعد|تساعد|تعمل|بتعمل)",
    r"(ايش|إيش|شو|ما)\s*(تعرف|تقدر|بتعرف|خدماتك)",
    r"ساعدني|مساعدة\s*عامة",
    r"هل\s*أنت\s*(ذكاء|روبوت|بوت)",
    r"شكرا|شكراً|ممنون|يسلمو|يعطيك",
    r"(حلو|ممتاز|رائع|كويس|زين|منيح)\s*$",
    r"(غلط|غلطت|مش\s*صح|خطأ)\s*$",
]

ACADEMIC_KEYWORDS = [
    "تسجيل", "مادة", "مواد", "ساعات", "فصل", "منهج", "دكتور", "أستاذ",
    "قسم", "جامعة", "كلية", "شعبة", "درجة", "علامة", "غياب", "امتحان",
    "مشروع", "تخرج", "خطة", "تأديب", "انتساب", "معدل", "إيميل", "مكتب",
    "نظام", "برنامج", "بحث", "وثيقة", "إجراء", "طلب", "استمارة",
    "شهادة", "توثيق", "رسوم", "مالية", "منحة", "محاضرة", "اختبار",
]

def _matches(text: str, patterns: list) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)

def classify_intent(query: str) -> str:
    if _matches(query, META_PATTERNS):
        return "meta"
    if _matches(query, CONVERSATIONAL_PATTERNS):
        return "conversational"
    if any(kw in query for kw in ACADEMIC_KEYWORDS):
        return "academic"
    if len(query.split()) <= 5:
        return "conversational"
    return "unknown"

# ==========================================
# 7. FAISS Search
# ==========================================
STOP_WORDS = {
    "ما","هو","هي","في","من","على","إلى","عن","مع","هل","كيف",
    "أين","متى","لماذا","الجامعي","الجامعية","للدكتور","للدكتورة",
    "مكتب","الإيميل","ايميل","ال","و","يتم","يمكن","هذا","هذه",
}

def word_overlap(q: str, stored: str) -> float:
    qw = set(q.split()) - STOP_WORDS
    sw = set(str(stored).split()) - STOP_WORDS
    if not qw or not sw:
        return 0.0
    return len(qw & sw) / len(qw | sw)

def hybrid_search(query: str, top_k: int = 10):
    norm  = normalize(query)
    q_emb = (sentence_model.encode([norm], normalize_embeddings=True,
                                   convert_to_tensor=True)
             .cpu().numpy().astype("float32"))
    k                  = min(top_k, len(df))
    distances, indices = faiss_index.search(q_emb, k)
    best_score, best_i = -1, indices[0][0]
    for rank in range(k):
        sem   = float(distances[0][rank])
        ri    = indices[0][rank]
        olap  = word_overlap(norm, df.iloc[ri]["Question"])
        score = 0.5 * sem + 0.5 * olap
        if score > best_score:
            best_score, best_i = score, ri
    return best_score, best_i

# ==========================================
# 8. Ollama Callers
# ==========================================
BOT_PERSONA = (
    "أنت مرشد أكاديمي ذكي اسمك 'المرشد الأكاديمي'، "
    "متخصص في قسم علم البيانات بجامعة العلوم والتكنولوجيا. "
    "أجب دائماً بالعربية بأسلوب ودي، واضح، ومختصر. "
)

def _call_ollama(messages: list) -> str | None:
    if not is_ollama_available():
        return None
    try:
        resp = ollama.chat(model=OLLAMA_MODEL, messages=messages,
                           options={"temperature": 0.3})
        return resp["message"]["content"].strip()
    except Exception as e:
        print(f"🚨 Ollama error: {e}")
        _ollama_cache["ts"] = 0
        return None

def ollama_conversational(query: str, history: list) -> str:
    """Free-form: greetings, feedback, clarifications, meta-chat."""
    system = (
        BOT_PERSONA +
        "يمكنك الإجابة على الأسئلة العامة والمحادثات اليومية بشكل طبيعي. "
        "إذا سألك أحد عن قدراتك، اشرح أنك تساعد في شؤون قسم علم البيانات. "
        "إذا انتقد المستخدم إجابة سابقة، اعتذر ووضح بشكل أفضل. "
        "لا تختلق معلومات أكاديمية غير موجودة في قاعدة بياناتك."
    )
    msgs = [{"role": "system", "content": system}]
    for m in history[-6:]:
        msgs.append(m if isinstance(m, dict) else {"role": "user", "content": m[0]})
    msgs.append({"role": "user", "content": query})
    return _call_ollama(msgs) or _rule_based_conversational(query)

def ollama_academic(query: str, history: list, data_context: str) -> str | None:
    """Strict academic answering. Returns None if model says unknown."""
    system = (
        BOT_PERSONA +
        "استخدم فقط البيانات الرسمية التالية للإجابة بدقة وإيجاز:\n\n"
        f"{data_context}\n\n"
        "إذا لم تجد إجابة واضحة في البيانات، أجب بكلمة 'مجهول' فقط."
    )
    msgs = [{"role": "system", "content": system}]
    for m in history[-4:]:
        msgs.append(m if isinstance(m, dict) else {"role": "user", "content": m[0]})
    msgs.append({"role": "user", "content": query})
    result = _call_ollama(msgs)
    if result and "مجهول" not in result:
        return result
    return None

def _rule_based_conversational(query: str) -> str:
    """Fallback when Ollama is offline."""
    q = query.lower()
    if any(w in q for w in ["ليش", "لماذا", "ليه"]):
        return (
            "عذراً إذا لم تكن إجابتي السابقة واضحة! 😊 "
            "أنا أبحث في قاعدة البيانات الأكاديمية للعثور على أقرب إجابة. "
            "هل يمكنك إعادة صياغة سؤالك بشكل أكثر تفصيلاً؟"
        )
    if any(w in q for w in ["كيف", "تساعد", "تقدر", "ساعدني", "خدمات"]):
        return (
            "يسعدني مساعدتك في 🎓:\n"
            "• تسجيل المواد والجداول الدراسية\n"
            "• معلومات الدكاترة والمكاتب\n"
            "• الأنظمة والتعليمات الجامعية\n"
            "• متطلبات التخرج والخطط الدراسية\n\n"
            "فقط اسألني!"
        )
    if any(w in q for w in ["مين معي", "من أنت", "اسمك", "انت شو"]):
        return (
            "أنا المرشد الأكاديمي الذكي 🤖\n"
            "مساعدك في قسم علم البيانات بجامعة العلوم والتكنولوجيا.\n"
            "كيف أقدر أساعدك اليوم؟"
        )
    if any(w in q for w in ["شكرا", "ممنون", "يسلمو"]):
        return "العفو! يسعدني خدمتك دائماً 😊 هل تحتاج شيئاً آخر؟"
    return (
        "أنا هنا لمساعدتك! 😊 "
        "يمكنك سؤالي عن أي شيء يخص قسم علم البيانات."
    )

# ==========================================
# 9. Meta Answers (no Ollama needed)
# ==========================================
def meta_answer(query: str) -> str | None:
    q = query.lower()
    if any(w in q for w in ["اسمك", "من أنت", "من انت", "مين انت", "انت شو", "شو انت", "مين معي"]):
        return (
            "أنا **المرشد الأكاديمي الذكي** 🎓\n"
            "مساعد آلي مخصص لقسم علم البيانات في جامعة العلوم والتكنولوجيا.\n"
            "أستطيع الإجابة على أسئلتك الأكاديمية المتعلقة بالقسم."
        )
    if any(w in q for w in ["موديل", "نموذج", "ollama", "llm", "شغال", "يشتغل"]):
        ok = is_ollama_available()
        return (
            f"**النموذج:** {OLLAMA_MODEL} عبر Ollama\n"
            f"**حالة Ollama:** {'✅ متصل ويعمل' if ok else '❌ غير متصل حالياً'}\n"
            f"**نموذج التضمين:** paraphrase-multilingual-MiniLM-L12-v2\n"
            f"**قاعدة البيانات:** {len(df)} سؤال وجواب"
        )
    if re.search(r"كيف\s*(تشتغل|تعمل)", q):
        return (
            "أعمل بثلاث طبقات 🔍:\n"
            "**1️⃣ FAISS:** أبحث في قاعدة الأسئلة الأكاديمية\n"
            "**2️⃣ Ollama:** إذا احتجت فهماً أعمق أستشير النموذج\n"
            "**3️⃣ Escalation:** إذا كان السؤال جديداً، أحوّله لرئيس القسم"
        )
    if re.search(r"مين\s*(صمم|برمجك|عملك)", q):
        return "تم تطويري كمشروع تخرج في قسم علم البيانات 🎓"
    return None

# ==========================================
# 10. Main Bot Logic
# ==========================================
def ask_smart_bot(user_query: str, history: list) -> str:
    try:
        query  = user_query.strip()
        intent = classify_intent(query)
        print(f"🧠 Intent={intent!r} | Q={query[:70]}")

        # Step 1 — Meta (bot identity / tech info)
        if intent == "meta":
            ans = meta_answer(query)
            return ans if ans else ollama_conversational(query, history)

        # Step 2 — Greetings
        GREETINGS = {"هلا", "مرحبا", "السلام عليكم", "يعطيك العافية",
                     "شكرا", "شكراً", "أهلا", "مرحباً", "صباح الخير",
                     "مساء الخير", "ممنون", "يسلمو", "هلو", "هاي"}
        if any(g in query for g in GREETINGS) and len(query.split()) <= 5:
            return ollama_conversational(query, history)

        # Step 3 — Conversational / feedback (never escalate these)
        if intent == "conversational":
            return ollama_conversational(query, history)

        # Step 4 — Build enriched search query (add context if needed)
        search_query  = query
        CONTEXT_WORDS = {"طيب", "شو", "وين", "كيف", "ها", "عن", "اين", "أين", "هيك"}
        is_contextual = any(w in query for w in CONTEXT_WORDS) or len(query.split()) <= 3
        if is_contextual and history:
            for msg in reversed(history):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    lq = msg.get("content", "")
                    if lq and lq != query:
                        search_query = f"{query} (سياق: {lq})"
                    break

        # Step 5 — FAISS
        best_score, best_idx = hybrid_search(search_query)
        fallback_answer = (
            str(df.iloc[best_idx]["Answer"]) if best_score >= THRESHOLD_MEDIUM else None
        )

        # Step 6 — Direct match
        if best_score >= THRESHOLD_DIRECT:
            print(f"✅ Direct hit ({best_score:.2f})")
            return str(fallback_answer)

        # Step 7 — Medium match → Ollama with focused context
        if best_score >= THRESHOLD_MEDIUM:
            print(f"🔶 Medium ({best_score:.2f}) → Ollama")
            norm  = normalize(search_query)
            q_emb = (sentence_model.encode([norm], normalize_embeddings=True,
                                           convert_to_tensor=True)
                     .cpu().numpy().astype("float32"))
            dists, idxs = faiss_index.search(q_emb, min(5, len(df)))
            rows = [df.iloc[idxs[0][i]] for i in range(len(idxs[0]))
                    if dists[0][i] >= THRESHOLD_MEDIUM]
            ctx  = "\n".join(f"س: {r['Question']} | ج: {r['Answer']}" for r in rows)
            ans  = ollama_academic(query, history, ctx)
            return ans if ans else str(fallback_answer)

        # Step 8 — Low score
        print(f"🔴 Low score ({best_score:.2f}), intent={intent}")

        if intent == "academic":
            # Real academic gap → escalate to department head
            cursor.execute(
                "INSERT INTO pending_questions (user_question, status) VALUES (?, ?)",
                (query, "Pending")
            )
            conn.commit()
            return (
                "سؤالك وصلني ✅\n"
                "لا أملك معلومات رسمية عن هذا الموضوع حالياً، "
                "لذا سأحوّله لرئيس القسم للرد عليك قريباً.\n"
                "هل هناك شيء آخر يمكنني مساعدتك به؟"
            )

        # Unknown intent with low academic signal → try Ollama freely, ask for clarification if it fails
        full_ctx = "\n".join(f"س: {r['Question']} | ج: {r['Answer']}" for _, r in df.iterrows())
        ans = ollama_academic(query, history, full_ctx)
        if ans:
            return ans
        return (
            "لم أفهم سؤالك بشكل كافٍ 🤔\n"
            "هل يمكنك توضيح ما تقصده؟ "
            "أنا متخصص في شؤون قسم علم البيانات ويسعدني مساعدتك."
        )

    except Exception as e:
        print(f"🚨 General Error: {e}")
        traceback.print_exc()
        return "أواجه مشكلة تقنية مؤقتة، يرجى المحاولة مجدداً."

# ==========================================
# 11. DB Helper
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
# 12. API Routes
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
    return jsonify({
        "ollama_online": is_ollama_available(),
        "model":         OLLAMA_MODEL,
        "rows_loaded":   len(df),
    })

if __name__ == "__main__":
    app.run(port=5000, debug=False)
