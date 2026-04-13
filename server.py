import sys
from flask import Flask, request, jsonify
import pandas as pd
import sqlite3
import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import faiss
import numpy as np
import torch

app = Flask(__name__)

# ==========================================
# 1. Flask and Model Options
# ==========================================
HF_TOKEN   = os.getenv("HF_TOKEN", "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")
model_id   = "meta-llama/Llama-3.3-70B-Instruct"
EXCEL_FILE = "project_data.xlsx"

# ==========================================
# 2. Database Setup
# ==========================================
conn = sqlite3.connect('unanswered_questions.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS pending_questions
                  (id INTEGER PRIMARY KEY AUTOINCREMENT,
                   user_question TEXT, status TEXT)''')
conn.commit()

# ------------------------------------------------------------------
# FIX: load_excel_data() now reads ALL sheets and merges them.
# Original code: pd.read_excel(EXCEL_FILE)
#   → reads only the first sheet, all other sheets are invisible.
# Fix: loop over every sheet, keep the 3 columns we need
#   (Question, Answer, Keywords), then concatenate everything.
# ------------------------------------------------------------------
def load_excel_data():
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame({'Question': ['متى ببدأ تسجيل المواد؟'],
                           'Answer':   ['يبدأ التسجيل الأسبوع القادم.'],
                           'Keywords': ['']})
        df.to_excel(EXCEL_FILE, index=False)
        return df

    xl     = pd.ExcelFile(EXCEL_FILE)
    frames = []
    for sheet in xl.sheet_names:
        raw   = pd.read_excel(xl, sheet_name=sheet)
        chunk = pd.DataFrame({
            'Question': raw['Question'] if 'Question' in raw.columns else pd.Series(dtype=str),
            'Answer':   raw['Answer']   if 'Answer'   in raw.columns else pd.Series(dtype=str),
            'Keywords': raw['Keywords'] if 'Keywords' in raw.columns else pd.Series(dtype=str),
        })
        frames.append(chunk)

    df = (pd.concat(frames, ignore_index=True)
            .dropna(subset=['Question', 'Answer']))
    df = df[df['Answer'].astype(str).str.strip() != ''].reset_index(drop=True)
    print(f"✅ Loaded {len(df)} rows from {len(xl.sheet_names)} sheet(s): {xl.sheet_names}")
    return df

df = load_excel_data()

print("⚙️ Loading embedding model and building FAISS index...")
sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Combine Question + Keywords → richer embeddings, better fuzzy matching
def get_embed_texts(dataframe):
    texts = []
    for _, row in dataframe.iterrows():
        q  = str(row['Question'])
        kw = str(row.get('Keywords', ''))
        texts.append(f"{q} {kw}".strip() if kw and kw.lower() != 'nan' else q)
    return texts

def build_index(dataframe):
    emb    = sentence_model.encode(get_embed_texts(dataframe),
                                   normalize_embeddings=True,
                                   convert_to_tensor=True)
    emb_np = emb.cpu().numpy().astype('float32')
    index  = faiss.IndexFlatIP(emb_np.shape[1])
    index.add(emb_np)
    return index

faiss_index = build_index(df)
client      = InferenceClient(model=model_id, token=HF_TOKEN)
print("✅ Ready.")

# ==========================================
# 3. Helper Functions
# ==========================================
def submit_head_answer(q_id, answer):
    try:
        global df, faiss_index
        cursor.execute("SELECT user_question FROM pending_questions WHERE id=?", (q_id,))
        question = cursor.fetchone()[0]
        new_row  = pd.DataFrame({'Question': [question], 'Answer': [answer], 'Keywords': ['']})
        df       = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False)
        cursor.execute("UPDATE pending_questions SET status='Answered' WHERE id=?", (q_id,))
        conn.commit()
        faiss_index = build_index(df)
        return "✅ تم حفظ الإجابة وتدريب البوت بنجاح!", 200
    except Exception as e:
        return f"❌ خطأ: {e}", 400

# ==========================================
# 4. Bot Brain
# ==========================================
# Three-tier threshold system:
#   ≥ 0.85  → near-exact FAISS match, return directly, no LLM call
#   ≥ 0.65  → decent match, send top-5 rows to LLM as focused context
#   < 0.65  → weak match, send full dataset to LLM for max coverage;
#             if LLM still says مجهول → log as pending, NEVER return a
#             low-confidence FAISS answer (that was the random answer bug)
THRESHOLD_DIRECT = 0.75   # CHANGED from 0.85 — catches dialect/formal variations (شو vs ما)
THRESHOLD_MEDIUM = 0.60   # CHANGED from 0.65 — slightly wider net before going to LLM


# ------------------------------------------------------------------
# FIX: Hybrid scoring — semantic (FAISS) + word overlap (Jaccard)
#
# Problem: all email/office questions share identical structure
#   "ما هو الإيميل الجامعي للدكتور X" — the embedding model treats
#   them as nearly the same vector and picks wrong names.
#   Pure semantic search can't tell "امل الزعبي" from "علاء الهويدي".
#
# Fix: after FAISS returns top-K candidates, re-rank them using a
#   combined score: 50% semantic + 50% word-overlap (Jaccard).
#   Word overlap is zero when names don't match → kills wrong rows.
# ------------------------------------------------------------------
# Dialect-to-formal word map so "شو/وين/شلون" match stored questions
DIALECT_MAP = {
    "شو": "ما", "وين": "أين", "اين": "أين", "كيفش": "كيف",
    "شلون": "كيف", "ليش": "لماذا", "هيك": "هكذا",
    "بدي": "أريد", "بقدر": "أستطيع", "لازم": "يجب",
    "هلأ": "الآن", "هلق": "الآن",
}

def normalize_query(text):
    """Replace dialect words with their formal equivalents."""
    words = text.split()
    return " ".join(DIALECT_MAP.get(w, w) for w in words)

def word_overlap(query, stored_question):
    """Jaccard similarity on word sets — catches name mismatches."""
    q_words = set(query.split())
    s_words = set(str(stored_question).split())
    # Remove very common Arabic stop-words so they don't inflate the score
    stops = {"ما","هو","هي","في","من","على","إلى","عن","مع","هل","كيف",
             "أين","متى","لماذا","الجامعي","الجامعية","للدكتور","للدكتورة",
             "مكتب","الإيميل","ايميل","ال","و","يتم","يمكن"}
    q_words -= stops
    s_words -= stops
    if not q_words or not s_words:
        return 0.0
    return len(q_words & s_words) / len(q_words | s_words)

def hybrid_search(user_query, top_k=10):
    """
    1. Normalize dialect words in the query.
    2. FAISS semantic search for top_k candidates.
    3. Re-rank by: 0.5 * semantic_score + 0.5 * word_overlap_score.
    4. Return (best_combined_score, best_row_index).
    """
    normalized = normalize_query(user_query)
    q_emb = (sentence_model
             .encode([normalized], normalize_embeddings=True, convert_to_tensor=True)
             .cpu().numpy().astype('float32'))

    k                  = min(top_k, len(df))
    distances, indices = faiss_index.search(q_emb, k)

    best_combined = -1
    best_idx      = indices[0][0]

    for rank in range(k):
        sem_score  = float(distances[0][rank])
        row_idx    = indices[0][rank]
        stored_q   = df.iloc[row_idx]['Question']
        overlap    = word_overlap(normalized, stored_q)
        combined   = 0.5 * sem_score + 0.5 * overlap
        if combined > best_combined:
            best_combined = combined
            best_idx      = row_idx

    return best_combined, best_idx, float(distances[0][0])   # combined, best row, raw top-1 sem score


def ask_smart_bot(user_query, history):
    greetings = ["هلا", "مرحبا", "السلام عليكم", "يعطيك العافية", "شكرا", "أهلا", "مرحباً"]
    if any(g in user_query for g in greetings) and len(user_query.split()) <= 3:
        return "أهلاً بك! أنا المرشد الأكاديمي الذكي 🎓. كيف بقدر أساعدك اليوم؟"

    search_query  = user_query
    context_words = ["طيب", "ما هي", "شو", "وين", "كيف", "ها", "ه", "عن", "ما", "اين", "أين"]
    is_contextual = any(w in user_query for w in context_words) or len(user_query.split()) <= 3

    if is_contextual and history:
        for msg in reversed(history):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_q = msg.get("content", "")
                if last_q:
                    search_query = f"{user_query} (سياق: {last_q})"
                break

    best_score, best_idx, sem_score = hybrid_search(search_query)

    # Tier 1 — strong combined match → return directly, no LLM
    if best_score >= THRESHOLD_DIRECT:
        return df.iloc[best_idx]['Answer']

    # Build LLM context from top-5 semantic candidates
    q_emb = (sentence_model
             .encode([normalize_query(search_query)], normalize_embeddings=True,
                     convert_to_tensor=True).cpu().numpy().astype('float32'))
    distances, indices = faiss_index.search(q_emb, min(5, len(df)))

    if best_score >= THRESHOLD_MEDIUM:
        rows         = [df.iloc[indices[0][i]] for i in range(len(indices[0]))
                        if distances[0][i] >= THRESHOLD_MEDIUM]
        data_context = "\n".join(f"س: {r['Question']} | ج: {r['Answer']}" for r in rows)
    else:
        data_context = "\n".join(f"س: {r['Question']} | ج: {r['Answer']}"
                                 for _, r in df.iterrows())

    messages = [{"role": "system", "content": (
        "أنت مرشد أكاديمي ذكي لقسم علم البيانات في جامعة العلوم والتكنولوجيا. "
        "استخدم فقط البيانات الرسمية التالية للإجابة بدقة وإيجاز:\n\n"
        f"{data_context}\n\n"
        "إذا لم تجد إجابة واضحة في البيانات أعلاه، أجب بكلمة 'مجهول' فقط."
    )}]
    for msg in history:
        messages.append(msg if isinstance(msg, dict) else {"role": "user", "content": msg[0]})
    messages.append({"role": "user", "content": user_query})

    try:
        response  = client.chat_completion(messages=messages, max_tokens=300, temperature=0.1)
        bot_reply = response.choices[0].message.content.strip()

        if "مجهول" in bot_reply or not bot_reply:
            if best_score >= THRESHOLD_MEDIUM:
                return df.iloc[best_idx]['Answer']
            cursor.execute("INSERT INTO pending_questions (user_question, status) VALUES (?, ?)",
                           (user_query, "Pending"))
            conn.commit()
            return "عذراً، سؤالك جديد وسأقوم بتحويله لرئيس القسم للرد عليه قريباً."
        return bot_reply

    except Exception as e:
        print(f"🚨 LLM Error: {e}")
        if best_score >= THRESHOLD_MEDIUM:
            return df.iloc[best_idx]['Answer']
        return "أواجه مشكلة تقنية حالياً."

# ==========================================
# 5. API Routes
# ==========================================
@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json
    return jsonify({"reply": ask_smart_bot(data.get("query"), data.get("history", []))})

@app.route('/api/pending', methods=['GET'])
def pending_api():
    df_p = pd.read_sql_query(
        "SELECT id, user_question FROM pending_questions WHERE status='Pending'", conn)
    return jsonify(df_p.to_dict(orient='records'))

@app.route('/api/answer', methods=['POST'])
def answer_api():
    data = request.json
    msg, code = submit_head_answer(data.get("id"), data.get("answer"))
    return jsonify({"success": code == 200, "message": msg}), code

if __name__ == "__main__":
    app.run(port=5000)