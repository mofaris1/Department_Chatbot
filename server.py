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
HF_TOKEN = os.getenv("HF_TOKEN", "hf_xxxxxxxxxxxxxxx") 
model_id = "meta-llama/Llama-3.3-70B-Instruct"
EXCEL_FILE = "data.xlsx"

# ==========================================
# 2. Databases and FAISS and Model Setup
# ==========================================
conn = sqlite3.connect('unanswered_questions.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS pending_questions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_question TEXT, status TEXT)''')
conn.commit()

def load_excel_data():
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE)
        return df.dropna(subset=['Question', 'Answer'])
    else:
        df = pd.DataFrame({'Question': ['متى ببدأ تسجيل المواد؟'], 'Answer': ['يبدأ التسجيل الأسبوع القادم.']})
        df.to_excel(EXCEL_FILE, index=False)
        return df

df = load_excel_data()

print("⚙️ Models and FAISS are starting...")
sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def build_index(dataframe):
    embeddings = sentence_model.encode(dataframe['Question'].tolist(), normalize_embeddings=True, convert_to_tensor=True)
    embeddings_np = embeddings.cpu().numpy().astype('float32')
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_np)
    return index

faiss_index = build_index(df)
client = InferenceClient(model=model_id, token=HF_TOKEN)

# ==========================================
# 3. Helper Functions
# ==========================================
def submit_head_answer(q_id, answer):
    try:
        global df, faiss_index
        cursor.execute("SELECT user_question FROM pending_questions WHERE id=?", (q_id,))
        question = cursor.fetchone()[0]
        
        # excel update
        new_row = pd.DataFrame({'Question': [question], 'Answer': [answer]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False)
        
        # Index update
        cursor.execute("UPDATE pending_questions SET status='Answered' WHERE id=?", (q_id,))
        conn.commit()
        faiss_index = build_index(df)
        return "✅ تم حفظ الإجابة وتدريب البوت بنجاح!", 200
    except Exception as e:
        return f"❌ خطأ: {e}", 400

# ==========================================
# 4. Bot Brain (RAG Logic)
# ==========================================
def ask_smart_bot(user_query, history, threshold=0.60):
    greetings = ["هلا", "مرحبا", "السلام عليكم", "يعطيك العافية", "شكرا", "أهلا", "مرحباً"]
    if any(greet in user_query for greet in greetings) and len(user_query.split()) <= 3:
        return "أهلاً بك! أنا المرشد الأكاديمي الذكي 🎓. كيف بقدر أساعدك اليوم؟"

    search_query = user_query
    context_words = ["طيب", "ما هي", "شو", "وين", "كيف", "ها", "ه", "عن", "ما"]
    is_contextual = any(word in user_query for word in context_words) or len(user_query.split()) <= 3
    
    if is_contextual and len(history) > 0:
        last_q = ""
        for msg in reversed(history):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_q = msg.get("content", "")
                break
        if last_q: search_query = f"{user_query} (سياق: {last_q})"

    query_embedding = sentence_model.encode([search_query], normalize_embeddings=True, convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().numpy().astype('float32')
    distances, indices = faiss_index.search(query_embedding_np, 1)
    best_score = distances[0][0]

    # direct answer
    if best_score >= 0.85:
        return df.iloc[indices[0][0]]['Answer']

    data_context = "\n".join([f"س: {row['Question']} | ج: {row['Answer']}" for _, row in df.iterrows()])
    
    messages = [{"role": "system", "content": f"أنت مرشد أكاديمي ذكي. استخدم هذه البيانات الرسمية فقط للإجابة:\n{data_context}\n\nإذا لم تجد الإجابة، قل 'مجهول'."}]
    for msg in history:
        messages.append(msg if isinstance(msg, dict) else {"role": "user", "content": msg[0]})
    messages.append({"role": "user", "content": user_query})

    try:
        response = client.chat_completion(messages=messages, max_tokens=150, temperature=0.1)
        bot_reply = response.choices[0].message.content.strip()
        
        if "مجهول" in bot_reply or not bot_reply:
            if best_score >= threshold: return df.iloc[indices[0][0]]['Answer']
            cursor.execute("INSERT INTO pending_questions (user_question, status) VALUES (?, ?)", (user_query, "Pending"))
            conn.commit()
            return "عذراً، سؤالك جديد وسأقوم بتحويله لرئيس القسم للرد عليه قريباً."
        return bot_reply
    except Exception as e:
        print(f"🚨 Error: {e}")
        # if api faild
        if best_score >= 0.50: return df.iloc[indices[0][0]]['Answer']
        return "أواجه مشكلة تقنية حالياً."

# ==========================================
# 5. API Routes
# ==========================================
@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json
    bot_reply = ask_smart_bot(data.get("query"), data.get("history", []))
    return jsonify({"reply": bot_reply})

@app.route('/api/pending', methods=['GET'])
def pending_api():
    query = "SELECT id, user_question FROM pending_questions WHERE status='Pending'"
    df_pending = pd.read_sql_query(query, conn)
    return jsonify(df_pending.to_dict(orient='records'))

@app.route('/api/answer', methods=['POST'])
def answer_api():
    data = request.json
    result_message, status_code = submit_head_answer(data.get("id"), data.get("answer"))
    return jsonify({"success": status_code==200, "message": result_message}), status_code

if __name__ == "__main__":
    app.run(port=5000)
