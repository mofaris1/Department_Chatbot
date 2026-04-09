from flask import Flask, request, jsonify
import pandas as pd
import sqlite3
import os
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import faiss
import numpy as np
import torch

# ==========================================
# 1. Flask and Model Options
# ==========================================
app = Flask(__name__)

HF_TOKEN = os.getenv("HF_TOKEN", "hf_xxxxxxxxxxxxxxxxxxxxxxxx")
model_id = "meta-llama/Llama-3.3-70B-Instruct"
EXCEL_FILE = "data.xlsx"

# ==========================================
# 2. Databases and FAISS
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

question_embeddings = sentence_model.encode(df['Question'].tolist(), normalize_embeddings=True, convert_to_tensor=True)
question_embeddings_np = question_embeddings.cpu().numpy().astype('float32')

embedding_dim = question_embeddings_np.shape[1]
faiss_index = faiss.IndexFlatIP(embedding_dim)
faiss_index.add(question_embeddings_np)

client = InferenceClient(model=model_id, token=HF_TOKEN)

# ==========================================
# 3. Bot Brain
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
            elif isinstance(msg, (list, tuple)) and msg[0]:
                last_q = msg[0]
                break
        
        if last_q:
            search_query = f"{user_query} (سياق: {last_q})"
            print(f"🔗 البحث المحسن بالسياق: {search_query}")

    
    query_embedding = sentence_model.encode([search_query], normalize_embeddings=True, convert_to_tensor=True)
    query_embedding_np = query_embedding.cpu().numpy().astype('float32')
    
    distances, indices = faiss_index.search(query_embedding_np, 1)
    best_score = distances[0][0]
    
    
    if best_score >= 0.85:
        print(f"✅ تم العثور على إجابة مباشرة بدقة: {best_score}")
        return df.iloc[indices[0][0]]['Answer']

    
    system_prompt = f"""أنت مرشد أكاديمي في جامعة العلوم والتكنولوجيا.
استخدم المعلومات التالية للإجابة: {df['Question'].iloc[0]}: {df['Answer'].iloc[0]} ... (وهكذا لجميع البيانات)
إذا كان السؤال "تكملة" لحديث سابق ، ابحث عن العقوبة في بياناتك وأجب بها.
إذا لم تجد الإجابة في البيانات، اكتب كلمة واحدة: مجهول"""

    
    data_context = "\n".join([f"س: {row['Question']} | ج: {row['Answer']}" for _, row in df.iterrows()])
    
    messages = [
    {
        "role": "system", 
        "content": f"""أنت مرشد أكاديمي ذكي. استخدم هذه البيانات الرسمية فقط للإجابة:
        {data_context}
        
        مهم جداً: إذا كان السؤال عن موضوع موجود في البيانات (مثل التدخين)، أجب من البيانات مباشرة ولا تقل مجهول.
        فقط إذا كان السؤال عن موضوع مختلف تماماً وغير موجود نهائياً، قل 'مجهول'."""
    },
]
    
    for msg in history:
        messages.append(msg if isinstance(msg, dict) else {"role": "user", "content": msg[0]})
    
    messages.append({"role": "user", "content": user_query})

    try:
        response = client.chat_completion(messages=messages, max_tokens=150, temperature=0.1)
        bot_reply = response.choices[0].message.content.strip()
        
        if "مجهول" in bot_reply or not bot_reply:
            
            if best_score >= threshold:
                return df.iloc[indices[0][0]]['Answer']
            
            cursor.execute("INSERT INTO pending_questions (user_question, status) VALUES (?, ?)", (user_query, "Pending"))
            conn.commit()
            return "عذراً، سؤالك جديد تماماً أو يحتاج لتفاصيل. تم حفظه وسيتم الرد عليه من رئيس القسم قريباً."
        return bot_reply
    
    except Exception as e:
        print(f"🚨 هذا هو الخطأ: {e}") 
        return "أواجه مشكلة تقنية حالياً."

# ==========================================
# 4. API Routes
# ==========================================
@app.route('/api/chat', methods=['POST'])
def chat_api():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "الرجاء إرسال السؤال (query)"}), 400
        
    user_query = data.get("query")
    history = data.get("history", [])
    
    bot_reply = ask_smart_bot(user_query, history)
    
    return jsonify({"reply": bot_reply})


@app.route('/api/pending', methods=['GET'])
def pending_api():
    query = "SELECT id, user_question FROM pending_questions WHERE status='Pending'"
    df_pending = pd.read_sql_query(query, conn)
    return jsonify(df_pending.to_dict(orient='records'))


@app.route('/api/answer', methods=['POST'])
def answer_api():
    data = request.json
    q_id = data.get("id")
    new_answer = data.get("answer")
    
    result_message, _ = submit_head_answer(q_id, new_answer)
    
    if "✅" in result_message:
        return jsonify({"success": True, "message": result_message})
    else:
        return jsonify({"success": False, "message": result_message}), 400

if __name__ == "__main__":
    app.run(port=5000)