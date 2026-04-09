import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="المساعد الأكاديمي الذكي", page_icon="🎓", layout="wide")

API_URL = "http://127.0.0.1:5000/api/chat"

st.title("🎓 المساعد الأكاديمي الذكي - قسم علم البيانات - جامعة العلوم والتكنولوجيا")

tab1, tab2 = st.tabs(["💬 واجهة الطلاب", "👨‍🏫 لوحة تحكم رئيس القسم"])

# ---------------------------------------------------------
# 1. tab1: Bot
# ---------------------------------------------------------
with tab1:
    st.header("دردش مع المرشد الأكاديمي")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    if prompt := st.chat_input("اسألني عن أي شيء يخص القسم..."):

        with st.chat_message("user"):
            st.markdown(prompt)
        

        history_data = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]


        try:
            with st.spinner("جاري التفكير..."):
                response = requests.post(API_URL, json={
                    "query": prompt,
                    "history": history_data
                })
            
            if response.status_code == 200:
                answer = response.json().get("reply", "عذراً، لم أستطع فهم الإجابة.")
            else:
                answer = f"❌ خطأ في السيرفر: {response.status_code}"
                
        except Exception as e:
            answer = f"❌ فشل الاتصال بالسيرفر: {e}"


        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})


        with st.chat_message("assistant"):
            st.markdown(answer)
        

        st.rerun()

# ---------------------------------------------------------
# 2. tab2: Admin
# ---------------------------------------------------------
with tab2:
    st.header("👨‍🏫 لوحة تحكم رئيس القسم")
    
    try:
        pending_resp = requests.get("http://127.0.0.1:5000/api/pending")
        if pending_resp.status_code == 200:
            pending_data = pending_resp.json()
            if pending_data:
                df_p = pd.DataFrame(pending_data)
                st.write("### 📌 أسئلة الطلاب بانتظار ردك:")
                st.table(df_p)
                
                st.divider()
                st.write("### ✍️ إضافة إجابة رسمية")
                
                with st.form("answer_form"):
                    q_id = st.number_input("أدخل رقم السؤال (ID)", step=1, min_value=1)
                    official_answer = st.text_area("الإجابة النموذجية")
                    submit_answer = st.form_submit_button("حفظ وتدريب البوت ✅")
                    
                    if submit_answer:
                        if official_answer:
                            res = requests.post("http://127.0.0.1:5000/api/answer", 
                                             json={"id": q_id, "answer": official_answer})
                            if res.status_code == 200:
                                st.success(res.json()['message'])
                                st.rerun() 
                            else:
                                st.error(res.json()['message'])
                        else:
                            st.warning("الرجاء كتابة الإجابة أولاً")
            else:
                st.success("🎉 لا توجد أسئلة معلقة حالياً")
        else:
            st.error("فشل في جلب الأسئلة من السيرفر")
    except Exception as e:
        st.error(f"خطأ في الاتصال: {e}")