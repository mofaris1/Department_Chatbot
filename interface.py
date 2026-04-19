import streamlit as st
import requests
import pandas as pd

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="المرشد الأكاديمي | علم البيانات - JUST",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL     = "http://127.0.0.1:5000/api/chat"
STATUS_URL  = "http://127.0.0.1:5000/api/status"
PENDING_URL = "http://127.0.0.1:5000/api/pending"
ANSWER_URL  = "http://127.0.0.1:5000/api/answer"

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&family=Tajawal:wght@300;400;500;700&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stApp"] {
    background: #0a0f1e !important;
    color: #e8eaf0 !important;
    font-family: 'Cairo', 'Tajawal', sans-serif !important;
    direction: rtl !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e4d8c; border-radius: 4px; }

/* ══════════════════════════════════════════
   HERO HEADER
══════════════════════════════════════════ */
.just-hero {
    background: linear-gradient(135deg, #0d1b3e 0%, #0a3a6b 40%, #0d5c9e 100%);
    border-bottom: 2px solid #1a6fc4;
    padding: 0;
    margin-bottom: 0;
    position: relative;
    overflow: hidden;
}

.just-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 80% at 80% 50%, rgba(26,111,196,0.25) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 20% 30%, rgba(0,180,255,0.08) 0%, transparent 60%);
    pointer-events: none;
}

.hero-inner {
    display: flex;
    align-items: center;
    gap: 24px;
    padding: 20px 36px;
    position: relative;
    z-index: 1;
}

.hero-logo-ring {
    width: 72px; height: 72px; flex-shrink: 0;
    background: linear-gradient(135deg, #1a6fc4, #0a3a6b);
    border: 2px solid rgba(255,255,255,0.2);
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 36px;
    box-shadow: 0 8px 32px rgba(26,111,196,0.4);
}

.hero-text { flex: 1; text-align: right; }
.hero-title {
    font-size: 1.65rem; font-weight: 900; line-height: 1.2;
    color: #ffffff;
    letter-spacing: -0.5px;
}
.hero-title span { color: #5bb8ff; }
.hero-sub {
    font-size: 0.85rem; font-weight: 400;
    color: rgba(255,255,255,0.65);
    margin-top: 4px;
}

.hero-badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.75rem;
    color: #a8d4ff;
    white-space: nowrap;
    backdrop-filter: blur(8px);
}

/* ══════════════════════════════════════════
   STATUS BAR
══════════════════════════════════════════ */
.status-bar {
    background: #0d1428;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 8px 36px;
    display: flex;
    align-items: center;
    gap: 24px;
    font-size: 0.75rem;
    color: rgba(255,255,255,0.5);
}
.status-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    margin-left: 5px;
    vertical-align: middle;
}
.dot-on  { background: #2ddc74; box-shadow: 0 0 6px #2ddc74; }
.dot-off { background: #ff4d4d; box-shadow: 0 0 6px #ff4d4d; }

/* ══════════════════════════════════════════
   TABS
══════════════════════════════════════════ */
[data-testid="stTabs"] { background: transparent !important; }

[data-testid="stTabs"] > div:first-child {
    background: #0d1428 !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    padding: 0 36px !important;
    gap: 4px !important;
}

button[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    color: rgba(255,255,255,0.45) !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    padding: 14px 20px !important;
    transition: all .2s !important;
    direction: rtl !important;
}
button[data-baseweb="tab"]:hover {
    color: #a8d4ff !important;
    border-bottom-color: rgba(91,184,255,0.3) !important;
}
button[aria-selected="true"][data-baseweb="tab"] {
    color: #5bb8ff !important;
    border-bottom-color: #1a6fc4 !important;
    background: rgba(26,111,196,0.08) !important;
}

/* ══════════════════════════════════════════
   CHAT AREA
══════════════════════════════════════════ */
.chat-wrap {
    max-width: 860px;
    margin: 0 auto;
    padding: 24px 16px 120px;
}

/* Streamlit message containers */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 6px 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"] .stMarkdown,
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) .stMarkdown p {
    background: linear-gradient(135deg, #1a3d6b, #0d5c9e) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 12px 18px !important;
    color: #e8f4ff !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    max-width: 75% !important;
    margin-right: auto !important;
}

/* Assistant bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) .stMarkdown p {
    background: #131d35 !important;
    border: 1px solid rgba(26,111,196,0.25) !important;
    border-radius: 18px 18px 18px 4px !important;
    padding: 12px 18px !important;
    color: #d0dff5 !important;
    font-size: 0.95rem !important;
    line-height: 1.8 !important;
    max-width: 80% !important;
    margin-left: auto !important;
}

/* Avatar icons */
[data-testid="chatAvatarIcon-user"] {
    background: linear-gradient(135deg, #1a6fc4, #0a3a6b) !important;
    border-radius: 10px !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: linear-gradient(135deg, #0d5c9e, #063060) !important;
    border-radius: 10px !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #0d1428 !important;
    border-top: 1px solid rgba(26,111,196,0.3) !important;
    padding: 16px 36px !important;
}
[data-testid="stChatInput"] textarea {
    background: #131d35 !important;
    border: 1px solid rgba(26,111,196,0.4) !important;
    border-radius: 14px !important;
    color: #e8eaf0 !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 0.95rem !important;
    direction: rtl !important;
    padding: 12px 18px !important;
    transition: border-color .2s !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #1a6fc4 !important;
    box-shadow: 0 0 0 3px rgba(26,111,196,0.15) !important;
}
[data-testid="stChatInput"] button {
    background: linear-gradient(135deg, #1a6fc4, #0a3a6b) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #5bb8ff !important; }

/* ══════════════════════════════════════════
   QUICK QUESTIONS
══════════════════════════════════════════ */
.quick-label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.4);
    text-align: center;
    margin-bottom: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.quick-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-bottom: 28px;
}
.quick-btn {
    background: rgba(26,111,196,0.1);
    border: 1px solid rgba(26,111,196,0.3);
    border-radius: 20px;
    padding: 7px 16px;
    font-size: 0.82rem;
    color: #a8d4ff;
    cursor: pointer;
    font-family: 'Cairo', sans-serif;
    transition: all .2s;
    white-space: nowrap;
}
.quick-btn:hover {
    background: rgba(26,111,196,0.25);
    border-color: #1a6fc4;
    color: #fff;
    transform: translateY(-1px);
}

/* ══════════════════════════════════════════
   WELCOME CARD (empty state)
══════════════════════════════════════════ */
.welcome-card {
    text-align: center;
    padding: 48px 24px;
    max-width: 580px;
    margin: 0 auto;
}
.welcome-icon {
    font-size: 56px;
    margin-bottom: 20px;
    display: block;
    filter: drop-shadow(0 0 20px rgba(26,111,196,0.5));
}
.welcome-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 10px;
}
.welcome-sub {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.5);
    line-height: 1.7;
    margin-bottom: 28px;
}
.capability-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    text-align: right;
}
.cap-item {
    background: rgba(26,111,196,0.08);
    border: 1px solid rgba(26,111,196,0.2);
    border-radius: 12px;
    padding: 12px 14px;
    font-size: 0.82rem;
    color: rgba(255,255,255,0.7);
    display: flex;
    align-items: center;
    gap: 8px;
}
.cap-icon { font-size: 1.1rem; }

/* ══════════════════════════════════════════
   ADMIN PANEL
══════════════════════════════════════════ */
.admin-wrap {
    max-width: 900px;
    margin: 0 auto;
    padding: 28px 16px;
}
.admin-section-title {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #5bb8ff;
    margin-bottom: 14px;
}
.stat-row {
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
}
.stat-card {
    flex: 1;
    background: #0d1428;
    border: 1px solid rgba(26,111,196,0.2);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}
.stat-num {
    font-size: 2rem;
    font-weight: 900;
    color: #5bb8ff;
    line-height: 1;
}
.stat-lbl {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.45);
    margin-top: 4px;
}
.pending-table-wrap {
    background: #0d1428;
    border: 1px solid rgba(26,111,196,0.2);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 24px;
}
[data-testid="stDataFrame"] {
    background: transparent !important;
    color: #e8eaf0 !important;
    font-family: 'Cairo', sans-serif !important;
    font-size: 0.88rem !important;
    direction: rtl !important;
}

/* ── Forms ── */
.stForm {
    background: #0d1428 !important;
    border: 1px solid rgba(26,111,196,0.2) !important;
    border-radius: 14px !important;
    padding: 24px !important;
}
.stNumberInput input, .stTextArea textarea {
    background: #131d35 !important;
    border: 1px solid rgba(26,111,196,0.3) !important;
    border-radius: 10px !important;
    color: #e8eaf0 !important;
    font-family: 'Cairo', sans-serif !important;
    direction: rtl !important;
}
label { color: rgba(255,255,255,0.65) !important; font-family: 'Cairo',sans-serif !important; }

/* Submit button */
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #1a6fc4, #0a3a6b) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Cairo', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 10px 28px !important;
    transition: all .2s !important;
}
[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(26,111,196,0.4) !important;
}

/* Alerts */
.stSuccess { background: rgba(45,220,116,0.1) !important; border-color: #2ddc74 !important; }
.stError   { background: rgba(255,77,77,0.1)  !important; border-color: #ff4d4d !important; }
.stWarning { background: rgba(255,179,0,0.1)  !important; border-color: #ffb300 !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 24px 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def get_status():
    try:
        r = requests.get(STATUS_URL, timeout=4)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def get_pending():
    try:
        r = requests.get(PENDING_URL, timeout=4)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


# ─── Hero Header ────────────────────────────────────────────────────────────
status = get_status()
ollama_on = status.get("ollama_online", False)
rows = status.get("rows_loaded", "—")

st.markdown(f"""
<div class="just-hero">
  <div class="hero-inner">
    <div class="hero-logo-ring">🎓</div>
    <div class="hero-text">
      <div class="hero-title">المرشد الأكاديمي الذكي <span>| علم البيانات</span></div>
      <div class="hero-sub">جامعة العلوم والتكنولوجيا الأردنية &nbsp;·&nbsp; Jordan University of Science and Technology</div>
    </div>
    <div class="hero-badge">JUST · DS Dept</div>
  </div>
</div>
<div class="status-bar">
  <span>
    <span class="status-dot {'dot-on' if ollama_on else 'dot-off'}"></span>
    {'النموذج: ' + status.get('model','—') if ollama_on else 'النموذج غير متصل'}
  </span>
  <span>📊 قاعدة البيانات: {rows} سؤال</span>
  <span style="margin-right:auto; color:rgba(255,255,255,0.25)">Academic Advisor · قسم علم البيانات</span>
</div>
""", unsafe_allow_html=True)


# ─── TABS ────────────────────────────────────────────────────────────────────
tab_student, tab_admin = st.tabs(["💬  واجهة الطالب", "👨‍🏫  لوحة الإدارة"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — STUDENT CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_student:
    st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

    # ── Session state ──────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    # ── Empty state / welcome ──────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
          <span class="welcome-icon">🤖</span>
          <div class="welcome-title">أهلاً بك في المرشد الأكاديمي!</div>
          <div class="welcome-sub">
            مساعدك الذكي المخصص لقسم علم البيانات في جامعة العلوم والتكنولوجيا.<br>
            اسألني عن أي شيء يخص قسمك أو مساقاتك أو أعضاء هيئة التدريس.
          </div>
          <div class="capability-grid">
            <div class="cap-item"><span class="cap-icon">📋</span> تسجيل المواد والجداول</div>
            <div class="cap-item"><span class="cap-icon">👨‍🏫</span> معلومات الدكاترة والمكاتب</div>
            <div class="cap-item"><span class="cap-icon">📧</span> إيميلات هيئة التدريس</div>
            <div class="cap-item"><span class="cap-icon">🎓</span> متطلبات التخرج</div>
            <div class="cap-item"><span class="cap-icon">📅</span> الأنظمة والتعليمات</div>
            <div class="cap-item"><span class="cap-icon">💬</span> دردشة أكاديمية حرة</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick questions
        st.markdown('<div class="quick-label">أسئلة شائعة</div>', unsafe_allow_html=True)
        quick_qs = [
            "كيف أسجل مواد؟",
            "ما هو إيميل رئيس القسم؟",
            "كم ساعة يحتاج التخرج؟",
            "ما هي مواد الخطة الدراسية؟",
            "ما هي ساعات الدوام؟",
            "كيف أتواصل مع الإرشاد الأكاديمي؟",
        ]
        cols = st.columns(3)
        for i, q in enumerate(quick_qs):
            if cols[i % 3].button(q, key=f"quick_{i}", use_container_width=True):
                st.session_state.pending_prompt = q
                st.rerun()

    # ── Render history ─────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Handle quick-question injection ───────────────────────────────────
    if st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
    else:
        prompt = st.chat_input("اسألني عن أي شيء يخص قسم علم البيانات...")

    # ── Process message ────────────────────────────────────────────────────
    if prompt:
        with st.chat_message("user"):
            st.markdown(prompt)

        history_data = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        try:
            with st.spinner("جاري التفكير..."):
                resp = requests.post(
                    API_URL,
                    json={"query": prompt, "history": history_data},
                    timeout=60,
                )
            answer = (
                resp.json().get("reply", "عذراً، لم أستطع فهم الإجابة.")
                if resp.status_code == 200
                else f"❌ خطأ في السيرفر: {resp.status_code}"
            )
        except Exception as e:
            answer = f"❌ فشل الاتصال بالسيرفر: {e}"

        st.session_state.messages.append({"role": "user",      "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.rerun()

    # ── Clear chat button ─────────────────────────────────────────────────
    if st.session_state.messages:
        if st.button("🗑️ مسح المحادثة", key="clear"):
            st.session_state.messages = []
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ADMIN PANEL
# ══════════════════════════════════════════════════════════════════════════════
with tab_admin:
    st.markdown('<div class="admin-wrap">', unsafe_allow_html=True)

    # ── Stats row ─────────────────────────────────────────────────────────
    pending_data = get_pending()
    n_pending = len(pending_data)

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card">
        <div class="stat-num">{n_pending}</div>
        <div class="stat-lbl">سؤال معلق</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">{rows}</div>
        <div class="stat-lbl">إجابة في قاعدة البيانات</div>
      </div>
      <div class="stat-card">
        <div class="stat-num">{'🟢' if ollama_on else '🔴'}</div>
        <div class="stat-lbl">{'النموذج متصل' if ollama_on else 'النموذج غير متصل'}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Pending questions ─────────────────────────────────────────────────
    st.markdown('<div class="admin-section-title">📌 أسئلة الطلاب المعلقة</div>',
                unsafe_allow_html=True)

    if pending_data:
        df_p = pd.DataFrame(pending_data).rename(
            columns={"id": "رقم", "user_question": "سؤال الطالب"}
        )
        st.markdown('<div class="pending-table-wrap">', unsafe_allow_html=True)
        st.dataframe(df_p, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.success("🎉 لا توجد أسئلة معلقة حالياً — ممتاز!")

    st.divider()

    # ── Answer form ───────────────────────────────────────────────────────
    st.markdown('<div class="admin-section-title">✍️ إضافة إجابة رسمية وتدريب البوت</div>',
                unsafe_allow_html=True)

    with st.form("answer_form", clear_on_submit=True):
        col_id, col_ans = st.columns([1, 3])
        with col_id:
            q_id = st.number_input("رقم السؤال (ID)", step=1, min_value=1)
        with col_ans:
            official_answer = st.text_area(
                "الإجابة الرسمية",
                height=120,
                placeholder="اكتب الإجابة النموذجية هنا...",
            )
        submitted = st.form_submit_button("💾 حفظ وتدريب البوت", use_container_width=False)

        if submitted:
            if official_answer.strip():
                try:
                    res = requests.post(
                        ANSWER_URL,
                        json={"id": int(q_id), "answer": official_answer.strip()},
                        timeout=15,
                    )
                    if res.status_code == 200:
                        st.success(res.json().get("message", "✅ تم الحفظ!"))
                        st.cache_data.clear()
                        st.rerun()
                    else:
                        st.error(res.json().get("message", "❌ حدث خطأ"))
                except Exception as e:
                    st.error(f"❌ خطأ في الاتصال: {e}")
            else:
                st.warning("⚠️ الرجاء كتابة الإجابة أولاً")

    st.divider()

    # ── System info ───────────────────────────────────────────────────────
    st.markdown('<div class="admin-section-title">⚙️ معلومات النظام</div>',
                unsafe_allow_html=True)
    if status:
        c1, c2 = st.columns(2)
        c1.markdown(f"**النموذج:** `{status.get('model','—')}`")
        c2.markdown(f"**حالة Ollama:** {'✅ متصل' if ollama_on else '❌ غير متصل'}")
    else:
        st.error("تعذّر الاتصال بالخادم")

    st.markdown('</div>', unsafe_allow_html=True)
