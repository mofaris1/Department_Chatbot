import sys, os, re, io, time, asyncio, traceback, sqlite3, requests
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import ollama
from difflib import SequenceMatcher
from urllib.parse import quote_plus
try:
    import fitz; _FITZ_OK = True
except: _FITZ_OK = False
try:
    import edge_tts; _EDGE_TTS_OK = True
except: _EDGE_TTS_OK = False
app = Flask(__name__); CORS(app)
# ═══════════════════════════════════════
# 1. CONFIG
# ═══════════════════════════════════════
OLLAMA_MODEL = "deepseek-r1:8b"
EXCEL_FILE = "project_data_V2.xlsx"
OLLAMA_HOST = "http://localhost:11434"
JUST_BASE = "https://www.just.edu.jo"
TTS_VOICE = "ar-JO-SanaNeural"
HIGH_CONF = 0.60; MED_CONF = 0.35; LOW_FAB = 0.35
# ═══════════════════════════════════════
# 2. DB
# ═══════════════════════════════════════
conn = sqlite3.connect("unanswered_questions.db", check_same_thread=False)
cur = conn.cursor()
cur.execute("CREATE TABLE IF NOT EXISTS pending_questions (id INTEGER PRIMARY KEY AUTOINCREMENT, user_question TEXT, status TEXT)")
conn.commit()
# ═══════════════════════════════════════
# 3. DATA & EMBEDDINGS
# ═══════════════════════════════════════
def load_excel():
    if not os.path.exists(EXCEL_FILE):
        df = pd.DataFrame({"Question":["متى يبدأ التسجيل؟"],"Answer":["الأسبوع القادم."],"Keywords":[""]})
        df.to_excel(EXCEL_FILE, index=False); return df
    xl = pd.ExcelFile(EXCEL_FILE); frames = []
    for s in xl.sheet_names:
        raw = pd.read_excel(xl, sheet_name=s)
        frames.append(pd.DataFrame({"Question":raw.get("Question",pd.Series(dtype=str)),"Answer":raw.get("Answer",pd.Series(dtype=str)),"Keywords":raw.get("Keywords",pd.Series(dtype=str))}))
    r = pd.concat(frames, ignore_index=True).dropna(subset=["Question","Answer"])
    r = r[r["Answer"].astype(str).str.strip()!=""].reset_index(drop=True)
    print(f"✅ Loaded {len(r)} rows from {len(xl.sheet_names)} sheet(s)"); return r
df = load_excel()
print("⚙️  Loading embedding model …")
smodel = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
def _embed_texts(d):
    ts = []
    for _, r in d.iterrows():
        q, kw, a = str(r["Question"]), str(r.get("Keywords","")), str(r["Answer"])[:200]
        t = f"{q} {kw}".strip() if kw and kw.lower()!="nan" else q
        ts.append(f"{t} {a}")
    return ts
def _build_idx(d):
    e = smodel.encode(_embed_texts(d), normalize_embeddings=True, convert_to_tensor=True)
    ea = e.cpu().numpy().astype("float32"); idx = faiss.IndexFlatIP(ea.shape[1]); idx.add(ea); return idx
fidx = _build_idx(df); print("✅ FAISS ready.")
# ═══════════════════════════════════════
# 3b. PROFESSOR INFO INDEX
_prof_info = {}
_TITLES_RE = r'^(الدكتور[ةه]?|الأستاذ[ةه]?|الاستاذ[ةه]?|دكتور[ةه]?|استاذ[ةه]?|المهندس[ةه]?|مهندس[ةه]?)\s+'
_PFX_TITLES_RE = r'(لل|بال|وال|فال|كال|ال)(الدكتور[ةه]?|الأستاذ[ةه]?|الاستاذ[ةه]?|دكتور[ةه]?|استاذ[ةه]?|المهندس[ةه]?|مهندس[ةه]?)\s+([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)'
_NP = re.compile(r'(الدكتور[ةه]?|الأستاذ[ةه]?|الاستاذ[ةه]?|دكتور[ةه]?|استاذ[ةه]?|المهندس[ةه]?|مهندس[ةه]?)\s+([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)', re.UNICODE)

def _build_prof_index():
    global _prof_info; _prof_info = {}
    for idx, row in df.iterrows():
        q, a = str(row["Question"]), str(row["Answer"])
        name = None
        m = _NP.search(q) or _NP.search(a)
        if m: name = f"{m.group(1)} {m.group(2).strip()}"
        else:
            pm = re.search(_PFX_TITLES_RE, q, re.UNICODE) or re.search(_PFX_TITLES_RE, a, re.UNICODE)
            if pm: name = f"{pm.group(2)} {pm.group(3).strip()}"
        if not name:
            if re.search(r'رئيس\s*(?:ال)?قسم', q) or re.search(r'رئيس\s*(?:ال)?قسم', a): name = "رئيس القسم"
            elif 'عميد' in q or 'عميد' in a: name = "العميد"
            else: continue
        nn = norm_ar(name); bare = re.sub(_TITLES_RE, '', name).strip()
        if bare == name or len(bare.split()) < 2:
            bare2 = re.sub(r'^(دكتور[ةه]?|استاذ[ةه]?|مهندس[ةه]?)\s+', '', name).strip()
            if len(bare2.split()) >= 2 and len(bare2) < len(bare): bare = bare2
        nb = norm_ar(bare); keys = set()
        if nb and len(nb) >= 3: keys.add(nb)
        if nn and len(nn) >= 3: keys.add(nn)
        if not nn.startswith("ال") and len(norm_ar("ال"+name)) >= 3: keys.add(norm_ar("ال"+name))
        tonly = re.sub(r'^(ال)', '', name) if name.startswith("ال") else None
        if tonly and len(tonly.split()) >= 2: keys.add(norm_ar(tonly))
        keys = {k for k in keys if len(k) >= 3}
        if not keys: continue
        mk = nb if nb and len(nb) >= 3 else next(iter(keys))
        if mk not in _prof_info:
            _prof_info[mk] = {"name":None,"email":None,"office":None,"role":None,"phone":None,"department":None,"rows":[]}
        master = _prof_info[mk]
        if bare and name != "رئيس القسم": master["name"] = bare
        for key in keys:
            if key not in _prof_info: _prof_info[key] = master
        em = re.search(r'[\w.\-+]+@[\w.\-]+\.\w{2,}', a)
        if em and not master["email"]: master["email"] = em.group(0)
        of = re.search(r'(مبنى\s*\S+\s*الطابق\s*\S+(?:\s*\([^)]+\))?)', a)
        if of and not master["office"]: master["office"] = of.group(0).strip()
        of2 = re.search(r'(مبنى\s*\S+(?:\s*\([^)]+\))?)', a)
        if of2 and not master["office"]: master["office"] = of2.group(0).strip()
        ph = re.search(r'(\+?962\s*\d[\d\s\-]{6,12}\d|0[7-9]\d[\d\s\-]{6,8}\d)', a)
        if ph and not master["phone"]: master["phone"] = ph.group(0).strip()
        dep = re.search(r'قسم\s*([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)', a)
        if dep and not master["department"]: master["department"] = dep.group(1).strip()
        if re.search(r'رئيس\s*(?:ال)?قسم', a) or re.search(r'رئيس\s*(?:ال)?قسم', q): master["role"] = "رئيس القسم"
        if 'عميد' in a or 'عميد' in q: master["role"] = "عميد"
        master["rows"].append({"q":q,"a":a})
        if bare and len(bare.split()) >= 2:
            for w in bare.split():
                nw = norm_ar(w)
                if len(nw) >= 3 and nw not in _prof_info: _prof_info[nw] = master
    print(f"🧑‍🏫 Professor index: {len(_prof_info)} keys")

# ═══════════════════════════════════════
# 4. PDF KNOWLEDGE BASE
# ═══════════════════════════════════════
PDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdfs")
_pdf_ch = []; _pdf_ix = None; _pdf_ok = False
def _garbled(ln):
    ar=[t for t in ln.split() if re.search(r'[\u0600-\u06ff]',t)]; return len(ar)>=3 and (sum(len(t) for t in ar)/len(ar)<3.2 or sum(1 for t in ar if len(t)<=2)/len(ar)>0.45)
def _load_pdfs():
    global _pdf_ch, _pdf_ix, _pdf_ok
    os.makedirs(PDF_DIR, exist_ok=True)
    pfs = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pfs: _pdf_ok = False; return
    pages = []
    for src in pfs:
        try:
            doc = fitz.open(src)
            for i, pg in enumerate(doc, 1):
                t = pg.get_text().strip()
                if t: pages.append({"text": t, "source": src, "page": i})
            doc.close()
        except: pass
    chunks = []
    for p in pages:
        t = "\n".join(ln for ln in p["text"].splitlines() if not _garbled(ln)).strip()
        if not t: continue
        s = 0
        while s < len(t):
            c = t[s:min(s+400, len(t))].strip()
            if c: chunks.append({"text":c,"source":p["source"],"page":p["page"]})
            s += 320
    _pdf_ch = chunks
    if chunks:
        em = smodel.encode([c["text"] for c in chunks], normalize_embeddings=True, convert_to_tensor=True)
        ea = em.cpu().numpy().astype("float32"); _pdf_ix = faiss.IndexFlatIP(ea.shape[1]); _pdf_ix.add(ea)
    _pdf_ok = bool(chunks)
    print(f"📄 PDF: {len(chunks)} chunks from {len(pfs)} file(s)")
_load_pdfs()
def search_pdf(q, top_k=4):
    if not _pdf_ok or not _pdf_ix: return []
    n = normalize(q); qe = smodel.encode([n], normalize_embeddings=True, convert_to_tensor=True).cpu().numpy().astype("float32")
    k = min(top_k, len(_pdf_ch)); ds, ids = _pdf_ix.search(qe, k)
    return [{"score":float(ds[0][i]),"text":_pdf_ch[ids[0][i]]["text"],"source":_pdf_ch[ids[0][i]]["source"],"page":_pdf_ch[ids[0][i]]["page"]} for i in range(k) if float(ds[0][i])>=0.35]
# ═══════════════════════════════════════
# 5. OLLAMA HEALTH
# ═══════════════════════════════════════
_oc = {"ok": False, "ts": 0}
def ollama_ok():
    now = time.time()
    if now - _oc["ts"] < 10: return _oc["ok"]
    for _ in range(2):
        try:
            r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=10)
            if r.status_code == 200: _oc["ok"]=True; _oc["ts"]=now; return True
        except: time.sleep(1)
    _oc["ok"]=False; _oc["ts"]=now; return False
ollama_ok()
# ═══════════════════════════════════════
# 6. TEXT NORMALIZATION
# ═══════════════════════════════════════
DMAP = {"شو":"ما","وين":"أين","اين":"أين","كيفش":"كيف","شلون":"كيف","ليش":"لماذا","هيك":"هكذا",
    "بدي":"أريد","بقدر":"أستطيع","لازم":"يجب","اسجل":"تسجيل","سجل":"تسجيل","يسجل":"تسجيل",
    "للجامعه":"في الجامعة","للجامعة":"في الجامعة","ايش":"ما","إيش":"ما","وش":"ما","زين":"جيد",
    "مو":"ليس","مين":"من","احكيلي":"أخبرني","حكيلي":"أخبرني","گلي":"أخبرني","قلي":"أخبرني",
    "خبرني":"أخبرني","أخبرني":"أخبرني","اخبرني":"أخبرني","اعطيني":"أعطني","عطيني":"أعطني",
    "ايميل":"إيميل","الايميل":"الإيميل","دكتوره":"دكتور","الدكتوره":"الدكتور","دكتورة":"دكتور","الدكتورة":"الدكتور","بقدرش":"هل يمكن",
    "اون لاين":"إلكتروني","اونلاين":"إلكتروني","اية":"أي","لاين":"إلكتروني","online":"إلكتروني",
    "طيب":"","يعطيك":"","العافية":"","اهه":"أريد مزيد من التفاصيل","آه":"أريد مزيد من التفاصيل",
    "تمام":"","ماشي":"","اوك":"","اوكي":"","ok":"","okay":"",
    "نحرم":"حرمان","بنحرم":"حرمان","بحرم":"حرمان","انحرم":"حرمان","حرماني":"حرمان",
    "سحب":"سحب وإضافة","اضافه":"إضافة","سحبنا":"سحب وإضافة","نضيف":"إضافة",
    "هاذ":"هذا","هاذا":"هذا","هاظ":"هذا","هاي":"هذه","هاد":"هذا",
    "بفتح":"يفتح","بيفتح":"يفتح","بيبدأ":"يبدأ","ببدأ":"يبدأ","فصل":"فصل دراسي",
    "قدر":"أستطيع","مش":"ليس","مش عارف":"لا أعرف","بغى":"أريد","ابغى":"أريد","بغي":"أريد",
    "بامتحان":"خلال الامتحان","بمتحن":"في الامتحان","شخبارك":"كيف حالك",
    "مشان":"بسبب","عشان":"لأن","بس":"لكن","لاكن":"لكن","كتر":"كثير","كتير":"كثير","هلق":"الآن","هلئ":"الآن",
    "بهمني":"يهمني","بهمك":"يهمني","ازا":"إذا","أزا":"إذا"}
_PUNCT = re.compile(r'[؟?،,\.!؛;:\-\(\)"\'«»\u200c\u200d]+')
def strip_punct(t): return _PUNCT.sub(' ', t)
def norm_ar(t):
    t = strip_punct(t); t = re.sub(r'[\u064B-\u065F\u0670]','',t)
    t = re.sub(r'[أإآٱ]','ا',t); t = re.sub(r'ؤ','و',t); t = re.sub(r'ئ','ي',t)
    t = re.sub(r'ة','ه',t); t = re.sub(r'ى','ي',t); return re.sub(r'\s+',' ',t).strip()
_PFX = ["لل","بال","وال","فال","كال","ال","ل","ب","و","ف","ك"]
def strip_pfx(w):
    for p in _PFX:
        if w.startswith(p) and len(w) > len(p)+1: return w[len(p):]
    return w
def normalize(t):
    ws = [DMAP.get(w,w) for w in t.split()]; return norm_ar(" ".join(w for w in ws if w))
_RSTOP = {"ما","هو","هي","في","من","على","إلى","عن","مع","هل","كيف","أين","اين","متى","لماذا","الجامعي","الجامعية","ال","و","يتم","يمكن","هذا","هذه","كان","يكون"}
STOP_W = {norm_ar(w) for w in _RSTOP} | {strip_pfx(w) for w in {norm_ar(w) for w in _RSTOP}}
# ═══════════════════════════════════════
# 7. NAME MATCHING
# ═══════════════════════════════════════
_names_cache = []
def _build_names():
    global _names_cache; names = []; seen = set()
    rp = re.compile(r'(?:رئيس|رئيسه|عميد|عميده)\s*(?:القسم|قسم)?\s*(?:علم\s*البيانات)?\s*:?\s*(الدكتور[ةه]?|الأستاذ[ةه]?|الاستاذ[ةه]?|دكتور[ةه]?|استاذ[ةه]?|المهندس[ةه]?|مهندس[ةه]?)\s+([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)', re.UNICODE)
    for _, row in df.iterrows():
        for i,pat in enumerate([_NP, _NP, rp]):
            src = str(row["Answer"]) if i==2 else str(row["Question"])
            for title, name in pat.findall(src):
                for form in [f"{title} {name.strip()}", name.strip()]:
                    n = norm_ar(form)
                    if n not in seen and (len(name.strip().split())>=2 or title):
                        names.append(form); seen.add(n)
    _names_cache = names; print(f"📛 {len(names)} names indexed")
_build_names(); _build_prof_index()
# Cross-ref: link رئيس القسم entry with actual person name
for _k, _v in _prof_info.items():
    if _v.get("role") == "رئيس القسم" and not _v.get("name"):
        for _, _row in df.iterrows():
            _q2, _a2 = str(_row["Question"]), str(_row["Answer"])
            if re.search(r'رئيس\s*(?:ال)?قسم', _q2):
                _m2 = _NP.search(_a2)
                if _m2:
                    _pn = f"{_m2.group(1)} {_m2.group(2).strip()}"
                    _v["name"] = re.sub(_TITLES_RE, '', _pn).strip()
                    _nb = norm_ar(re.sub(_TITLES_RE, '', _pn).strip())
                    if _nb in _prof_info:
                        if not _prof_info[_nb].get("role"): _prof_info[_nb]["role"] = "رئيس القسم"
                        for _f in ("email","office","phone"):
                            if _v.get(_f) and not _prof_info[_nb].get(_f): _prof_info[_nb][_f] = _v[_f]
                    break

def lookup_prof(query):
    if not _prof_info: return None
    nq = norm_ar(query)
    if nq in _prof_info: return _prof_info[nq]
    name = _ext_person(query)
    if name:
        nn = norm_ar(name)
        if nn in _prof_info: return _prof_info[nn]
        bare = re.sub(_TITLES_RE, '', name).strip(); nb = norm_ar(bare)
        if nb in _prof_info: return _prof_info[nb]
    qw = set(nq.split()); bm = None; bs = 0
    for key, info in _prof_info.items():
        kw = set(key.split()); ol = len(qw & kw) / max(len(qw | kw), 1)
        if ol > bs and ol >= 0.4: bs = ol; bm = info
    if bm: return bm
    fm, fs = fuzzy_name(query, 0.45)
    if fm:
        for n in [norm_ar(fm), norm_ar(re.sub(_TITLES_RE, '', fm).strip())]:
            if n in _prof_info: return _prof_info[n]
    return None

def fuzzy_name(q, thr=0.45):
    if not _names_cache: return None, 0.0
    cq = re.sub(r'^(مين|من|هو|هي)\s+','',q.strip(),flags=re.UNICODE); nq = norm_ar(cq)
    bs=0.0; bn=None
    for nm in _names_cache:
        nn = norm_ar(nm); sc = SequenceMatcher(None,nq,nn).ratio()
        qw, nw = set(nq.split()), set(nn.split())
        wo = len(qw&nw)/max(len(qw),1); cb = 0.5*sc + 0.5*wo
        if qw<=nw: cb = max(cb, 0.85)
        if cb>bs: bs=cb; bn=nm
    return (bn, bs) if bs>=thr else (None, 0.0)

# ═══════════════════════════════════════
# 8. PERSON EXTRACTION
# ═══════════════════════════════════════
def _ext_person(q):
    q = q.strip()
    for pat, action in [
        (r'^مين\s+(رئيس|رئيسه|عميد|عميده)\s*(.*)', lambda m: f"{m.group(1)} {m.group(2).strip()}" if m.group(2).strip() else m.group(1)),
        (r'^مين\s+([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)', lambda m: m.group(1).strip() if not re.match(r'^(رئيس|رئيسه|عميد|عميده)', m.group(1)) else None),
        (r'^من\s+(?:هو\s+|هي\s+)?([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)', lambda m: m.group(1).strip()),
        (r'(?:في\s+)?حدا\s+اسمه\s+(.*)', lambda m: m.group(1).strip()),
        (r'(?:احكيلي|حكيلي|أخبرني|اخبرني)\s+عن\s+(.*)', lambda m: m.group(1).strip() if m.group(1).strip() not in ("ه","ها","هو","هي","عنه","عنها") else None),
        (r'(الدكتور|الأستاذ|الاستاذ|دكتور|استاذ|المهندس|مهندس)\s+([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)', lambda m: f"{m.group(1)} {m.group(2).strip()}"),
    ]:
        m = re.search(pat, q, re.UNICODE)
        if m:
            r = action(m)
            if r: return r
    m = re.match(r'^([\u0600-\u06ff]{2,}\s+[\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)', q, re.UNICODE)
    return m.group(1).strip() if m else None

def _ext_person_hist(hist):
    for msg in reversed(hist[-10:]):
        if not isinstance(msg, dict): continue
        for m in _NP.finditer(msg.get("content","")): return f"{m.group(1)} {m.group(2).strip()}"
    if _names_cache:
        for msg in reversed(hist[-10:]):
            if not isinstance(msg, dict): continue
            nc = norm_ar(msg.get("content",""))
            for nm in _names_cache:
                bare = re.sub(r'^(الدكتور|الأستاذ|الاستاذ|دكتور|استاذ|المهندس|مهندس|الدكتوره|الأستاذة)\s+','',nm).strip()
                if bare and len(bare.split())>=2 and norm_ar(bare) in nc: return nm
    return None

def _collect_person(pname):
    nn = norm_ar(pname); bare = re.sub(_TITLES_RE, '', pname).strip(); nb = norm_ar(bare)
    is_role = any(w in nn for w in ["رئيس","عميد"])
    nw = set(nn.split() if is_role else nb.split()) - {"من","في","عن","علي","مع","هو","هي"}
    nw |= {strip_pfx(w) for w in list(nw) if len(strip_pfx(w))>=2}
    hits=[]; seen=set()
    for idx, row in df.iterrows():
        q,a = str(row["Question"]), str(row["Answer"]); nq,na = norm_ar(q), norm_ar(a)
        sc = 0.0
        if nb and len(nb.split())>=2 and (nb in nq or nb in na): sc=1.0
        if is_role and nn and (nn in nq or nn in na): sc=1.0
        if nw:
            oq = len(nw & (set(nq.split())|{strip_pfx(w) for w in set(nq.split()) if len(strip_pfx(w))>=2}))/max(len(nw),1)
            oa = len(nw & (set(na.split())|{strip_pfx(w) for w in set(na.split()) if len(strip_pfx(w))>=2}))/max(len(nw),1)
            sc = max(sc, max(oq,oa))
        sc = max(sc, max(SequenceMatcher(None,nn,nq).ratio(), SequenceMatcher(None,nn,na).ratio())*0.8)
        if norm_ar(pname) in nq or norm_ar(pname) in na: sc=1.0
        if not is_role and nb and len(nb.split())>=2:
            bp = set(nb.split())
            if bp<=(set(nq.split())|{strip_pfx(w) for w in set(nq.split()) if len(strip_pfx(w))>=2}) or bp<=(set(na.split())|{strip_pfx(w) for w in set(na.split()) if len(strip_pfx(w))>=2}): sc=max(sc,0.9)
        if sc>=0.35 and idx not in seen: seen.add(idx); hits.append({"question":q,"answer":a,"score":sc})
    hits.sort(key=lambda x: x["score"], reverse=True); return hits

# ═══════════════════════════════════════
# 9. INTENT DETECTION
# ═══════════════════════════════════════
META_P = [r"مين\s*معي",r"من\s*(أنت|انت)",r"اسمك",r"انت\s*شو",r"شو\s*انت",r"كيف\s*(تشتغل|تعمل)",r"ollama",r"\bllm\b",r"\bai\b"]
CONV_P = [r"^(كيف\s*(حالك|الحال))",r"^(شو|ايش|إيش)\s*(اخبارك|أخبارك)",r"شكرا|شكراً|ممنون|يسلمو|يعطيك",r"(حلو|ممتاز|رائع|كويس|زين|منيح)\s*$"]
ACAD_KW = ["تسجيل","مادة","مواد","دكتور","أستاذ","استاذ","مهندس","قسم","جامعة","كلية","امتحان","معدل","إيميل","مكتب","رئيس","مين","حدا","غش","عقوبة",
    "حرمان","سحب","إضافة","حذف","فصل","جدول","ساعات","متطلب","خطة","مساق","شعبة",
    "نجاح","رسوب","اعتذار","تأديب","إنذار","فصل مؤقت","فصل نهائي","تأمين","سكن",
    "منحة","رسوم","قبول","تخرج","مشروع","تدريب","تعليم","إلكتروني","متزامن","غير متزامن","نظام","سياسة","تعليمات"]
TELL_P = [r'احكيلي\s+عن',r'حكيلي\s+عن',r'أخبرني\s+عن',r'اخبرني\s+عن',r'مين\s+(رئيس|رئيسه|عميد)',r'^مين\s+[\u0600-\u06ff]{2,}\s+[\u0600-\u06ff]{2,}',r'^في\s+حدا\s+اسمه']
EXP_P = [r'^(اهه|آه|أه|ها)\s*$',r'(اعطيني|أعطني)\s*(تفاصيل|معلومات|المزيد)',r'^(تفاصيل|المزيد|اكثر)\s*$',r'(وضح|اشرح|شرحلي)\s*(لي|أكثر)?',r'(اكمل|أكمل|كمل|استمر)']
PERS_P = [r'^مين\s+[\u0600-\u06ff]{2,}',r'^من\s+[\u0600-\u06ff]{2,}',r'احكيلي\s+عن',r'مين\s+هو',r'في\s+حدا\s+اسمه']
ESCAL_P = [r'حوّ?ل(ها?)?\s*(السؤال|سؤالي|سؤالك)?\s*(ل|إلى|الى)\s*رئيس\s*(القسم|قسم)?',
    r'ابعث?\s*(السؤال\s*)?(ل|الى|إلى)\s*رئيس',r'ارسل?\s*(السؤال\s*)?(ل|الى|إلى)\s*رئيس',
    r'اطلع\s*رئيس\s*القسم',r'تواصل\s*(مع)?\s*رئيس',r'خلي\s*رئيس\s*القسم',
    r'ابلغ?\s*رئيس\s*القسم',r'نبه?\s*رئيس\s*القسم',r'بلغ?\s*رئيس',
    r'ابغى\s*احكي\s*رئيس\s*(القسم|قسم)',r'ابي\s*اتكلم\s*مع\s*رئيس']
def _mp(t, pats): return any(re.search(p, t.lower()) for p in pats)
def is_tell_about(q): return _mp(q.strip(), TELL_P) or _mp(normalize(q), TELL_P)
def _is_person_q(q):
    if _mp(q, PERS_P) or _mp(normalize(q), PERS_P): return True
    if _names_cache:
        nq = norm_ar(q)
        for nm in _names_cache:
            bare = re.sub(r'^(الدكتور|الأستاذ|الاستاذ|دكتور|استاذ|المهندس|مهندس|الدكتوره|الأستاذة)\s+','',nm).strip()
            if bare and len(bare.split())>=2 and (norm_ar(bare) in nq or len(set(norm_ar(bare).split())&set(nq.split()))>=2): return True
        fm,fs = fuzzy_name(q, 0.50)
        if fm: return True
    return False
def classify(q):
    nq = normalize(q)
    if _mp(q,META_P) or _mp(nq,META_P): return "meta"
    if _mp(q,ESCAL_P) or _mp(nq,ESCAL_P): return "escalate"
    if _is_person_q(q): return "academic"
    if _mp(q,CONV_P) or _mp(nq,CONV_P): return "conversational"
    if any(kw in q for kw in ACAD_KW) or any(kw in nq for kw in ACAD_KW): return "academic"
    if len(q.split())<=2: return "conversational"
    return "unknown"

# ═══════════════════════════════════════
# 9b. KEYWORD-BASED DB SEARCH
# ═══════════════════════════════════════
_TOPIC_SYN = {"تسجيل":["تسجيل","سجل","سجلوا","أسجل","تسجل"],"مواد":["مواد","مادة","مساق","مساقات","شعبة"],
    "سحب وإضافة":["سحب","إضافة","اضافه","سحب وإضافة","تعديل جدول"],"حرمان":["حرمان","حرم","نحرم","بحرم","بنحرم","انحرم"],
    "فصل":["فصل","فصل مؤقت","فصل نهائي"],"امتحان":["امتحان","امتحانات","اختبار","شفاوي","عملي"],
    "تأديب":["تأديب","عقوبة","تنبيه","إنذار"],"رسوب":["رسوب","راسب"],"اعتذار":["اعتذار","انسحاب"],
    "رسوم":["رسوم","أقساط","دفع"],"قبول":["قبول","مفاضلة"],"سكن":["سكن","إسكان"],
    "تأمين":["تأمين","صحي"],"تعليم إلكتروني":["إلكتروني","اونلاين","متزامن","عن بعد"]}
def _keyword_search(q, top_k=8):
    nq = norm_ar(q); qw = set(nq.split()) - STOP_W
    for w in nq.split():
        for syns in _TOPIC_SYN.values():
            if norm_ar(w) in [norm_ar(s) for s in syns]: qw |= set(norm_ar(s) for s in syns)
    qw |= {strip_pfx(w) for w in qw if len(w)>=2}
    scored = []
    for idx, row in df.iterrows():
        rq, ra = norm_ar(str(row["Question"])), norm_ar(str(row["Answer"]))
        rk = norm_ar(str(row.get("Keywords","")))
        db_all = set(rq.split())|set(ra.split())|set(rk.split())|{strip_pfx(w) for w in set(rq.split())|set(ra.split()) if len(w)>=2}
        if not qw: continue
        score = len(qw & db_all) / max(len(qw), 1)
        q_key = qw - STOP_W; qm = len(q_key & (set(rq.split())|{strip_pfx(w) for w in set(rq.split()) if len(w)>=2}))
        if q_key: score = max(score, 0.5*score + 0.5*(qm/len(q_key)))
        if nq and len(nq)>3 and (nq in rq or nq in ra): score = max(score, 0.9)
        if score >= 0.15: scored.append({"score":score,"question":str(row["Question"]),"answer":str(row["Answer"])})
    scored.sort(key=lambda x: x["score"], reverse=True); return scored[:top_k]

# ═══════════════════════════════════════
# 10. FAISS RETRIEVAL
# ═══════════════════════════════════════
_NTRIG = {"الدكتور","الدكتورة","الاستاذ","دكتور","دكتورة","استاذ","الدكتوره","المهندس","مهندس","رئيس","رئيسه","عميد","عميده"}
def word_olap(q, s):
    nq, ns = norm_ar(q), norm_ar(str(s))
    qw = {strip_pfx(w) for w in nq.split()} - STOP_W; sw = {strip_pfx(w) for w in ns.split()} - STOP_W
    return len(qw&sw)/len(qw|sw) if qw and sw else 0.0
def retrieve(q, top_k=5):
    n = normalize(q); is_nm = any(w in n for w in _NTRIG)
    sw = 0.15 if is_nm else 0.50; ow = 0.85 if is_nm else 0.50
    qe = smodel.encode([n], normalize_embeddings=True, convert_to_tensor=True).cpu().numpy().astype("float32")
    km = 20 if is_nm else 2; k = min(max(top_k*km,10), len(df))
    ds, ids = fidx.search(qe, k); scored = []
    for r in range(k):
        sem = float(ds[0][r]); ri = ids[0][r]; sq = df.iloc[ri]["Question"]
        ol = word_olap(n, sq); sc = sw*sem + ow*ol; scored.append((sc, ri))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score":s,"question":str(df.iloc[ri]["Question"]),"answer":str(df.iloc[ri]["Answer"])} for s,ri in scored[:top_k]]

# ═══════════════════════════════════════
# 11. WEB SEARCH
# ═══════════════════════════════════════
_WHDR = {"User-Agent":"Mozilla/5.0 Chrome/120.0.0.0","Accept-Language":"ar,en;q=0.9","Accept":"text/html"}
def web_search(q):
    cq = " ".join(w for w in q.split() if w not in {"طيب","بتعرف","يعني","شو","ايش","هل","من","هو"})
    snips = []
    try:
        s = requests.Session(); s.headers.update(_WHDR)
        r = s.get("https://www.bing.com/search?q="+quote_plus(f"site:just.edu.jo {cq}"), timeout=10)
        if r.status_code == 200:
            for b in re.split(r'<li class="b_algo', r.text)[1:]:
                um = re.search(r'href="([^"]+)"', b)
                sm2 = re.search(r'<p[^>]*>(.*?)</p>', b, re.IGNORECASE|re.DOTALL)
                if um and sm2:
                    sn = re.sub(r'<[^>]+>','',sm2.group(1)); sn = re.sub(r'\s+',' ',sn).strip()
                    if sn and len(sn)>20 and not any(x["snippet"]==sn for x in snips):
                        snips.append({"snippet":sn,"url":um.group(1)})
    except: pass
    return snips[:4]

# ═══════════════════════════════════════
# 12. PERSONA & LLM
# ═══════════════════════════════════════
PERSONA = (
    "أنت مرشد أكاديمي ذكي اسمه 'المرشد الأكاديمي'، متخصص في قسم علم البيانات بجامعة العلوم والتكنولوجيا الأردنية.\n"
    "CRITICAL: Reply in Arabic ONLY. NEVER use English, Russian, or any non-Arabic words except URLs and emails.\n"
    "CRITICAL: NEVER use Cyrillic, Chinese, or any non-Arabic script.\n"
    "CRITICAL: NEVER output HTML tags, buttons, forms.\n"
    "CRITICAL: لا تبدأ إجاباتك بكلمة 'مرحبا' أو أي تحية إلا إذا كان هذا أول سؤال يطرحه المستخدم (سؤال ترحيبي). "
    "في جميع الإجابات اللاحقة لا تكتب أي تحية — ابدأ بالإجابة مباشرة.\n"
    "CRITICAL: NEVER add religious supplications or prayers.\n"
    "CRITICAL: NEVER add 'ملاحظة' or disclaimer about Arabic ability.\n"
    "CRITICAL: إذا سُئلت عن شخص → اذكر إيميله ومكتبه ورقمه من السياق. لا تختلق سيرة ذاتية.\n"
    "CRITICAL: لا تستنتج اسم الشخص من بريده الإلكتروني — استخدم الاسم المذكور في السياق فقط.\n"
    "CRITICAL: إذا لم تجد الإجابة في السياق اكتب 'مجهول' فقط.\n"
    "CRITICAL: NEVER invent or fabricate information not in context.\n"
    "CRITICAL: Do NOT write incomplete phone numbers. If unknown write 'مجهول'.\n"
    "CRITICAL: When mentioning an email, write the FULL address: name@just.edu.jo\n"
    "CRITICAL: NEVER leave URL or email blank. If unknown → 'مجهول'.\n"
    "═══ قاعدة البيانات ═══\n"
    "1. أجب مباشرة من السياق — لا توجّه الطالب للبحث بنفسه.\n"
    "2. محظور: 'يُنصح بالتوجه' أو 'راجع الموقع' أو 'للحصول على معلومات' أو 'يمكنك الاطلاع' أو 'زيارة بوابة'.\n"
    "3. إذا كانت الإجابة في السياق، اكتبها كاملة وبالتفصيل بأسلوب احترافي وسلس.\n"
    "4. يمكنك إضافة رابط البوابة في النهاية كمصدر إضافي فقط بعد الإجابة الكاملة.\n"
    "5. إذا سألك عن شخص → اذكر كل معلوماته من السياق فوراً بصيغة طبيعية ومفيدة.\n"
    "6. ادمج المعلومات من السياق بإجابة متماسكة واحترافية — لا تسردها كقائمة جافة.\n"
    "═══ معلومات الجامعة ═══\n"
    "- الموقع: https://www.just.edu.jo\n- البوابة: https://student.just.edu.jo\n"
    "- أعضاء هيئة التدريس: https://www.just.edu.jo/FacultiesAndDepartments/FacultyofComputer/Departments/DataScience/Pages/Staff.aspx\n"
    "- صفحة القسم: https://www.just.edu.jo/FacultiesAndDepartments/FacultyofComputer/Departments/DataScience/Pages/Home.aspx\n"
    "═══════════════════════\n"
)
def _clean_llm(t):
    if not t: return t
    t = re.sub(r'</?(?:button|form|input|script|style|iframe|div|span|table|tr|td|img|a|p|h[1-6])[^>]*>','',t,flags=re.IGNORECASE)
    t = re.sub(r'<br\s*/?>','\n',t,flags=re.IGNORECASE)
    for p in ["تأكيد الدخول","اضغط هنا للتسجيل","هل تريد أن تستمر","أسل الله العظيم","اسل الله العظيم","أسال الله العظيم"]:
        t = t.replace(p,'')
    t = re.sub(r'(مرحبا[ًّ]?\s*){2,}','مرحباً ',t)
    t = re.sub(r'(أهلا[ًّ]?\s*){2,}','أهلاً ',t)
    t = re.sub(r'\*?\*?ملاحظة\*?\*?[:\s]+[^\n]*(أتحدث|الفصح|لغت|محدودي)[^\n]*\n?','',t,flags=re.IGNORECASE)
    for rng in [r'\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff', r'\u0400-\u04ff\u0500-\u052f',
                r'\uac00-\ud7af\u1100-\u11ff', r'\u0e00-\u0e7f',
                r'\u0980-\u09ff\u0a00-\u0a7f\u0a80-\u0aff\u0b00-\u0b7f\u0b80-\u0bff\u0c00-\u0c7f\u0c80-\u0cff\u0d00-\u0d7f',
                r'\u10a0-\u10ff', r'\u0530-\u058f']:
        t = re.sub(f'[{rng}]+','',t)
    t = re.sub(r'\*?\*?حول\s+(الدكتور|الأستاذ|المهندس)[^\n]*\n[^\n]*\n?','',t,flags=re.IGNORECASE)
    t = re.sub(r'<think.*?>','',t,flags=re.DOTALL); t = re.sub(r'</think\s*>','',t,flags=re.DOTALL)
    prot = []
    def _st(m): prot.append(m.group(0)); return f"\x00P{len(prot)-1}\x00"
    t = re.sub(r'\[[^\]]*\]\([^)]+\)', _st, t)
    t = re.sub(r'https?://[^\s)\]>"\']+', _st, t)
    t = re.sub(r'[\w.\-+]+@[\w.\-]+\.\w{2,}', _st, t)
    _KEEP = {'pdf','vpn','wifi','toefl','ds','cs','it','ai','iot','url','edu','jo','com','org','net','ac','gov',
             'http','https','www','just','mail','web','home','staff','data','science','aspx','html','php','student','system','office','dept','fac','admin','page'}
    def _kl(m):
        w=m.group(0); pos=m.start()
        if '\x00P' in t[max(0,pos-20):pos+len(w)+20]: return w
        if re.search(r'\d',w) or '@' in w or '://' in w or w.lower() in _KEEP: return w
        return ' '
    t = re.sub(r'[a-zA-Z][a-zA-Z0-9\-\.]*', _kl, t)
    t = re.sub(r'(?<![\w@.])([\u0600-\u06ff]+[a-zA-Z]+|[a-zA-Z]+[\u0600-\u06ff]+)(?![\w@.])',' ',t)
    for i,o in enumerate(prot): t = t.replace(f"\x00P{i}\x00", o)
    return re.sub(r'[ \t]+',' ',t).strip()

def _call_llm(msgs):
    if not ollama_ok(): return None
    try:
        r = ollama.chat(model=OLLAMA_MODEL, messages=msgs, options={"temperature":0.1,"num_predict":2048})
        raw = (r.message.content if hasattr(r,"message") else r["message"]["content"]).strip()
        return _clean_llm(raw)
    except Exception as e:
        print(f"🚨 LLM error: {e}"); _oc["ts"]=0; return None

def llm_chat(q, hist, ctx=None, followup=False, expand=False):
    cb = ""
    if ctx:
        fr = ("3. وسّع الإجابة بتفاصيل أكثر.\n" if expand
              else "3. السؤال متابعة. استخرج المعلومة وأجب مباشرة.\n" if followup
              else "3. استخدم السياق السابق لتحديد المقصود بالضمير.\n")
        wp = ctx.startswith("[ 🌐")
        sr = ("4. يمكنك الإجابة من معلوماتك العامة إذا تكاملت مع السياق.\n" if wp and not expand
              else "4. إذا لم تجد الإجابة في السياق اكتب مجهول فقط.\n")
        cb = (f"\n\n══ السياق الرسمي ══\n1. أجب فقط من هذا السياق.\n2. روابط → [النص](https://url) | إيميلات → كاملة\n"
              + fr + sr +
              "5. محظور: 'يُنصح بالتوجه' أو 'راجع الموقع' — ممنوع.\n"
              "6. أجب بالعربية فقط بأسلوب احترافي وسلس.\n"
              f"{ctx}\n═══════════════════\n")
    sys = (PERSONA + "يمكنك الإجابة على أي سؤال.\n"
           + "• إذا سأل عن شخص → اذكر إيميله ومكتبه من السياق فقط.\n"
           + "• إذا كان أكاديمياً وليس في السياق → 'مجهول'.\n"
           + "• لا تبدأ الإجابة بكلمة 'مرحبا' — ابدأ بالإجابة مباشرة.\n"
           + "• ادمج المعلومات بإجابة احترافية وليست مجرد قائمة.\n" + cb)
    ms = [{"role":"system","content":sys}]
    for m in hist[-10:]:
        if isinstance(m, dict): ms.append(m)
        elif isinstance(m, (list,tuple)) and len(m)>=2:
            ms.append({"role":"user","content":str(m[0])}); ms.append({"role":"assistant","content":str(m[1])})
    ms.append({"role":"user","content":q})
    return _call_llm(ms)

# ═══════════════════════════════════════
# 13. SELF-VERIFICATION
# ═══════════════════════════════════════
def _extract_fact(text, itype):
    if itype == "email":
        m = re.search(r'[\w.\-+]+@[\w.\-]+\.\w{2,}', text); return m.group(0) if m else None
    if itype == "office":
        m = re.search(r'(مبنى\s*\S+\s*الطابق\s*\S+(?:\s*\([^)]+\))?)', text)
        if not m: m = re.search(r'(مكتب[^\n.،]{0,60})', text)
        return m.group(0).strip() if m else None
    if itype == "phone":
        m = re.search(r'(\+?\d[\d\s\-]{6,14}\d)', text); return m.group(0).strip() if m else None
    return None

def _verify_answer(query, answer, hits, info_type):
    if not answer or info_type == "general": return answer
    llm_fact = _extract_fact(answer, info_type)
    _LBL = {"email":"الإيميل","office":"المكتب","phone":"الرقم"}
    if llm_fact:
        if hits:
            for h in hits:
                db_fact = _extract_fact(h.get("answer",""), info_type)
                if db_fact and db_fact == llm_fact: return answer
            for h in hits:
                db_fact = _extract_fact(h.get("answer",""), info_type)
                if db_fact and db_fact != llm_fact:
                    return f"**{_LBL.get(info_type,'المعلومة')}:** {db_fact}"
        pinfo = lookup_prof(query)
        if pinfo:
            prof_fact = pinfo.get(info_type)
            if prof_fact and prof_fact != llm_fact:
                return f"**{_LBL.get(info_type,'المعلومة')}:** {prof_fact}"
            elif prof_fact and prof_fact == llm_fact: return answer
        if not (hits and any(_extract_fact(h.get("answer",""), info_type) for h in hits)):
            if info_type == "phone" and re.search(r'\d{2,3}\s*[-–]\s*$', answer): return "مجهول"
    if not llm_fact and info_type in ("email","office","phone","name_info"):
        pinfo = lookup_prof(query)
        if pinfo and pinfo.get(info_type):
            return f"**{_LBL.get(info_type,'المعلومة')}:** {pinfo[info_type]}"
    return answer

def _detect_info_type(q):
    nq = normalize(q).lower()
    if re.search(r'(ايميل|إيميل|بريد|email)',nq): return "email"
    if re.search(r'(مكتب|غرف|office)',nq): return "office"
    if re.search(r'(رقم|هاتف|تلفون|phone)',nq): return "phone"
    if re.search(r'(مين هو|من هو|احكيلي عن|أخبرني عن|مين رئيس|رئيس|عميد|حدا اسمه|اسمه الدكتور)',nq): return "name_info"
    return "general"

# ═══════════════════════════════════════
# 14. POST-PROCESSING
# ═══════════════════════════════════════
_DS="https://www.just.edu.jo/FacultiesAndDepartments/FacultyofComputer/Departments/DataScience/Pages"
_JUST_URLS = {"الموقع الرسمي":"https://www.just.edu.jo","بوابة الطالب":"https://student.just.edu.jo",
    "أعضاء هيئة التدريس":f"{_DS}/Staff.aspx","قسم علم البيانات":f"{_DS}/Home.aspx"}
_REG_KW = ["تسجيل","سجل","مواد","مادة","جدول","إضافة","حذف","سحب","حرمان","فصل","امتحان","اعتذار"]
_GENERIC_REDIRECTS = [r'يُنصح\s*ب(?:التوجه|الزيارة|زيارة|الرجوع|مراجعة|الاطلاع)',
    r'للحصول\s*على\s*معلومات\s*حول',r'للمعلومات\s*حول',r'راجع\s*(?:الموقع|بوابة|النظام|الجهة)',
    r'يمكنك\s*(?:الاطلاع|متابعة|التواصل|زيارة)\s*(?:من|عبر|عن|من خلال)?\s*(?:خلال\s*)?(?:بوابة|الموقع)',
    r'زيارة\s*(?:بوابة|الموقع)',r'يُرجى\s*(?:الرجوع|مراجعة|التواصل|زيارة)',
    r'للمزيد\s*من\s*المعلومات\s*(?:يرجى|يمكنك|راجع)',
    r'عبر\s*بوابة\s*الطالب',r'التواصل\s*مع\s*(?:الجهة|القسم|عمادة)',
    r'يُفضل\s*(?:مراجعة|التواصل|زيارة)',r'التوجه\s*إلى\s*(?:بوابة|الموقع|الجهة)']

def _post_proc(text, q):
    text = re.sub(r'\[\s*\+?\d{2,3}\s*[-–]\s*\]','مجهول',text)
    text = re.sub(r'(\+?\d{2,3})\s*[-–]\s*(?=[\]\)\s]|$)','مجهول',text)
    text = re.sub(r'((?:رقم|هاتف)\s*[^:]*:\s*)\+?\d{2,3}\s*[-–]\s*','مجهول ',text)
    text = re.sub(r'((?:الإيميل|البريد|ايميل)\s*:\s*)@(?!\S)','مجهول',text)
    def _fl(m):
        lb, u = m.group(1).strip(), m.group(2).strip()
        if not u or u in ("://","/"," "):
            for k,v in _JUST_URLS.items():
                if k in lb: return f"[{lb}]({v})"
            return f"[{lb or 'الموقع الرسمي'}](https://www.just.edu.jo)"
        return m.group(0)
    text = re.sub(r'\[([^\]]*)\]\(([^)]*)\)', _fl, text)
    nq = normalize(q)
    if any(kw in nq for kw in _REG_KW) and not re.search(r'https?://', text):
        text = text.rstrip() + f"\n\n🔗 [بوابة الطالب](https://student.just.edu.jo)"
    for pat in _GENERIC_REDIRECTS:
        text = re.sub(pat + '[^\n]*', '', text)
    # Aggressive redirect cleanup: if entire response is just a redirect, clear it
    text_stripped = text.strip()
    if text_stripped and len(text_stripped) < 250:
        _RED_PHRASES = ['يُنصح','يُرجى','الرجوع إلى','راجع الموقع','راجع بوابة','التوجه إلى','بوابة الطالب','الموقع الرسمي']
        _INFO_PHRASES = ['يُمنع','يُسمح','نعم','لا ','تشمل','تُطبق','يتم','يجب','يمكن','يحق','تُعد','حسب','وفقاً','المادة','نص']
        red_count = sum(1 for p in _RED_PHRASES if p in text_stripped)
        info_count = sum(1 for p in _INFO_PHRASES if p in text_stripped)
        if red_count >= 2 and info_count == 0: text = ''
    _NON_AR = r'[\u0400-\u052f\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u1100-\u11ff\u0e00-\u0e7f\u0980-\u0d7f]+'
    text = re.sub(_NON_AR,'',text)
    text = re.sub(r'\n{3,}','\n\n',text)
    _is_greet = any(g in q for g in ["كيف حال","شخبارك","شو اخبار","هلا","مرحبا","أهلا","السلام","صباح","مساء"]) and len(q.split()) <= 6
    if not _is_greet:
        text = re.sub(r'^مرحبا[ًّ]?\s*','',text); text = re.sub(r'^أهلا[ًّ]?\s*','',text)
    for p in ["تأكيد الدخول","اضغط هنا","هل تريد أن تستمر","أسل الله","اسل الله"]: text = text.replace(p,'')
    text = re.sub(r'-\s*(?:رقم|هاتف)[^\n]*مجهول[^\n]*\n?','',text)
    return re.sub(r'\n{3,}','\n\n',text).strip()

# ═══════════════════════════════════════
# 15. BUILD PROF DATA (for LLM context)
# ═══════════════════════════════════════
def _build_prof_ctx(pinfo, extra_rows, pname):
    """Build a context string from prof index + DB rows for LLM to blend."""
    actual = pinfo.get("name") if pinfo else None
    display = f"{actual} — {pname}" if actual and actual != pname else pname
    lines = [f"المعلومات المتوفرة عن {display}:"]
    if actual and actual != pname:
        lines.append(f"مهم: اسم الشخص هو '{actual}' — لا تستنتج اسمًا من الإيميل، استخدم هذا الاسم فقط.")
    if pinfo:
        if pinfo.get("role"): lines.append(f"- المنصب: {pinfo['role']}")
        if pinfo.get("department"): lines.append(f"- القسم: {pinfo['department']}")
        if pinfo.get("email"): lines.append(f"- الإيميل: {pinfo['email']}")
        if pinfo.get("office"): lines.append(f"- المكتب: {pinfo['office']}")
        if pinfo.get("phone"): lines.append(f"- الرقم: {pinfo['phone']}")
    all_text = " ".join(lines)
    for h in extra_rows:
        ra = h.get("answer","")
        for fld,pat,lbl in [("email",r'[\w.\-+]+@[\w.\-]+\.\w{2,}',"الإيميل"),("office",r'(مبنى\s*\S+\s*الطابق\s*\S+(?:\s*\([^)]+\))?)',"المكتب"),("phone",r'(\+?962\s*\d[\d\s\-]{6,12}\d|0[7-9]\d[\d\s\-]{6,8}\d)',"الرقم"),("department",r'قسم\s*([\u0600-\u06ff]{2,}(?:\s+[\u0600-\u06ff]{2,})*)',"القسم")]:
            if (not pinfo or not pinfo.get(fld)) and lbl not in all_text:
                m = re.search(pat, ra)
                if m:
                    val = m.group(1).strip() if fld=="department" else m.group(0).strip()
                    lines.append(f"- {lbl}: {val}"); all_text += val
                    if pinfo: pinfo[fld] = val
    # Add extra unique info from rows
    for r in (pinfo.get("rows",[]) if pinfo else []) + [{"q":h.get("question",""),"a":h.get("answer","")} for h in extra_rows]:
        ra = r.get("a", r.get("answer",""))
        if ra.strip() and norm_ar(ra) not in norm_ar(all_text):
            has_new = (re.search(r'[\w.\-+]+@[\w.\-]+\.\w{2,}', ra) and '@' not in all_text) or \
                      (re.search(r'مبنى', ra) and 'مبنى' not in all_text) or \
                      (re.search(r'قسم', ra) and 'قسم' not in all_text)
            if has_new: lines.append(f"- معلومة إضافية: {ra.strip()}")
    lines.append("أجب عن سؤال المستخدم باستخدام هذه المعلومات بأسلوب احترافي وسلس، ولا تذكر أي شيء ليس هنا.")
    return "\n".join(lines)

# ═══════════════════════════════════════
# 16. MAIN BOT LOGIC
# ═══════════════════════════════════════
def _last_bot(h):
    return next((m.get("content","") for m in reversed(h[-10:]) if isinstance(m,dict) and m.get("role")=="assistant"), None)
def _last_user(h):
    u=[m for m in h if isinstance(m,dict) and m.get("role")=="user"]; return u[-2].get("content","") if len(u)>=2 else (u[-1].get("content","") if u else None)

def ask_bot(q, hist):
    try:
        query = q.strip(); intent = classify(query)
        print(f"🧠 Intent={intent} | Q={query[:60]}")
        # 1. Escalation to department head
        if intent == "escalate":
            lq = _last_user(hist) if hist else None
            if not lq: lq = query
            cur.execute("INSERT INTO pending_questions (user_question,status) VALUES (?,?)",(lq,"Forwarded")); conn.commit()
            head = lookup_prof("رئيس القسم")
            head_info = ""
            if head:
                if head.get("email"): head_info += f"\n📧 الإيميل: {head['email']}"
                if head.get("office"): head_info += f"\n🏢 المكتب: {head['office']}"
                if head.get("phone"): head_info += f"\n📞 الرقم: {head['phone']}"
            
            # الرسالة الجديدة التي تعكس نموذج التعلم الذاتي للبوت
            reply_msg = (
                f"✅ تم تحويل سؤالك إلى رئيس القسم لتحديث معلوماتي.{head_info}\n\n"
                f"يرجى سؤالي عنه مجدداً لاحقاً (مثلاً غداً) لأعطيك الإجابة المعتمدة بعد أن أتعلمها."
            )
            return reply_msg, "high"
        # 2. Meta
        if intent == "meta":
            nq = normalize(query).lower()
            if any(w in nq for w in ["اسمك","من انت","مين انت","انت شو","شو انت"]):
                return "أنا **المرشد الأكاديمي الذكي** 🎓\nمساعد آلي لقسم علم البيانات في جامعة العلوم والتكنولوجيا.", "high"
            if any(w in nq for w in ["موديل","نموذج","ollama","llm"]):
                return f"**النموذج:** {OLLAMA_MODEL} عبر Ollama\n**القاعدة:** {len(df)} سؤال وجواب", "high"
            if re.search(r"مين\s*(صمم|برمجك|عملك)", nq):
                return "تم تطويري كمشروع تخرج في قسم علم البيانات 🎓", "high"
            a = llm_chat(query, hist)
            return a or "أنا المرشد الأكاديمي الذكي.", "medium"
        # 3. Greetings
        GR = {"هلا","مرحبا","السلام عليكم","يعطيك العافية","أهلا","مرحباً","صباح الخير","مساء الخير","هلو","هاي","أهلاً","أهلين"}
        if any(g in query for g in GR) and len(query.split())<=4:
            a = llm_chat(query, hist)
            return a or "أهلاً وسهلاً! 😊 كيف يمكنني مساعدتك؟", "high"
        # 4. Expansion
        if not is_tell_about(query) and _mp(query, EXP_P) and hist:
            la = _last_bot(hist); lq = _last_user(hist)
            if la and lq:
                ectx = f"[ 💬 السؤال السابق ]\n{lq}\n\n[ ✅ إجابتك السابقة — وسّعها ]\n{la}\n\n"
                ph = search_pdf(lq, 4)
                if ph: ectx += "[ 📄 معلومات إضافية ]\n" + "\n".join(p["text"] for p in ph)
                a = llm_chat(f"أعطني تفاصيل أكثر عن: {lq}", hist, ectx, expand=True)
                return (a or la, "medium")
        # 5. DIRECT DB LOOKUP → Mix with LLM
        itype = _detect_info_type(query)
        if itype in ("email", "office", "phone", "name_info"):
            pinfo = lookup_prof(query)
            pname = _ext_person(query)
            if not pname:
                fm,fs = fuzzy_name(query, 0.45)
                if fm: pname = fm
            extra_rows = _collect_person(pname) if pname else []
            if pinfo or extra_rows:
                print(f"🗄️ Direct DB lookup hit for: {query[:60]}")
                prof_ctx = _build_prof_ctx(pinfo, extra_rows, pname or "الشخص")
                # Mix DB data with LLM for professional response
                a = llm_chat(query, hist, f"[ 👤 معلومات من قاعدة البيانات ]\n{prof_ctx}")
                if a and "مجهول" not in a[:30]: return _post_proc(a, query), "high"
                # Fallback to structured if LLM fails
                fb = "\n".join(f"• {h['answer']}" for h in extra_rows[:5]) if extra_rows else ""
                return _post_proc(fb, query) if fb else ("لا توجد معلومات كافية.", "low")
        # 6. Tell-about handling
        if is_tell_about(query):
            pn = _ext_person(query)
            if not pn or pn in ("عنه","عنها","هو","هي","ه","ها"): pn = _ext_person_hist(hist)
            if not pn:
                fm,fs = fuzzy_name(query)
                if fm: pn = fm
            if pn:
                print(f"👤 Tell-about: {pn}")
                ah = _collect_person(pn)
                if not ah:
                    fm,fs = fuzzy_name(pn)
                    if fm: ah = _collect_person(fm)
                if ah:
                    pinfo = lookup_prof(pn)
                    prof_ctx = _build_prof_ctx(pinfo, ah, pn)
                    # Add PDF + web data
                    ph = search_pdf(pn, 4)
                    if ph: prof_ctx += "\n[ 📄 من الوثائق ]\n" + "\n".join(p["text"] for p in ph)
                    try:
                        wr = web_search(pn)
                        if wr: prof_ctx += "\n[ 🌐 من الموقع الرسمي ]\n" + "\n".join(f"- {r['snippet']}" for r in wr)
                    except: pass
                    a = llm_chat(f"أخبرني بكل المعلومات عن: {pn}", hist, prof_ctx)
                    if a and "مجهول" not in a[:50]: return _post_proc(a, query), "medium"
                    fb = "\n".join(f"• {h['answer']}" for h in ah[:8])
                    return _post_proc(fb, query), "medium"
                else:
                    try:
                        wr = web_search(pn)
                        if wr:
                            wctx = "\n".join(f"- {r['snippet']}" for r in wr)
                            a = llm_chat(f"أخبرني عن: {pn}", hist, f"[ 🌐 من الموقع ]\n{wctx}")
                            if a: return _post_proc(a, query), "medium"
                    except: pass
        # 7. KEYWORD + FAISS retrieval
        sq = query
        FW = {"ايميله","ايميلها","إيميله","إيميلها","مكتبه","مكتبها","رقمه","رقمها","عنه","عنها","احكيلي عنه","حكيلي عنه"}
        is_fu = any(w in query for w in FW)
        if is_fu and hist:
            np_ = re.compile(r'(الدكتور|الأستاذ|دكتور|استاذ|المهندس)\s+[\u0600-\u06ff]{2,}\s*[\u0600-\u06ff]*', re.UNICODE)
            for m in reversed(hist[-10:]):
                if not isinstance(m, dict): continue
                mt = np_.search(m.get("content",""))
                if mt: sq = f"{mt.group(0)} {query}"; break
        kw_hits = _keyword_search(sq, 8) if intent == "academic" else []
        hits = retrieve(sq, 5)
        if kw_hits and kw_hits[0]["score"] >= 0.3:
            seen_q = set(); merged = []
            for h in kw_hits:
                nq_h = norm_ar(h["question"])
                if nq_h not in seen_q: seen_q.add(nq_h); merged.append(h)
            for h in hits:
                nq_h = norm_ar(h["question"])
                if nq_h not in seen_q: seen_q.add(nq_h); merged.append(h)
            hits = merged[:8]
        bs = hits[0]["score"] if hits else -1
        nq = normalize(query); is_nq = any(w in nq for w in _NTRIG) or _is_person_q(query)
        # 8. Fuzzy name suggestion
        if is_nq and bs < 0.50:
            fm,fs = fuzzy_name(query)
            if not fm or fs < 0.55:
                en = _ext_person(query)
                if en: fm2,fs2 = fuzzy_name(en);
                if fm2 and fs2>fs: fm,fs=fm2,fs2
            if fm and fs>=0.45:
                chi = retrieve(fm, 5); api = _collect_person(fm)
                cs = chi[0]["score"] if chi else 0
                if api and len(api)>1:
                    pinfo = lookup_prof(fm)
                    prof_ctx = _build_prof_ctx(pinfo, api, fm)
                    a = llm_chat(f"أخبرني عن: {fm}", hist, prof_ctx)
                    if a: return _post_proc(f"هل تقصد **{fm}**؟ 🔍\n\n{a}", query), "medium"
                if cs >= 0.35:
                    return _post_proc(f"هل تقصد **{fm}**؟ 🔍\n\n{chi[0]['answer']}", query), "medium"
                return f"هل تقصد **{fm}**؟ 🔍 اكتب الاسم كاملاً وسأساعدك.", "low"
        # 8b. KEYWORD-DIRECT: keyword search found good match → use DB directly via LLM
        if intent == "academic" and kw_hits and kw_hits[0]["score"] >= 0.35 and not is_nq:
            print(f"🔑 KW-DIRECT ({kw_hits[0]['score']:.3f}): DB answer via LLM")
            db_ctx = "[ ✅ معلومات موثوقة من قاعدة البيانات — أجب بها مباشرة وبالتفصيل بأسلوب احترافي وسلس، محظور توجيه الطالب للبوابة أو الموقع ]\n"
            for h in kw_hits[:5]:
                db_ctx += f"س: {h['question']}\nج: {h['answer']}\n"
            ph = search_pdf(sq, 3)
            if ph: db_ctx += "[ 📄 من الوثائق ]\n" + "\n".join(p["text"] for p in ph)
            a = llm_chat(query, hist, db_ctx)
            if a and "مجهول" not in a[:30]:
                # Check if LLM still generated redirect → fallback to raw DB
                def _is_red(t):
                    if not t: return False
                    return any(f in t.lower() for f in ['يُنصح ب','يُرجى','راجع الموقع','للمعلومات حول','للحصول على معلومات','يمكنك الاطلاع'])
                if _is_red(a):
                    print("🔴 LLM redirect in KW-DIRECT → raw DB fallback")
                    return _post_proc(kw_hits[0]["answer"], query), "medium"
                return _post_proc(a, query), "medium"
            # LLM failed → return raw DB answer
            return _post_proc(kw_hits[0]["answer"], query), "medium"
        # 9. HIGH confidence → Mix with LLM
        if hits and bs >= HIGH_CONF and intent == "academic":
            print(f"🟢 HIGH ({bs:.3f}): DB+LLM blend"); r = hits[0]
            # Build context from top hits
            db_ctx = "[ ✅ معلومات من قاعدة البيانات — أجب بأسلوب احترافي ]\n"
            db_ctx += f"س: {r['question']}\nج: {r['answer']}\n"
            if len(hits) >= 2:
                for h in hits[1:3]:
                    if h["score"] >= 0.5 and norm_ar(h["answer"]) != norm_ar(r["answer"]):
                        db_ctx += f"س: {h['question']}\nج: {h['answer']}\n"
            # Add last bot message for context continuity
            if hist:
                lb = _last_bot(hist)
                if lb: db_ctx += f"[ 💬 إجابتك السابقة للسياق ]\n{lb[:200]}\n"
            a = llm_chat(query, hist, db_ctx)
            if a: la = a
            else: la = r['answer']
            itype = _detect_info_type(query)
            la = _verify_answer(query, la, hits, itype)
            return _post_proc(la, query), "high"
        # 10. Web search
        wr = []; ws = []
        if intent in ["academic","unknown"] or is_nq:
            wr = web_search(query); ws = [r["snippet"] for r in wr]
        # 11. Build context
        ctx = []
        if ws:
            ctx.append("[ 🌐 من الموقع الرسمي ]")
            ctx.extend(f"- {s}" for s in ws)
            ctx.append(f"روابط: {JUST_BASE} | https://student.just.edu.jo")
        if is_fu and hist:
            lb = _last_bot(hist)
            if lb: ctx.append(f"[ 💬 إجابتك السابقة ]\n{lb}")
        dbt = 0.15 if kw_hits else (0.35 if is_nq else 0.45)
        if hits and hits[0]["score"] >= dbt:
            ctx.append("[ ✅ من قاعدة البيانات — أجب بأسلوب احترافي ]")
            for h in hits[:6]:
                if h["score"] >= dbt: ctx.append(f"س: {h['question']}\nج: {h['answer']}")
        elif kw_hits and kw_hits[0]["score"] >= 0.2:
            ctx.append("[ ✅ من قاعدة البيانات — أجب بأسلوب احترافي ]")
            for h in kw_hits[:6]: ctx.append(f"س: {h['question']}\nج: {h['answer']}")
        if intent=="academic" and not is_nq and not is_fu:
            ph = search_pdf(sq, 4)
            if ph: ctx.append("[ 📄 من الوثائق ]"); ctx.extend(p["text"] for p in ph)
        # Add conversation context
        if hist:
            lb = _last_bot(hist); lu = _last_user(hist)
            if lb and not is_fu: ctx.append(f"[ 💬 السياق السابق — السؤال: {lu} | الإجابة: {lb[:150]} ]")
        dbctx = "\n".join(ctx) if ctx else None
        # 12. Call LLM
        lq = re.sub(r'اية\s*لاين','اون لاين', query)
        la = llm_chat(lq, hist, dbctx, followup=is_fu)
        # 12b. Redirect guard — if LLM gives redirect but DB has answer, use DB instead
        def _is_redirect(t):
            if not t: return False
            t_low = t.lower()
            _RED_FLAGS = ['يُنصح ب','يُرجى','راجع الموقع','راجع بوابة','يمكنك الاطلاع','يمكنك متابعة',
                'للحصول على معلومات','للمعلومات حول','زيارة بوابة','التوجه إلى بوابة','التواصل مع الجهة',
                'يُفضل مراجعة','عبر بوابة الطالب','من خلال بوابة','الرجوع إلى الموقع','مراجعة النظام']
            return any(f in t_low for f in _RED_FLAGS)
        if la and _is_redirect(la) and (kw_hits or (hits and bs >= 0.20)):
            # LLM gave redirect but DB has data → use DB answer directly
            print("🔴 Redirect blocked → using DB answer")
            db_ans = None
            if kw_hits and kw_hits[0]["score"] >= 0.25:
                db_ans = kw_hits[0]["answer"]
            elif hits and bs >= 0.20:
                db_ans = hits[0]["answer"]
            if db_ans:
                # Try to rephrase DB answer with LLM but enforce no redirect
                db_ctx2 = f"[ ✅ الإجابة من قاعدة البيانات — أجب بها مباشرة وبالتفصيل، محظور التوجيه للبوابة ]\nس: {kw_hits[0]['question'] if kw_hits else hits[0]['question']}\nج: {db_ans}"
                if len(kw_hits) > 1 or len(hits) > 1:
                    for h in (kw_hits[1:3] if kw_hits else hits[1:3]):
                        db_ctx2 += f"\nس: {h['question']}\nج: {h['answer']}"
                la2 = llm_chat(query, hist, db_ctx2)
                if la2 and not _is_redirect(la2) and "مجهول" not in la2[:30]:
                    return _post_proc(la2, query), "medium"
                # Fallback: return raw DB answer
                return _post_proc(db_ans, query), "medium"
        # 13. Hallucination guard
        if la and is_nq and hits and bs < LOW_FAB:
            fab = ["تخصصه","تخصصها","حاصل على","حاصلة على","يدرّس","تدرّس","يدرس","تدرس","أبحاثه","خبرته","ماجستير في","دكتوراه في"]
            if any(m in la for m in fab) and not ws:
                print("🔴 Hallucination blocked")
                if hits[0]["answer"] and bs>=0.45: return hits[0]["answer"], "medium"
                return "لا تتوفر لدي معلومات كافية عن هذا الشخص. 🤔\nجرّب ذكر الاسم الكامل.", "low"
        # 14. Ollama offline
        if la is None:
            if hits and bs>=0.45: return hits[0]["answer"], "medium"
            if intent=="conversational": return "أهلاً! 😊 يمكنني مساعدتك في الأسئلة الأكاديمية.", "low"
            return "عذراً، النظام يعمل بوضع محدود. يرجى المحاولة مجدداً.", "low"
        # 15. Self-verify
        la = _verify_answer(query, la, hits, _detect_info_type(query))
        # 16. Unknown handling
        def _is_unknown(t):
            t = t.strip()
            if len(t)<=15 and "مجهول" in t: return True
            return any(p in t for p in ["لا أملك معلومات","لا تتوفر لدي","غير مذكور في السياق"]) and len(t)<150
        if _is_unknown(la):
            if ws:
                wc = "\n".join(f"- {s}" for s in ws)
                da = llm_chat(query, hist, f"[ 🌐 من الموقع ]\n{wc}\nأجب بناءً على ما سبق فقط.")
                if da and not _is_unknown(da): return _post_proc(da, query), "medium"
                return "وجدت معلومات من الموقع الرسمي:\n\n" + wc, "low"
            fs = PERSONA + "أجب من معلوماتك العامة عن الجامعة. إذا لم تعرف، قل ذلك بوضوح."
            fm = [{"role":"system","content":fs}]
            for m in hist[-6:]:
                if isinstance(m, dict): fm.append(m)
            fm.append({"role":"user","content":query})
            fa = _call_llm(fm)
            if fa and not _is_unknown(fa): return _post_proc(fa, query), "low"
            if intent=="academic":
                cur.execute("INSERT INTO pending_questions (user_question,status) VALUES (?,?)",(query,"Pending")); conn.commit()
                return "سؤالك وصلني ✅\nلا أملك إجابة دقيقة، سأحوّله لرئيس القسم.", "low"
            return "لم أجد معلومات كافية 🤔 هل يمكنك توضيح سؤالك؟", "low"
        # 17. Append sources
        src = ""
        ph = search_pdf(sq, 2) if intent=="academic" else []
        if ph: src += "\n".join(f"📄 {p['source']} — صفحة {p['page']}" for p in ph) + "\n"
        if wr: src += "\n".join(f"🔗 [الرابط]({r['url']})" for r in wr[:2]) + "\n"
        if src: la = f"{la}\n\n**المصادر:**\n{src.strip()}"
        return _post_proc(la, query), "medium"
    except Exception as e:
        print(f"🚨 Error: {e}"); traceback.print_exc()
        return "أواجه مشكلة تقنية مؤقتة، يرجى المحاولة مجدداً.", "low"

# ═══════════════════════════════════════
# 17. API ROUTES
# ═══════════════════════════════════════
@app.route("/")
@app.route("/interface")
def serve_ui():
    d = os.path.dirname(os.path.abspath(__file__))
    for n in ("index.html","interface_v3.html","interface_v2.html","interface.html"):
        if os.path.exists(os.path.join(d,n)): return send_from_directory(d,n)
    return "<h2>No interface HTML found.</h2>", 404

@app.route("/api/chat", methods=["POST"])
def chat_api():
    d = request.json; q = d.get("query","")
    r, c = ask_bot(q, d.get("history",[]))
    return jsonify({"reply": _post_proc(r, q), "confidence": c})

@app.route("/api/pending", methods=["GET"])
def pending_api():
    query = "SELECT id, user_question, status FROM pending_questions WHERE status IN ('Pending', 'Forwarded')"
    return jsonify(pd.read_sql_query(query, conn).to_dict(orient="records"))

@app.route("/api/answer", methods=["POST"])
def answer_api():
    d = request.json; msg, code = _submit_answer(d.get("id"), d.get("answer"))
    return jsonify({"success":code==200,"message":msg}), code

@app.route("/api/answer-bulk", methods=["POST"])
def answer_bulk_api():
    d = request.json; ids, ans = d.get("ids",[]), d.get("answer","").strip()
    if not ids or not ans: return jsonify({"success":False,"message":"❌ يرجى تحديد الأسئلة وكتابة الإجابة"}), 400
    msg, code = _submit_bulk(ids, ans); return jsonify({"success":code==200,"message":msg}), code

@app.route("/api/status", methods=["GET"])
def status_api():
    return jsonify({"ollama_online":ollama_ok(),"model":OLLAMA_MODEL,"rows":len(df)})

@app.route("/api/tts", methods=["POST"])
def api_tts():
    if not _EDGE_TTS_OK: return jsonify({"error":"edge-tts not installed"}), 503
    d = request.get_json(silent=True) or {}; text = d.get("text","").strip()
    if not text: return jsonify({"error":"No text"}), 400
    text = text[:4000]
    clean = re.sub(r'\*\*(.*?)\*\*', r'\1', text); clean = re.sub(r'\*([^*]+)\*', r'\1', clean)
    clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean); clean = re.sub(r'<[^>]+>','',clean)
    clean = re.sub(r'#{1,6}\s?','',clean); clean = re.sub(r'[📄🌐🔗✅❌⚠️🤖🎓😊🔍💬•◆🔊⏸⏹]','',clean)
    clean = re.sub(r'https?://\S+','',clean); clean = re.sub(r'\s{2,}',' ',clean).strip()
    if not clean: return jsonify({"error":"Empty text"}), 400
    voice = TTS_VOICE  # Female only — ar-JO-SanaNeural
    rate = f"+{int(d.get('rate',0))}%" if int(d.get('rate',0))>=0 else f"{int(d.get('rate',0))}%"
    try:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        async def _gen():
            c = edge_tts.Communicate(clean, voice, rate=rate); buf = io.BytesIO()
            async for ch in c.stream():
                if ch["type"]=="audio": buf.write(ch["data"])
            buf.seek(0); return buf
        buf = loop.run_until_complete(_gen()); loop.close()
        if buf.getbuffer().nbytes==0: return jsonify({"error":"No audio"}), 500
        return send_file(buf, mimetype="audio/mpeg", as_attachment=False, download_name="tts.mp3")
    except Exception as e: return jsonify({"error":str(e)}), 500

@app.route("/api/tts/voices", methods=["GET"])
def tts_voices():
    return jsonify({"available":_EDGE_TTS_OK,"voices":[{"id":"female","name":"Sana","language":"ar-JO"}],"default":"female"})

# ═══════════════════════════════════════
# 18. DB HELPERS
# ═══════════════════════════════════════
def _submit_answer(qid, answer):
    global df, fidx
    try:
        cur.execute("SELECT user_question FROM pending_questions WHERE id=?",(qid,)); row = cur.fetchone()
        if not row: return "❌ السؤال غير موجود", 404
        df = pd.concat([df, pd.DataFrame({"Question":[row[0]],"Answer":[answer],"Keywords":[""]})], ignore_index=True)
        df.to_excel(EXCEL_FILE, index=False)
        cur.execute("UPDATE pending_questions SET status='Answered' WHERE id=?",(qid,)); conn.commit()
        fidx = _build_idx(df); _build_names(); _build_prof_index(); return "✅ تم الحفظ والتدريب!", 200
    except Exception as e: return f"❌ خطأ: {e}", 400

def _submit_bulk(ids, answer):
    global df, fidx
    rows, nf = [], []
    for qid in ids:
        cur.execute("SELECT user_question FROM pending_questions WHERE id=?",(qid,)); row = cur.fetchone()
        if row: rows.append({"Question":row[0],"Answer":answer,"Keywords":""}); cur.execute("UPDATE pending_questions SET status='Answered' WHERE id=?",(qid,))
        else: nf.append(qid)
    if not rows: return "❌ لم يُعثر على أسئلة", 404
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True); df.to_excel(EXCEL_FILE, index=False); conn.commit()
    fidx = _build_idx(df); _build_names(); _build_prof_index()
    return f"✅ تم حفظ {len(rows)} إجابة!", 200

if __name__ == "__main__":
    app.run(port=5000, debug=False)
