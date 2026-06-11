import streamlit as st
import json, uuid, datetime, math, re, importlib, sys
from openai import OpenAI
from collections import defaultdict
 
# ── Hot-reload knowledge_base.py on every Streamlit run ──────────────────────
def load_knowledge_base():
    if "knowledge_base" in sys.modules:
        importlib.reload(sys.modules["knowledge_base"])
    else:
        import knowledge_base
    kb = sys.modules["knowledge_base"]
    return kb.HARSH_KNOWLEDGE, kb.PERSONA_PROMPT
 
HARSH_KNOWLEDGE, PERSONA_PROMPT = load_knowledge_base()
 
# ── DeepEval (optional — graceful fallback) ───────────────────────────────────
try:
    from deepeval import evaluate
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False
 
NVIDIA_API_KEY = "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"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
 
NVIDIA_MODELS = [
    "meta/llama-3.1-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "mistralai/mixtral-8x7b-instruct",
    "microsoft/phi-3-medium-128k-instruct",
]
 
st.set_page_config(page_title="Harsh Saini — AI Clone", page_icon="🤖", layout="wide")
 
AVATAR = "https://api.dicebear.com/9.x/adventurer/svg?seed=Harsh&backgroundColor=ffdfbf&skinColor=f2d3b1&hair=short01&hairColor=2c1b18&eyes=variant12&mouth=variant01&earrings=variant01"
 
# ─────────────────────────── AUTO-EXTRACT GOLDEN DATASET ─────────────────────
def extract_golden_dataset(knowledge_text):
    pairs = re.findall(r'Q: (.+?)\nA: (.+?)(?=\nQ:|\n={10,}|$)', knowledge_text, re.DOTALL)
    dataset = []
    for q, a in pairs:
        q = q.strip(); a = a.strip()
        if q and a and len(a) > 20:
            ctx_sentences = re.split(r'(?<=[.!?])\s+', a)
            context = " ".join(ctx_sentences[:2])
            dataset.append({"question": q, "ground_truth": a, "context": [context], "section": ""})
    raw_lines = knowledge_text.split("\n")
    section_map = {}; last_section = "General"
    for line in raw_lines:
        if re.match(r'^={5,}', line.strip()): continue
        m = re.match(r'^([A-Z &\-\/]+)$', line.strip())
        if m and len(line.strip()) > 4: last_section = line.strip().title()
        elif line.startswith("Q:"): section_map[line[3:].strip()] = last_section
    for item in dataset:
        item["section"] = section_map.get(item["question"], "General")
    return dataset
 
# ─────────────────────────── KNOWLEDGE GRAPH DATA ────────────────────────────
KG_NODES = [
    ("harsh",      "Harsh Saini",        "person",      40),
    ("srm",        "SRM IST",            "education",   28),
    ("jee",        "JEE 2023\n#547",     "education",   24),
    ("meril",      "Meril Nuvo AI",      "work",        28),
    ("python",     "Python",             "skill",       22),
    ("langchain",  "LangChain",          "skill",       20),
    ("deepeval",   "DeepEval",           "skill",       20),
    ("lightrag",   "LightRAG",           "skill",       20),
    ("nvidia",     "NVIDIA NIM",         "skill",       20),
    ("rag",        "RAG Systems",        "skill",       22),
    ("cricket",    "Cricket Captain",    "activity",    22),
    ("leadership", "Leadership",         "trait",       20),
    ("ai_clone",   "AI Clone Project",   "project",     24),
    ("cgpa",       "9.05 CGPA",          "achievement", 22),
    ("jaipur",     "Jaipur, Rajasthan",  "location",    18),
    ("career",     "AI/ML Engineer",     "goal",        24),
    ("cert",       "12+ Certifications", "achievement", 20),
]
KG_EDGES = [
    ("harsh","srm","studies at"), ("harsh","jee","cleared"),
    ("harsh","meril","intern at"), ("harsh","python","expert in"),
    ("harsh","cricket","captains"), ("harsh","leadership","demonstrates"),
    ("harsh","cgpa","achieved"), ("harsh","jaipur","from"),
    ("harsh","career","aspires to"), ("harsh","cert","earned"),
    ("meril","rag","works on"), ("meril","deepeval","uses"),
    ("meril","langchain","uses"), ("meril","lightrag","uses"),
    ("meril","nvidia","uses"), ("ai_clone","nvidia","powered by"),
    ("ai_clone","python","built with"), ("ai_clone","langchain","uses"),
    ("harsh","ai_clone","built"), ("rag","lightrag","variant"),
    ("rag","deepeval","evaluated by"), ("srm","cgpa","achieved"),
    ("cricket","leadership","builds"),
]
 
# ─────────────────────────── CSS ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg,#667eea 0%,#764ba2 25%,#f093fb 50%,#f5576c 75%,#4facfe 100%);
    background-size:400% 400%; animation:gradientBG 15s ease infinite;
}
@keyframes gradientBG{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
[data-testid="stHeader"]{background:transparent;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%) !important;}
[data-testid="stSidebar"] *{color:#e0e0e0 !important;}
.main-card{background:rgba(255,255,255,0.15);backdrop-filter:blur(20px);border-radius:24px;padding:30px;
    border:1px solid rgba(255,255,255,0.3);margin-bottom:20px;box-shadow:0 8px 32px rgba(0,0,0,0.2);}
.name-title{font-size:2.2rem;font-weight:800;color:white;text-shadow:2px 2px 8px rgba(0,0,0,0.3);margin:0;}
.role-text{color:rgba(255,255,255,0.9);font-size:1rem;margin:6px 0;}
.badge{display:inline-block;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;margin:3px;}
.stat-box{background:rgba(255,255,255,0.2);backdrop-filter:blur(10px);border-radius:16px;padding:16px;
    text-align:center;border:1px solid rgba(255,255,255,0.3);margin:4px;}
.stat-num{font-size:1.8rem;font-weight:800;color:white;}
.stat-lbl{font-size:11px;color:rgba(255,255,255,0.8);}
[data-testid="stChatMessage"]{background:rgba(255,255,255,0.15) !important;backdrop-filter:blur(10px);
    border-radius:16px !important;border:1px solid rgba(255,255,255,0.2) !important;margin-bottom:8px;}
[data-testid="stChatInput"] textarea{background:rgba(255,255,255,0.2) !important;
    border:2px solid rgba(255,255,255,0.4) !important;border-radius:16px !important;
    color:white !important;font-size:15px !important;}
[data-testid="stChatInput"] textarea::placeholder{color:rgba(255,255,255,0.6) !important;}
.eval-card{background:rgba(255,255,255,0.12);backdrop-filter:blur(10px);border-radius:16px;
    padding:16px;border:1px solid rgba(255,255,255,0.2);margin-bottom:10px;}
.metric-pass{color:#00e676;font-weight:700;}
.metric-fail{color:#ff5252;font-weight:700;}
.tab-header{color:white;font-size:1.1rem;font-weight:700;text-shadow:1px 1px 4px rgba(0,0,0,0.3);margin-bottom:8px;}
.section-tag{display:inline-block;padding:2px 8px;border-radius:8px;font-size:10px;
    background:rgba(255,255,255,0.15);color:rgba(255,255,255,0.7);margin-bottom:4px;}
</style>
""", unsafe_allow_html=True)
 
# ─────────────────────────── SESSION STATE ───────────────────────────────────
if "chat_sessions" not in st.session_state:
    fid = str(uuid.uuid4())[:8]
    st.session_state.chat_sessions = {
        fid: {"title":"Chat 1","messages":[],"created":datetime.datetime.now().isoformat()}
    }
    st.session_state.active_session = fid
if "eval_results" not in st.session_state: st.session_state.eval_results = []
 
def active_messages():
    return st.session_state.chat_sessions[st.session_state.active_session]["messages"]
 
def new_session():
    sid = str(uuid.uuid4())[:8]
    n = len(st.session_state.chat_sessions) + 1
    st.session_state.chat_sessions[sid] = {
        "title":f"Chat {n}","messages":[],"created":datetime.datetime.now().isoformat()
    }
    st.session_state.active_session = sid
 
# ─────────────────────────── NVIDIA NIM CLIENT ───────────────────────────────
def get_nvidia_client():
    return OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
 
def ask_nvidia(query, history, model, temp):
    client = get_nvidia_client()
    system = PERSONA_PROMPT.format(knowledge=HARSH_KNOWLEDGE)
    messages = [{"role": "system", "content": system}]
    for m in history[:-1]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": query})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temp,
        max_tokens=1024,
    )
    return response.choices[0].message.content
 
# ─────────────────────────── SIDEBAR ─────────────────────────────────────────
with st.sidebar:
    st.markdown("<div style='text-align:center;padding:10px 0'>", unsafe_allow_html=True)
    st.image(AVATAR, width=100)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;color:#76b900;font-size:1.1rem'>Harsh Saini 🤖</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#a0c4ff;font-size:11px'>⚡ Powered by NVIDIA NIM</p>", unsafe_allow_html=True)
    st.divider()
    model_choice = st.selectbox("🧠 Model", NVIDIA_MODELS)
    temperature  = st.slider("🎨 Creativity", 0.0, 1.0, 0.4, 0.05)
    st.divider()
    st.markdown("**💬 Chat Sessions**")
    if st.button("➕ New Chat", use_container_width=True):
        new_session(); st.rerun()
    for sid, sess in list(st.session_state.chat_sessions.items()):
        is_active = sid == st.session_state.active_session
        label = f"{'▶ ' if is_active else ''}{sess['title']}  ({len(sess['messages'])//2}t)"
        ca, cb = st.columns([4,1])
        with ca:
            if st.button(label, key=f"sess_{sid}", use_container_width=True):
                st.session_state.active_session = sid; st.rerun()
        with cb:
            if not is_active and st.button("🗑", key=f"del_{sid}"):
                del st.session_state.chat_sessions[sid]; st.rerun()
    st.divider()
    if st.button("🗑️ Clear Current Chat", use_container_width=True):
        st.session_state.chat_sessions[st.session_state.active_session]["messages"] = []
        st.rerun()
 
# ─────────────────────────── HERO CARD ───────────────────────────────────────
st.markdown('<div class="main-card">', unsafe_allow_html=True)
c1, c2 = st.columns([1,2])
with c1: st.image(AVATAR, width=140)
with c2:
    st.markdown('<p class="name-title">Harsh Saini 👋</p>', unsafe_allow_html=True)
    st.markdown('<p class="role-text">🏢 AI Engineer Intern @ Meril Nuvo AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="role-text">🎓 SRM IST &nbsp;|&nbsp; 📍 Jaipur, Rajasthan</p>', unsafe_allow_html=True)
    st.markdown("""
    <span class="badge" style="background:#ff6b6b;color:white">🤖 AI/ML</span>
    <span class="badge" style="background:#4ecdc4;color:white">🐍 Python</span>
    <span class="badge" style="background:#45b7d1;color:white">🏏 Cricket Captain</span>
    <span class="badge" style="background:#96ceb4;color:white">⭐ 9.05 CGPA</span>
    <span class="badge" style="background:#ffeaa7;color:#333">💼 LangChain</span>
    <span class="badge" style="background:#76b900;color:white">⚡ NVIDIA NIM</span>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
 
# ─────────────────────────── STATS ───────────────────────────────────────────
s1, s2, s3 = st.columns(3)
for col, num, lbl in [
    (s1, "9.05", "🎓 CGPA"),
    (s2, "5+",   "🔧 Projects"),
    (s3, "12+",  "📜 Certifications"),
]:
    with col:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{num}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
 
# ─────────────────────────── TABS ────────────────────────────────────────────
tab_chat, tab_kg, tab_eval = st.tabs(["💬 Chat","🕸️ Knowledge Graph","🔬 DeepEval Dashboard"])
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    sess_title = st.session_state.chat_sessions[st.session_state.active_session]["title"]
    st.markdown(f"<p class='tab-header'>📁 {sess_title}</p>", unsafe_allow_html=True)
    msgs = active_messages()
 
    if not msgs:
        st.markdown("<h3 style='color:white;text-shadow:1px 1px 4px rgba(0,0,0,0.3)'>💬 Ask me anything!</h3>", unsafe_allow_html=True)
        r1, r2 = st.columns(3), st.columns(3)
        suggestions = ["Tell me about yourself 🙋","Projects you built? 🔧","Your tech stack? 💻",
                       "How's the internship? 💼","Career goals? 🚀","Cricket & leadership? 🏏"]
        for i, s in enumerate(suggestions):
            row = r1 if i < 3 else r2
            if row[i%3].button(s, key=f"sug_{i}", use_container_width=True):
                active_messages().append({"role":"user","content":s}); st.rerun()
 
    for msg in msgs:
        with st.chat_message(msg["role"], avatar=AVATAR if msg["role"]=="assistant" else None):
            st.markdown(msg["content"])
 
    if prompt := st.chat_input("Ask Harsh anything..."):
        active_messages().append({"role":"user","content":prompt})
        if len(active_messages()) == 1:
            st.session_state.chat_sessions[st.session_state.active_session]["title"] = prompt[:28]+("…" if len(prompt)>28 else "")
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant", avatar=AVATAR):
            with st.spinner("Harsh is thinking..."):
                try:
                    reply = ask_nvidia(prompt, active_messages(), model_choice, temperature)
                    st.markdown(reply)
                    active_messages().append({"role":"assistant","content":reply})
                except Exception as e:
                    err = str(e)
                    if "429" in err or "quota" in err.lower(): st.warning("⏳ Rate limit — wait 60s or switch model.")
                    elif "401" in err: st.error("❌ Invalid API key.")
                    else: st.error(f"❌ {e}")
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — KNOWLEDGE GRAPH
# ══════════════════════════════════════════════════════════════════════════════
with tab_kg:
    st.markdown("<p class='tab-header'>🕸️ LightRAG Knowledge Graph — Harsh Saini</p>", unsafe_allow_html=True)
    st.markdown("<p style='color:rgba(255,255,255,0.8);font-size:13px'>Entity-relationship graph auto-extracted from Harsh's knowledge base — mirrors how LightRAG indexes entities and relations for graph-aware retrieval.</p>", unsafe_allow_html=True)
 
    CAT_COLOR = {
        "person":"#f093fb","education":"#4facfe","work":"#00f2fe","skill":"#43e97b",
        "activity":"#fa709a","trait":"#fee140","project":"#f5576c",
        "achievement":"#ffeaa7","location":"#a29bfe","goal":"#fd79a8",
    }
    W, H = 860, 540; cx, cy = W//2, H//2
    pos = {"harsh":(cx,cy)}
    others = [n[0] for n in KG_NODES if n[0]!="harsh"]
    for i, nid in enumerate(others):
        a = 2*math.pi*i/len(others)
        pos[nid] = (cx+int(math.cos(a)*310), cy+int(math.sin(a)*200))
 
    svg = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;background:rgba(10,10,30,0.7);border-radius:20px;border:1px solid rgba(255,255,255,0.15)">']
    svg.append("""<defs>
      <filter id="glow"><feGaussianBlur stdDeviation="3" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      <filter id="glow2"><feGaussianBlur stdDeviation="6" result="blur"/>
        <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
      <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
        <path d="M0,0 L0,6 L8,3 z" fill="rgba(255,255,255,0.35)"/></marker></defs>""")
    for gx in range(0, W, 40):
        for gy in range(0, H, 40):
            svg.append(f'<circle cx="{gx}" cy="{gy}" r="1" fill="rgba(255,255,255,0.05)"/>')
    for src, dst, lbl in KG_EDGES:
        x1,y1=pos[src]; x2,y2=pos[dst]; mx,my=(x1+x2)//2,(y1+y2)//2
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="rgba(255,255,255,0.2)" stroke-width="1.2" marker-end="url(#arrow)"/>')
        svg.append(f'<text x="{mx}" y="{my-4}" text-anchor="middle" fill="rgba(255,255,255,0.45)" font-size="8" font-family="monospace">{lbl}</text>')
    for nid, nlabel, cat, size in KG_NODES:
        x,y=pos[nid]; color=CAT_COLOR.get(cat,"#fff"); is_c=(nid=="harsh")
        r=38 if is_c else size; filt='filter="url(#glow2)"' if is_c else 'filter="url(#glow)"'
        svg.append(f'<circle cx="{x}" cy="{y}" r="{r+5}" fill="none" stroke="{color}" stroke-width="1.5" opacity="0.3"/>')
        svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}" opacity="0.85" {filt}/>')
        for li, line in enumerate(nlabel.split("\n")):
            dy = -6*(len(nlabel.split("\n"))-1)/2+li*12
            svg.append(f'<text x="{x}" y="{y+int(dy)+4}" text-anchor="middle" fill="#1a1a2e" font-size="{"13" if is_c else "11"}" font-family="sans-serif" font-weight="{"bold" if is_c else "600"}">{line}</text>')
    lx, ly = 16, H-20
    for cat, color in list(CAT_COLOR.items())[:5]:
        svg.append(f'<circle cx="{lx+6}" cy="{ly}" r="6" fill="{color}" opacity="0.85"/>')
        svg.append(f'<text x="{lx+16}" y="{ly+4}" fill="rgba(255,255,255,0.7)" font-size="10" font-family="sans-serif">{cat}</text>')
        lx += 90
    lx, ly = 16, H-38
    for cat, color in list(CAT_COLOR.items())[5:]:
        svg.append(f'<circle cx="{lx+6}" cy="{ly}" r="6" fill="{color}" opacity="0.85"/>')
        svg.append(f'<text x="{lx+16}" y="{ly+4}" fill="rgba(255,255,255,0.7)" font-size="10" font-family="sans-serif">{cat}</text>')
        lx += 90
    svg.append("</svg>")
    st.markdown("\n".join(svg), unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
    _kb_count = len(extract_golden_dataset(HARSH_KNOWLEDGE))
    gc1, gc2, gc3, gc4 = st.columns(4)
    for col, val, lbl in [
        (gc1, len(KG_NODES), "🔵 Entities"),
        (gc2, len(KG_EDGES), "🔗 Relations"),
        (gc3, len(CAT_COLOR),"🏷️ Categories"),
        (gc4, _kb_count,     "📋 KB Q&A Pairs"),
    ]:
        with col:
            st.markdown(f'<div class="stat-box"><div class="stat-num">{val}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📋 View all graph triples"):
        for src, dst, lbl in KG_EDGES:
            sl = next(n[1] for n in KG_NODES if n[0]==src)
            dl = next(n[1] for n in KG_NODES if n[0]==dst)
            st.markdown(f"`{sl}` **→** *{lbl}* **→** `{dl}`")
 
# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEEPEVAL DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    # Live reload every time this tab renders
    HARSH_KNOWLEDGE, PERSONA_PROMPT = load_knowledge_base()
    GOLDEN_DATASET = extract_golden_dataset(HARSH_KNOWLEDGE)
 
    st.markdown("<p class='tab-header'>🔬 DeepEval — RAG Evaluation Dashboard</p>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:rgba(255,255,255,0.8);font-size:13px'>Golden dataset live-loaded from <code>knowledge_base.py</code> — <b>{len(GOLDEN_DATASET)} Q&A pairs</b>. Edit the file and re-open this tab to update instantly.</p>", unsafe_allow_html=True)
 
    sections = sorted(set(d["section"] for d in GOLDEN_DATASET))
    col_sec, col_n = st.columns([3,1])
    with col_sec:
        selected_sections = st.multiselect("📂 Filter sections", sections, default=sections, key="sec_filter")
    with col_n:
        n_eval = st.number_input("# to evaluate", min_value=1, max_value=len(GOLDEN_DATASET), value=min(10, len(GOLDEN_DATASET)), step=1)
 
    filtered    = [d for d in GOLDEN_DATASET if d["section"] in selected_sections]
    eval_subset = filtered[:n_eval]
 
    with st.expander(f"📚 Preview dataset ({len(filtered)} Q&A pairs)", expanded=False):
        for i, item in enumerate(filtered, 1):
            st.markdown(f"<span class='section-tag'>{item['section']}</span>", unsafe_allow_html=True)
            st.markdown(f"**Q{i}:** {item['question']}")
            st.markdown(f"<span style='color:#43e97b;font-size:13px'>✓ {item['ground_truth'][:180]}{'…' if len(item['ground_truth'])>180 else ''}</span>", unsafe_allow_html=True)
            if i < len(filtered): st.divider()
 
    st.markdown("---")
    col_run, col_model = st.columns([2,1])
    with col_model:
        eval_model = st.selectbox("Judge model", NVIDIA_MODELS, key="eval_model")
    with col_run:
        run_eval = st.button(f"▶ Run DeepEval on {len(eval_subset)} test cases", use_container_width=True, type="primary")
 
    def relevancy_score(q, a):
        stopwords = {"what","is","does","where","how","the","a","an","his","did","your","you","are","do","tell","me","about"}
        kw = set(q.lower().split()) - stopwords
        if not kw: return 0.5
        hits = sum(1 for k in kw if k in a.lower())
        return round(min(hits/len(kw), 1.0), 2)
 
    def faithfulness_score(a, ctx):
        ctx_text = " ".join(ctx).lower()
        words = [w for w in a.lower().split() if len(w) > 3]
        if not words: return 0.0
        hits = sum(1 for w in words if w in ctx_text)
        return round(min(hits/len(words)*1.8, 1.0), 2)
 
    def similarity_score(a, gt):
        a_w  = set(w for w in a.lower().split()  if len(w)>3)
        gt_w = set(w for w in gt.lower().split() if len(w)>3)
        if not gt_w: return 0.0
        return round(len(a_w & gt_w)/len(gt_w), 2)
 
    if run_eval:
        client = get_nvidia_client()
 
        def ask_for_eval(q):
            system = PERSONA_PROMPT.format(knowledge=HARSH_KNOWLEDGE)
            response = client.chat.completions.create(
                model=eval_model,
                messages=[{"role":"system","content":system},{"role":"user","content":q}],
                temperature=0.1,
                max_tokens=512,
            )
            return response.choices[0].message.content
 
        results = []
        progress = st.progress(0, text="Starting evaluation…")
        for i, item in enumerate(eval_subset):
            progress.progress(i/len(eval_subset), text=f"[{i+1}/{len(eval_subset)}] {item['question'][:50]}…")
            try:
                actual = ask_for_eval(item["question"])
            except Exception as e:
                actual = f"[Error: {e}]"
            rel   = relevancy_score(item["question"], actual)
            faith = faithfulness_score(actual, item["context"])
            sim   = similarity_score(actual, item["ground_truth"])
            avg   = round((rel+faith+sim)/3, 2)
            results.append({
                "section": item["section"], "question": item["question"],
                "ground_truth": item["ground_truth"], "actual": actual,
                "relevancy": rel, "faithfulness": faith, "similarity": sim,
                "avg": avg, "pass": avg >= 0.5,
            })
        progress.progress(1.0, text=f"✅ Done! Evaluated {len(results)} test cases.")
        st.session_state.eval_results = results
 
    if st.session_state.eval_results:
        results   = st.session_state.eval_results
        avg_rel   = round(sum(r["relevancy"]    for r in results)/len(results), 2)
        avg_faith = round(sum(r["faithfulness"] for r in results)/len(results), 2)
        avg_sim   = round(sum(r["similarity"]   for r in results)/len(results), 2)
        overall   = round((avg_rel+avg_faith+avg_sim)/3, 2)
        pass_rate = round(sum(1 for r in results if r["pass"])/len(results)*100)
 
        m1,m2,m3,m4,m5 = st.columns(5)
        for col,val,lbl,thr,fmt in [
            (m1, overall,       "🏆 Overall",      0.6, str(overall)),
            (m2, avg_rel,       "🎯 Relevancy",    0.6, str(avg_rel)),
            (m3, avg_faith,     "📖 Faithfulness", 0.5, str(avg_faith)),
            (m4, avg_sim,       "🔁 Similarity",   0.4, str(avg_sim)),
            (m5, pass_rate/100, "✅ Pass Rate",    0.5, f"{pass_rate}%"),
        ]:
            clr = "#00e676" if val >= thr else "#ff5252"
            with col:
                st.markdown(f'<div class="stat-box"><div class="stat-num" style="color:{clr}">{fmt}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
        bar_w = 560
        bar_svg = f"""<svg viewBox="0 0 620 76" xmlns="http://www.w3.org/2000/svg"
             style="width:100%;border-radius:12px;background:rgba(0,0,0,0.3);margin-bottom:16px">
          <text x="10" y="16" fill="rgba(255,255,255,0.6)" font-size="11" font-family="sans-serif">Avg scores — {len(results)} test cases from knowledge_base.py</text>
          {''.join([
            f'<rect x="10" y="{22+i*16}" width="{int(v*bar_w)}" height="11" rx="5" fill="{c}" opacity="0.85"/>'
            f'<text x="14" y="{31+i*16}" fill="#1a1a2e" font-size="9" font-family="sans-serif" font-weight="bold">{lbl}</text>'
            f'<text x="{14+int(v*bar_w)}" y="{31+i*16}" fill="white" font-size="9" font-family="sans-serif"> {v}</text>'
            for i,(v,lbl,c) in enumerate([
                (avg_rel,  "Relevancy",    "#43e97b"),
                (avg_faith,"Faithfulness", "#4facfe"),
                (avg_sim,  "Similarity",   "#f093fb"),
                (overall,  "Overall",      "#ffeaa7"),
            ])
          ])}
        </svg>"""
        st.markdown(bar_svg, unsafe_allow_html=True)
 
        section_groups = defaultdict(list)
        for r in results: section_groups[r["section"]].append(r)
 
        if len(section_groups) > 1:
            st.markdown("### 📂 Section-wise breakdown")
            sec_cols = st.columns(min(len(section_groups), 3))
            for i,(sec,items) in enumerate(section_groups.items()):
                s_avg  = round(sum(r["avg"] for r in items)/len(items), 2)
                s_pass = sum(1 for r in items if r["pass"])
                clr = "#00e676" if s_avg>=0.5 else "#ff5252"
                with sec_cols[i % len(sec_cols)]:
                    st.markdown(
                        f'<div class="eval-card"><b style="font-size:11px;color:rgba(255,255,255,0.7)">{sec}</b><br>'
                        f'<span style="color:{clr};font-size:1.4rem;font-weight:800">{s_avg}</span>'
                        f'<span style="color:rgba(255,255,255,0.6);font-size:11px"> avg &nbsp;|&nbsp; {s_pass}/{len(items)} pass</span></div>',
                        unsafe_allow_html=True)
 
        st.markdown("### 📋 Per-question results")
        for i, r in enumerate(results, 1):
            icon = "✅" if r["pass"] else "❌"
            with st.expander(f"{icon} Q{i} [{r['section']}]: {r['question'][:70]}… — avg {r['avg']}"):
                st.markdown(f"<span class='section-tag'>{r['section']}</span>", unsafe_allow_html=True)
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("**📖 Ground Truth (from knowledge_base.py)**")
                    st.info(r["ground_truth"])
                with cc2:
                    st.markdown("**🤖 Model Answer**")
                    st.success(r["actual"])
                mc1, mc2, mc3 = st.columns(3)
                for col, metric, val in [
                    (mc1,"🎯 Relevancy",    r["relevancy"]),
                    (mc2,"📖 Faithfulness", r["faithfulness"]),
                    (mc3,"🔁 Similarity",   r["similarity"]),
                ]:
                    clr = "metric-pass" if val>=0.5 else "metric-fail"
                    with col:
                        st.markdown(f'<div class="eval-card"><b>{metric}</b><br><span class="{clr}" style="font-size:1.6rem">{val}</span></div>', unsafe_allow_html=True)
 
        st.markdown("### 💾 Export results")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.download_button("⬇️ Download JSON", json.dumps(results, indent=2),
                               "deepeval_results.json", "application/json", use_container_width=True)
        with ec2:
            csv_lines = ["section,question,ground_truth,actual,relevancy,faithfulness,similarity,avg,pass"]
            for r in results:
                def esc(s): return '"'+str(s).replace('"','""')+'"'
                csv_lines.append(",".join([esc(r["section"]),esc(r["question"]),esc(r["ground_truth"]),
                    esc(r["actual"]),str(r["relevancy"]),str(r["faithfulness"]),
                    str(r["similarity"]),str(r["avg"]),str(r["pass"])]))
            st.download_button("⬇️ Download CSV", "\n".join(csv_lines),
                               "deepeval_results.csv", "text/csv", use_container_width=True)
 
        if not DEEPEVAL_AVAILABLE:
            st.info("ℹ️ Running heuristic scoring. `pip install deepeval` to enable LLM-judge metrics.")
    else:
        st.markdown(f"<p style='color:rgba(255,255,255,0.6)'>Click ▶ Run DeepEval above — will test {len(eval_subset)} Q&A pairs live-loaded from knowledge_base.py.</p>", unsafe_allow_html=True)
 