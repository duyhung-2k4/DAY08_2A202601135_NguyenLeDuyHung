"""
ĐHQGHN Student Assistant — Premium RAG Chatbot
Streamlit app styled as a modern AI SaaS product (ChatGPT / Claude / Linear inspired).

Chạy:
    streamlit run app.py
"""

import html
import inspect
import os
import re
import sys
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="ĐHQGHN Student Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUGGESTIONS = [
    {
        "icon": "🎓",
        "category": "Tuition",
        "text": "Định mức học phí các chương trình đào tạo tại Trường Đại học Công nghệ năm học 2025-2026 là bao nhiêu?",
    },
    {
        "icon": "💰",
        "category": "Scholarship",
        "text": "Điều kiện và thủ tục xét duyệt học bổng tại ĐHQGHN như thế nào?",
    },
    {
        "icon": "🏠",
        "category": "Dormitory",
        "text": "Thủ tục đăng ký nội trú ký túc xá ĐHQGHN cần những giấy tờ gì?",
    },
    {
        "icon": "📚",
        "category": "Regulations",
        "text": "Quy định về thời gian học tập của sinh viên UET-VNU ra sao?",
    },
    {
        "icon": "📝",
        "category": "Procedures",
        "text": "Sinh viên đăng ký học lại, học cải thiện theo quy chế đào tạo như thế nào?",
    },
]

MAX_HISTORY_TURNS = 3
STOPWORDS = {
    "là", "của", "cho", "như", "thế", "nào", "bao", "nhiêu", "có", "các", "và",
    "với", "được", "tại", "về", "gì", "khi", "để", "một", "những", "này", "đó",
    "hay", "thì", "làm", "sao", "ở", "ra", "trong", "theo", "phải", "hỏi",
}

# =============================================================================
# PREMIUM SAAS CSS (light theme, Inter typeface, 8px spacing system)
# =============================================================================

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --primary: #4F46E5;
  --secondary: #6366F1;
  --accent: #3B82F6;
  --bg: #F8FAFC;
  --surface: #FFFFFF;
  --border: #E5E7EB;
  --text: #111827;
  --text-secondary: #6B7280;
  --success: #10B981;
  --warning: #F59E0B;
  --error: #EF4444;
  --radius-lg: 20px;
  --radius-md: 16px;
  --radius-sm: 10px;
  --shadow-sm: 0 1px 2px rgba(17, 24, 39, 0.04);
  --shadow-md: 0 4px 16px rgba(17, 24, 39, 0.06);
  --shadow-lg: 0 12px 32px rgba(79, 70, 229, 0.10);
}

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

[data-testid="stAppViewContainer"], .main {
    background: var(--bg);
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 6rem;
    max-width: 900px;
}

h1, h2, h3 { color: var(--text); font-weight: 700; }

/* -------------------- FADE-IN -------------------- */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}
.stChatMessage { animation: fadeInUp 260ms ease; }

/* -------------------- SIDEBAR -------------------- */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
    width: 280px !important;
}
section[data-testid="stSidebar"] > div {
    padding: 1.25rem 1rem;
}
section[data-testid="stSidebar"] .stMarkdown { color: var(--text); }

.sb-logo {
    display: flex;
    align-items: center;
    gap: .6rem;
    padding: .25rem 0 1.25rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}
.sb-logo .mark {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    box-shadow: var(--shadow-sm);
    flex: none;
}
.sb-logo .brand-name {
    font-weight: 700;
    font-size: .95rem;
    color: var(--text);
    line-height: 1.2;
}
.sb-logo .brand-sub {
    font-size: .72rem;
    color: var(--text-secondary);
}

.sb-label {
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: var(--text-secondary);
    margin: 1.1rem 0 .5rem 0;
}

.sb-history-item {
    font-size: .8rem;
    color: var(--text-secondary);
    padding: .5rem .6rem;
    border-radius: var(--radius-sm);
    margin-bottom: .3rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    border: 1px solid transparent;
    transition: all 150ms ease;
}
.sb-history-item:hover {
    background: var(--bg);
    border-color: var(--border);
    color: var(--text);
}

.status-card {
    display: flex;
    align-items: center;
    gap: .55rem;
    padding: .55rem .7rem;
    border-radius: var(--radius-sm);
    background: var(--bg);
    border: 1px solid var(--border);
    margin-bottom: .45rem;
    font-size: .78rem;
    color: var(--text);
    font-weight: 500;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex: none;
    box-shadow: 0 0 0 3px rgba(16, 185, 129, .15);
}
.status-dot.ok { background: var(--success); box-shadow: 0 0 0 3px rgba(16, 185, 129, .15); }
.status-dot.warn { background: var(--warning); box-shadow: 0 0 0 3px rgba(245, 158, 11, .15); }
.status-dot.err { background: var(--error); box-shadow: 0 0 0 3px rgba(239, 68, 68, .15); }

/* -------------------- SIDEBAR: NEW CHAT BUTTON -------------------- */
.st-key-new_chat button {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    box-shadow: var(--shadow-sm) !important;
    transition: transform 150ms ease, box-shadow 150ms ease !important;
}
.st-key-new_chat button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md) !important;
}
.st-key-new_chat button p { color: #FFFFFF !important; }

/* -------------------- GENERIC BUTTONS -------------------- */
.stButton > button {
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    font-size: .88rem;
    font-weight: 500;
    padding: .55rem 1rem;
    transition: all 180ms ease;
    box-shadow: var(--shadow-sm);
}
.stButton > button:hover {
    border-color: var(--primary);
    color: var(--primary);
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}

/* -------------------- SUGGESTION CARDS -------------------- */
.st-key-suggest_card button {
    text-align: left !important;
    justify-content: flex-start !important;
    white-space: normal !important;
    height: auto !important;
    min-height: 92px;
    padding: 1rem 1.1rem !important;
    border-radius: var(--radius-md) !important;
    line-height: 1.45;
    font-size: .87rem !important;
    color: var(--text) !important;
    position: relative;
}
.st-key-suggest_card button:hover {
    border-color: var(--primary) !important;
    box-shadow: var(--shadow-lg) !important;
    transform: translateY(-2px);
}
.st-key-suggest_card button::after {
    content: "→";
    position: absolute;
    right: 1rem;
    bottom: .85rem;
    color: var(--text-secondary);
    font-weight: 600;
    opacity: 0;
    transition: opacity 180ms ease, transform 180ms ease;
    transform: translateX(-4px);
}
.st-key-suggest_card button:hover::after {
    opacity: 1;
    transform: translateX(0);
    color: var(--primary);
}

/* -------------------- CHAT INPUT -------------------- */
.stChatInputContainer { padding: 1rem 0; }
.stChatInput {
    border-radius: 999px !important;
}
.stChatInput textarea, .stChatInput input {
    border-radius: 999px !important;
    border: 1px solid var(--border) !important;
    background: var(--surface) !important;
    color: var(--text) !important;
    padding: .85rem 1.4rem !important;
    font-size: .95rem !important;
    box-shadow: var(--shadow-md) !important;
    transition: all 180ms ease;
}
.stChatInput textarea:focus, .stChatInput input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.12), var(--shadow-md) !important;
}
.stChatInput textarea::placeholder, .stChatInput input::placeholder {
    color: #9CA3AF !important;
}

/* -------------------- CHAT MESSAGES -------------------- */
.stChatMessage {
    background: transparent !important;
    border: none !important;
    padding: .7rem 0 !important;
}
.stChatMessage [data-testid="stChatMessageContent"] {
    background: transparent;
    padding: 0 .5rem;
    color: var(--text);
}
.stChatMessage:has([data-testid="stChatMessageContentUser"]) [data-testid="stChatMessageContent"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: .7rem 1rem;
    box-shadow: var(--shadow-sm);
}
.stChatMessage:has([data-testid="stChatMessageContentUser"]) {
    margin-right: 0;
    margin-left: auto;
    max-width: 75%;
}
.stChatMessage:has([data-testid="stChatMessageContentAssistant"]) {
    margin-left: 0;
    max-width: 100%;
}

/* -------------------- MISC WIDGETS -------------------- */
.stSpinner { opacity: .7; }
details > summary {
    cursor: pointer;
    padding: .5rem 0;
    color: var(--text-secondary);
    font-weight: 600;
    font-size: .87rem;
}
details > summary:hover { color: var(--primary); }

.stToggle label { font-size: .87rem; font-weight: 500; color: var(--text); }
.stSlider > div > div > div { border-radius: 8px; }
.stDivider, hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

.stInfo, .stWarning, .stError {
    border-radius: var(--radius-sm);
    padding: .9rem 1rem;
    border-left: 4px solid;
}
.stInfo { border-left-color: var(--accent); }
.stWarning { border-left-color: var(--warning); }
.stError { border-left-color: var(--error); }

/* -------------------- HERO -------------------- */
.chat-hero {
    text-align: center;
    padding: 3rem 1.5rem 2.5rem 1.5rem;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(99,102,241,.10) 0%, rgba(99,102,241,0) 70%),
                linear-gradient(180deg, rgba(79,70,229,.04) 0%, rgba(79,70,229,0) 100%);
    border-radius: var(--radius-lg);
    margin-bottom: 2rem;
}
.chat-hero .orb {
    width: 64px;
    height: 64px;
    margin: 0 auto 1.25rem auto;
    border-radius: 20px;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    box-shadow: var(--shadow-lg);
}
.chat-hero h1 {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -.02em;
    margin-bottom: .6rem;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 60%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.chat-hero p {
    font-size: 1rem;
    color: var(--text-secondary);
    max-width: 560px;
    margin: 0 auto;
    line-height: 1.6;
}
.hero-badges {
    display: flex;
    justify-content: center;
    gap: .6rem;
    flex-wrap: wrap;
    margin-top: 1.5rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: .4rem;
    padding: .45rem .9rem;
    border-radius: 999px;
    background: var(--surface);
    border: 1px solid var(--border);
    font-size: .8rem;
    font-weight: 500;
    color: var(--text);
    box-shadow: var(--shadow-sm);
}

.section-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text);
    margin: 0 0 1rem 2px;
}

/* -------------------- SOURCE CARDS -------------------- */
.source-card {
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: .9rem 1rem;
    margin-bottom: .6rem;
    background: var(--surface);
    box-shadow: var(--shadow-sm);
}
.source-card .head {
    display: flex;
    align-items: center;
    gap: .5rem;
    margin-bottom: .5rem;
    font-size: .8rem;
}
.source-card .idx {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--primary), var(--accent));
    color: #FFF;
    font-size: .66rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex: none;
}
.source-card .name {
    font-weight: 600;
    color: var(--text);
    flex: 1;
    word-break: break-all;
}
.source-card .type {
    font-size: .68rem;
    padding: .15rem .5rem;
    border-radius: 999px;
    background: rgba(79, 70, 229, 0.10);
    color: var(--primary);
    font-weight: 600;
}
.source-card .score {
    font-size: .7rem;
    color: #9CA3AF;
    font-family: ui-monospace, monospace;
}
.source-card .progress {
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
    margin: .4rem 0;
}
.source-card .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--accent));
    border-radius: 2px;
}
.source-card .text {
    font-size: .82rem;
    line-height: 1.55;
    color: var(--text-secondary);
    margin-top: .4rem;
}
.source-card mark {
    background: #FDE68A;
    color: #111827;
    padding: 0 3px;
    border-radius: 3px;
}

/* -------------------- CHIPS -------------------- */
.chip {
    display: inline-block;
    padding: .3rem .7rem;
    margin-right: .4rem;
    margin-bottom: .3rem;
    font-size: .74rem;
    font-weight: 500;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--bg);
    color: var(--text-secondary);
}
.chip.active {
    background: rgba(79, 70, 229, 0.10);
    border-color: var(--primary);
    color: var(--primary);
}

/* -------------------- TYPING INDICATOR -------------------- */
.typing-dots { display: inline-flex; gap: 4px; padding: .3rem 0; }
.typing-dots span {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--primary);
    animation: typingBounce 1.1s infinite ease-in-out;
}
.typing-dots span:nth-child(2) { animation-delay: .15s; }
.typing-dots span:nth-child(3) { animation-delay: .3s; }
@keyframes typingBounce {
    0%, 60%, 100% { transform: translateY(0); opacity: .5; }
    30% { transform: translateY(-4px); opacity: 1; }
}
</style>
"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# =============================================================================
# HELPERS
# =============================================================================

def query_terms(query: str) -> list[str]:
    words = re.findall(r"\w+", query.lower(), flags=re.UNICODE)
    return [w for w in dict.fromkeys(words) if len(w) >= 2 and w not in STOPWORDS]


def highlight(text: str, terms: list[str]) -> str:
    safe = html.escape(text)
    if not terms:
        return safe
    pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", safe)


def render_sources(sources: list[dict], query: str) -> None:
    if not sources:
        return
    terms = query_terms(query)
    scores = [float(s.get("score", 0) or 0) for s in sources]
    top = max(scores) if scores and max(scores) > 0 else 1.0

    st.markdown("**References**", help="Sources used to generate this response")
    for i, src in enumerate(sources, 1):
        meta = src.get("metadata", {}) or {}
        name = html.escape(str(meta.get("source", "Unknown")))
        doc_type = str(meta.get("type", "unknown")).lower()
        score = float(src.get("score", 0) or 0)
        ratio = max(min(score / top, 1.0), 0.02) * 100
        content = src.get("content", "")
        if len(content) > 280:
            content = content[:280] + " …"

        st.markdown(
            f"""<div class="source-card">
  <div class="head">
    <span class="idx">{i}</span>
    <span class="name">{name}</span>
    <span class="type">{html.escape(doc_type)}</span>
    <span class="score">{score:.4f}</span>
  </div>
  <div class="progress"><div class="progress-fill" style="width:{ratio:.1f}%"></div></div>
  <div class="text">{highlight(content, terms)}</div>
</div>""",
            unsafe_allow_html=True,
        )


def build_history(messages: list[dict]) -> list[dict]:
    trimmed = messages[-(MAX_HISTORY_TURNS * 2):]
    return [{"role": m["role"], "content": m["content"]} for m in trimmed]


def call_rag(query: str, top_k: int, history: list[dict]) -> dict:
    from src.task10_generation import generate_with_citation
    kwargs = {"top_k": top_k}
    if history and "history" in inspect.signature(generate_with_citation).parameters:
        kwargs["history"] = history
    return generate_with_citation(query, **kwargs)


def demo_response(query: str) -> dict:
    return {
        "answer": (
            f"**Câu hỏi:** {query}\n\n"
            "**Câu trả lời:**\n\n"
            "Định mức học phí các chương trình đào tạo tại Trường Đại học Công nghệ được quy định theo từng năm học và tính theo tín chỉ đăng ký "
            "[Định mức học phí 2025-2026]. Sinh viên diện chính sách, có hoàn cảnh khó khăn hoặc đạt thành tích học tập xuất sắc có thể được xét cấp "
            "học bổng theo Quy định công tác quản lý và sử dụng học bổng tại ĐHQGHN [Quy định học bổng ĐHQGHN, 4618]. "
            "Về chỗ ở, sinh viên nộp hồ sơ xét duyệt nội trú tại Ký túc xá theo hướng dẫn của Trung tâm Hỗ trợ sinh viên [Ký túc xá ĐHQGHN]."
        ),
        "sources": [
            {
                "content": "Định mức học phí các chương trình đào tạo năm học 2025-2026 được tính theo số tín chỉ đăng ký của từng chương trình.",
                "score": 0.82,
                "metadata": {"source": "dinh-muc-hoc-phi-2025-2026.md", "type": "news"},
            },
            {
                "content": "Quy định công tác quản lý và sử dụng học bổng tại Đại học Quốc gia Hà Nội quy định về đối tượng, điều kiện và mức xét học bổng.",
                "score": 0.71,
                "metadata": {"source": "quy-dinh-hoc-bong-dhqghn-4618.pdf", "type": "legal"},
            },
            {
                "content": "Sinh viên nộp hồ sơ xét duyệt nội trú và làm thủ tục nhận phòng tại Ký túc xá theo hướng dẫn của Trung tâm Hỗ trợ sinh viên.",
                "score": 0.64,
                "metadata": {"source": "ky-tuc-xa-css-vnu.md", "type": "news"},
            },
        ],
        "retrieval_source": "hybrid",
    }


def export_markdown(messages: list[dict]) -> str:
    lines = ["# ĐHQGHN Student Assistant — Conversation History\n"]
    for m in messages:
        if m["role"] == "user":
            lines.append(f"\n## Q: {m['content']}\n")
        else:
            lines.append(f"**A:** {m['content']}\n")
            srcs = m.get("sources", []) or []
            if srcs:
                lines.append("\n**Sources:**\n")
                for s in srcs:
                    meta = s.get("metadata", {}) or {}
                    lines.append(
                        f"- {meta.get('source', 'Unknown')} "
                        f"({meta.get('type', 'unknown')}) `{float(s.get('score', 0) or 0):.4f}`"
                    )
                lines.append("")
    return "\n".join(lines)


# =============================================================================
# SESSION STATE
# =============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown(
        """<div class="sb-logo">
  <div class="mark">🎓</div>
  <div>
    <div class="brand-name">ĐHQGHN Assistant</div>
    <div class="brand-sub">Student Services RAG</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    if st.button("+ New Conversation", use_container_width=True, key="new_chat"):
        st.session_state.messages = []
        st.rerun()

    user_questions = [m["content"] for m in st.session_state.messages if m["role"] == "user"]
    if user_questions:
        st.markdown('<div class="sb-label">Chat History</div>', unsafe_allow_html=True)
        history_html = "".join(
            f'<div class="sb-history-item" title="{html.escape(q)}">{html.escape(q)}</div>'
            for q in reversed(user_questions[-8:])
        )
        st.markdown(history_html, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Settings</div>', unsafe_allow_html=True)
    top_k = st.slider(
        "Context size", 3, 10, 5,
        help="Number of document chunks to retrieve"
    )
    use_memory = st.toggle(
        "Conversation memory", value=True,
        help="Remember recent messages for follow-up questions"
    )
    demo_mode = st.toggle(
        "Demo mode", value=False,
        help="Show sample responses (pipeline not implemented yet)"
    )

    st.markdown('<div class="sb-label">System Status</div>', unsafe_allow_html=True)
    has_key = bool(os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
    has_db = (PROJECT_ROOT / "chroma_db").exists()
    citation_ready = has_key and has_db

    st.markdown(
        f"""<div class="status-card">
  <span class="status-dot {'ok' if has_key else 'err'}"></span>
  <span>API {"Connected" if has_key else "Key Missing"}</span>
</div>
<div class="status-card">
  <span class="status-dot {'ok' if has_db else 'warn'}"></span>
  <span>Vector Database {"Ready" if has_db else "Building"}</span>
</div>
<div class="status-card">
  <span class="status-dot {'ok' if citation_ready else 'warn'}"></span>
  <span>Citation Engine {"Active" if citation_ready else "Standby"}</span>
</div>""",
        unsafe_allow_html=True,
    )

    if st.session_state.messages:
        st.markdown('<div class="sb-label">Export</div>', unsafe_allow_html=True)
        st.download_button(
            "Export Conversation",
            data=export_markdown(st.session_state.messages),
            file_name="conversation.md",
            mime="text/markdown",
            use_container_width=True,
        )

# =============================================================================
# MAIN CHAT AREA
# =============================================================================

# Hero + suggestions when empty
if not st.session_state.messages:
    st.markdown(
        """<div class="chat-hero">
  <div class="orb">🎓</div>
  <h1>ĐHQGHN Student Assistant</h1>
  <p>Ask anything about tuition, scholarships, dormitories, academic regulations, and student services.</p>
  <div class="hero-badges">
    <span class="hero-badge">📄 Official Documents</span>
    <span class="hero-badge">⚡ AI-powered Search</span>
    <span class="hero-badge">✅ Verified Citations</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Suggested Questions</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for idx, item in enumerate(SUGGESTIONS):
        with cols[idx % 2]:
            label = f"{item['icon']}  {item['text']}"
            if st.button(label, key=f"suggest_card_{idx}", use_container_width=True):
                st.session_state.pending_query = item["text"]
                st.rerun()

    st.info(
        "Click a question above or type your own below. "
        "Every response includes citations from official university documents."
    )

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # Metadata chips
            if msg.get("retrieval_source") or msg.get("sources"):
                chips = []
                if msg.get("retrieval_source") == "hybrid":
                    chips.append('<span class="chip active">Hybrid Search</span>')
                elif msg.get("retrieval_source") == "pageindex":
                    chips.append('<span class="chip active">PageIndex Fallback</span>')
                if msg.get("sources"):
                    chips.append(f'<span class="chip">{len(msg["sources"])} sources</span>')
                if msg.get("latency"):
                    chips.append(f'<span class="chip">{msg["latency"]:.1f}s</span>')
                if chips:
                    st.markdown("".join(chips), unsafe_allow_html=True)
            render_sources(msg.get("sources", []), msg.get("query", ""))

# Chat input & processing
user_input = st.chat_input("Ask anything about student services...")
query = user_input or st.session_state.pending_query

if query:
    st.session_state.pending_query = None
    history = build_history(st.session_state.messages) if use_memory else []

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="typing-dots"><span></span><span></span><span></span></div>',
            unsafe_allow_html=True,
        )
        retrieval_source = ""
        started = time.perf_counter()
        try:
            response = demo_response(query) if demo_mode else call_rag(query, top_k, history)
            answer = response.get("answer") or "Unable to answer this question."
            sources = response.get("sources", []) or []
            retrieval_source = response.get("retrieval_source", "")
        except NotImplementedError:
            answer = "Pipeline not yet implemented. Enable **Demo mode** in sidebar to see sample responses."
            sources = []
        except Exception as e:
            answer = f"Error: {type(e).__name__}: {str(e)[:100]}"
            sources = []

        latency = time.perf_counter() - started
        placeholder.empty()

        st.markdown(answer)

        if retrieval_source or sources:
            chips = []
            if retrieval_source == "hybrid":
                chips.append('<span class="chip active">Hybrid Search</span>')
            elif retrieval_source == "pageindex":
                chips.append('<span class="chip active">PageIndex Fallback</span>')
            if sources:
                chips.append(f'<span class="chip">{len(sources)} sources</span>')
            if latency:
                chips.append(f'<span class="chip">{latency:.1f}s</span>')
            if chips:
                st.markdown("".join(chips), unsafe_allow_html=True)

        render_sources(sources, query)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
        "query": query,
        "latency": latency,
    })
