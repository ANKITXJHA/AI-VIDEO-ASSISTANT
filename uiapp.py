import contextlib
import io
import os
import tempfile
import time
import traceback

import streamlit as st

# Reuse the existing pipeline and chat function exactly as they are (no logic changes)
from main import run_pipeline
from core.rag_engine import ask_question

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1300px;}
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #4338ca 55%, #0ea5e9 100%);
    border-radius: 18px; padding: 1.8rem 2.1rem; color: #fff;
    margin-bottom: 1.2rem; box-shadow: 0 12px 30px rgba(15,23,42,.25);
}
.hero h1 {margin: 0; font-size: 2rem; font-weight: 800; color: #fff;}
.hero p {margin: .4rem 0 0; opacity: .9; font-size: 1rem;}
.title-card {
    border: 1px solid rgba(128,128,128,.25); border-left: 6px solid #6366f1;
    background: rgba(99,102,241,.08); border-radius: 12px;
    padding: 1rem 1.3rem; margin-bottom: 1rem;
}
.title-card .label {font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; opacity: .65; font-weight: 700;}
.title-card .value {font-size: 1.45rem; font-weight: 800; margin-top: .15rem;}
.feature {
    border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
    padding: 1rem 1.1rem; background: rgba(128,128,128,.06); height: 100%;
}
.feature .icon {font-size: 1.5rem;}
.feature .name {font-weight: 700; margin: .3rem 0 .15rem;}
.feature .desc {font-size: .85rem; opacity: .75;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
    padding: .7rem .9rem; background: rgba(128,128,128,.05);
}
div[data-testid="stTabs"] button {font-weight: 600;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

LANGUAGES = {
    "English": "english",
    "Hinglish (Hindi speech to English text)": "hinglish",
    "Hindi (Hindi text)": "hindi",
}
MEDIA_TYPES = ["mp4", "mkv", "mov", "avi", "webm", "mp3", "wav", "m4a", "aac", "flac", "ogg"]
SUGGESTIONS = [
    "What were the main topics discussed?",
    "Who is responsible for what?",
    "Were any decisions made?",
    "What questions are still open?",
]

# ---------- Session state ----------
st.session_state.setdefault("result", None)
st.session_state.setdefault("meta", {})
st.session_state.setdefault("chat", [])
st.session_state.setdefault("pending_question", None)


class LiveLog(io.StringIO):
    """Capture print() output from the pipeline and show the latest lines live in the UI."""

    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder

    def write(self, text):
        super().write(text)
        lines = self.getvalue().strip().splitlines()[-14:]
        self.placeholder.code("\n".join(lines), language="text")
        return len(text)


def save_upload(uploaded_file) -> str:
    """Save the uploaded file to a temp folder and return its path for process_input()."""
    folder = tempfile.mkdtemp(prefix="video_assistant_")
    path = os.path.join(folder, uploaded_file.name)
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def build_report(result: dict) -> str:
    return (
        f"# {result['title']}\n\n"
        f"## Summary\n{result['summary']}\n\n"
        f"## Action Items\n{result['action_items']}\n\n"
        f"## Key Decisions\n{result['key_decisions']}\n\n"
        f"## Open Questions\n{result['open_questions']}\n"
    )


def queue_question(question: str):
    st.session_state["pending_question"] = question


def reset_session():
    st.session_state["result"] = None
    st.session_state["meta"] = {}
    st.session_state["chat"] = []


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 🎬 AI Video Assistant")
    st.caption("Transcribe, summarize and chat with any video or audio file.")
    st.divider()

    mode = st.radio("Input source", ["Upload a file", "YouTube link"])
    uploaded, url = None, ""

    if mode == "Upload a file":
        uploaded = st.file_uploader("Video or audio file", type=MEDIA_TYPES)
        ready = uploaded is not None
    else:
        url = st.text_input("YouTube URL", placeholder="https://youtu.be/...")
        st.caption("YouTube downloads can get blocked (HTTP 429 or bot check). Uploading a file is more reliable.")
        ready = url.strip().startswith(("http://", "https://"))

    language_label = st.selectbox("Spoken language", list(LANGUAGES))
    run_clicked = st.button("🚀 Analyze", type="primary", use_container_width=True, disabled=not ready)

    if st.session_state["result"] is not None:
        st.button("🔄 New analysis", use_container_width=True, on_click=reset_session)

# ---------- Run pipeline ----------
if run_clicked:
    source = save_upload(uploaded) if mode == "Upload a file" else url.strip()
    language = LANGUAGES[language_label]
    started = time.time()
    st.session_state["chat"] = []

    try:
        with st.status("Analyzing... this can take a few minutes on CPU.", expanded=True) as status:
            log_box = st.empty()
            with contextlib.redirect_stdout(LiveLog(log_box)):
                result = run_pipeline(source, language)
            status.update(label="Analysis complete", state="complete", expanded=False)

        st.session_state["result"] = result
        st.session_state["meta"] = {
            "elapsed": int(time.time() - started),
            "language": language_label.split(" (")[0],
            "source": uploaded.name if mode == "Upload a file" else "YouTube",
        }
        st.toast("Analysis ready!", icon="✅")
    except Exception as e:
        st.session_state["result"] = None
        st.error(f"Pipeline failed: {e}")
        if "429" in str(e) or "Sign in to confirm" in str(e):
            st.info("YouTube blocked the download. Try uploading the video file instead.")
        with st.expander("Error details (traceback)"):
            st.code(traceback.format_exc(), language="text")

# ---------- Empty state ----------
result = st.session_state["result"]

if result is None:
    st.markdown(
        """
<div class="hero">
    <h1>AI Video Assistant</h1>
    <p>Drop in a video or audio file. Get a transcript, summary, action items, decisions, open questions, and a chat that answers from the transcript.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("🎙️", "Transcribe", "Speech to text with Whisper, in English or Hindi"),
        ("📋", "Summarize", "A title and a clean bullet summary"),
        ("✅", "Extract", "Action items, decisions and open questions"),
        ("💬", "Chat", "Ask questions, answered only from the transcript"),
    ]
    for col, (icon, name, desc) in zip((c1, c2, c3, c4), features):
        col.markdown(
            f'<div class="feature"><div class="icon">{icon}</div>'
            f'<div class="name">{name}</div><div class="desc">{desc}</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    st.info("Choose a file or paste a link in the sidebar, then press **Analyze**.")
    st.stop()

# ---------- Results ----------
meta = st.session_state["meta"]
title = str(result["title"]).strip().strip('"').strip("'")

st.markdown(
    f'<div class="title-card"><div class="label">Title</div><div class="value">{title}</div></div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Words transcribed", f"{len(result['transcript'].split()):,}")
m2.metric("Processing time", f"{meta.get('elapsed', 0)}s")
m3.metric("Language", meta.get("language", "-"))
m4.metric("Source", meta.get("source", "-"))
st.write("")

left, right = st.columns([3, 2], gap="large")

with left:
    tab_sum, tab_act, tab_dec, tab_q, tab_tr = st.tabs(
        ["📋 Summary", "✅ Action items", "🔑 Decisions", "❓ Open questions", "📝 Transcript"]
    )
    with tab_sum:
        with st.container(border=True):
            st.markdown(result["summary"])
    with tab_act:
        with st.container(border=True):
            st.markdown(result["action_items"])
    with tab_dec:
        with st.container(border=True):
            st.markdown(result["key_decisions"])
    with tab_q:
        with st.container(border=True):
            st.markdown(result["open_questions"])
    with tab_tr:
        st.text_area("Full transcript", result["transcript"], height=420, disabled=True, label_visibility="collapsed")

    d1, d2 = st.columns(2)
    d1.download_button(
        "⬇️ Download report (.md)",
        data=build_report(result),
        file_name="video_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    d2.download_button(
        "⬇️ Download transcript (.txt)",
        data=result["transcript"],
        file_name="transcript.txt",
        mime="text/plain",
        use_container_width=True,
    )

with right:
    st.markdown("### 💬 Chat with your video")
    st.caption("Answers come only from the transcript.")

    history_box = st.container(height=440, border=True)

    s1, s2 = st.columns(2)
    for i, suggestion in enumerate(SUGGESTIONS):
        (s1 if i % 2 == 0 else s2).button(
            suggestion, key=f"sg_{i}", on_click=queue_question, args=(suggestion,), use_container_width=True
        )

    typed = st.chat_input("Ask something about the video...")
    question = typed or st.session_state.pop("pending_question", None)

    if question:
        st.session_state["chat"].append({"role": "user", "content": question})
        try:
            with st.spinner("Thinking..."):
                answer = ask_question(result["rag_chain"], question)
        except Exception as e:
            answer = f"Sorry, something went wrong: {e}"
        st.session_state["chat"].append({"role": "assistant", "content": answer})

    with history_box:
        if not st.session_state["chat"]:
            st.caption("No messages yet. Pick a suggestion or type your own question.")
        for message in st.session_state["chat"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])