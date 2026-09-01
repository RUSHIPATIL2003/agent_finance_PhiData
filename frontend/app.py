"""High-Performance Financial AI Chatbot Streamlit Application."""

import os
import sys
import uuid
from pathlib import Path
import requests
import streamlit as st

# Ensure project root directory is always on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Page Configuration
st.set_page_config(
    page_title="Financial AI Agent",
    page_icon="💹",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Load secrets from Streamlit Cloud Secrets Manager into os.environ if present
for key in ["GOOGLE_API_KEY", "GROQ_API_KEY", "PHI_API_KEY", "OPENAI_API_KEY", "FASTAPI_BACKEND_URL"]:
    try:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

# Backend URL Configuration
BACKEND_URL = os.getenv("FASTAPI_BACKEND_URL", "http://127.0.0.1:8000")


def load_css(css_file_path: str):
    """Inject custom CSS styles into the Streamlit app."""
    if os.path.exists(css_file_path):
        with open(css_file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load CSS styles
css_path = os.path.join(os.path.dirname(__file__), "styles.css")
load_css(css_path)


@st.cache_data(ttl=15, show_spinner=False)
def check_backend_health() -> tuple[bool, str]:
    """Check whether the FastAPI backend service or local agent is healthy with minimal latency."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=0.8)
        if response.status_code == 200:
            data = response.json()
            model_name = data.get("model_id", "Active")
            return True, f"FastAPI Online ({model_name})"
    except Exception:
        pass

    # Direct in-process check
    try:
        from src.rag_agent.agent import create_model_instance
        _, model_info = create_model_instance()
        return True, f"High-Speed Engine ({model_info})"
    except Exception:
        return False, "Agent Offline (Set API Key)"


# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"

# Sidebar - Minimal Controls
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    is_healthy, health_msg = check_backend_health()
    if is_healthy:
        st.markdown(
            f'<div class="status-pill status-online"><div class="status-dot"></div>Engine: {health_msg}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status-pill status-offline"><div class="status-dot"></div>Engine: {health_msg}</div>',
            unsafe_allow_html=True,
        )
        st.warning("Please configure your API keys in `.env` (e.g. `GOOGLE_API_KEY` or `GROQ_API_KEY`)")

    st.markdown("---")
    st.markdown("### 💬 Chat Management")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
        st.rerun()

    st.markdown("---")
    st.caption("Phase 1: High-Speed Financial Assistant")
    st.caption("Powered by PhiData, DuckDuckGo & YFinance")

# Main Header & Hero
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">⚡ Real-Time Market Intelligence</div>
        <h1 class="hero-title">Financial AI Assistant</h1>
        <p class="hero-subtitle">
            Autonomous multi-agent system for stock metrics, analyst consensus, company news, and market synthesis.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick Starter Prompt Chips (only visible when chat history is empty)
if not st.session_state.messages:
    st.markdown("##### 💡 Suggested Questions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "📈 Summarize NVDA analyst recommendations and latest news",
            key="p1",
            use_container_width=True,
        ):
            st.session_state.starter_prompt = (
                "Summarize analyst recommendations and share the latest news for Nvidia"
            )
        if st.button(
            "💰 Compare Apple (AAPL) and Microsoft (MSFT) fundamentals",
            key="p2",
            use_container_width=True,
        ):
            st.session_state.starter_prompt = (
                "Compare the fundamentals and financial metrics of Apple (AAPL) and Microsoft (MSFT)"
            )

    with col2:
        if st.button(
            "📊 Tesla (TSLA) current stock price and key ratios",
            key="p3",
            use_container_width=True,
        ):
            st.session_state.starter_prompt = (
                "Give me Tesla's (TSLA) current stock price, analyst consensus, and key financial ratios in tables."
            )
        if st.button(
            "📰 Latest major financial news in the semiconductor sector",
            key="p4",
            use_container_width=True,
        ):
            st.session_state.starter_prompt = (
                "What is the latest market news and analyst sentiment regarding the semiconductor sector?"
            )

# Render Existing Chat History
for message in st.session_state.messages:
    role = message["role"]
    avatar = "👤" if role == "user" else "💹"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])


def stream_agent_response(prompt: str):
    """Stream response tokens dynamically for immediate visual feedback."""
    try:
        from src.rag_agent.agent import run_financial_agent_stream
        for chunk in run_financial_agent_stream(prompt):
            yield chunk
    except Exception:
        # Fallback to standard execution if streaming encounters an error
        from src.rag_agent.agent import run_financial_agent
        full_text, _ = run_financial_agent(prompt)
        yield full_text


# Handle Starter Prompt Trigger
prompt_to_send = None
if "starter_prompt" in st.session_state and st.session_state.starter_prompt:
    prompt_to_send = st.session_state.starter_prompt
    st.session_state.starter_prompt = None

# Handle User Input from Chat Bar
user_input = st.chat_input("Ask about stock fundamentals, analyst ratings, financial news...")
if user_input:
    prompt_to_send = user_input

# Process Message Submission with Real-Time Streaming
if prompt_to_send:
    # 1. Append and render user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt_to_send})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_to_send)

    # 2. Stream assistant response in real-time
    with st.chat_message("assistant", avatar="💹"):
        response_generator = stream_agent_response(prompt_to_send)
        complete_response = st.write_stream(response_generator)

    # 3. Store full assistant response in history
    st.session_state.messages.append({"role": "assistant", "content": complete_response})
