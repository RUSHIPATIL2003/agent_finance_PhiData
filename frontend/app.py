"""Modern Financial AI Chatbot Streamlit Application."""

import os
import uuid
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Financial AI Agent",
    page_icon=" ",
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


def check_backend_health() -> tuple[bool, str]:
    """Check whether the FastAPI backend service is reachable and healthy."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            model_name = data.get("model_id", "Active")
            return True, f"FastAPI Online ({model_name})"
        return False, f"FastAPI Degraded ({response.status_code})"
    except Exception:
        # Fallback to local agent engine (Streamlit Cloud / Standalone mode)
        try:
            from src.rag_agent.agent import create_model_instance
            _, model_info = create_model_instance()
            return True, f"Cloud / Local Agent ({model_info})"
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
        st.warning("Please configure your API keys (e.g. `GOOGLE_API_KEY` or `GROQ_API_KEY`)")

    st.markdown("---")
    st.markdown("### 💬 Chat Management")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = f"sess_{uuid.uuid4().hex[:8]}"
        st.rerun()

    st.markdown("---")
    st.caption("Phase 1: Multi-Agent Financial Assistant")
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

# Render Chat History
for message in st.session_state.messages:
    role = message["role"]
    avatar = "👤" if role == "user" else "💹"
    with st.chat_message(role, avatar=avatar):
        st.markdown(message["content"])


# Query Routing (FastAPI API -> Local Agent Fallback)
def query_agent(prompt: str) -> str:
    """Send user prompt to FastAPI backend or fallback to local agent execution."""
    # 1. Try FastAPI Backend if running
    try:
        payload = {
            "message": prompt,
            "session_id": st.session_state.session_id,
        }
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response generated.")
    except Exception:
        pass

    # 2. Direct In-Process Execution (Streamlit Cloud & Standalone fallback)
    try:
        from src.rag_agent.agent import run_financial_agent
        agent_response, _ = run_financial_agent(prompt)
        return agent_response
    except Exception as e:
        return f"⚠️ **Agent Execution Error**: {str(e)}"


# Handle Starter Prompt Trigger
prompt_to_send = None
if "starter_prompt" in st.session_state and st.session_state.starter_prompt:
    prompt_to_send = st.session_state.starter_prompt
    st.session_state.starter_prompt = None

# Handle User Input from Chat Bar
user_input = st.chat_input("Ask about stock fundamentals, analyst ratings, financial news...")
if user_input:
    prompt_to_send = user_input

# Process Message Submission
if prompt_to_send:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt_to_send})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt_to_send)

    # Generate assistant response
    with st.chat_message("assistant", avatar="💹"):
        with st.spinner("🤖 Coordinating Web Search and Financial Analytics agents..."):
            agent_response = query_agent(prompt_to_send)
            st.markdown(agent_response)

    # Store assistant response in history
    st.session_state.messages.append({"role": "assistant", "content": agent_response})
    st.rerun()
