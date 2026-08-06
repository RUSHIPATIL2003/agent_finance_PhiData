"""Streamlit frontend for the Multimodal RAG Chatbot."""

import os
import sys
import uuid
import logging
from typing import Any

# Ensure project root is on path when Streamlit changes cwd to app/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.config import get_settings
from app.chains import create_rag_chain
from app.ingestion import ingest_document, reindex_document
from app.utils.logging import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Multimodal RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .source-citation {
        font-size: 0.85rem;
        color: #666;
        background-color: #f5f5f5;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
    }
    .source-citation:hover {
        background-color: #eaeaea;
    }
    .retrieved-context {
        font-size: 0.8rem;
        background-color: #fafafa;
        border: 1px solid #eee;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin-top: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_rag_chain(session_id: str, retriever_type: str, **kwargs):
    """Get or create RAG chain (cached)."""
    return create_rag_chain(
        session_id=session_id,
        retriever_type=retriever_type,
        **kwargs
    )


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "retriever_type" not in st.session_state:
        st.session_state.retriever_type = "vector"
    if "retrieval_params" not in st.session_state:
        st.session_state.retrieval_params = {
            "top_k": 5,
            "similarity_threshold": 0.7,
            "use_hybrid": False,
        }


def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Retriever settings
        st.subheader("Retrieval Settings")
        retriever_type = st.selectbox(
            "Retriever Type",
            options=["vector", "hybrid", "metadata_aware"],
            index=["vector", "hybrid", "metadata_aware"].index(st.session_state.retriever_type),
            help="Vector: semantic search only. Hybrid: vector + keyword. Metadata-aware: with filters."
        )
        
        top_k = st.slider(
            "Top K Results",
            min_value=1,
            max_value=20,
            value=st.session_state.retrieval_params["top_k"],
            help="Number of chunks to retrieve"
        )
        
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.retrieval_params["similarity_threshold"],
            step=0.05,
            help="Minimum similarity score for retrieval"
        )
        
        use_hybrid = st.checkbox(
            "Enable Hybrid Search",
            value=st.session_state.retrieval_params["use_hybrid"],
            help="Combine vector and keyword search"
        )
        
        # Update settings if changed
        if (retriever_type != st.session_state.retriever_type or
            top_k != st.session_state.retrieval_params["top_k"] or
            similarity_threshold != st.session_state.retrieval_params["similarity_threshold"] or
            use_hybrid != st.session_state.retrieval_params["use_hybrid"]):
            
            st.session_state.retriever_type = retriever_type
            st.session_state.retrieval_params = {
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
                "use_hybrid": use_hybrid,
            }
            st.session_state.rag_chain = None  # Force recreation
            st.rerun()
        
        st.divider()
        
        # Document management
        st.subheader("📄 Document Management")
        settings = get_settings()
        
        if st.button("🔄 Re-index Document", type="secondary"):
            with st.spinner("Re-indexing document..."):
                try:
                    result = reindex_document(settings.document_path)
                    st.success(f"Re-indexed: {result['chunks_inserted']} chunks inserted")
                    st.session_state.rag_chain = None
                except Exception as e:
                    st.error(f"Re-indexing failed: {e}")
                    logger.error("Re-indexing failed: %s", e)
        
        if st.button("📥 Ingest Document (if new)", type="secondary"):
            with st.spinner("Ingesting document..."):
                try:
                    result = ingest_document(settings.document_path)
                    if result["status"] == "success":
                        st.success(f"Ingested: {result['chunks_inserted']} chunks")
                        st.session_state.rag_chain = None
                    else:
                        st.info(result["message"])
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
                    logger.error("Ingestion failed: %s", e)
        
        st.divider()
        
        # Session management
        st.subheader("💬 Session")
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.messages = []
            if st.session_state.rag_chain:
                st.session_state.rag_chain.clear_history()
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
        
        # Display current session info
        st.caption(f"Session ID: {st.session_state.session_id[:8]}...")
        
        st.divider()
        
        # System info
        st.subheader("ℹ️ System Info")
        st.caption(f"Embedding Model: {settings.embedding_model_name}")
        st.caption(f"LLM Model: {settings.llm_model_name}")
        st.caption(f"Chunk Size: {settings.chunk_size}")
        st.caption(f"Chunk Overlap: {settings.chunk_overlap}")


def render_chat_interface():
    """Render the main chat interface."""
    # Initialize RAG chain if needed
    if st.session_state.rag_chain is None:
        st.session_state.rag_chain = get_rag_chain(
            session_id=st.session_state.session_id,
            retriever_type=st.session_state.retriever_type,
            **st.session_state.retrieval_params
        )
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        page = source.get("page_number", "N/A")
                        section = source.get("section_title", "N/A")
                        ctype = source.get("content_type", "text")
                        sim = source.get("similarity", 0)
                        
                        st.markdown(f"""
                        <div class="source-citation">
                            <strong>Source {i}</strong> | Page: {page} | Section: {section} | Type: {ctype} | Similarity: {sim:.2f}
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.rag_chain.invoke(prompt)
                    
                    # Display answer
                    st.markdown(result["answer"])
                    
                    # Display sources
                    if result["sources"]:
                        with st.expander("📚 Sources", expanded=True):
                            for i, source in enumerate(result["sources"], 1):
                                page = source.get("page_number", "N/A")
                                section = source.get("section_title", "N/A")
                                ctype = source.get("content_type", "text")
                                sim = source.get("similarity", 0)
                                
                                st.markdown(f"""
                                <div class="source-citation">
                                    <strong>Source {i}</strong> | Page: {page} | Section: {section} | Type: {ctype} | Similarity: {sim:.2f}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")
                    logger.error("Response generation failed: %s", e)


def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.title("🤖 Multimodal RAG Chatbot")
    st.caption("Chat with your documents using Retrieval-Augmented Generation")
    
    # Render sidebar
    render_sidebar()
    
    # Render chat interface
    render_chat_interface()


if __name__ == "__main__":
    main()