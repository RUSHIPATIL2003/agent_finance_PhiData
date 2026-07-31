"""LangChain RAG chains and pipeline."""

import logging
import uuid
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.database import get_conversation_history, save_conversation
from app.embeddings import get_embedding_model
from app.prompts import (
    CHAT_RAG_PROMPT,
    format_context,
    format_sources,
    get_rag_prompt,
)
from app.retrievers import create_retriever

logger = logging.getLogger(__name__)


class RAGChain:
    """Main RAG chain with conversational memory."""
    
    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        retriever_type: str = "vector",
        session_id: Optional[str] = None,
        **retriever_kwargs
    ):
        self.settings = get_settings()
        self.session_id = session_id or str(uuid.uuid4())
        self.llm = llm or self._create_default_llm()
        self.retriever = create_retriever(retriever_type, **retriever_kwargs)
        self.chain = self._build_chain()
    
    def _create_default_llm(self) -> BaseChatModel:
        """Create default LLM from settings."""
        return ChatOpenAI(
            api_key=self.settings.llm_api_key,
            base_url=self.settings.llm_base_url,
            model=self.settings.llm_model_name,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )
    
    def _build_chain(self) -> Runnable:
        """Build the RAG chain."""
        prompt = get_rag_prompt(include_history=True)
        
        def retrieve_and_format(inputs: dict[str, Any]) -> dict[str, Any]:
            question = inputs["question"]
            docs = self.retriever.invoke(question)
            context = format_context([
                {
                    "content": d.page_content,
                    "page_number": d.metadata.get("page_number"),
                    "section_title": d.metadata.get("section_title"),
                    "content_type": d.metadata.get("content_type"),
                    "similarity": d.metadata.get("similarity"),
                    "chunk_id": d.metadata.get("chunk_id"),
                    "metadata": d.metadata,
                }
                for d in docs
            ])
            sources = format_sources([
                {
                    "page_number": d.metadata.get("page_number"),
                    "section_title": d.metadata.get("section_title"),
                    "content_type": d.metadata.get("content_type"),
                    "similarity": d.metadata.get("similarity"),
                    "chunk_id": d.metadata.get("chunk_id"),
                }
                for d in docs
            ])
            return {
                "context": context,
                "question": question,
                "sources": sources,
                "retrieved_docs": docs,
            }
        
        chain = (
            RunnablePassthrough.assign(
                context_and_sources=RunnableLambda(retrieve_and_format)
            )
            | RunnableLambda(lambda x: {
                "context": x["context_and_sources"]["context"],
                "question": x["context_and_sources"]["question"],
                "chat_history": x.get("chat_history", []),
            })
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    def invoke(self, question: str) -> dict[str, Any]:
        """Invoke the RAG chain with a question."""
        # Get chat history
        history = get_conversation_history(self.session_id, limit=10)
        chat_history = []
        for h in reversed(history):
            chat_history.append(("human", h["user_message"]))
            chat_history.append(("ai", h["assistant_message"]))
        
        # Run chain
        result = self.chain.invoke({
            "question": question,
            "chat_history": chat_history,
        })
        
        # Extract sources from the chain's internal state
        # We need to run retrieval again to get sources
        docs = self.retriever.invoke(question)
        sources = format_sources([
            {
                "page_number": d.metadata.get("page_number"),
                "section_title": d.metadata.get("section_title"),
                "content_type": d.metadata.get("content_type"),
                "similarity": d.metadata.get("similarity"),
                "chunk_id": d.metadata.get("chunk_id"),
            }
            for d in docs
        ])
        
        # Save conversation
        retrieved_chunk_ids = [d.metadata.get("chunk_id") for d in docs]
        save_conversation(
            session_id=self.session_id,
            user_message=question,
            assistant_message=result,
            retrieved_chunks=retrieved_chunk_ids,
            metadata={"model": self.settings.llm_model_name}
        )
        
        return {
            "answer": result,
            "sources": sources,
            "session_id": self.session_id,
            "retrieved_chunks": len(docs),
        }
    
    async def ainvoke(self, question: str) -> dict[str, Any]:
        """Async invoke the RAG chain."""
        # For now, delegate to sync
        return self.invoke(question)
    
    def stream(self, question: str):
        """Stream the response."""
        history = get_conversation_history(self.session_id, limit=10)
        chat_history = []
        for h in reversed(history):
            chat_history.append(("human", h["user_message"]))
            chat_history.append(("ai", h["assistant_message"]))
        
        # Stream the chain
        for chunk in self.chain.stream({
            "question": question,
            "chat_history": chat_history,
        }):
            yield chunk
    
    def get_history(self) -> list[dict[str, Any]]:
        """Get conversation history."""
        return get_conversation_history(self.session_id)
    
    def clear_history(self) -> None:
        """Clear conversation history (start new session)."""
        self.session_id = str(uuid.uuid4())


def create_rag_chain(
    llm: Optional[BaseChatModel] = None,
    retriever_type: str = "vector",
    session_id: Optional[str] = None,
    **kwargs
) -> RAGChain:
    """Factory function to create RAG chains."""
    return RAGChain(
        llm=llm,
        retriever_type=retriever_type,
        session_id=session_id,
        **kwargs
    )


def create_llm(
    provider: str = "openai",
    **kwargs
) -> BaseChatModel:
    """Factory function to create LLM instances (for swapping providers)."""
    settings = get_settings()
    
    if provider == "openai":
        return ChatOpenAI(
            api_key=kwargs.get("api_key", settings.llm_api_key),
            base_url=kwargs.get("base_url", settings.llm_base_url),
            model=kwargs.get("model", settings.llm_model_name),
            temperature=kwargs.get("temperature", settings.llm_temperature),
            max_tokens=kwargs.get("max_tokens", settings.llm_max_tokens),
        )
    elif provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                api_key=kwargs.get("api_key"),
                model=kwargs.get("model", "claude-3-haiku-20240307"),
                temperature=kwargs.get("temperature", settings.llm_temperature),
                max_tokens=kwargs.get("max_tokens", settings.llm_max_tokens),
            )
        except ImportError:
            raise ImportError("langchain-anthropic not installed")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")