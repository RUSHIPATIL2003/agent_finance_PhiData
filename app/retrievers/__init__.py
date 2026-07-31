"""Retriever implementations for the RAG pipeline."""

import logging
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.config import get_settings
from app.database import hybrid_search, similarity_search
from app.embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class PGVectorRetriever(BaseRetriever):
    """Custom retriever using PostgreSQL with pgvector."""
    
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity score")
    document_id: Optional[str] = Field(default=None, description="Filter by document ID")
    use_hybrid: bool = Field(default=False, description="Use hybrid search (vector + keyword)")
    vector_weight: float = Field(default=0.7, description="Weight for vector search in hybrid")
    keyword_weight: float = Field(default=0.3, description="Weight for keyword search in hybrid")
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Retrieve relevant documents for a query."""
        settings = get_settings()
        embedding_model = get_embedding_model()
        
        # Generate query embedding
        query_embedding = embedding_model.embed_query(query)
        
        # Perform search
        if self.use_hybrid:
            results = hybrid_search(
                query_embedding=query_embedding,
                query_text=query,
                match_threshold=self.similarity_threshold,
                match_count=self.top_k,
                document_id=self.document_id,
                vector_weight=self.vector_weight,
                keyword_weight=self.keyword_weight,
            )
        else:
            results = similarity_search(
                query_embedding=query_embedding,
                match_threshold=self.similarity_threshold,
                match_count=self.top_k,
                document_id=self.document_id,
            )
        
        # Convert to LangChain Documents
        documents = []
        for row in results:
            doc = Document(
                page_content=row["content"],
                metadata={
                    "chunk_id": str(row["chunk_id"]),
                    "document_id": str(row["document_id"]),
                    "page_number": row["page_number"],
                    "section_title": row["section_title"],
                    "content_type": row["content_type"],
                    "heading_hierarchy": row["heading_hierarchy"],
                    "similarity": row["similarity"],
                    "bbox": row.get("bbox"),
                    **row.get("metadata", {}),
                }
            )
            documents.append(doc)
        
        logger.debug("Retrieved %d documents for query: %s", len(documents), query[:50])
        return documents
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Async version of retrieve (delegates to sync for now)."""
        return self._get_relevant_documents(query, run_manager=run_manager)


class MetadataAwareRetriever(PGVectorRetriever):
    """Retriever that supports metadata filtering."""
    
    content_types: Optional[list[str]] = Field(default=None, description="Filter by content types")
    page_range: Optional[tuple[int, int]] = Field(default=None, description="Filter by page range")
    section_titles: Optional[list[str]] = Field(default=None, description="Filter by section titles")
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Retrieve with metadata filtering."""
        # For now, use base retriever and filter post-retrieval
        # In production, this would push filters to SQL
        documents = super()._get_relevant_documents(query, run_manager=run_manager)
        
        # Apply metadata filters
        if self.content_types:
            documents = [d for d in documents if d.metadata.get("content_type") in self.content_types]
        
        if self.page_range:
            start, end = self.page_range
            documents = [
                d for d in documents 
                if d.metadata.get("page_number") and start <= d.metadata["page_number"] <= end
            ]
        
        if self.section_titles:
            documents = [
                d for d in documents 
                if d.metadata.get("section_title") in self.section_titles
            ]
        
        return documents


def create_retriever(
    retriever_type: str = "vector",
    **kwargs
) -> BaseRetriever:
    """Factory function to create retrievers (for swapping implementations)."""
    settings = get_settings()
    
    default_kwargs = {
        "top_k": settings.retrieval_top_k,
        "similarity_threshold": settings.retrieval_similarity_threshold,
        "use_hybrid": False,
    }
    default_kwargs.update(kwargs)
    
    if retriever_type == "vector":
        return PGVectorRetriever(**default_kwargs)
    elif retriever_type == "hybrid":
        default_kwargs["use_hybrid"] = True
        return PGVectorRetriever(**default_kwargs)
    elif retriever_type == "metadata_aware":
        return MetadataAwareRetriever(**default_kwargs)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


def create_langchain_retriever(
    vectorstore: Any,  # LangChain vectorstore interface
    **kwargs
) -> BaseRetriever:
    """Create a LangChain-compatible retriever from a vectorstore."""
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": kwargs.get("top_k", settings.retrieval_top_k),
            "score_threshold": kwargs.get("similarity_threshold", settings.retrieval_similarity_threshold),
        }
    )