"""Prompt templates for the RAG pipeline."""

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)


# System prompt for RAG
RAG_SYSTEM_PROMPT = """You are a knowledgeable AI assistant that answers questions based strictly on the provided document context. 
Your responses should be accurate, well-structured, and grounded in the retrieved information.

Guidelines:
1. Only use information from the provided context to answer questions
2. If the context doesn't contain enough information, say so honestly
3. Cite sources using page numbers and section titles when available
4. Be concise but thorough
5. Don't make assumptions beyond what's in the context
6. If multiple sources conflict, mention this
7. Structure your response with clear sections when appropriate"""

# Main RAG prompt template
RAG_PROMPT_TEMPLATE = """Context from document:
{context}

Question: {question}

Answer the question based on the provided context. Include citations with page numbers and section titles where relevant."""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)

# Chat prompt with history
CHAT_RAG_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(RAG_SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template(RAG_PROMPT_TEMPLATE),
])

# Prompt for query rewriting (for better retrieval)
QUERY_REWRITE_PROMPT = PromptTemplate(
    template="""Given the following conversation history and a follow-up question, 
rewrite the follow-up question to be a standalone question that captures all relevant context.

Conversation history:
{chat_history}

Follow-up question: {question}

Standalone question:""",
    input_variables=["chat_history", "question"],
)

# Prompt for response formatting
RESPONSE_FORMAT_PROMPT = PromptTemplate(
    template="""Format the following answer in a clear, well-structured way:

Answer: {answer}

Sources: {sources}

Formatted response:""",
    input_variables=["answer", "sources"],
)

# Prompt for reranking (if enabled)
RERANK_PROMPT = PromptTemplate(
    template="""Given a query and a list of document chunks, rank the chunks by relevance to the query.

Query: {query}

Chunks:
{chunks}

Rank the chunks from most to least relevant (output only the chunk indices in order, comma-separated):""",
    input_variables=["query", "chunks"],
)

# Prompt for metadata-aware retrieval
METADATA_AWARE_PROMPT = PromptTemplate(
    template="""Given a query and available document metadata, determine the best retrieval strategy.

Query: {query}

Available metadata filters:
- Content types: {content_types}
- Page range: {page_range}
- Section titles: {section_titles}

Recommend retrieval parameters (top_k, filters, search_type):""",
    input_variables=["query", "content_types", "page_range", "section_titles"],
)

# Specialized prompts for different content types
TABLE_QA_PROMPT = PromptTemplate(
    template="""Answer the question based on the table data provided.

Table:
{table}

Question: {question}

Answer:""",
    input_variables=["table", "question"],
)

IMAGE_QA_PROMPT = PromptTemplate(
    template="""Answer the question based on the image description/OCR text provided.

Image description: {description}

Question: {question}

Answer:""",
    input_variables=["description", "question"],
)

FORMULA_QA_PROMPT = PromptTemplate(
    template="""Explain or answer questions about the mathematical formula provided.

Formula: {formula}

Question: {question}

Answer:""",
    input_variables=["formula", "question"],
)


def get_rag_prompt(include_history: bool = False) -> ChatPromptTemplate | PromptTemplate:
    """Get the appropriate RAG prompt template."""
    if include_history:
        return CHAT_RAG_PROMPT
    return RAG_PROMPT


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into context string."""
    if not chunks:
        return "No relevant context found."
    
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {})
        page = chunk.get("page_number", "Unknown")
        section = chunk.get("section_title", "Unknown")
        content_type = chunk.get("content_type", "text")
        
        header = f"[Source {i}: Page {page}, Section: {section}, Type: {content_type}]"
        context_parts.append(f"{header}\n{chunk.get('content', '')}")
    
    return "\n\n---\n\n".join(context_parts)


def format_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format sources for citation."""
    sources = []
    for chunk in chunks:
        sources.append({
            "page_number": chunk.get("page_number"),
            "section_title": chunk.get("section_title"),
            "content_type": chunk.get("content_type"),
            "similarity": chunk.get("similarity"),
            "chunk_id": chunk.get("chunk_id"),
        })
    return sources