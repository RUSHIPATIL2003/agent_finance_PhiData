# Multimodal RAG Chatbot 

A production-ready, scalable, and memory-efficient Multimodal Retrieval-Augmented Generation (RAG) chatbot built with Python, Streamlit, PostgreSQL (pgvector), LangChain, and Hugging Face.

## Features

- **Multimodal Document Processing**: Extracts text, tables, images (with OCR), formulas, and structured layouts from PDFs
- **Semantic Chunking**: Intelligent chunking with configurable size and overlap, preserving document structure
- **Vector Storage**: PostgreSQL with pgvector for efficient similarity search (HNSW/IVFFlat indexes)
- **Deduplication**: Document hashing prevents re-processing of unchanged documents
- **Incremental Indexing**: Only processes new or modified documents
- **Flexible Retrieval**: Vector search, hybrid search (vector + keyword), metadata-aware filtering
- **Conversational Memory**: Session-based chat history with context awareness
- **Source Citations**: Page numbers, section titles, and similarity scores for every response
- **Clean Architecture**: Modular design with swappable components (LLMs, embeddings, retrievers, vector stores)
- **Configuration Management**: Centralized `.env` configuration with validation
- **Production Ready**: Connection pooling, caching, comprehensive logging, error handling

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│   LangChain      │────▶│   PostgreSQL    │
│   Frontend      │     │   RAG Chain      │     │   + pgvector    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              ┌──────────┐ ┌─────────┐ ┌──────────┐
              │Embeddings│ │Retriever│ │ Prompts  │
              └──────────┘ └─────────┘ └──────────┘
```

## Project Structure

```
rag_agent/
├── app/
│   ├── config/           # Configuration management
│   ├── database/         # Database connection pool & operations
│   ├── ingestion/        # Multimodal PDF ingestion pipeline
│   ├── embeddings/       # Hugging Face embedding models
│   ├── retrievers/       # Custom retrievers (vector, hybrid, metadata)
│   ├── prompts/          # Prompt templates
│   ├── chains/           # LangChain RAG chains
│   ├── models/           # Data models
│   ├── utils/            # Utilities (logging, config, helpers)
│   └── streamlit_app.py  # Streamlit frontend
├── sql/
│   └── init_database.sql # Database initialization script
├── Data/
│   └── sample.pdf        # Document to process
├── .env.example          # Environment template
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## Prerequisites

- **Python 3.14** (or 3.11+)
- **PostgreSQL 15+** with **pgvector 0.5+** extension
- **Hugging Face Account** (for embedding models)
- **OpenAI API Key** (or compatible LLM provider)

## Installation

### 1. Clone and Setup Environment

```bash
cd D:\Projects\rag_agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 2. PostgreSQL Setup

#### Windows (using Chocolatey)
```powershell
choco install postgresql
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

#### macOS (using Homebrew)
```bash
brew install postgresql
brew services start postgresql
```

### 3. Install pgvector Extension

#### From Source (Recommended for latest version)
```bash
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
make install  # May require sudo
```

#### Using Package Manager
```bash
# Ubuntu/Debian
sudo apt install postgresql-16-pgvector

# macOS (Homebrew)
brew install pgvector
```

### 4. Initialize Database

```bash
# Connect to PostgreSQL as superuser
psql -U postgres

# Run initialization script
\i sql/init_database.sql

# Or from command line
psql -U postgres -f sql/init_database.sql
```

### 5. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your values
# Required: POSTGRES_PASSWORD, HF_API_TOKEN, LLM_API_KEY
```

**Example `.env` configuration:**
```env
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_agent
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password

# Hugging Face
HF_API_TOKEN=hf_your_token_here
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# LLM (OpenAI compatible)
LLM_API_KEY=sk-your-openai-key
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-3.5-turbo
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048

# Document Processing
CHUNK_SIZE=512
CHUNK_OVERLAP=50
DOCUMENT_PATH=Data/sample.pdf

# Retrieval
RETRIEVAL_TOP_K=5
RETRIEVAL_SIMILARITY_THRESHOLD=0.7
ENABLE_RERANKING=false

# Application
APP_SECRET_KEY=your-secret-key-change-in-production
LOG_LEVEL=INFO
CACHE_ENABLED=true
```

### 6. Place Your Document

Place your PDF document at `Data/sample.pdf` (or update `DOCUMENT_PATH` in `.env`).

## Usage

### 1. Ingest Document (First Run)

```bash
# Run ingestion manually
python -m app.ingestion

# Or via Streamlit UI (see below)
```

### 2. Start the Chatbot

```bash
streamlit run app/streamlit_app.py
```

The application will be available at `http://localhost:8501`

### 3. Using the Interface

1. **Ask Questions**: Type questions in the chat input
2. **View Sources**: Expand "Sources" to see page numbers, sections, and similarity scores
3. **Configure Retrieval**: Use sidebar to adjust:
   - Retriever type (vector/hybrid/metadata-aware)
   - Top-K results
   - Similarity threshold
   - Hybrid search toggle
4. **Re-index Document**: Click "Re-index Document" in sidebar after updating the PDF
5. **Clear History**: Click "Clear Chat History" to start fresh session

## API Reference

### Ingestion Pipeline

```python
from app.ingestion import ingest_document, reindex_document

# Ingest new document
result = ingest_document("Data/sample.pdf")
print(result)  # {'status': 'success', 'chunks_inserted': 42, ...}

# Force re-index
result = reindex_document("Data/sample.pdf")
```

### RAG Chain

```python
from app.chains import create_rag_chain

# Create chain
chain = create_rag_chain(
    retriever_type="hybrid",
    top_k=10,
    similarity_threshold=0.7
)

# Ask question
result = chain.invoke("What is the main topic of the document?")
print(result["answer"])
print(result["sources"])
```

### Custom Retriever

```python
from app.retrievers import create_retriever

# Vector search only
retriever = create_retriever("vector", top_k=5, similarity_threshold=0.7)

# Hybrid search
retriever = create_retriever("hybrid", top_k=5, vector_weight=0.7, keyword_weight=0.3)

# Metadata-aware
retriever = create_retriever(
    "metadata_aware",
    content_types=["table", "text"],
    page_range=(1, 10)
)

# Retrieve
docs = retriever.invoke("machine learning")
```

### Custom Embeddings

```python
from app.embeddings import create_embeddings_instance

# Use different model
embeddings = create_embeddings_instance(
    model_name="sentence-transformers/all-mpnet-base-v2"
)
```

### Custom LLM

```python
from app.chains import create_llm

# OpenAI (default)
llm = create_llm("openai", model="gpt-4", temperature=0)

# Anthropic
llm = create_llm("anthropic", model="claude-3-haiku-20240307")
```

## Configuration Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_HOST` | PostgreSQL host | `localhost` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | Database name | `rag_agent` |
| `POSTGRES_USER` | Database user | `postgres` |
| `POSTGRES_PASSWORD` | Database password | *required* |
| `HF_API_TOKEN` | Hugging Face token | *optional* |
| `EMBEDDING_MODEL_NAME` | Embedding model | `all-MiniLM-L6-v2` |
| `LLM_API_KEY` | LLM API key | *required* |
| `LLM_BASE_URL` | LLM API base URL | `https://api.openai.com/v1` |
| `LLM_MODEL_NAME` | LLM model name | `gpt-3.5-turbo` |
| `LLM_TEMPERATURE` | LLM temperature | `0.1` |
| `LLM_MAX_TOKENS` | LLM max tokens | `2048` |
| `CHUNK_SIZE` | Text chunk size | `512` |
| `CHUNK_OVERLAP` | Chunk overlap | `50` |
| `DOCUMENT_PATH` | Path to PDF | `Data/sample.pdf` |
| `RETRIEVAL_TOP_K` | Top-K retrieval | `5` |
| `RETRIEVAL_SIMILARITY_THRESHOLD` | Similarity threshold | `0.7` |
| `ENABLE_RERANKING` | Enable reranking | `false` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `CACHE_ENABLED` | Enable embedding cache | `true` |

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
# Windows
sc query postgresql-x64-16

# Linux
sudo systemctl status postgresql

# macOS
brew services list | grep postgresql
```

### pgvector Not Found

```sql
-- In psql, check extension
SELECT * FROM pg_extension WHERE extname = 'vector';

-- If missing, create it
CREATE EXTENSION IF NOT EXISTS vector;
```

### Import Errors

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check Python version
python --version  # Should be 3.11+
```

### Memory Issues

- Reduce `CHUNK_SIZE` in `.env`
- Reduce `DB_POOL_MAX_CONN`
- Use smaller embedding model (e.g., `all-MiniLM-L6-v2` is 384-dim)

### OCR Not Working

```bash
# Install Tesseract OCR
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
# Linux: sudo apt install tesseract-ocr
# macOS: brew install tesseract
```

## Performance Optimization

1. **Use HNSW Index** (default in init script) for faster searches
2. **Enable Caching**: `CACHE_ENABLED=true` caches embeddings
3. **Connection Pooling**: Adjust `DB_POOL_MIN_CONN`/`DB_POOL_MAX_CONN`
4. **Batch Processing**: Ingestion uses batch inserts
5. **Lazy Loading**: Components loaded on demand

## Extending the System

### Add New Content Type

1. Add extractor in `app/ingestion/__init__.py`
2. Add content type to `DocumentChunk.content_type`
3. Add specialized prompt in `app/prompts/__init__.py`
4. Update retriever filters if needed

### Add New Vector Store

1. Implement `BaseRetriever` in `app/retrievers/__init__.py`
2. Add factory method in `create_retriever()`
3. Update chain to use new retriever

### Add New LLM Provider

1. Add case in `create_llm()` in `app/chains/__init__.py`
2. Install required LangChain integration package

## License

MIT License - Feel free to use and modify for your projects.

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## Support

For issues and questions, please check the troubleshooting section or create an issue in the repository.
