# AGENTS.md - Multimodal RAG Chatbot

## Quick Commands

```bash
# Setup (first time)
python -m venv .venv            # README uses `venv`; both names work (both gitignored)
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Verify Python version (3.14 validated; existing venv uses 3.14.6)
python --version

# NOTE: the committed `venv/` is Windows-style (uses `Scripts/`, not `bin/`).
# On WSL/Linux, recreate it: rm -rf venv && python -m venv venv && source venv/bin/activate

# Install pgvector extension (required before db init)
# Windows: build from source or use package manager
# Linux: sudo apt install postgresql-16-pgvector
# macOS: brew install pgvector

# Initialize database (run as postgres superuser)
psql -U postgres -f sql/init_database.sql

# Configure environment
cp .env.example .env
# Edit .env with: POSTGRES_PASSWORD, HF_API_TOKEN, LLM_API_KEY

# Full initialization (db + ingestion) - alternative to separate steps
python main.py

# Ingest document only (run once, or after PDF changes)
python -m app.ingestion

# Start app
streamlit run app/streamlit_app.py
# → http://localhost:8501
```

## Architecture Essentials

| Component | Module | Key Fact |
|-----------|--------|----------|
| Config | `app/config/__init__.py` | Pydantic Settings, loads `.env`, validates all vars |
| Database | `app/database/__init__.py` | psycopg pool (min=2, max=10), pgvector HNSW index |
| Ingestion | `app/ingestion/__init__.py` | PyMuPDF + OCR, semantic chunking, SHA256 dedup |
| Embeddings | `app/embeddings/__init__.py` | Cached HF embeddings (sentence-transformers) |
| Retrievers | `app/retrievers/__init__.py` | Vector / Hybrid / Metadata-aware (factory) |
| Chains | `app/chains/__init__.py` | LangChain RAG chain with session memory |
| Prompts | `app/prompts/__init__.py` | Templates for RAG, chat, table/image/formula QA |

**Swappable by design**: `create_retriever()`, `create_embeddings_instance()`, `create_llm()`, `create_rag_chain()` factory functions.

## Critical Config (`.env`)

Required (no defaults):
- `POSTGRES_PASSWORD`
- `HF_API_TOKEN` (for private models)
- `LLM_API_KEY` (OpenAI or compatible)

Optional with defaults:
- `EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2`
- `LLM_MODEL_NAME=gpt-3.5-turbo`
- `CHUNK_SIZE=512`, `CHUNK_OVERLAP=50`
- `RETRIEVAL_TOP_K=5`, `RETRIEVAL_SIMILARITY_THRESHOLD=0.7`
- `DB_POOL_MIN_CONN=2`, `DB_POOL_MAX_CONN=10`, `DB_POOL_TIMEOUT=30`
- `CACHE_ENABLED=true`, `CACHE_TTL_SECONDS=3600`
- `ENABLE_RERANKING=false`, `RERANKER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2`
- `LOG_LEVEL=INFO`, `APP_SECRET_KEY=change-me-in-production`
- `STREAMLIT_SERVER_PORT=8501`, `STREAMLIT_SERVER_ADDRESS=0.0.0.0`
- `DOCUMENT_PATH=Data/sample.pdf`

**Note**: `LLM_BASE_URL` defaults to `https://api.openai.com/v1` but `.env.example` shows Groq (`https://api.groq.com/openai/v1`). Adjust for your provider.

## Dependency Management

- **Source of truth**: `requirements.in` (direct deps only)
- **Lock file**: `requirements.txt` (generated via `pip-compile`)
- **Workflow**: Edit `requirements.in` → `pip-compile requirements.in` → commit both
- Python 3.14 validated; numpy 2.x, torch 2.13, transformers 4.48

## Development Gotchas

1. **Virtual env**: both `venv/` and `.venv/` are gitignored, so either name works. README uses `venv`; the committed `venv/` here is Windows-style (`Scripts/`), not Linux (`bin/`) — recreate on WSL/Linux.
2. **OCR requires system tesseract**: `brew install tesseract` / `apt install tesseract-ocr`
3. **pgvector extension must exist** before `init_database.sql` runs
4. **Embedding dimension is 384** (hardcoded in `sql/init_database.sql` for `all-MiniLM-L6-v2`) — change SQL if using different model
5. **Document path** is `Data/sample.pdf` by default — update `DOCUMENT_PATH` in `.env` if different
6. **Ingestion is idempotent**: SHA256 hash prevents re-embedding unchanged PDFs
7. **Re-index**: Use `python -m app.ingestion` again, or click "Re-index Document" in Streamlit sidebar
8. **main.py** runs full init (db + ingest) — useful for CI/automation
9. **`create_llm("anthropic")` is wired but not installable** by default — `langchain-anthropic` is not in `requirements.in`; `RAGChain` itself uses `_create_default_llm()` (OpenAI-compatible only), not `create_llm()`

## Testing

No test suite currently exists. If adding:
- Unit: `tests/unit/test_<module>.py`
- Integration: `tests/integration/test_rag_pipeline.py` (needs running Postgres)
- Fixtures: `tests/fixtures/sample_documents/` (small PDFs only)

## File Ownership

- `app/config/*` — Configuration, env loading
- `app/database/*` — Pool, CRUD, SQL functions
- `app/ingestion/*` — PDF extraction, chunking, OCR
- `app/embeddings/*` — HF embeddings, caching
- `app/retrievers/*` — Vector/hybrid/metadata search
- `app/chains/*` — LangChain RAG, memory, LLM factory
- `app/prompts/*` — All prompt templates
- `app/models/*` — Dataclasses (Document, Chunk, RAGResponse, IngestionResult)
- `app/utils/*` — Logging, helpers, config utils
- `sql/init_database.sql` — Schema, indexes, functions

## Extending

| Goal | Where |
|------|-------|
| New content extractor | `app/ingestion/__init__.py` → add to `process_pdf()` |
| New retriever type | `app/retrievers/__init__.py` → add to `create_retriever()` |
| New LLM provider | `app/chains/__init__.py` → add to `create_llm()` |
| New embedding model | `app/embeddings/__init__.py` → `create_embeddings_instance()` |
| New prompt | `app/prompts/__init__.py` |

## Troubleshooting Quick Reference

| Issue | Fix |
|-------|-----|
| `pgvector` not found | `CREATE EXTENSION IF NOT EXISTS vector;` in psql |
| OCR fails | Install tesseract system package |
| Import errors | `pip install --force-reinstall -r requirements.txt` |
| Memory issues | Reduce `CHUNK_SIZE`, `DB_POOL_MAX_CONN` |
| DB connection | Check `POSTGRES_*` vars, pg running, pgvector installed |
| Wrong embedding dim | Update `VECTOR(384)` in `sql/init_database.sql` to match model |

## Do Not Commit

See `.gitignore` — notably: `.env`, `venv/`, `.venv/`, `__pycache__/`, `Data/*.pdf`, `extracted/`, `chunks/`, `embeddings/`, `pgdata/`, `.streamlit/secrets.toml`, `*.log`, model weights (`*.bin`, `*.safetensors`, `*.pt`).

**Note**: `.gitignore` line 24 also lists `AGENTS.md` — remove that line if you want this file tracked.