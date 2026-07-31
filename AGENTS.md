# AGENTS.md - Multimodal RAG Chatbot

## Quick Commands

```bash
# Setup (first time)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Install pgvector extension (required before db init)
# Windows: build from source or use package manager
# Linux: sudo apt install postgresql-16-pgvector
# macOS: brew install pgvector

# Initialize database (run as postgres superuser)
psql -U postgres -f sql/init_database.sql

# Configure environment
cp .env.example .env
# Edit .env with: POSTGRES_PASSWORD, HF_API_TOKEN, LLM_API_KEY

# Ingest document (run once, or after PDF changes)
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

## Dependency Management

- **Source of truth**: `requirements.in` (direct deps only)
- **Lock file**: `requirements.txt` (generated via `pip-compile`)
- **Workflow**: Edit `requirements.in` → `pip-compile requirements.in` → commit both
- Python 3.14 validated; numpy 2.x, torch 2.13, transformers 4.48

## Development Gotchas

1. **Virtual env is `.venv`** (not `venv`) — already in `.gitignore`
2. **OCR requires system tesseract**: `brew install tesseract` / `apt install tesseract-ocr`
3. **pgvector extension must exist** before `init_database.sql` runs
4. **Document path** is `Data/sample.pdf` by default — update `DOCUMENT_PATH` in `.env` if different
5. **Ingestion is idempotent**: SHA256 hash prevents re-embedding unchanged PDFs
6. **Re-index**: Use `python -m app.ingestion` again, or click "Re-index Document" in Streamlit sidebar

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
- `app/models/*` — Dataclasses (Document, Chunk, Response)
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

## Do Not Commit

See `.gitignore` — notably: `.env`, `.venv/`, `__pycache__/`, `Data/*.pdf`, `extracted/`, `chunks/`, `embeddings/`, `pgdata/`, `.streamlit/secrets.toml`, `*.log`, model weights (`*.bin`, `*.safetensors`, `*.pt`).