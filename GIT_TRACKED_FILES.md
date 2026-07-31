# Git-Tracked File List - Multimodal RAG Chatbot
# =============================================================================
# This document lists all files that SHOULD be committed to the repository
# and files that MUST remain local. Use this as a reference for code reviews
# and onboarding new team members.
# =============================================================================

## ✅ FILES TO COMMIT (Tracked in Git)
### Core Application Source Code
```
app/
├── __init__.py
├── config/
│   ├── __init__.py          # Pydantic settings management
├── database/
│   ├── __init__.py          # Connection pool & CRUD operations
├── ingestion/
│   ├── __init__.py          # Multimodal PDF processing pipeline
├── embeddings/
│   ├── __init__.py          # HuggingFace embeddings with caching
├── retrievers/
│   ├── __init__.py          # Vector, Hybrid, Metadata-aware retrievers
├── prompts/
│   ├── __init__.py          # RAG prompt templates
├── chains/
│   ├── __init__.py          # LangChain RAG chain with memory
├── models/
│   ├── __init__.py          # Data models (Document, Chunk, etc.)
├── utils/
│   ├── __init__.py          # Utilities package
│   ├── config.py            # Config loading helpers
│   ├── helpers.py           # Helper functions
│   └── logging.py           # Logging configuration
├── streamlit_app.py         # Streamlit frontend
└── main.py                  # Application entry point
```

### Database Schema & Initialization
```
sql/
└── init_database.sql        # PostgreSQL + pgvector schema (tracked!)
```

### Configuration Templates (NO SECRETS)
```
.env.example                 # Template with placeholder values - COMMIT THIS
.env.template                # Alternative template name
config/
├── settings.example.yaml    # Optional YAML config template
└── logging.example.yaml     # Optional logging config template
```

### Documentation
```
README.md                    # Complete project documentation
CHANGELOG.md                 # Version history
CONTRIBUTING.md              # Contribution guidelines
LICENSE                      # License file (MIT/Apache-2.0/etc.)
docs/
├── architecture.md          # System architecture docs
├── api.md                   # API reference
├── deployment.md            # Deployment guide
└── troubleshooting.md       # Common issues & solutions
```

### Dependency Management
```
requirements.in              # Direct dependencies (source of truth)
requirements.txt             # Fully resolved lock file - COMMIT THIS
pyproject.toml               # Project metadata (if using modern packaging)
```

### CI/CD & Development
```
.github/
├── workflows/
│   ├── ci.yml               # GitHub Actions CI pipeline
│   ├── cd.yml               # Deployment workflow
│   └── dependabot.yml       # Dependency updates
.gitlab/
├── ci/
│   └── pipeline.yml         # GitLab CI pipeline
.pre-commit-config.yaml      # Pre-commit hooks
Makefile                     # Common development tasks
justfile                     # Alternative to Makefile
```

### Testing
```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration
├── unit/
│   ├── test_config.py
│   ├── test_database.py
│   ├── test_ingestion.py
│   ├── test_embeddings.py
│   ├── test_retrievers.py
│   ├── test_chains.py
│   └── test_utils.py
├── integration/
│   ├── test_rag_pipeline.py
│   └── test_database_integration.py
├── fixtures/
│   ├── sample_documents/    # SMALL test PDFs only (<100KB)
│   └── mock_responses.py
└── test_requirements.txt    # Test-only dependencies
```

### Sample Data (SMALL ONLY)
```
data/
├── sample.pdf               # Small sample PDF for demos (<500KB)
├── sample_chunks.json       # Pre-chunked sample for testing
└── sample_embeddings.npy    # Tiny embedding sample for CI tests
```

### Scripts & Utilities
```
scripts/
├── setup.sh                 # Environment setup script
├── ingest.py                # Standalone ingestion script
├── reindex.py               # Re-indexing utility
├── migrate.py               # Database migration runner
├── backup_db.sh             # Database backup script
└── health_check.py          # Health check endpoint
```

---

## ❌ FILES THAT MUST NEVER BE COMMITTED (Ignored by .gitignore)

### Secrets & Credentials
```
.env                         # ACTUAL environment with real secrets
.env.local                   # Local overrides
.env.production              # Production secrets
.env.*.secret                # Any secret files
*.key                        # Private keys
*.pem                        # Certificates
*.crt                        # Certificates
secrets/                     # Secrets directory
credentials/                 # Credentials directory
```

### Virtual Environments & Python Cache
```
venv/                        # Virtual environment
.env/                        # Virtual env (alternative name)
.venv/                       # Virtual env
__pycache__/                 # Bytecode cache
*.pyc                        # Compiled Python
*.pyo                        # Optimized bytecode
.pytest_cache/               # Pytest cache
.mypy_cache/                 # MyPy cache
.ruff_cache/                 # Ruff cache
.coverage                    # Coverage data
htmlcov/                     # HTML coverage report
```

### Model & Embedding Caches (Large, Machine-Specific)
```
~/.cache/huggingface/        # HF model cache
~/.cache/torch/              # PyTorch cache
~/.cache/sentence_transformers/  # ST cache
.cache/huggingface/          # Project-local cache
.cache/torch/
embedding_cache/             # Local embedding cache
model_cache/                 # Local model cache
models/                      # Downloaded models
weights/                     # Model weights
*.bin                        # Model binaries
*.safetensors                # Safetensors format
*.pt                         # PyTorch models
*.pth                        # PyTorch checkpoints
*.onnx                       # ONNX models
*.gguf                       # GGUF quantized models
```

### Vector Database & Indexes
```
pgdata/                      # PostgreSQL data directory
postgresql_data/             # PG data (alternative)
data/postgres/               # PG data (alternative)
*.dump                       # Database dumps
*.sql.backup                 # SQL backups
vector_index/                # Vector indexes
*.index                      # Index files
*.faiss                      # FAISS indexes
*.hnsw                       # HNSW indexes
*.ivf                        # IVF indexes
chroma_db/                   # ChromaDB data
faiss_index/                 # FAISS index data
```

### Document Processing Outputs
```
extracted/                   # Extracted PDF content
extracted_images/            # Extracted images
extracted_tables/            # Extracted tables (CSV/JSON)
extracted_text/              # Extracted text files
thumbnails/                  # Generated thumbnails
previews/                    # Document previews
ocr_output/                  # OCR results
chunks/                      # Generated chunks
processed/                   # Processed documents
chunked_documents/           # Chunked output
document_chunks/             # Document chunks
*.chunks.json                # Chunk files
*.chunks.pkl                 # Pickled chunks
*.chunks.parquet             # Parquet chunks
```

### Large Data Files
```
Data/*.pdf                   # Original large PDFs
data/*.pdf                   # Data directory PDFs
*.pdf                        # Any PDF in root
embeddings/                  # Generated embeddings
*.embeddings.npy             # NumPy embeddings
*.embeddings.pkl             # Pickled embeddings
*.embeddings.parquet         # Parquet embeddings
vectors/                     # Vector files
*.vectors.npy                # NumPy vectors
datasets/                    # Large datasets
*.csv                        # Large CSV files
*.parquet                    # Parquet files
*.jsonl                      # JSONL files
*.ndjson                     # NDJSON files
```

### Streamlit & Runtime
```
.streamlit/
├── config.toml              # Local Streamlit config
├── secrets.toml             # Streamlit secrets (NEVER COMMIT)
└── credentials.toml         # Streamlit credentials
static/                      # Streamlit static cache
streamlit_cache/             # Streamlit cache
run/                         # Runtime directory
var/                         # Variable data
*.pid                        # Process ID files
*.sock                       # Unix sockets
```

### IDE & Editor Files
```
.vscode/settings.json        # Local VS Code settings
.vscode/launch.json          # Local launch config
.idea/                       # PyCharm/IntelliJ
*.iml                        # IntelliJ modules
*.swp                        # Vim swap
*.swo                        # Vim swap
*~                           # Backup files
.ipynb_checkpoints/          # Jupyter checkpoints
```

### OS Files
```
.DS_Store                    # macOS metadata
Thumbs.db                    # Windows thumbnails
ehthumbs.db                  # Windows thumbnails
Desktop.ini                  # Windows folder config
$RECYCLE.BIN/                # Windows recycle bin
._*                          # macOS resource forks
.Spotlight-V100              # macOS Spotlight
.Trashes                     # macOS trash
.AppleDouble                 # macOS AppleDouble
.LSOverride                  # macOS LSOverride
Icon?                        # macOS custom icons
```

### Logs & Temporary Files
```
logs/                        # Log directory
*.log                        # Log files
*.log.*                      # Rotated logs
log/                         # Log directory (alternative)
logging/                     # Logging directory
nohup.out                    # Nohup output
*.tmp                        # Temporary files
*.temp                       # Temporary files
tmp/                         # Temp directory
temp/                        # Temp directory
.tmp/                        # Hidden temp
```

### Build & Distribution
```
dist/                        # Distribution package
build/                       # Build directory
*.egg-info/                  # Egg metadata
*.egg                        # Egg files
*.whl                        # Wheel files
*.tar.gz                     # Source distributions
pip-wheel-metadata/          # Pip metadata
.sdist/                      # Source dist
```

### Profiling & Debug
```
*.prof                       # Profile data
*.profraw                    # Raw profile
.profiler/                   # Profiler output
*.lprof                      # Line profiler
*.stat                       # Stats files
py-spy/                      # Py-spy output
*.speedscope.json            # Speedscope profiles
```

### Documentation Build
```
docs/_build/                 # Sphinx build output
docs/build/                  # Build output
site/                        # MkDocs site
_html/                       # HTML output
```

### CI/CD Local Overrides
```
docker-compose.override.yml  # Local Docker override
docker-compose.local.yml     # Local compose
Dockerfile.local             # Local Dockerfile
.dockerignore.local          # Local dockerignore
.kube/                       # Local kube config
kubeconfig                   # Kubernetes config
*.kubeconfig                 # Kubeconfig files
.terraform/                  # Terraform state
*.tfstate                    # Terraform state
*.tfvars                     # Terraform variables
*.tfplan                     # Terraform plan
```

---

## 📋 COMMIT CHECKLIST FOR CODE REVIEWS

Before merging any PR, verify:

- [ ] No `.env` or secret files in diff
- [ ] No `venv/`, `__pycache__/`, `.pytest_cache/` in diff
- [ ] No model weights (`.bin`, `.safetensors`, `.pt`, `.onnx`) in diff
- [ ] No vector indexes (`*.index`, `*.faiss`, `*.hnsw`) in diff
- [ ] No extracted PDF content (`extracted/`, `chunks/`) in diff
- [ ] No large PDFs (`*.pdf` > 500KB) in diff
- [ ] No database dumps (`*.dump`, `*.sql.backup`) in diff
- [ ] No IDE files (`.vscode/settings.json`, `.idea/`) in diff
- [ ] No OS files (`.DS_Store`, `Thumbs.db`) in diff
- [ ] No log files (`*.log`) in diff
- [ ] `requirements.txt` updated if `requirements.in` changed
- [ ] `.env.example` updated if new config vars added
- [ ] Documentation updated for new features
- [ ] Tests added for new functionality

---

## 🔒 SECRET SCANNING REMINDER

This repository should have secret scanning enabled:
- **GitHub**: Enable "Secret scanning" and "Push protection" in repo settings
- **GitLab**: Enable "Secret detection" in CI/CD settings
- **Pre-commit**: Use `detect-secrets` or `truffleHog` in pre-commit hooks

```bash
# Install pre-commit with secret detection
pip install pre-commit detect-secrets
pre-commit install

# Scan history for secrets (run periodically)
detect-secrets scan --all-files
trufflehog git file://. --since-commit=HEAD~100
```

---

## 📦 REPOSITORY STRUCTURE SUMMARY

```
rag_agent/
├── .gitignore                 # ← THIS FILE (tracked)
├── .env.example               # ← TRACKED (template)
├── requirements.in            # ← TRACKED (direct deps)
├── requirements.txt           # ← TRACKED (resolved lock)
├── pyproject.toml             # ← TRACKED (project metadata)
├── README.md                  # ← TRACKED
├── CHANGELOG.md               # ← TRACKED
├── LICENSE                    # ← TRACKED
├── Makefile                   # ← TRACKED
├── app/                       # ← TRACKED (all source)
├── sql/                       # ← TRACKED (schema)
├── tests/                     # ← TRACKED (test code)
├── docs/                      # ← TRACKED (documentation)
├── scripts/                   # ← TRACKED (utilities)
├── data/                      # ← TRACKED (SMALL samples only)
├── .github/                   # ← TRACKED (CI/CD)
├── .gitlab/                   # ← TRACKED (CI/CD)
├── .pre-commit-config.yaml    # ← TRACKED
│
├── .env                       # ← IGNORED (secrets)
├── venv/                      # ← IGNORED
├── __pycache__/               # ← IGNORED
├── Data/sample.pdf            # ← IGNORED (large PDF)
├── extracted/                 # ← IGNORED
├── chunks/                    # ← IGNORED
├── embeddings/                # ← IGNORED
├── pgdata/                    # ← IGNORED
├── .streamlit/secrets.toml    # ← IGNORED
├── .vscode/settings.json      # ← IGNORED
└── .DS_Store                  # ← IGNORED
```

---

*Generated for Multimodal RAG Chatbot v1.0.0*
*Last updated: 2026-01-01*
*Compatible with: Python 3.14, Git 2.40+*