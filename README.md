# Local RAG System with Ollama + ChromaDB

This repository contains a local Retrieval-Augmented Generation (RAG) workflow for PDF documents using:

- Ollama for embeddings + generation
- ChromaDB for vector storage + retrieval

## Current State

Week 2 blockers are resolved in the main `app/` flow:

- Generic ingestion metadata (no filename-topic guessing)
- Idempotent ingestion for reruns and document edits
- Source-aware context formatting in prompts
- Embedding evaluation script for model comparison

## Main Files

- `app/config.py`: central settings (models, Chroma endpoints, chunking, docs path, `TOP_K`)
- `app/ingest_documents.py`: ingests all PDFs from `data/docs/`
- `app/full_rag.py`: retrieval + prompt assembly + answer generation
- `app/eval_embeddings.py`: embedding model comparison on `data/eval/retrieval_eval.jsonl`
- `data/eval/retrieval_eval.jsonl`: evaluation queries and expected labels

## Prerequisites

- Python 3.10+ (for local CLI / host-run API)
- Docker Desktop (for Ollama and ChromaDB containers)
- Optional: Docker Compose (included with modern Docker Desktop)

## Ingestion Behavior (Important)

Ingestion is intentionally simple and generic:

- Metadata per chunk:
  - `source_document`: original filename
  - `file_name`: original filename
  - `chunk_id`: stable chunk id (`<slug>_chunk_<index>`)
- No hardcoded topic keywords or filename-topic inference

Ingestion is idempotent:

- Before re-indexing a document, old chunks for that `source_document` are deleted
- New chunks are then written
- `upsert` is used for document writes in `full_rag.py`
- Re-running ingestion is safe and refreshes changed content

## Retrieval + Prompt Behavior

- Retrieval uses `TOP_K` from `app/config.py` (currently `5`)
- Context sent to the LLM includes all retrieved chunks
- Each chunk is formatted with source:

```text
[Source: <source_document>]
<chunk text>
```

- Prompt instructs the model to combine details across chunks and provide complete answers

## Run Commands

1. Ingest documents:

```bash
python app/ingest_documents.py
```

2. Ask a question (one-shot):

```bash
python app/full_rag.py "What are breach notification obligations under GDPR?"
```

3. Ask a question (interactive):

```bash
python app/full_rag.py
```

4. Run embedding comparison (full):

```bash
python app/eval_embeddings.py --models nomic-embed-text bge-m3 --top-k 3 --output data/eval/embedding_comparison_results.json
```

5. Optional smoke run:

```bash
python app/eval_embeddings.py --models nomic-embed-text bge-m3 --max-chunks 8 --top-k 3 --output data/eval/embedding_comparison_results_smoke.json
```

6. Run prompt variant evaluation for Week 3:

```bash
python scripts/evaluate_prompts.py --top-k 3
```

Use `--top-k 5` to rerun the same dataset with a larger retrieval set:

```bash
python scripts/evaluate_prompts.py --top-k 5
```

Prompt evaluation notes:

- Dataset: `scripts/eval_dataset.json`
- Results folder: `scripts/results/`
- Detailed outputs:
  - `scripts/results/prompt_eval_results.csv`
  - `scripts/results/prompt_eval_results.json`
  - `scripts/results/prompt_eval_summary.csv`
- Prompt variants:
  - `baseline`: current production prompt from `app/full_rag.py`
  - `quality_focused`: longer prompt emphasizing completeness and grounded detail
  - `short_optimized`: shorter prompt intended to reduce prompt overhead while preserving grounding rules

## FastAPI Layer

The existing RAG logic is exposed as an API without duplicating the core RAG pipeline.

Default FastAPI port is `8001` so it does not conflict with Chroma (`8000`).

### Environment Configuration

Service connections and API behavior are env-driven:

- `OLLAMA_BASE_URL`
- `CHROMA_HOST`
- `CHROMA_PORT`
- `API_HOST`
- `API_PORT`
- `API_DEV_MODE`
- `API_RELOAD`
- `HEALTH_INCLUDE_ERROR_DETAILS`
- `INGEST_PROTECTION_ENABLED`
- `INGEST_API_KEY`
- `INGEST_API_KEY_HEADER`

`mode 1` (`FastAPI on host, Ollama + Chroma in Docker`) `.env` example:

```env
OLLAMA_BASE_URL=http://localhost:11434
CHROMA_HOST=localhost
CHROMA_PORT=8000
API_HOST=127.0.0.1
API_PORT=8001
API_DEV_MODE=true
API_RELOAD=true
HEALTH_INCLUDE_ERROR_DETAILS=true
INGEST_PROTECTION_ENABLED=true
INGEST_API_KEY=dev-ingest-key
INGEST_API_KEY_HEADER=X-API-Key
```

`mode 2` (`FastAPI in Docker, Ollama + Chroma manually managed in Docker Desktop`) `.env` example:

```env
OLLAMA_BASE_URL=http://host.docker.internal:11434
CHROMA_HOST=host.docker.internal
CHROMA_PORT=8000
API_HOST=0.0.0.0
API_PORT=8001
API_DEV_MODE=false
API_RELOAD=false
HEALTH_INCLUDE_ERROR_DETAILS=false
INGEST_PROTECTION_ENABLED=true
INGEST_API_KEY=dev-ingest-key
INGEST_API_KEY_HEADER=X-API-Key
```

`future clean mode` (`FastAPI + Ollama + Chroma all managed by compose`) `.env` example:

```env
OLLAMA_BASE_URL=http://ollama:11434
CHROMA_HOST=chromadb
CHROMA_PORT=8000
API_HOST=0.0.0.0
API_PORT=8001
API_DEV_MODE=false
API_RELOAD=false
HEALTH_INCLUDE_ERROR_DETAILS=false
INGEST_PROTECTION_ENABLED=true
INGEST_API_KEY=dev-ingest-key
INGEST_API_KEY_HEADER=X-API-Key
```

Note: current `docker-compose.yml` is intentionally set to temporary integration mode for `api` (`host.docker.internal`) so it can connect to your existing manually managed `ollama` and `chromadb` containers.

### Docker Setup

`Dockerfile`:

- Builds a minimal Python image for this project
- Reuses current app code (`python -m app.api.run`)
- Exposes FastAPI on `8001`

`docker-compose.yml` services:

- `chromadb` -> `8000:8000`
- `ollama` -> `11434:11434`
- `api` -> `8001:8001`

### Run Mode 1: FastAPI on Host, Ollama + Chroma in Docker

If your containers are already running in Docker Desktop, keep them as-is and run API locally.

Or start them from compose:

```bash
docker compose up -d chromadb ollama
```

Run FastAPI on host:

```bash
pip install -r requirements.txt
python -m app.api.run
```

API docs:

- Swagger UI: `http://127.0.0.1:8001/docs`
- ReDoc: `http://127.0.0.1:8001/redoc`

### Run Mode 2 (Temporary): API in Docker, Ollama + Chroma Manually Managed

This mode is for your current setup. It does not require recreating your existing Ollama and Chroma containers.

```bash
docker compose up --build -d api
```

Useful checks:

```bash
docker compose ps
docker compose logs -f api
```

### Future Clean Mode: All Services Managed by Compose

After you safely migrate Ollama models and are ready to let compose manage all services:

1. Update `api` environment in `docker-compose.yml` to use:
   - `OLLAMA_BASE_URL=http://ollama:11434`
   - `CHROMA_HOST=chromadb`
2. Re-add `depends_on` for `api` on `ollama` and `chromadb`.
3. Run:

```bash
docker compose up --build -d
```

Stop stack:

```bash
docker compose down
```

Ingest protection:

- Enabled by default (`INGEST_PROTECTION_ENABLED=true`)
- Requires `INGEST_API_KEY` in request header `X-API-Key` (or custom `INGEST_API_KEY_HEADER`)
- To disable protection for local-only testing, set `INGEST_PROTECTION_ENABLED=false`

### Endpoint Summary

- `GET /health`
  - Real connectivity checks for Ollama and ChromaDB
- `POST /api/v1/rag/query`
  - Full RAG (retrieve + generate)
  - Body: `question`, optional `top_k`, optional `session_id`
  - Returns: `status`, `answer`, `sources`, timing metrics, `session_id`
- `POST /api/v1/rag/search`
  - Retrieval only (no generation)
  - Body: `query`, optional `top_k`
  - Returns: `status`, retrieved chunks + metadata + timing
- `POST /api/v1/rag/ingest`
  - Triggers ingestion from configured docs directory (`DOCS_DIR`) or optional `docs_dir` override
  - Protected by API key header by default

### cURL Examples

Health:

```bash
curl -X GET "http://127.0.0.1:8001/health"
```

Query:

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What are breach notification obligations under GDPR?\",\"top_k\":5,\"session_id\":\"demo-session\"}"
```

Search only:

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/rag/search" \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"What are breach notification obligations under GDPR?\",\"top_k\":5}"
```

Ingest:

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/rag/ingest" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-ingest-key" \
  -d "{}"
```

## Notes

- Use the `app/` scripts as the active workflow.
- `query.py` and `ingest.py` are older prototype scripts.
