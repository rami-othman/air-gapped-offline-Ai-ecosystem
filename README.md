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

- Ollama running on `http://localhost:11434`
- ChromaDB running on `localhost:8000`
- Python packages installed: `chromadb`, `requests`, `pypdf`

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

## FastAPI Layer

The existing RAG logic is now also exposed as an API (no duplicated core logic).

### Run API

```bash
pip install -r requirements.txt
uvicorn app.api.main:app --reload
```

API docs:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### Endpoint Summary

- `GET /health`
  - Basic health status
- `POST /api/v1/rag/query`
  - Full RAG (retrieve + generate)
  - Body: `question`, optional `top_k`, optional `session_id`
  - Returns: `answer`, `sources`, timing metrics, `session_id`
- `POST /api/v1/rag/search`
  - Retrieval only (no generation)
  - Body: `question`, optional `top_k`
  - Returns: retrieved chunks + metadata + timing
- `POST /api/v1/rag/ingest`
  - Triggers ingestion from configured docs directory (`DOCS_DIR`) or optional `docs_dir` override

### cURL Examples

Health:

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

Query:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What are breach notification obligations under GDPR?\",\"top_k\":5,\"session_id\":\"demo-session\"}"
```

Search only:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/rag/search" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What are breach notification obligations under GDPR?\",\"top_k\":5}"
```

Ingest:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/rag/ingest" \
  -H "Content-Type: application/json" \
  -d "{}"
```

## Notes

- Use the `app/` scripts as the active workflow.
- `query.py` and `ingest.py` are older prototype scripts.
