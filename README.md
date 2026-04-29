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
- `ADMIN_PROTECTION_ENABLED`
- `ADMIN_API_KEY`
- `ADMIN_API_KEY_HEADER`

### Performance Architecture

The API uses a process-local limiter for `/api/v1/rag/query`. Retrieval caching, response caching, Ollama warm-up, and in-process background jobs are available; vLLM production migration is not implemented.

User RAG request flow:

```text
User Request
-> FastAPI
-> Concurrency limiter
-> Response cache check
-> Retrieval cache / ChromaDB
-> Ollama generation
-> Chat logging
-> Response metrics
```

Admin maintenance flow:

```text
Admin Request
-> FastAPI
-> Background job queue
-> Heavy operation
-> Job status endpoint
```

Related final docs:

- [Performance report template](docs/performance_report_template.md)
- [AI performance tasks summary](docs/ai_performance_tasks_summary.md)

#### Configuration Reference

| Variable | Default | Purpose | When to change |
| --- | --- | --- | --- |
| `MAX_CONCURRENT_LLM_REQUESTS` | `2` | Caps simultaneous API-level RAG generations. | Increase on stronger hardware; reduce if Ollama becomes unstable. |
| `MAX_WAITING_RAG_REQUESTS` | `20` | Bounds the number of queued RAG requests. | Tune for expected user burst size. |
| `MAX_QUEUE_WAIT_SECONDS` | `45` | Max time a request waits for a generation slot. | Increase for patient internal users; reduce for faster overload feedback. |
| `OLLAMA_KEEP_ALIVE` | `30m` | Keeps Ollama models resident after use. | Increase to avoid cold starts; reduce to save RAM/VRAM. |
| `WARMUP_ON_STARTUP` | `false` | Runs Ollama warm-up during API startup. | Enable on a dedicated server where slower startup is acceptable. |
| `MODEL_NUM_CTX` | `4096` | Ollama context window option. | Increase for larger prompts if hardware supports it. |
| `MODEL_NUM_PREDICT` | `512` | Max generated tokens for Ollama generation. | Increase for longer answers; reduce for speed. |
| `RAG_RETRIEVAL_CACHE_ENABLED` | `false` | Enables in-memory Chroma retrieval result cache. | Enable for repeated document questions. |
| `RAG_RESPONSE_CACHE_ENABLED` | `false` | Enables exact repeated final-answer cache. | Enable for demos or repeated workloads; disable for raw generation benchmarks. |
| `RAG_CACHE_TTL_SECONDS` | `600` | TTL for retrieval and response cache entries. | Increase for stable corpora; reduce when documents change often. |
| `RAG_INDEX_VERSION` | `dev` | Version label included in cache keys and responses. | Change after re-indexing or changing document corpus. |
| `RAG_PROMPT_VERSION` | `v1` | Version label included in response metadata and response cache keys. | Change after prompt edits. |
| `BACKGROUND_JOBS_ENABLED` | `true` | Enables async heavy admin job endpoints. | Disable if all maintenance should be synchronous. |
| `BACKGROUND_JOB_WORKERS` | `1` | Number of in-process background worker threads. | Increase cautiously for independent admin jobs. |
| `BACKGROUND_JOB_MAX_QUEUE` | `20` | Max queued/running background jobs per process. | Tune for admin workload size. |
| `LLM_BACKEND` | `ollama` | Configures LLM client factory default. Production RAG still uses Ollama path. | Set only for explicit backend experiments. |
| `VLLM_BASE_URL` | `http://127.0.0.1:8002/v1` | OpenAI-compatible vLLM endpoint base URL. | Set to the PC/GPU vLLM server URL. |
| `VLLM_MODEL` | empty | Model name served by vLLM. | Required before running vLLM benchmark requests. |

#### Concurrency Limiter

The limiter protects `/api/v1/rag/query` with a process-local semaphore and bounded waiting queue. Successful responses include `queue_wait_time_sec`, `active_llm_requests`, and `waiting_rag_requests`.

Clean overload errors:

- `server_busy` (`503`): active slots are full and the waiting queue is full.
- `queue_timeout` (`503`): the request waited longer than `MAX_QUEUE_WAIT_SECONDS`.

Validation helper:

```bash
python scripts/validate_rag_concurrency.py --requests 3
```

- `MAX_CONCURRENT_LLM_REQUESTS`: cap for simultaneous API-level RAG generations.
- `MAX_WAITING_RAG_REQUESTS`: cap for queued RAG requests when all execution slots are busy.
- `MAX_QUEUE_WAIT_SECONDS`: maximum time a queued request may wait for an execution slot.
- `OLLAMA_KEEP_ALIVE`: Ollama model keep-alive value sent with generation requests.
- `MODEL_NUM_CTX`: Ollama context window option.
- `MODEL_NUM_PREDICT`: Ollama maximum generated tokens option.
- `RAG_RETRIEVAL_CACHE_ENABLED`: enables in-memory caching for ChromaDB retrieval results.
- `RAG_RESPONSE_CACHE_ENABLED`: enables in-memory caching for exact repeated RAG answers.
- `RAG_INDEX_VERSION`: index version label returned in RAG query metadata.
- `RAG_PROMPT_VERSION`: prompt version label returned in RAG query metadata.

#### Ollama Warm-up

Warm-up preloads the configured Ollama generation and embedding models with tiny requests so the first real RAG query avoids model cold-start delay. It does not write chat logs, call ChromaDB, or run retrieval.

Run manually:

```bash
python scripts/warmup_ollama.py
```

Enable warm-up during API startup:

```env
WARMUP_ON_STARTUP=true
```

Warm-up keeps models loaded according to `OLLAMA_KEEP_ALIVE` and may use RAM/VRAM while the models stay resident.

#### Retrieval Cache

The retrieval cache stores ChromaDB retrieval results only. It does not cache generated answers, full RAG responses, chat logs, or errors.

Enable it with:

```env
RAG_RETRIEVAL_CACHE_ENABLED=true
```

The cache is in-memory and process-local, uses `RAG_CACHE_TTL_SECONDS` and `RAG_CACHE_MAX_ITEMS`, and is cleared after document or chat-history ingestion. Repeated matching RAG questions can return `cache_hit=true` and `cache_type="retrieval"` while the answer is still generated normally.

#### Response Cache

The response cache stores final successful answers for exact repeated RAG questions. It is separate from the retrieval cache: a response cache hit skips retrieval and Ollama generation, while a retrieval cache hit only skips ChromaDB retrieval.

Enable it with:

```env
RAG_RESPONSE_CACHE_ENABLED=true
```

The cache is in-memory and process-local, uses `RAG_CACHE_TTL_SECONDS` and `RAG_CACHE_MAX_ITEMS`, and is cleared after document or chat-history ingestion. Use it carefully because document changes can make old answers outdated until ingestion clears the cache or `RAG_INDEX_VERSION` changes.

#### Background Jobs

Background jobs are for heavy admin or maintenance operations. They run in-process and are process-local; Redis/Celery is not used yet.

Config:

- `BACKGROUND_JOBS_ENABLED`: enable async job endpoints.
- `BACKGROUND_JOB_WORKERS`: worker threads for background jobs.
- `BACKGROUND_JOB_MAX_QUEUE`: maximum queued/running jobs in this process.
- `BACKGROUND_JOB_RETENTION_SECONDS`: how long completed jobs stay queryable.

Async endpoints:

- `POST /api/v1/rag/ingest/async`
- `POST /api/v1/admin/migrate-chat-history/async`
- `POST /api/v1/admin/ingest-chat-history/async`

Example async chat-history ingestion:

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/admin/ingest-chat-history/async" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-ingest-key" \
  -d "{}"
```

Check status:

```bash
curl -X GET "http://127.0.0.1:8001/api/v1/admin/jobs/<job_id>" \
  -H "X-API-Key: dev-ingest-key"
```

List recent jobs:

```bash
curl -X GET "http://127.0.0.1:8001/api/v1/admin/jobs" \
  -H "X-API-Key: dev-ingest-key"
```

#### vLLM Benchmark Layer

Ollama remains the default production backend. vLLM is optional and should be used for benchmark comparison only unless `LLM_BACKEND=vllm` is explicitly configured. Test vLLM on the PC/GPU machine; actual vLLM server and model setup is separate and should follow official vLLM documentation.

Config:

- `LLM_BACKEND`: `ollama` by default.
- `VLLM_BASE_URL`: OpenAI-compatible vLLM base URL.
- `VLLM_API_KEY`: bearer token for the vLLM-compatible endpoint.
- `VLLM_MODEL`: vLLM-served model name; required for vLLM requests.
- `VLLM_TIMEOUT_SECONDS`: vLLM HTTP timeout.

Ollama-only benchmark:

```bash
python scripts/benchmark_llm_backends.py --backends ollama --requests-per-backend 2
```

Later, when vLLM is running:

```bash
python scripts/benchmark_llm_backends.py --backends ollama vllm --requests-per-backend 5 --skip-unavailable
```

Reports are written to `results/performance/` as `llm_backend_benchmark_*` raw and summary JSON/CSV files.

#### Recommended Benchmark Workflow

A. Laptop smoke test:

```bash
python scripts/load_test_rag_api.py --preset smoke
```

B. PC raw performance test:

Disable caches:

```env
RAG_RETRIEVAL_CACHE_ENABLED=false
RAG_RESPONSE_CACHE_ENABLED=false
```

Then run:

```bash
python scripts/warmup_ollama.py
python scripts/load_test_rag_api.py --preset full --warmup
```

C. PC optimized performance test:

Enable caches:

```env
RAG_RETRIEVAL_CACHE_ENABLED=true
RAG_RESPONSE_CACHE_ENABLED=true
```

Then run:

```bash
python scripts/load_test_rag_api.py --preset full --warmup
```

D. Ollama-only backend benchmark:

```bash
python scripts/benchmark_llm_backends.py --backends ollama --requests-per-backend 2 --skip-unavailable
```

E. Future Ollama vs vLLM benchmark:

```bash
python scripts/benchmark_llm_backends.py --backends ollama vllm --requests-per-backend 5 --skip-unavailable
```

Use [docs/performance_report_template.md](docs/performance_report_template.md) to capture the final numbers.

#### Quick Validation Checklist

- [ ] API starts with `python -m app.api.run`.
- [ ] `GET /health` returns healthy Ollama and Chroma status.
- [ ] `POST /api/v1/rag/query` returns an answer.
- [ ] Repeated query shows `cache_hit=true` and `cache_type="response"` when response cache is enabled.
- [ ] `GET /api/v1/admin/jobs` returns job history.
- [ ] `POST /api/v1/admin/migrate-chat-history/async` queues a job.
- [ ] `python scripts/load_test_rag_api.py --preset smoke` generates reports under `results/performance/`.
- [ ] `python scripts/benchmark_llm_backends.py --backends ollama --requests-per-backend 1 --skip-unavailable` runs and writes benchmark reports.

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
ADMIN_PROTECTION_ENABLED=true
ADMIN_API_KEY=dev-ingest-key
ADMIN_API_KEY_HEADER=X-API-Key
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
ADMIN_PROTECTION_ENABLED=true
ADMIN_API_KEY=dev-ingest-key
ADMIN_API_KEY_HEADER=X-API-Key
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
ADMIN_PROTECTION_ENABLED=true
ADMIN_API_KEY=dev-ingest-key
ADMIN_API_KEY_HEADER=X-API-Key
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

Admin protection:

- Enabled by default (`ADMIN_PROTECTION_ENABLED=true`)
- Requires `ADMIN_API_KEY` in request header `X-API-Key` (or custom `ADMIN_API_KEY_HEADER`)
- If `ADMIN_API_KEY` is not set, it falls back to `INGEST_API_KEY`
- The admin endpoints trigger Week 4 maintenance operations and should not be exposed without this header

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
- `POST /api/v1/rag/ingest/async`
  - Queues document ingestion as a background job
  - Protected by API key header by default
- `GET /api/v1/admin/jobs`
  - Lists recent background jobs
  - Protected by admin API key header by default
- `GET /api/v1/admin/jobs/{job_id}`
  - Returns background job status/result/error
  - Protected by admin API key header by default
- `GET /api/v1/admin/jobs-stats`
  - Returns background job queue stats
  - Protected by admin API key header by default
- `POST /api/v1/admin/migrate-chat-history`
  - Migrates `data/chat_logs.jsonl` to timestamped Week 4 migration JSON
  - Body: optional `output_dir`, optional `write_latest`
  - Protected by API key header by default
- `POST /api/v1/admin/migrate-chat-history/async`
  - Queues chat history migration as a background job
  - Protected by admin API key header by default
- `POST /api/v1/admin/ingest-chat-history`
  - Ingests migrated chat history into ChromaDB using the existing RAG upsert path
  - Body: optional `input_file`, optional `dry_run`
  - Defaults to `scripts/results/migrations/chat_history_migrated_latest.json`
  - Protected by API key header by default
- `POST /api/v1/admin/ingest-chat-history/async`
  - Queues migrated chat history ingestion as a background job
  - Protected by admin API key header by default

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

Concurrency limiter validation:

Set these values, restart the API, then run the helper:

```env
MAX_CONCURRENT_LLM_REQUESTS=1
MAX_WAITING_RAG_REQUESTS=1
MAX_QUEUE_WAIT_SECONDS=3
```

```bash
python scripts/validate_rag_concurrency.py --requests 3
```

Expected behavior: one request runs, one waits, and extra simultaneous requests return a clean `503` with `server_busy` or `queue_timeout`. Successful responses include `queue_wait_time_sec`, `active_llm_requests`, and `waiting_rag_requests`.

Limiter error meanings:

- `server_busy` (`503`): all active slots are full and the bounded waiting queue is already full.
- `queue_timeout` (`503`): the request entered the waiting queue but no slot opened before `MAX_QUEUE_WAIT_SECONDS`.

The validation helper prints per-request JSON plus totals for successes, `server_busy`, `queue_timeout`, other errors, and average successful queue wait time.

### Load Testing

Laptop smoke test:

```bash
python scripts/load_test_rag_api.py --preset smoke
```

Full PC test:

```bash
python scripts/load_test_rag_api.py --preset full --warmup
```

Reports are written to `results/performance/` as timestamped raw JSON/CSV and summary JSON/CSV files, plus `load_test_latest_summary.json` and `load_test_latest_summary.csv`.

Key metrics:

- `queue_wait_time_sec`: time spent waiting for a RAG execution slot.
- `generation_time_sec`: server-side Ollama generation time.
- `retrieval_time_sec`: server-side retrieval time.
- `cache_hit` / `cache_type`: whether retrieval or response cache helped.
- `server_busy`: bounded waiting queue was full.
- `queue_timeout`: request waited longer than `MAX_QUEUE_WAIT_SECONDS`.

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

Admin migrate chat history:

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/admin/migrate-chat-history" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-ingest-key" \
  -d "{\"write_latest\":true}"
```

Example response:

```json
{
  "status": "success",
  "operation": "migrate_chat_history",
  "items_migrated": 10,
  "output_file": "scripts/results/migrations/chat_history_migrated_20260424_181234.json",
  "latest_file": "scripts/results/migrations/chat_history_migrated_latest.json"
}
```

Admin ingest migrated chat history:

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/admin/ingest-chat-history" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-ingest-key" \
  -d "{}"
```

Optional dry run:

```bash
curl -X POST "http://127.0.0.1:8001/api/v1/admin/ingest-chat-history" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-ingest-key" \
  -d "{\"dry_run\":true}"
```

Example response:

```json
{
  "status": "success",
  "operation": "ingest_chat_history",
  "records_loaded": 10,
  "records_upserted": 10,
  "records_skipped": 0,
  "collection": "documents"
}
```

Swagger/Postman testing:

- Swagger UI: open `http://127.0.0.1:8001/docs`, expand the `admin` tag, choose an endpoint, click "Try it out", add the JSON body, and add the `X-API-Key` header.
- Postman: use `POST`, set `Content-Type: application/json`, add header `X-API-Key: dev-ingest-key`, and send `{}` or the optional fields shown above.

## Notes

- Use the `app/` scripts as the active workflow.
- `query.py` and `ingest.py` are older prototype scripts.
