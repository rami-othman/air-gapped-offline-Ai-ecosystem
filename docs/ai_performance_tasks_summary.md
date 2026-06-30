# AI Performance Tasks Summary

This document summarizes the implemented AI/performance work. It is a final reference for what is production-ready, what is benchmark-only, and what remains limited by the current in-process architecture.

## Implemented Tasks

### Task 9 - Performance Config And Metrics Foundation

- Added performance, Ollama runtime, and cache-related environment settings.
- Added optional RAG response metadata fields for queue, cache, model, prompt, and index information.
- Added safe default metadata values before real queue/cache behavior existed.
- Updated Ollama runtime options to read from config.

### Task 10 - API-Level Semaphore And Bounded Queue

- Added an API-level process-local concurrency limiter for `/api/v1/rag/query`.
- Used a semaphore and bounded waiting count to cap active and waiting RAG requests.
- Returned real queue wait, active request, and waiting request metrics in successful RAG responses.

### Task 11 - Clean Busy And Timeout Errors

- Standardized `server_busy` and `queue_timeout` responses.
- Added structured error responses with safe details and HTTP `503`.
- Kept raw Python exceptions out of API responses while preserving server logs.

### Task 12 - Ollama Warm-Up

- Added config for manual and startup warm-up.
- Added `app/ollama_warmup.py` and `scripts/warmup_ollama.py`.
- Warm-up calls Ollama generation and embedding endpoints only; it does not touch ChromaDB or chat logs.

### Task 14 - Retrieval Cache

- Added a thread-safe in-memory TTL cache.
- Cached ChromaDB retrieval results only.
- Added cache keys based on normalized query, `top_k`, collection, embedding model, and index version.
- Cleared retrieval cache after document and chat-history ingestion.

### Task 15 - Response Cache

- Added a separate in-memory TTL cache for exact repeated successful RAG answers.
- Response cache hits skip retrieval and Ollama generation.
- Response cache keys include normalized question, `top_k`, model, prompt version, index version, model runtime options, collection, and embedding model.
- Cleared response cache after ingestion.

### Task 16 - Load-Test Reports

- Added `scripts/load_test_rag_api.py`.
- Added `scripts/load_test_questions.json`.
- Added smoke and full presets.
- Captured queue, cache, retrieval, generation, error, and response-size metrics.
- Wrote raw and summary JSON/CSV reports to `results/performance/`.

### Task 17 - Background Jobs For Heavy Admin Tasks

- Added an in-process background job manager.
- Added async endpoints for document ingestion, chat-history migration, and chat-history ingestion.
- Added job list, job status, and job stats endpoints.
- Preserved synchronous endpoints.

### Task 18 - vLLM Benchmark Layer

- Added LLM client abstractions for Ollama and vLLM.
- Added optional vLLM OpenAI-compatible benchmark client.
- Added `scripts/benchmark_llm_backends.py`.
- Kept production RAG on the existing Ollama path.

## Production-Ready Now

- FastAPI RAG query endpoint with queue protection.
- Clean overload errors for busy and timed-out requests.
- Ollama warm-up script.
- Retrieval cache and response cache for in-process deployments.
- Admin background jobs for heavy maintenance operations.
- API load-test reporting.

## Benchmark Or Experimental

- vLLM client and backend benchmark script.
- `LLM_BACKEND=vllm` as a config concept; production RAG has not migrated to vLLM.
- Full PC benchmark workflow until run on the target hardware.

## Known Limitations

- Caches are in-memory and process-local.
- Background jobs are in-process and not persistent.
- The concurrency limiter is process-local.
- API restarts lose job history and cache state.
- Multiple API worker processes each get separate limiters, caches, and job queues.
- vLLM is benchmark-only, not a production migration.
- Response cache can hide raw generation cost during performance tests.
- Cached final answers can become stale if documents change outside known ingestion paths or if `RAG_INDEX_VERSION` is not updated.

## Operational Notes

- Use raw performance tests with both caches disabled when measuring baseline model speed.
- Use optimized tests with caches enabled when measuring expected repeated-query behavior.
- Change `RAG_INDEX_VERSION` after manual re-indexing or corpus replacement.
- Change `RAG_PROMPT_VERSION` after prompt changes that should invalidate response cache entries.
- Use `docs/performance_report_template.md` for final benchmark writeups.
