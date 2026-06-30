# Performance Report Template

Use this template after running laptop smoke tests, PC raw tests, PC optimized tests, and optional backend benchmarks. Do not fill in numbers unless they came from a saved report under `results/performance/`.

## Test Environment

- Date:
- Tester:
- Machine:
- Operating system:
- API run mode:
- Ollama location:
- ChromaDB location:
- Notes:

## Hardware Specs

- CPU:
- RAM:
- GPU:
- VRAM:
- Storage:
- Thermal/power mode:

## Model Used

- Generation model:
- Embedding model:
- Quantization:
- Ollama keep-alive:
- vLLM model, if tested:

## Config Values

Record the active values from `.env`:

| Variable | Value |
| --- | --- |
| `MAX_CONCURRENT_LLM_REQUESTS` | |
| `MAX_WAITING_RAG_REQUESTS` | |
| `MAX_QUEUE_WAIT_SECONDS` | |
| `MODEL_NUM_CTX` | |
| `MODEL_NUM_PREDICT` | |
| `RAG_RETRIEVAL_CACHE_ENABLED` | |
| `RAG_RESPONSE_CACHE_ENABLED` | |
| `RAG_CACHE_TTL_SECONDS` | |
| `RAG_INDEX_VERSION` | |
| `RAG_PROMPT_VERSION` | |
| `BACKGROUND_JOB_WORKERS` | |
| `LLM_BACKEND` | |

## Raw Performance Test Results

Command:

```bash
python scripts/load_test_rag_api.py --preset full --warmup
```

Cache state:

```env
RAG_RETRIEVAL_CACHE_ENABLED=false
RAG_RESPONSE_CACHE_ENABLED=false
```

Report files:

- Raw JSON:
- Raw CSV:
- Summary JSON:
- Summary CSV:

Key observations:

- Success rate:
- Average generation time:
- Average retrieval time:
- Average queue wait:
- Overload errors:

## Optimized Cache Test Results

Command:

```bash
python scripts/load_test_rag_api.py --preset full --warmup
```

Cache state:

```env
RAG_RETRIEVAL_CACHE_ENABLED=true
RAG_RESPONSE_CACHE_ENABLED=true
```

Report files:

- Raw JSON:
- Raw CSV:
- Summary JSON:
- Summary CSV:

Key observations:

- Cache hit rate:
- Response cache hit count:
- Retrieval cache hit count:
- Average generation time:
- Average client elapsed time:

## Queue Behavior

- Highest concurrency level tested:
- Max `queue_wait_time_sec`:
- Average `queue_wait_time_sec`:
- `server_busy` count:
- `queue_timeout` count:
- Recommended limiter settings:

## Error Behavior

- Validation errors observed:
- `server_busy` behavior:
- `queue_timeout` behavior:
- Backend errors:
- Notes:

## Cache Behavior

- Retrieval cache enabled:
- Response cache enabled:
- Observed retrieval cache hits:
- Observed response cache hits:
- Did ingestion clear caches:
- Notes about stale-answer risk:

## Bottlenecks Found

- Generation bottlenecks:
- Retrieval bottlenecks:
- Queue/concurrency bottlenecks:
- Memory/VRAM bottlenecks:
- Disk/network bottlenecks:

## Recommendations

- Recommended default config:
- Recommended laptop config:
- Recommended PC config:
- Cache recommendations:
- Operational notes:

## Next Steps

- Short-term:
- Medium-term:
- Future backend experiments:
