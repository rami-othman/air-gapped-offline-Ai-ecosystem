"""Small manual validation helper for the RAG API concurrency limiter.

Recommended .env values before running:
MAX_CONCURRENT_LLM_REQUESTS=1
MAX_WAITING_RAG_REQUESTS=1
MAX_QUEUE_WAIT_SECONDS=3
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Barrier
from typing import Any


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _send_query(api_url: str, index: int, barrier: Barrier) -> dict[str, Any]:
    payload = json.dumps(
        {
            "question": f"Concurrency validation request {index}: summarize the available policy context briefly.",
            "top_k": 1,
            "session_id": f"concurrency-validation-{index}",
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/v1/rag/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    barrier.wait()
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            body = json.loads(response.read().decode("utf-8"))
            return {
                "request": index,
                "http_status": response.status,
                "duration_sec": round(time.perf_counter() - started, 4),
                "error": None,
                "queue_wait_time_sec": body.get("queue_wait_time_sec"),
                "active_llm_requests": body.get("active_llm_requests"),
                "waiting_rag_requests": body.get("waiting_rag_requests"),
                "message": body.get("message"),
            }
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode("utf-8")
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            body = {"message": raw_body}
        return {
            "request": index,
            "http_status": exc.code,
            "duration_sec": round(time.perf_counter() - started, 4),
            "error": body.get("error"),
            "message": body.get("message"),
            "details": body.get("details"),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RAG API concurrency limiter behavior.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8001")
    parser.add_argument("--requests", type=int, default=3)
    args = parser.parse_args()

    barrier = Barrier(args.requests)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.requests) as executor:
        futures = [
            executor.submit(_send_query, args.api_url, index, barrier)
            for index in range(1, args.requests + 1)
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, indent=2, sort_keys=True))

    success_results = [result for result in results if result.get("http_status") == 200]
    success_queue_waits = [
        queue_wait
        for queue_wait in (_to_float(result.get("queue_wait_time_sec")) for result in success_results)
        if queue_wait is not None
    ]
    avg_queue_wait = (
        sum(success_queue_waits) / len(success_queue_waits)
        if success_queue_waits
        else 0.0
    )
    server_busy_count = sum(1 for result in results if result.get("error") == "server_busy")
    queue_timeout_count = sum(1 for result in results if result.get("error") == "queue_timeout")
    other_error_count = sum(
        1
        for result in results
        if result.get("http_status") != 200
        and result.get("error") not in {"server_busy", "queue_timeout"}
    )

    print("\nSummary")
    print(f"total_requests={len(results)}")
    print(f"success_count={len(success_results)}")
    print(f"server_busy_count={server_busy_count}")
    print(f"queue_timeout_count={queue_timeout_count}")
    print(f"other_error_count={other_error_count}")
    print(f"avg_success_queue_wait_time_sec={avg_queue_wait:.4f}")


if __name__ == "__main__":
    main()
