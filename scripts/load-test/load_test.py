import csv
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Make `app/` importable when this script runs from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from full_rag import run_rag_query

MODEL_NAME = "gemma3:12b"
CONCURRENCY_LEVELS = [1, 2, 5, 10 , 12]
QUESTIONS = [
    "Explain GDPR breach notification obligations",
    "What are the key steps in incident response?",
    "What does the cybersecurity policy say about password requirements?",
    "Summarize the NIST cybersecurity framework",
]

DETAILS_OUTPUT_CSV = PROJECT_ROOT / "scripts" / "load-test" / "load_test_results.csv"
SUMMARY_OUTPUT_CSV = PROJECT_ROOT / "scripts" / "load-test" / "load_test_summary.csv"


def get_gpu_memory_mb() -> Optional[int]:
    """Return current GPU memory usage (MB) from nvidia-smi, or None if unavailable."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if not lines:
            return None
        return int(lines[0])
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        return None


def safe_round(value) -> str | float:
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    return ""


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_single_request(
    request_id: int,
    concurrency_level: int,
    question: str,
    run_timestamp: str,
    shared_results: list[dict],
    results_lock: threading.Lock,
    start_event: threading.Event,
) -> None:
    """
    Execute one RAG request and append a benchmark row in a thread-safe way.
    Threads wait on start_event so requests start nearly simultaneously.
    """
    start_event.wait()

    request_start = time.perf_counter()
    answer = ""
    retrieval_time_sec = ""
    generation_time_sec = ""
    retrieved_sources = []
    status = "fail"
    error_message = ""

    try:
        result = run_rag_query(
            question=question,
            model_name=MODEL_NAME,
            model_options=None,  # Keep default runtime options for load testing.
        )
        answer = result.get("answer", "") or ""
        retrieval_time_sec = safe_round(result.get("retrieval_time_sec"))
        generation_time_sec = safe_round(result.get("generation_time_sec"))
        retrieved_sources = result.get("retrieved_sources", []) or []
        status = result.get("status", "success")
        if status != "success":
            error_message = result.get("error_message", "") or "RAG query returned non-success status"
    except Exception as exc:
        status = "fail"
        error_message = str(exc)

    total_response_time_sec = time.perf_counter() - request_start
    response_length = len(answer)
    tokens_per_sec = (
        response_length / total_response_time_sec if total_response_time_sec > 0 else 0.0
    )

    row = {
        "test_run_timestamp": run_timestamp,
        "concurrency_level": concurrency_level,
        "request_id": request_id,
        "model": MODEL_NAME,
        "question": question,
        "total_response_time_sec": round(total_response_time_sec, 4),
        "retrieval_time_sec": retrieval_time_sec,
        "generation_time_sec": generation_time_sec,
        "response_length": response_length,
        "tokens_per_sec": round(tokens_per_sec, 4),
        "retrieved_sources": "; ".join(str(src) for src in retrieved_sources),
        "status": status,
        "error_message": error_message,
    }

    with results_lock:
        shared_results.append(row)

    print(
        f"[request {request_id}/{concurrency_level}] "
        f"completed in {total_response_time_sec:.2f}s (status={status})"
    )


def _avg(values: list[float]) -> str | float:
    if not values:
        return ""
    return round(sum(values) / len(values), 4)


def build_summary_row(
    run_timestamp: str,
    concurrency_level: int,
    request_rows: list[dict],
    vram_before_mb: Optional[int],
    vram_after_mb: Optional[int],
) -> dict:
    success_count = sum(1 for row in request_rows if row.get("status") == "success")
    fail_count = len(request_rows) - success_count

    total_times = [
        float(row["total_response_time_sec"])
        for row in request_rows
        if isinstance(row.get("total_response_time_sec"), (int, float))
    ]
    retrieval_times = [
        float(row["retrieval_time_sec"])
        for row in request_rows
        if isinstance(row.get("retrieval_time_sec"), (int, float))
    ]
    generation_times = [
        float(row["generation_time_sec"])
        for row in request_rows
        if isinstance(row.get("generation_time_sec"), (int, float))
    ]

    vram_delta_mb = ""
    if vram_before_mb is not None and vram_after_mb is not None:
        vram_delta_mb = vram_after_mb - vram_before_mb

    return {
        "test_run_timestamp": run_timestamp,
        "concurrency_level": concurrency_level,
        "total_requests": len(request_rows),
        "success_count": success_count,
        "fail_count": fail_count,
        "avg_total_response_time_sec": _avg(total_times),
        "avg_retrieval_time_sec": _avg(retrieval_times),
        "avg_generation_time_sec": _avg(generation_times),
        "min_total_response_time_sec": round(min(total_times), 4) if total_times else "",
        "max_total_response_time_sec": round(max(total_times), 4) if total_times else "",
        "vram_before_mb": vram_before_mb if vram_before_mb is not None else "",
        "vram_after_mb": vram_after_mb if vram_after_mb is not None else "",
        "vram_delta_mb": vram_delta_mb,
    }


def run_concurrency_batch(concurrency_level: int, run_timestamp: str) -> tuple[list[dict], dict]:
    print(f"Starting concurrency level {concurrency_level}...")

    results: list[dict] = []
    results_lock = threading.Lock()
    start_event = threading.Event()
    threads: list[threading.Thread] = []

    vram_before_mb = get_gpu_memory_mb()

    for request_id in range(1, concurrency_level + 1):
        question = QUESTIONS[(request_id - 1) % len(QUESTIONS)]
        thread = threading.Thread(
            target=_run_single_request,
            args=(
                request_id,
                concurrency_level,
                question,
                run_timestamp,
                results,
                results_lock,
                start_event,
            ),
            daemon=False,
        )
        threads.append(thread)
        thread.start()

    # Release all workers so they fire requests at nearly the same time.
    start_event.set()

    for thread in threads:
        thread.join()

    vram_after_mb = get_gpu_memory_mb()

    # Keep results stable and easy to scan.
    results.sort(key=lambda row: int(row["request_id"]))

    summary_row = build_summary_row(
        run_timestamp=run_timestamp,
        concurrency_level=concurrency_level,
        request_rows=results,
        vram_before_mb=vram_before_mb,
        vram_after_mb=vram_after_mb,
    )

    print(f"Finished concurrency level {concurrency_level}")
    return results, summary_row


def run_load_test() -> None:
    run_timestamp = datetime.now().isoformat(timespec="seconds")
    all_request_rows: list[dict] = []
    all_summary_rows: list[dict] = []

    for concurrency_level in CONCURRENCY_LEVELS:
        request_rows, summary_row = run_concurrency_batch(
            concurrency_level=concurrency_level,
            run_timestamp=run_timestamp,
        )
        all_request_rows.extend(request_rows)
        all_summary_rows.append(summary_row)

    details_fields = [
        "test_run_timestamp",
        "concurrency_level",
        "request_id",
        "model",
        "question",
        "total_response_time_sec",
        "retrieval_time_sec",
        "generation_time_sec",
        "response_length",
        "tokens_per_sec",
        "retrieved_sources",
        "status",
        "error_message",
    ]
    summary_fields = [
        "test_run_timestamp",
        "concurrency_level",
        "total_requests",
        "success_count",
        "fail_count",
        "avg_total_response_time_sec",
        "avg_retrieval_time_sec",
        "avg_generation_time_sec",
        "min_total_response_time_sec",
        "max_total_response_time_sec",
        "vram_before_mb",
        "vram_after_mb",
        "vram_delta_mb",
    ]

    write_csv(DETAILS_OUTPUT_CSV, details_fields, all_request_rows)
    write_csv(SUMMARY_OUTPUT_CSV, summary_fields, all_summary_rows)

    print(f"Detailed results saved to: {DETAILS_OUTPUT_CSV}")
    print(f"Summary results saved to: {SUMMARY_OUTPUT_CSV}")


if __name__ == "__main__":
    run_load_test()
