"""API-level load test for the RAG query endpoint."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
from typing import Any
from urllib import error, parse, request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "http://127.0.0.1:8001/api/v1/rag/query"
DEFAULT_QUESTIONS_FILE = PROJECT_ROOT / "scripts" / "load_test_questions.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "performance"
SMOKE_CONCURRENCY = [1, 2]
FULL_CONCURRENCY = [1, 2, 5, 10, 12]

RAW_FIELDS = [
    "timestamp",
    "concurrency_level",
    "request_index",
    "question",
    "status_code",
    "success",
    "error",
    "message",
    "client_elapsed_time_sec",
    "api_total_time_sec",
    "retrieval_time_sec",
    "generation_time_sec",
    "queue_wait_time_sec",
    "active_llm_requests",
    "waiting_rag_requests",
    "cache_hit",
    "cache_type",
    "model_name",
    "top_k",
    "prompt_version",
    "index_version",
    "answer_length",
    "sources_count",
    "session_id",
]

SUMMARY_FIELDS = [
    "timestamp",
    "concurrency_level",
    "total_requests",
    "success_count",
    "failed_count",
    "success_rate",
    "server_busy_count",
    "queue_timeout_count",
    "other_error_count",
    "avg_client_elapsed_time_sec",
    "max_client_elapsed_time_sec",
    "avg_api_total_time_sec",
    "avg_retrieval_time_sec",
    "avg_generation_time_sec",
    "avg_queue_wait_time_sec",
    "max_queue_wait_time_sec",
    "cache_hit_count",
    "cache_hit_rate",
    "response_cache_hit_count",
    "retrieval_cache_hit_count",
    "avg_answer_length",
    "avg_sources_count",
]


def _round(value: Any) -> float | str:
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    return ""


def _to_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _avg(values: list[float]) -> float | str:
    if not values:
        return ""
    return round(statistics.fmean(values), 4)


def _max(values: list[float]) -> float | str:
    if not values:
        return ""
    return round(max(values), 4)


def load_questions(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if isinstance(payload, dict):
        payload = payload.get("questions", [])

    if not isinstance(payload, list):
        raise ValueError("Questions file must contain a JSON list or an object with a 'questions' list.")

    questions = [str(item).strip() for item in payload if str(item).strip()]
    if not questions:
        raise ValueError("Questions file does not contain any non-empty questions.")
    return questions


def derive_health_url(query_url: str) -> str:
    parsed = parse.urlparse(query_url)
    return parse.urlunparse((parsed.scheme, parsed.netloc, "/health", "", "", ""))


def parse_json_object(body: str) -> dict[str, Any]:
    if not body:
        return {}

    parsed = json.loads(body)
    if isinstance(parsed, dict):
        return parsed
    return {"data": parsed}


def http_json(method: str, url: str, payload: dict[str, Any] | None, timeout: int) -> tuple[int, dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return response.status, parse_json_object(body)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            response_payload = parse_json_object(body)
        except json.JSONDecodeError:
            response_payload = {"message": body}
        return exc.code, response_payload


def check_health(query_url: str, timeout: int) -> bool:
    health_url = derive_health_url(query_url)
    try:
        status_code, payload = http_json("GET", health_url, None, min(timeout, 30))
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: health check failed for {health_url}: {exc}")
        return False

    print(f"Health check {health_url}: status_code={status_code} body={payload}")
    return 200 <= status_code < 300


def run_warmup() -> None:
    warmup_script = PROJECT_ROOT / "scripts" / "warmup_ollama.py"
    if not warmup_script.exists():
        print("Warning: warm-up requested but scripts/warmup_ollama.py was not found.")
        return

    print("Running Ollama warm-up before load test...")
    try:
        completed = subprocess.run(
            [sys.executable, str(warmup_script)],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            timeout=600,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: warm-up could not be started: {exc}")
        return

    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip())
    if completed.returncode != 0:
        print(f"Warning: warm-up exited with code {completed.returncode}; continuing load test.")


def run_single_request(
    *,
    url: str,
    timeout: int,
    concurrency_level: int,
    request_index: int,
    question: str,
    session_id: str,
    start_event: threading.Event,
) -> dict[str, Any]:
    payload = {
        "question": question,
        "session_id": session_id,
    }
    start_event.wait()
    started = time.perf_counter()
    timestamp = datetime.now().isoformat(timespec="seconds")

    status_code = 0
    body: dict[str, Any] = {}
    transport_error = ""
    try:
        status_code, body = http_json("POST", url, payload, timeout)
    except Exception as exc:  # noqa: BLE001
        transport_error = str(exc)

    elapsed = time.perf_counter() - started
    success = 200 <= status_code < 300 and body.get("status") == "success"
    answer = str(body.get("answer", "") or "")
    sources = body.get("sources", [])
    if not isinstance(sources, list):
        sources = []

    return {
        "timestamp": timestamp,
        "concurrency_level": concurrency_level,
        "request_index": request_index,
        "question": question,
        "status_code": status_code,
        "success": success,
        "error": body.get("error") or (transport_error if transport_error else ""),
        "message": body.get("message", ""),
        "client_elapsed_time_sec": round(elapsed, 4),
        "api_total_time_sec": _round(body.get("total_time_sec")),
        "retrieval_time_sec": _round(body.get("retrieval_time_sec")),
        "generation_time_sec": _round(body.get("generation_time_sec")),
        "queue_wait_time_sec": _round(body.get("queue_wait_time_sec")),
        "active_llm_requests": body.get("active_llm_requests", ""),
        "waiting_rag_requests": body.get("waiting_rag_requests", ""),
        "cache_hit": bool(body.get("cache_hit", False)),
        "cache_type": body.get("cache_type") or "",
        "model_name": body.get("model_name", ""),
        "top_k": body.get("top_k", ""),
        "prompt_version": body.get("prompt_version", ""),
        "index_version": body.get("index_version", ""),
        "answer_length": len(answer),
        "sources_count": len(sources),
        "session_id": session_id,
    }


def run_level(
    *,
    url: str,
    questions: list[str],
    concurrency_level: int,
    requests_per_level: int,
    timeout: int,
    session_prefix: str,
) -> list[dict[str, Any]]:
    print(f"Running concurrency level {concurrency_level} with {requests_per_level} requests...")
    rows: list[dict[str, Any]] = []
    max_workers = max(1, concurrency_level)
    start_event = threading.Event()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for request_index in range(1, requests_per_level + 1):
            question = questions[(request_index - 1) % len(questions)]
            session_id = f"{session_prefix}-c{concurrency_level}-r{request_index}"
            futures.append(
                executor.submit(
                    run_single_request,
                    url=url,
                    timeout=timeout,
                    concurrency_level=concurrency_level,
                    request_index=request_index,
                    question=question,
                    session_id=session_id,
                    start_event=start_event,
                )
            )

        start_event.set()
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            status = "success" if row["success"] else row["error"] or row["status_code"]
            print(
                f"  request {row['request_index']}/{requests_per_level}: "
                f"{status} in {row['client_elapsed_time_sec']:.2f}s"
            )

    rows.sort(key=lambda item: int(item["request_index"]))
    return rows


def summarize_level(timestamp: str, concurrency_level: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    success_rows = [row for row in rows if row.get("success")]
    failed_count = total - len(success_rows)
    server_busy_count = sum(1 for row in rows if row.get("error") == "server_busy")
    queue_timeout_count = sum(1 for row in rows if row.get("error") == "queue_timeout")
    other_error_count = sum(
        1
        for row in rows
        if not row.get("success") and row.get("error") not in {"server_busy", "queue_timeout"}
    )

    client_times = [_to_float(row.get("client_elapsed_time_sec")) for row in rows]
    api_total_times = [_to_float(row.get("api_total_time_sec")) for row in success_rows]
    retrieval_times = [_to_float(row.get("retrieval_time_sec")) for row in success_rows]
    generation_times = [_to_float(row.get("generation_time_sec")) for row in success_rows]
    queue_wait_times = [_to_float(row.get("queue_wait_time_sec")) for row in success_rows]
    answer_lengths = [_to_float(row.get("answer_length")) for row in success_rows]
    source_counts = [_to_float(row.get("sources_count")) for row in success_rows]

    client_times_clean = [value for value in client_times if value is not None]
    api_total_clean = [value for value in api_total_times if value is not None]
    retrieval_clean = [value for value in retrieval_times if value is not None]
    generation_clean = [value for value in generation_times if value is not None]
    queue_wait_clean = [value for value in queue_wait_times if value is not None]
    answer_length_clean = [value for value in answer_lengths if value is not None]
    source_count_clean = [value for value in source_counts if value is not None]

    cache_hit_count = sum(1 for row in success_rows if row.get("cache_hit"))
    response_cache_hit_count = sum(1 for row in success_rows if row.get("cache_type") == "response")
    retrieval_cache_hit_count = sum(1 for row in success_rows if row.get("cache_type") == "retrieval")

    return {
        "timestamp": timestamp,
        "concurrency_level": concurrency_level,
        "total_requests": total,
        "success_count": len(success_rows),
        "failed_count": failed_count,
        "success_rate": round(len(success_rows) / total, 4) if total else 0.0,
        "server_busy_count": server_busy_count,
        "queue_timeout_count": queue_timeout_count,
        "other_error_count": other_error_count,
        "avg_client_elapsed_time_sec": _avg(client_times_clean),
        "max_client_elapsed_time_sec": _max(client_times_clean),
        "avg_api_total_time_sec": _avg(api_total_clean),
        "avg_retrieval_time_sec": _avg(retrieval_clean),
        "avg_generation_time_sec": _avg(generation_clean),
        "avg_queue_wait_time_sec": _avg(queue_wait_clean),
        "max_queue_wait_time_sec": _max(queue_wait_clean),
        "cache_hit_count": cache_hit_count,
        "cache_hit_rate": round(cache_hit_count / len(success_rows), 4) if success_rows else 0.0,
        "response_cache_hit_count": response_cache_hit_count,
        "retrieval_cache_hit_count": retrieval_cache_hit_count,
        "avg_answer_length": _avg(answer_length_clean),
        "avg_sources_count": _avg(source_count_clean),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_summary_table(summary_rows: list[dict[str, Any]]) -> None:
    headers = [
        "Concurrency",
        "Requests",
        "Success",
        "Failed",
        "Avg Time",
        "Max Time",
        "Avg Queue",
        "Cache Hit Rate",
        "server_busy",
        "queue_timeout",
    ]
    table_rows = []
    for row in summary_rows:
        table_rows.append(
            [
                row["concurrency_level"],
                row["total_requests"],
                row["success_count"],
                row["failed_count"],
                row["avg_client_elapsed_time_sec"],
                row["max_client_elapsed_time_sec"],
                row["avg_queue_wait_time_sec"],
                row["cache_hit_rate"],
                row["server_busy_count"],
                row["queue_timeout_count"],
            ]
        )

    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in table_rows))
        for index, header in enumerate(headers)
    ]
    print("")
    print(" | ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in table_rows:
        print(" | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load test the RAG API query endpoint.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--concurrency-levels", nargs="+", type=int, default=None)
    parser.add_argument("--requests-per-level", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--session-prefix", default="load-test")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--preset", choices=["smoke", "full"], default="smoke")
    return parser.parse_args()


def resolve_plan(args: argparse.Namespace) -> tuple[list[int], int, int]:
    if args.concurrency_levels is not None:
        concurrency_levels = args.concurrency_levels
    elif args.preset == "full":
        concurrency_levels = FULL_CONCURRENCY
    else:
        concurrency_levels = SMOKE_CONCURRENCY

    default_requests = 5 if args.preset == "full" else 2
    default_timeout = 300 if args.preset == "full" else 180
    requests_per_level = args.requests_per_level or default_requests
    timeout = args.timeout or default_timeout

    return concurrency_levels, requests_per_level, timeout


def main() -> int:
    args = parse_args()
    concurrency_levels, requests_per_level, timeout = resolve_plan(args)

    try:
        questions = load_questions(args.questions_file)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not load questions file: {exc}")
        return 2

    health_ok = check_health(args.url, timeout)
    if args.warmup:
        run_warmup()
        if not health_ok:
            print("Warning: API health check failed before warm-up; continuing because load requests may still fail cleanly.")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    print(
        f"Starting RAG API load test. levels={concurrency_levels} "
        f"requests_per_level={requests_per_level} timeout={timeout}s"
    )
    for concurrency_level in concurrency_levels:
        level_rows = run_level(
            url=args.url,
            questions=questions,
            concurrency_level=concurrency_level,
            requests_per_level=requests_per_level,
            timeout=timeout,
            session_prefix=f"{args.session_prefix}-{run_timestamp}",
        )
        all_raw_rows.extend(level_rows)
        summary_rows.append(summarize_level(run_timestamp, concurrency_level, level_rows))

    output_dir = args.output_dir
    raw_json_path = output_dir / f"load_test_{run_timestamp}_raw.json"
    summary_json_path = output_dir / f"load_test_{run_timestamp}_summary.json"
    raw_csv_path = output_dir / f"load_test_{run_timestamp}_raw.csv"
    summary_csv_path = output_dir / f"load_test_{run_timestamp}_summary.csv"
    latest_summary_json_path = output_dir / "load_test_latest_summary.json"
    latest_summary_csv_path = output_dir / "load_test_latest_summary.csv"

    write_json(raw_json_path, all_raw_rows)
    write_json(summary_json_path, summary_rows)
    write_csv(raw_csv_path, all_raw_rows, RAW_FIELDS)
    write_csv(summary_csv_path, summary_rows, SUMMARY_FIELDS)
    write_json(latest_summary_json_path, summary_rows)
    write_csv(latest_summary_csv_path, summary_rows, SUMMARY_FIELDS)

    print_summary_table(summary_rows)
    print("")
    print(f"Raw JSON: {raw_json_path}")
    print(f"Raw CSV: {raw_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"Latest summary JSON: {latest_summary_json_path}")
    print(f"Latest summary CSV: {latest_summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
