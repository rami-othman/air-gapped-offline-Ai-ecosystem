"""Benchmark Ollama and optional vLLM generation backends."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.llm_clients.base import LLMClientError  # noqa: E402
from app.llm_clients.factory import get_llm_client  # noqa: E402

DEFAULT_QUESTIONS_FILE = PROJECT_ROOT / "scripts" / "load_test_questions.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "performance"

RAW_FIELDS = [
    "timestamp",
    "backend",
    "request_index",
    "question",
    "success",
    "model_name",
    "generation_time_sec",
    "client_elapsed_time_sec",
    "answer_length",
    "error",
]

SUMMARY_FIELDS = [
    "timestamp",
    "backend",
    "total_requests",
    "success_count",
    "failed_count",
    "success_rate",
    "avg_generation_time_sec",
    "max_generation_time_sec",
    "min_generation_time_sec",
    "avg_answer_length",
    "errors_by_type",
]


def load_questions(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if isinstance(payload, dict):
        payload = payload.get("questions", [])
    if not isinstance(payload, list):
        raise ValueError("Questions file must contain a JSON list or an object with a questions list.")

    questions = [str(item).strip() for item in payload if str(item).strip()]
    if not questions:
        raise ValueError("Questions file does not contain any non-empty questions.")
    return questions


def _avg(values: list[float]) -> float | str:
    if not values:
        return ""
    return round(statistics.fmean(values), 4)


def _max(values: list[float]) -> float | str:
    if not values:
        return ""
    return round(max(values), 4)


def _min(values: list[float]) -> float | str:
    if not values:
        return ""
    return round(min(values), 4)


def _error_type(message: str) -> str:
    lowered = message.lower()
    if "vllm_model" in lowered or "model is required" in lowered:
        return "missing_model"
    if "connection" in lowered or "refused" in lowered or "timed out" in lowered:
        return "unavailable"
    if not message:
        return ""
    return "generation_error"


def run_backend(
    *,
    backend: str,
    questions: list[str],
    requests_per_backend: int,
    timeout: int,
    skip_unavailable: bool,
    model_override: str | None,
    timestamp: str,
) -> list[dict[str, Any]]:
    print(f"Benchmarking backend={backend} requests={requests_per_backend}")
    try:
        client = get_llm_client(backend, timeout_seconds=timeout)
    except Exception as exc:  # noqa: BLE001
        return [
            {
                "timestamp": timestamp,
                "backend": backend,
                "request_index": 1,
                "question": "",
                "success": False,
                "model_name": model_override or "",
                "generation_time_sec": "",
                "client_elapsed_time_sec": 0.0,
                "answer_length": 0,
                "error": str(exc),
            }
        ]

    rows: list[dict[str, Any]] = []
    for request_index in range(1, requests_per_backend + 1):
        question = questions[(request_index - 1) % len(questions)]
        started = time.perf_counter()
        try:
            result = client.generate(question, model=model_override)
            elapsed = time.perf_counter() - started
            text = result.get("text", "") or ""
            row = {
                "timestamp": timestamp,
                "backend": backend,
                "request_index": request_index,
                "question": question,
                "success": True,
                "model_name": result.get("model", model_override or ""),
                "generation_time_sec": round(float(result.get("generation_time_sec", elapsed)), 4),
                "client_elapsed_time_sec": round(elapsed, 4),
                "answer_length": len(text),
                "error": "",
            }
            print(f"  {backend} request {request_index}: success in {elapsed:.2f}s")
        except LLMClientError as exc:
            elapsed = time.perf_counter() - started
            row = {
                "timestamp": timestamp,
                "backend": backend,
                "request_index": request_index,
                "question": question,
                "success": False,
                "model_name": model_override or "",
                "generation_time_sec": "",
                "client_elapsed_time_sec": round(elapsed, 4),
                "answer_length": 0,
                "error": str(exc),
            }
            print(f"  {backend} request {request_index}: failed ({exc})")
            rows.append(row)
            if skip_unavailable and _error_type(str(exc)) in {"missing_model", "unavailable"}:
                print(f"  {backend} unavailable; skipping remaining requests.")
                break
            continue

        rows.append(row)

    return rows


def summarize_backend(timestamp: str, backend: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    success_rows = [row for row in rows if row.get("success")]
    generation_times = [
        float(row["generation_time_sec"])
        for row in success_rows
        if isinstance(row.get("generation_time_sec"), (int, float))
    ]
    answer_lengths = [
        float(row["answer_length"])
        for row in success_rows
        if isinstance(row.get("answer_length"), (int, float))
    ]
    errors: dict[str, int] = {}
    for row in rows:
        if row.get("success"):
            continue
        error_type = _error_type(str(row.get("error", "")))
        errors[error_type] = errors.get(error_type, 0) + 1

    return {
        "timestamp": timestamp,
        "backend": backend,
        "total_requests": total,
        "success_count": len(success_rows),
        "failed_count": total - len(success_rows),
        "success_rate": round(len(success_rows) / total, 4) if total else 0.0,
        "avg_generation_time_sec": _avg(generation_times),
        "max_generation_time_sec": _max(generation_times),
        "min_generation_time_sec": _min(generation_times),
        "avg_answer_length": _avg(answer_lengths),
        "errors_by_type": errors,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Ollama and optional vLLM generation backends.")
    parser.add_argument("--backends", nargs="+", default=["ollama"], choices=["ollama", "vllm"])
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--requests-per-backend", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--skip-unavailable", action="store_true")
    parser.add_argument("--model-ollama", default=None)
    parser.add_argument("--model-vllm", default=None)
    return parser.parse_args()


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("")
    print("Backend | Requests | Success | Failed | Avg Gen | Max Gen | Avg Length | Errors")
    print("--------+----------+---------+--------+---------+---------+------------+-------")
    for row in summary_rows:
        print(
            f"{row['backend']:<7} | {row['total_requests']:<8} | {row['success_count']:<7} | "
            f"{row['failed_count']:<6} | {row['avg_generation_time_sec']!s:<7} | "
            f"{row['max_generation_time_sec']!s:<7} | {row['avg_answer_length']!s:<10} | "
            f"{row['errors_by_type']}"
        )


def main() -> int:
    args = parse_args()
    questions = load_questions(args.questions_file)
    requests_per_backend = max(1, args.requests_per_backend)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for backend in args.backends:
        model_override = args.model_ollama if backend == "ollama" else args.model_vllm
        backend_rows = run_backend(
            backend=backend,
            questions=questions,
            requests_per_backend=requests_per_backend,
            timeout=args.timeout,
            skip_unavailable=args.skip_unavailable,
            model_override=model_override,
            timestamp=timestamp,
        )
        raw_rows.extend(backend_rows)
        summary_rows.append(summarize_backend(timestamp, backend, backend_rows))

    output_dir = args.output_dir
    raw_json_path = output_dir / f"llm_backend_benchmark_{timestamp}_raw.json"
    raw_csv_path = output_dir / f"llm_backend_benchmark_{timestamp}_raw.csv"
    summary_json_path = output_dir / f"llm_backend_benchmark_{timestamp}_summary.json"
    summary_csv_path = output_dir / f"llm_backend_benchmark_{timestamp}_summary.csv"
    latest_summary_json_path = output_dir / "llm_backend_benchmark_latest_summary.json"
    latest_summary_csv_path = output_dir / "llm_backend_benchmark_latest_summary.csv"

    write_json(raw_json_path, raw_rows)
    write_csv(raw_csv_path, raw_rows, RAW_FIELDS)
    write_json(summary_json_path, summary_rows)
    write_csv(summary_csv_path, summary_rows, SUMMARY_FIELDS)
    write_json(latest_summary_json_path, summary_rows)
    write_csv(latest_summary_csv_path, summary_rows, SUMMARY_FIELDS)

    print_summary(summary_rows)
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
