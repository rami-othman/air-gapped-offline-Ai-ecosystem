import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Make `app/` importable when this script runs from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from full_rag import GENERAL_GENERATION_MODEL, run_rag_query

MODEL = "gemma3:12b"
QUESTIONS = [
    "Explain GDPR breach notification obligations",
    "What are the key steps in incident response?",
    "What does the cybersecurity policy say about password requirements?",
    "Summarize the NIST cybersecurity framework",
]
OUTPUT_CSV = Path("scripts/benchmarks/benchmark_results.csv")


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


def benchmark_question(run_name: str, question: str) -> dict:
    """Benchmark one full RAG query using app/full_rag.py."""
    vram_before = get_gpu_memory_mb()
    total_start = time.perf_counter()

    response_text = ""
    retrieval_time_sec = None
    generation_time_sec = None
    retrieved_sources = []
    status = "fail"

    try:
        result = run_rag_query(question)
        response_text = result.get("answer", "")
        retrieval_time_sec = result.get("retrieval_time_sec")
        generation_time_sec = result.get("generation_time_sec")
        retrieved_sources = result.get("retrieved_sources", [])
        status = result.get("status", "fail")
    except Exception:
        status = "fail"

    total_response_time_sec = time.perf_counter() - total_start
    vram_after = get_gpu_memory_mb()

    response_length = len(response_text)
    tokens_per_sec = (
        response_length / total_response_time_sec if total_response_time_sec > 0 else 0.0
    )

    vram_delta = None
    if vram_before is not None and vram_after is not None:
        vram_delta = vram_after - vram_before

    return {
        "run_name": run_name,
        "model": MODEL,
        "question": question,
        "total_response_time_sec": round(total_response_time_sec, 4),
        "retrieval_time_sec": round(retrieval_time_sec, 4)
        if isinstance(retrieval_time_sec, (int, float))
        else "",
        "generation_time_sec": round(generation_time_sec, 4)
        if isinstance(generation_time_sec, (int, float))
        else "",
        "response_length": response_length,
        "tokens_per_sec": round(tokens_per_sec, 4),
        "vram_before_mb": vram_before if vram_before is not None else "",
        "vram_after_mb": vram_after if vram_after is not None else "",
        "vram_delta_mb": vram_delta if vram_delta is not None else "",
        "retrieved_sources": "; ".join(retrieved_sources),
        "status": status,
    }


def write_results_csv(rows: list[dict], output_path: Path) -> None:
    """Append benchmark rows to CSV, preserving previous runs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "model",
        "question",
        "total_response_time_sec",
        "retrieval_time_sec",
        "generation_time_sec",
        "response_length",
        "tokens_per_sec",
        "vram_before_mb",
        "vram_after_mb",
        "vram_delta_mb",
        "retrieved_sources",
        "status",
    ]

    file_exists = output_path.exists()
    should_write_header = (not file_exists) or output_path.stat().st_size == 0

    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if GENERAL_GENERATION_MODEL != MODEL:
        print(
            f"Warning: full_rag uses model '{GENERAL_GENERATION_MODEL}', "
            f"but benchmark label is '{MODEL}'."
        )

    run_name = input("Enter benchmark run name (e.g., my-laptop-baseline): ").strip()
    if not run_name:
        run_name = f"run-{int(time.time())}"
        print(f"No name entered. Using: {run_name}")

    rows = []
    total = len(QUESTIONS)

    for idx, question in enumerate(QUESTIONS, start=1):
        print(f"Testing {MODEL} | Question {idx}/{total}...")
        rows.append(benchmark_question(run_name, question))

    write_results_csv(rows, OUTPUT_CSV)
    print(f"Benchmark finished. Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
