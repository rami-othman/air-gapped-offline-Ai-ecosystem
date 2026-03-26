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

from full_rag import run_rag_query

# Keep this list easy to edit based on your locally available Ollama tags.
MODEL_VARIANTS = [
    "gemma3:12b",
    "gemma3:12b-q4",
    "gemma3:12b-q5",
    "gemma3:12b-q8",
]

QUESTIONS = [
    "Explain GDPR breach notification obligations",
    "Summarize the NIST cybersecurity framework",
]

OUTPUT_CSV = PROJECT_ROOT / "scripts" / "benchmarks" / "quantization_results.csv"


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


def csv_fieldnames() -> list[str]:
    return [
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
        "error_message",
    ]


def ensure_csv_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames())
        writer.writeheader()


def append_result_row(path: Path, row: dict) -> None:
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames())
        writer.writerow(row)


def safe_round(value) -> str | float:
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    return ""


def benchmark_single(model_name: str, question: str) -> dict:
    """
    Benchmark one full RAG query for a single model variant.
    Uses default model runtime settings (no custom generation options).
    """
    vram_before = get_gpu_memory_mb()
    start_total = time.perf_counter()

    answer = ""
    retrieved_sources = []
    retrieval_time_sec = ""
    generation_time_sec = ""
    status = "fail"
    error_message = ""

    try:
        result = run_rag_query(question=question, model_name=model_name, model_options=None)
        answer = result.get("answer", "") or ""
        retrieved_sources = result.get("retrieved_sources", []) or []
        retrieval_time_sec = safe_round(result.get("retrieval_time_sec"))
        generation_time_sec = safe_round(result.get("generation_time_sec"))
        status = result.get("status", "success")
        if status != "success":
            error_message = result.get("error_message", "") or "RAG query returned non-success status"
    except Exception as exc:
        status = "fail"
        error_message = str(exc)

    total_response_time_sec = time.perf_counter() - start_total
    vram_after = get_gpu_memory_mb()

    response_length = len(answer)
    tokens_per_sec = (
        response_length / total_response_time_sec if total_response_time_sec > 0 else 0.0
    )

    vram_delta_mb = ""
    if vram_before is not None and vram_after is not None:
        vram_delta_mb = vram_after - vram_before

    return {
        "model": model_name,
        "question": question,
        "total_response_time_sec": round(total_response_time_sec, 4),
        "retrieval_time_sec": retrieval_time_sec,
        "generation_time_sec": generation_time_sec,
        "response_length": response_length,
        "tokens_per_sec": round(tokens_per_sec, 4),
        "vram_before_mb": vram_before if vram_before is not None else "",
        "vram_after_mb": vram_after if vram_after is not None else "",
        "vram_delta_mb": vram_delta_mb,
        "retrieved_sources": "; ".join(str(src) for src in retrieved_sources),
        "status": status,
        "error_message": error_message,
    }


def run_quantization_benchmark() -> None:
    ensure_csv_header(OUTPUT_CSV)

    total_runs = len(MODEL_VARIANTS) * len(QUESTIONS)
    current_run = 0

    for model_name in MODEL_VARIANTS:
        for question_index, question in enumerate(QUESTIONS, start=1):
            current_run += 1
            print(
                f"[{current_run}/{total_runs}] Testing {model_name} | "
                f"Question {question_index}/{len(QUESTIONS)}"
            )
            row = benchmark_single(model_name=model_name, question=question)
            append_result_row(OUTPUT_CSV, row)

    print(f"Quantization benchmark complete. Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_quantization_benchmark()
