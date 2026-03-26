import csv
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Optional

# Make `app/` importable when this script runs from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from full_rag import run_rag_query

MODEL = "gemma3:12b"
NUM_CTX_VALUES = [2048, 4096, 8192]
NUM_BATCH_VALUES = [16, 32, 64]
NUM_THREAD_VALUES = [8, 16 , 32]
QUESTIONS = [
    "Explain GDPR breach notification obligations",
    # "What are the key steps in incident response?",
    # "What does the cybersecurity policy say about password requirements?",
    "Summarize the NIST cybersecurity framework",
]
OUTPUT_CSV = PROJECT_ROOT / "scripts" / "benchmarks" / "config_sweep_results.csv"


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


def get_fieldnames() -> list[str]:
    return [
        "model",
        "num_ctx",
        "num_batch",
        "num_thread",
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


def ensure_csv_with_header(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return

    with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=get_fieldnames())
        writer.writeheader()


def append_row(csv_path: Path, row: dict) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=get_fieldnames())
        writer.writerow(row)


def safe_round(value) -> str | float:
    if isinstance(value, (int, float)):
        return round(float(value), 4)
    return ""


def benchmark_single_query(
    question: str,
    model: str,
    num_ctx: int,
    num_batch: int,
    num_thread: int,
) -> dict:
    vram_before = get_gpu_memory_mb()
    total_start = time.perf_counter()

    retrieval_time_sec = ""
    generation_time_sec = ""
    answer = ""
    retrieved_sources = []
    status = "fail"
    error_message = ""

    try:
        result = run_rag_query(
            question=question,
            model_options={
                "model": model,
                "num_ctx": num_ctx,
                "num_batch": num_batch,
                "num_thread": num_thread,
            },
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

    total_response_time_sec = time.perf_counter() - total_start
    vram_after = get_gpu_memory_mb()

    response_length = len(answer)
    tokens_per_sec = (
        response_length / total_response_time_sec if total_response_time_sec > 0 else 0.0
    )

    vram_delta_mb = ""
    if vram_before is not None and vram_after is not None:
        vram_delta_mb = vram_after - vram_before

    return {
        "model": model,
        "num_ctx": num_ctx,
        "num_batch": num_batch,
        "num_thread": num_thread,
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


def run_config_sweep() -> None:
    ensure_csv_with_header(OUTPUT_CSV)

    configs = list(product(NUM_CTX_VALUES, NUM_BATCH_VALUES, NUM_THREAD_VALUES))
    total_runs = len(configs) * len(QUESTIONS)
    run_idx = 0

    for num_ctx, num_batch, num_thread in configs:
        for question_idx, question in enumerate(QUESTIONS, start=1):
            run_idx += 1
            print(
                f"[{run_idx}/{total_runs}] {MODEL} | "
                f"ctx={num_ctx} batch={num_batch} threads={num_thread} | "
                f"Q{question_idx}/{len(QUESTIONS)}"
            )
            row = benchmark_single_query(
                question=question,
                model=MODEL,
                num_ctx=num_ctx,
                num_batch=num_batch,
                num_thread=num_thread,
            )
            append_row(OUTPUT_CSV, row)

    print(f"Config sweep complete. Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    run_config_sweep()
