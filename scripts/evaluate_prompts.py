import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# Make `app/` importable when this script runs from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from full_rag import DEFAULT_MODEL_OPTIONS, build_prompt, run_rag_query

MODEL_NAME = "gemma3:12b"
FIXED_MODEL_OPTIONS = dict(DEFAULT_MODEL_OPTIONS)
DEFAULT_DATASET_PATH = PROJECT_ROOT / "scripts" / "eval_dataset.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "scripts" / "results"
DEFAULT_TOP_K = 3


def build_quality_focused_prompt(context: str, question: str) -> str:
    return f"""
You are an AI assistant for [your system name].

Your job:
- Answer the user's question accurately.
- Use the provided context as the main source.
- Be clear, helpful, and direct.

Rules:
- Do not make up facts.
- If the context does not contain the answer, say that clearly.
- Do not claim certainty when uncertain.
- Keep the answer focused on the question.
- Do not repeat the context unnecessarily.
- If the user asks outside the provided context, say what is missing.

Answer style:
- Use clear professional language.
- Keep the answer concise unless the user asks for detail.
- When possible, give the final answer first.
- If useful, use short bullet points.

Provided context:
{context}

User question:
{question}

Write the best possible answer.
"""


def build_short_optimized_prompt(context: str, question: str) -> str:
    return f"""
Answer using only the context.

Rules:
- No outside knowledge.
- If context is insufficient, say what is missing.
- Be concise and accurate.

Format:
Answer:
<answer>

Sources:
- <file_name>

Context:
{context}

Question:
{question}
"""


PROMPT_VARIANTS: list[dict[str, object]] = [
    {
        "name": "baseline",
        "description": "Current production prompt from app/full_rag.py",
        "builder": build_prompt,
    },
    {
        "name": "quality_focused",
        "description": "Longer prompt that emphasizes completeness, synthesis, and grounded detail.",
        "builder": build_quality_focused_prompt,
    },
    {
        "name": "short_optimized",
        "description": "Shorter prompt that keeps the same grounding rules with fewer instruction tokens.",
        "builder": build_short_optimized_prompt,
    },
]


def load_dataset(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_text(value: str) -> str:
    lowered = (value or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def ensure_results_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def serialize_sources(sources: list[str]) -> str:
    return "; ".join(str(item) for item in sources)


def build_phrase_matchers(requirement) -> tuple[str, list[str]]:
    if isinstance(requirement, str):
        return requirement, [requirement]

    if isinstance(requirement, dict):
        label = requirement.get("label") or requirement.get("any", [""])[0]
        variants = requirement.get("any", [])
        return str(label), [str(variant) for variant in variants if str(variant).strip()]

    raise TypeError(f"Unsupported must_include requirement: {requirement!r}")


def score_must_include(answer: str, requirements: list) -> tuple[int, int, float, list[str], list[str]]:
    normalized_answer = normalize_text(answer)
    matched_labels: list[str] = []
    missing_labels: list[str] = []

    for requirement in requirements:
        label, variants = build_phrase_matchers(requirement)
        normalized_variants = [normalize_text(variant) for variant in variants]
        is_match = any(variant and variant in normalized_answer for variant in normalized_variants)
        if is_match:
            matched_labels.append(label)
        else:
            missing_labels.append(label)

    total = len(requirements)
    matched = len(matched_labels)
    score = round(matched / total, 4) if total else 0.0
    return total, matched, score, matched_labels, missing_labels


def evaluate_single_question(
    item: dict,
    prompt_name: str,
    prompt_builder: Callable[[str, str], str],
    top_k: int,
) -> dict:
    question = item["question"]
    expected_source = item["source_file"]

    row = {
        "prompt_name": prompt_name,
        "question_id": item["id"],
        "question": question,
        "answer": "",
        "retrieved_sources": [],
        "expected_source": expected_source,
        "correct_source": 0,
        "must_include_total": 0,
        "must_include_matched": 0,
        "must_include_score": 0.0,
        "retrieval_time_sec": 0.0,
        "generation_time_sec": 0.0,
        "total_time_sec": 0.0,
        "difficulty": item.get("difficulty", ""),
        "expected_answer": item.get("expected_answer", ""),
        "matched_must_include": [],
        "missing_must_include": [],
        "manual_notes": "",
        "status": "fail",
        "error_message": "",
    }

    try:
        result = run_rag_query(
            question=question,
            model_name=MODEL_NAME,
            model_options=FIXED_MODEL_OPTIONS,
            top_k=top_k,
            prompt_builder=prompt_builder,
        )
    except Exception as exc:  # noqa: BLE001
        row["error_message"] = str(exc)
        return row

    answer = result.get("answer", "") or ""
    retrieved_sources = result.get("retrieved_sources", []) or []
    retrieval_time_sec = float(result.get("retrieval_time_sec", 0.0))
    generation_time_sec = float(result.get("generation_time_sec", 0.0))
    total_time_sec = float(result.get("total_time_sec", retrieval_time_sec + generation_time_sec))

    must_include_total, must_include_matched, must_include_score, matched_labels, missing_labels = (
        score_must_include(answer, item.get("must_include", []))
    )

    row.update(
        {
            "answer": answer,
            "retrieved_sources": retrieved_sources,
            "correct_source": int(expected_source in retrieved_sources),
            "must_include_total": must_include_total,
            "must_include_matched": must_include_matched,
            "must_include_score": must_include_score,
            "retrieval_time_sec": round(retrieval_time_sec, 4),
            "generation_time_sec": round(generation_time_sec, 4),
            "total_time_sec": round(total_time_sec, 4),
            "matched_must_include": matched_labels,
            "missing_must_include": missing_labels,
            "status": result.get("status", "success"),
            "error_message": result.get("error_message", ""),
        }
    )
    return row


def summarize_results(rows: list[dict]) -> list[dict]:
    prompt_names = []
    for row in rows:
        prompt_name = row["prompt_name"]
        if prompt_name not in prompt_names:
            prompt_names.append(prompt_name)

    summary_rows = []
    for prompt_name in prompt_names:
        prompt_rows = [row for row in rows if row["prompt_name"] == prompt_name]
        count = len(prompt_rows) or 1

        summary_rows.append(
            {
                "prompt_name": prompt_name,
                "questions_evaluated": len(prompt_rows),
                "average_retrieval_time_sec": round(
                    sum(float(row["retrieval_time_sec"]) for row in prompt_rows) / count,
                    4,
                ),
                "average_generation_time_sec": round(
                    sum(float(row["generation_time_sec"]) for row in prompt_rows) / count,
                    4,
                ),
                "average_total_time_sec": round(
                    sum(float(row["total_time_sec"]) for row in prompt_rows) / count,
                    4,
                ),
                "average_must_include_score": round(
                    sum(float(row["must_include_score"]) for row in prompt_rows) / count,
                    4,
                ),
                "source_accuracy_rate": round(
                    sum(int(row["correct_source"]) for row in prompt_rows) / count,
                    4,
                ),
            }
        )

    return summary_rows


def write_results_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "prompt_name",
        "question_id",
        "question",
        "answer",
        "retrieved_sources",
        "expected_source",
        "correct_source",
        "must_include_total",
        "must_include_matched",
        "must_include_score",
        "retrieval_time_sec",
        "generation_time_sec",
        "total_time_sec",
        "difficulty",
        "manual_notes",
        "status",
        "error_message",
    ]

    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {field: row.get(field, "") for field in fieldnames}
            csv_row["retrieved_sources"] = serialize_sources(row.get("retrieved_sources", []))
            writer.writerow(csv_row)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "prompt_name",
        "questions_evaluated",
        "average_retrieval_time_sec",
        "average_generation_time_sec",
        "average_total_time_sec",
        "average_must_include_score",
        "source_accuracy_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_results_json(
    path: Path,
    dataset_path: Path,
    top_k: int,
    rows: list[dict],
    summary_rows: list[dict],
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "model": MODEL_NAME,
        "runtime_options": FIXED_MODEL_OPTIONS,
        "top_k": top_k,
        "prompt_variants": [
            {
                "name": variant["name"],
                "description": variant["description"],
            }
            for variant in PROMPT_VARIANTS
        ],
        "results": rows,
        "summary": summary_rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt variants for the existing RAG pipeline using a fixed model and runtime config."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory for CSV/JSON outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved chunks per query. Use 3 or 5 for the planned Week 3 comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    results_dir = Path(args.results_dir)
    dataset = load_dataset(dataset_path)
    ensure_results_dir(results_dir)

    print(f"Prompt evaluation dataset: {dataset_path}")
    print(f"Model: {MODEL_NAME}")
    print(f"Runtime options: {FIXED_MODEL_OPTIONS}")
    print(f"TOP_K: {args.top_k}")

    results: list[dict] = []
    total_runs = len(dataset) * len(PROMPT_VARIANTS)
    run_index = 0

    for item in dataset:
        for variant in PROMPT_VARIANTS:
            run_index += 1
            prompt_name = str(variant["name"])
            print(
                f"[{run_index}/{total_runs}] "
                f"{prompt_name} | {item['id']} | top_k={args.top_k}"
            )
            row = evaluate_single_question(
                item=item,
                prompt_name=prompt_name,
                prompt_builder=variant["builder"],
                top_k=args.top_k,
            )
            results.append(row)

    summary_rows = summarize_results(results)

    results_csv_path = results_dir / "prompt_eval_results.csv"
    results_json_path = results_dir / "prompt_eval_results.json"
    summary_csv_path = results_dir / "prompt_eval_summary.csv"

    write_results_csv(results_csv_path, results)
    write_results_json(results_json_path, dataset_path, args.top_k, results, summary_rows)
    write_summary_csv(summary_csv_path, summary_rows)

    print(f"Saved detailed CSV to: {results_csv_path}")
    print(f"Saved detailed JSON to: {results_json_path}")
    print(f"Saved summary CSV to: {summary_csv_path}")


if __name__ == "__main__":
    main()
