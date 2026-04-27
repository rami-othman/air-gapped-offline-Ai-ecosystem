import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from chat_log_store import CHAT_LOG_PATH, load_normalized_chat_logs

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "scripts" / "results" / "finetune"


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _resolve_output_dir(output_dir: str | Path | None = None) -> Path:
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    return DEFAULT_OUTPUT_DIR


def _build_output_paths(output_dir: Path) -> dict[str, Path]:
    timestamp = _timestamp()
    return {
        "jsonl_output_file": output_dir / f"finetune_dataset_{timestamp}.jsonl",
        "latest_jsonl_file": output_dir / "finetune_dataset_latest.jsonl",
        "csv_output_file": output_dir / f"finetune_dataset_{timestamp}.csv",
        "latest_csv_file": output_dir / "finetune_dataset_latest.csv",
        "summary_file": output_dir / f"finetune_dataset_summary_{timestamp}.json",
    }


def _build_finetune_record(question: str, answer: str) -> dict[str, str]:
    return {
        "instruction": question,
        "input": "",
        "output": answer,
    }


def _to_csv_helpful(helpful: bool | None) -> str:
    if helpful is True:
        return "true"
    if helpful is False:
        return "false"
    return "null"


def _to_export_status(helpful: bool | None) -> str:
    if helpful is True:
        return "exported_helpful"
    return "exported_unrated"


def _build_csv_row(item: dict, question: str, answer: str) -> dict[str, str]:
    helpful = item.get("helpful")
    return {
        "id": str(item.get("id", "")).strip(),
        "timestamp": str(item.get("timestamp", "")).strip(),
        "question": question,
        "answer": answer,
        "helpful": _to_csv_helpful(helpful),
        "retrieved_sources": json.dumps(item.get("retrieved_sources", []), ensure_ascii=False),
        "export_status": _to_export_status(helpful),
    }


def build_finetune_dataset(
    interactions: list[dict],
    *,
    only_helpful: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    csv_rows: list[dict[str, str]] = []
    stats = {
        "total_logs_read": len(interactions),
        "exported_count": 0,
        "exported_helpful": 0,
        "exported_unrated": 0,
        "skipped_unrated": 0,
        "skipped_unhelpful": 0,
        "skipped_missing_content": 0,
    }

    for item in interactions:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        helpful = item.get("helpful")

        if not question or not answer:
            stats["skipped_missing_content"] += 1
            continue

        if helpful is False:
            stats["skipped_unhelpful"] += 1
            continue

        if helpful is None and only_helpful:
            stats["skipped_unrated"] += 1
            continue

        records.append(_build_finetune_record(question=question, answer=answer))
        csv_rows.append(_build_csv_row(item=item, question=question, answer=answer))
        stats["exported_count"] += 1
        if helpful is True:
            stats["exported_helpful"] += 1
        else:
            stats["exported_unrated"] += 1

    return records, csv_rows, stats


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as file_obj:
        for record in records:
            file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    fieldnames = [
        "id",
        "timestamp",
        "question",
        "answer",
        "helpful",
        "retrieved_sources",
        "export_status",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_finetune_dataset(
    *,
    output_dir: str | Path | None = None,
    only_helpful: bool = False,
) -> dict[str, Any]:
    active_output_dir = _resolve_output_dir(output_dir)
    active_output_dir.mkdir(parents=True, exist_ok=True)

    interactions, malformed_lines = load_normalized_chat_logs(CHAT_LOG_PATH)
    records, csv_rows, stats = build_finetune_dataset(
        interactions,
        only_helpful=only_helpful,
    )

    paths = _build_output_paths(active_output_dir)
    write_jsonl(records, paths["jsonl_output_file"])
    write_jsonl(records, paths["latest_jsonl_file"])
    write_csv(csv_rows, paths["csv_output_file"])
    write_csv(csv_rows, paths["latest_csv_file"])

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_log_path": str(CHAT_LOG_PATH),
        "only_helpful": only_helpful,
        "total_logs_read": stats["total_logs_read"],
        "exported_count": stats["exported_count"],
        "exported_helpful": stats["exported_helpful"],
        "exported_unrated": stats["exported_unrated"],
        "skipped_unrated": stats["skipped_unrated"],
        "skipped_unhelpful": stats["skipped_unhelpful"],
        "skipped_missing_content": stats["skipped_missing_content"],
        "skipped_malformed_lines": malformed_lines,
        "jsonl_output_file": str(paths["jsonl_output_file"]),
        "csv_output_file": str(paths["csv_output_file"]),
        "latest_jsonl_file": str(paths["latest_jsonl_file"]),
        "latest_csv_file": str(paths["latest_csv_file"]),
    }

    paths["summary_file"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["summary_file"] = str(paths["summary_file"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export approved and unrated chat interactions as a future fine-tuning JSONL dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to scripts/results/finetune.",
    )
    parser.add_argument(
        "--only-helpful",
        action="store_true",
        help="Export only helpful=true records. helpful=false records are always excluded.",
    )
    args = parser.parse_args()

    summary = export_finetune_dataset(
        output_dir=args.output_dir,
        only_helpful=args.only_helpful,
    )

    print("")
    print("Fine-tuning dataset export summary")
    print(f"Only helpful: {summary['only_helpful']}")
    print(f"Total logs read: {summary['total_logs_read']}")
    print(f"Exported records: {summary['exported_count']}")
    print(f"Exported helpful: {summary['exported_helpful']}")
    print(f"Exported unrated: {summary['exported_unrated']}")
    print(f"Skipped unrated: {summary['skipped_unrated']}")
    print(f"Skipped unhelpful: {summary['skipped_unhelpful']}")
    print(f"Skipped missing content: {summary['skipped_missing_content']}")
    print(f"Skipped malformed lines: {summary['skipped_malformed_lines']}")
    print(f"JSONL output file: {summary['jsonl_output_file']}")
    print(f"CSV output file: {summary['csv_output_file']}")
    print(f"Latest JSONL file: {summary['latest_jsonl_file']}")
    print(f"Latest CSV file: {summary['latest_csv_file']}")
    print(f"Summary file: {summary['summary_file']}")


if __name__ == "__main__":
    main()
