import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_FILE_PATH = PROJECT_ROOT / "data" / "chat_logs.jsonl"
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
EXPORT_JSON_PATH = EXPORT_DIR / "chat_logs.json"
EXPORT_CSV_PATH = EXPORT_DIR / "chat_logs.csv"


def load_jsonl_logs(log_path: Path) -> list[dict]:
    """Load chat interactions from JSONL, skipping malformed lines safely."""
    if not log_path.exists():
        return []

    rows: list[dict] = []
    with log_path.open("r", encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
            except json.JSONDecodeError:
                print(f"[Warning] Skipping malformed JSON at line {line_number}.")
    return rows


def normalize_rows(rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for row in rows:
        sources = row.get("retrieved_sources", [])
        if not isinstance(sources, list):
            sources = [str(sources)] if sources else []
        normalized.append(
            {
                "timestamp": row.get("timestamp", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "retrieved_sources": sources,
            }
        )
    return normalized


def export_json(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(rows, output_file, ensure_ascii=False, indent=2)


def export_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = ["timestamp", "question", "answer", "retrieved_sources"]
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "timestamp": row.get("timestamp", ""),
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "retrieved_sources": "; ".join(
                        str(src) for src in row.get("retrieved_sources", [])
                    ),
                }
            )


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl_logs(LOG_FILE_PATH)
    normalized_rows = normalize_rows(rows)

    export_json(normalized_rows, EXPORT_JSON_PATH)
    export_csv(normalized_rows, EXPORT_CSV_PATH)

    print(f"Loaded interactions: {len(normalized_rows)}")
    print(f"JSON export saved to: {EXPORT_JSON_PATH}")
    print(f"CSV export saved to: {EXPORT_CSV_PATH}")


if __name__ == "__main__":
    main()
