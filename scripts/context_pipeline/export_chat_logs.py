import csv
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from chat_log_store import CHAT_LOG_FIELDS, CHAT_LOG_PATH, load_normalized_chat_logs

LOG_FILE_PATH = CHAT_LOG_PATH
EXPORT_DIR = PROJECT_ROOT / "data" / "exports"
EXPORT_HISTORY_DIR = EXPORT_DIR / "history"
EXPORT_JSON_PATH = EXPORT_DIR / "chat_logs.json"
EXPORT_CSV_PATH = EXPORT_DIR / "chat_logs.csv"
LATEST_JSON_PATH = EXPORT_DIR / "chat_logs_latest.json"
LATEST_CSV_PATH = EXPORT_DIR / "chat_logs_latest.csv"


def _build_timestamped_export_paths() -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        EXPORT_HISTORY_DIR / f"chat_logs_{timestamp}.json",
        EXPORT_HISTORY_DIR / f"chat_logs_{timestamp}.csv",
    )

def export_json(rows: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(rows, output_file, ensure_ascii=False, indent=2)


def export_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = list(CHAT_LOG_FIELDS)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            helpful = row.get("helpful")
            if helpful is True:
                helpful_str = "true"
            elif helpful is False:
                helpful_str = "false"
            else:
                helpful_str = "null"

            writer.writerow(
                {
                    "id": row.get("id", ""),
                    "timestamp": row.get("timestamp", ""),
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "retrieved_sources": "; ".join(
                        str(src) for src in row.get("retrieved_sources", [])
                    ),
                    "helpful": helpful_str,
                }
            )


def main() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    normalized_rows, malformed_lines = load_normalized_chat_logs(LOG_FILE_PATH)

    timestamped_json_path, timestamped_csv_path = _build_timestamped_export_paths()

    export_json(normalized_rows, EXPORT_JSON_PATH)
    export_csv(normalized_rows, EXPORT_CSV_PATH)
    export_json(normalized_rows, LATEST_JSON_PATH)
    export_csv(normalized_rows, LATEST_CSV_PATH)
    export_json(normalized_rows, timestamped_json_path)
    export_csv(normalized_rows, timestamped_csv_path)

    print(f"Loaded interactions: {len(normalized_rows)}")
    print(f"Skipped malformed lines: {malformed_lines}")
    print(f"JSON export saved to: {EXPORT_JSON_PATH}")
    print(f"CSV export saved to: {EXPORT_CSV_PATH}")
    print(f"Latest JSON export saved to: {LATEST_JSON_PATH}")
    print(f"Latest CSV export saved to: {LATEST_CSV_PATH}")
    print(f"Timestamped JSON export saved to: {timestamped_json_path}")
    print(f"Timestamped CSV export saved to: {timestamped_csv_path}")


if __name__ == "__main__":
    main()
