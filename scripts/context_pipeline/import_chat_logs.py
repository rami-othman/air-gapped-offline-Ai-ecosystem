import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from full_rag import add_document

RAW_LOG_PATH = PROJECT_ROOT / "data" / "chat_logs.jsonl"
EXPORTED_JSON_PATH = PROJECT_ROOT / "data" / "exports" / "chat_logs.json"


def _load_from_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _load_from_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
            except json.JSONDecodeError:
                print(f"[Warning] Skipping malformed JSONL line {line_number}.")
    return rows


def load_interactions() -> list[dict]:
    """
    Prefer exported JSON for stable import snapshots.
    Fallback to raw JSONL if export file is not present.
    """
    exported_rows = _load_from_json(EXPORTED_JSON_PATH)
    if exported_rows:
        print(f"Using exported log file: {EXPORTED_JSON_PATH}")
        return exported_rows

    raw_rows = _load_from_jsonl(RAW_LOG_PATH)
    if raw_rows:
        print(f"Using raw log file: {RAW_LOG_PATH}")
    return raw_rows


def _build_chat_doc_id(timestamp: str, question: str, answer: str) -> str:
    """
    Stable deterministic ID so re-import updates same record instead of duplicating.
    """
    base = f"{timestamp}|{question}|{answer}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"chat_history_{digest}"


def import_interactions(interactions: list[dict]) -> tuple[int, int]:
    imported = 0
    failed = 0

    for idx, item in enumerate(interactions, start=1):
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        timestamp = str(item.get("timestamp", "")).strip()

        if not question or not answer:
            print(f"[Warning] Skipping row {idx}: missing question or answer.")
            failed += 1
            continue

        doc_text = f"Q: {question}\nA: {answer}"
        doc_id = _build_chat_doc_id(timestamp, question, answer)
        metadata = {
            "source": "chat_history",
            "source_document": "chat_history",
            "source_type": "chat",
            "timestamp": timestamp,
            "question": question,
        }

        try:
            add_document(doc_id=doc_id, text=doc_text, metadata=metadata)
            imported += 1
        except Exception as exc:
            print(f"[Error] Failed to import row {idx}: {exc}")
            failed += 1

    return imported, failed


def main() -> None:
    interactions = load_interactions()
    if not interactions:
        print("No chat logs found to import.")
        print(f"Checked: {EXPORTED_JSON_PATH} and {RAW_LOG_PATH}")
        return

    imported, failed = import_interactions(interactions)
    print(f"Import completed. Imported: {imported}, Failed: {failed}")


if __name__ == "__main__":
    main()
