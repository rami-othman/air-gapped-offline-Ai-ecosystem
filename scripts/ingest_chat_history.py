import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import CHROMA_COLLECTION

DEFAULT_MIGRATED_PATH = (
    PROJECT_ROOT / "scripts" / "results" / "migrations" / "chat_history_migrated_latest.json"
)
LEGACY_MIGRATED_PATH = PROJECT_ROOT / "scripts" / "results" / "chat_history_migrated.json"


def _resolve_input_path(input_file: str | None) -> Path:
    if input_file:
        return Path(input_file).expanduser().resolve()
    if DEFAULT_MIGRATED_PATH.exists():
        return DEFAULT_MIGRATED_PATH
    return LEGACY_MIGRATED_PATH


def _load_migrated_items(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Migrated chat history file not found: {path}")

    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if isinstance(payload, dict):
        items = payload.get("items", [])
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return []

    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    return []


def _get_add_document():
    from full_rag import add_document

    return add_document


def _to_chroma_metadata_value(value: Any) -> str | int | float | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    if value is None:
        return "null"
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        key_text = str(key).strip()
        if not key_text:
            continue
        sanitized[key_text] = _to_chroma_metadata_value(value)
    return sanitized


def _normalize_record(record: dict) -> tuple[str | None, str | None, dict[str, Any] | None]:
    doc_id = str(record.get("doc_id", "")).strip()
    text = str(record.get("text", "")).strip()
    metadata = record.get("metadata", {})

    if not doc_id or not text or not isinstance(metadata, dict):
        return None, None, None

    normalized_metadata = _sanitize_metadata(metadata)
    if "source_document" not in normalized_metadata:
        normalized_metadata["source_document"] = "chat_history"
    if "source" not in normalized_metadata:
        normalized_metadata["source"] = "chat_history"

    return doc_id, text, normalized_metadata


def ingest_migrated_items(items: list[dict], add_document_fn=None) -> tuple[int, int]:
    active_add_document = add_document_fn or _get_add_document()
    upserted = 0
    skipped = 0
    seen_ids: set[str] = set()

    for index, record in enumerate(items, start=1):
        doc_id, text, metadata = _normalize_record(record)
        if not doc_id or not text or metadata is None:
            skipped += 1
            print(f"[Warning] Skipping row {index}: invalid doc_id/text/metadata.")
            continue

        if doc_id in seen_ids:
            skipped += 1
            print(f"[Warning] Skipping row {index}: duplicate doc_id in input ({doc_id}).")
            continue
        seen_ids.add(doc_id)

        try:
            active_add_document(doc_id=doc_id, text=text, metadata=metadata)
            upserted += 1
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"[Error] Failed to ingest row {index} ({doc_id}): {exc}")

    return upserted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest migrated chat history into ChromaDB using existing RAG embedding/upsert path.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Path to migrated chat history JSON. Defaults to latest migration output.",
    )
    args = parser.parse_args()

    input_path = _resolve_input_path(args.input_file)
    items = _load_migrated_items(input_path)
    loaded = len(items)
    upserted, skipped = ingest_migrated_items(items)

    print("")
    print("Chat history ingestion summary")
    print(f"Input file: {input_path}")
    print(f"Target collection: {CHROMA_COLLECTION}")
    print(f"Records loaded: {loaded}")
    print(f"Records ingested/upserted: {upserted}")
    print(f"Records skipped: {skipped}")


if __name__ == "__main__":
    main()