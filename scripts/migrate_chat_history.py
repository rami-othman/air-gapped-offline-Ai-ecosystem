import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from chat_log_store import CHAT_LOG_PATH, compute_reuse_weight, load_normalized_chat_logs

MIGRATIONS_DIR = PROJECT_ROOT / "scripts" / "results" / "migrations"
LATEST_OUTPUT_PATH = MIGRATIONS_DIR / "chat_history_migrated_latest.json"


def _build_timestamped_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return MIGRATIONS_DIR / f"chat_history_migrated_{timestamp}.json"


def _build_migrated_text(question: str, answer: str) -> str:
    return f"Question:\n{question}\n\nAnswer:\n{answer}"


def migrate_interactions(interactions: list[dict]) -> tuple[list[dict], int]:
    migrated_items: list[dict] = []
    skipped_missing_content = 0

    for item in interactions:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            skipped_missing_content += 1
            continue

        interaction_id = str(item.get("id", "")).strip()
        helpful = item.get("helpful")
        migrated_items.append(
            {
                "doc_id": f"chat_history_{interaction_id}",
                "text": _build_migrated_text(question=question, answer=answer),
                "metadata": {
                    "source": "chat_history",
                    "source_document": "chat_history",
                    "interaction_id": interaction_id,
                    "helpful": helpful,
                    "reuse_weight": compute_reuse_weight(helpful),
                    "timestamp": str(item.get("timestamp", "")).strip(),
                    "retrieved_sources": item.get("retrieved_sources", []),
                },
            }
        )

    return migrated_items, skipped_missing_content


def run_migration(output_dir: str | Path | None = None, write_latest: bool = True) -> dict:
    active_output_dir = Path(output_dir).expanduser().resolve() if output_dir else MIGRATIONS_DIR
    active_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_path = active_output_dir / f"chat_history_migrated_{timestamp}.json"
    latest_output_path = active_output_dir / "chat_history_migrated_latest.json"

    interactions, malformed_lines = load_normalized_chat_logs(CHAT_LOG_PATH)
    migrated_items, skipped_missing_content = migrate_interactions(interactions)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_log_path": str(CHAT_LOG_PATH),
        "stats": {
            "total_loaded": len(interactions),
            "migrated": len(migrated_items),
            "skipped_missing_content": skipped_missing_content,
            "skipped_malformed_lines": malformed_lines,
        },
        "items": migrated_items,
    }

    serialized_payload = json.dumps(payload, ensure_ascii=False, indent=2)

    timestamped_output_path.write_text(serialized_payload, encoding="utf-8")
    if write_latest:
        latest_output_path.write_text(serialized_payload, encoding="utf-8")

    return {
        "status": "success",
        "operation": "migrate_chat_history",
        "items_migrated": len(migrated_items),
        "output_file": str(timestamped_output_path),
        "latest_file": str(latest_output_path) if write_latest else None,
        "total_loaded": len(interactions),
        "skipped_missing_content": skipped_missing_content,
        "skipped_malformed_lines": malformed_lines,
    }


def main() -> None:
    result = run_migration(output_dir=MIGRATIONS_DIR, write_latest=True)

    print(f"Loaded interactions: {result['total_loaded']}")
    print(f"Migrated interactions: {result['items_migrated']}")
    print(f"Skipped missing content: {result['skipped_missing_content']}")
    print(f"Skipped malformed lines: {result['skipped_malformed_lines']}")
    print(f"Output saved to: {result['output_file']}")
    print(f"Latest output saved to: {result['latest_file']}")


if __name__ == "__main__":
    main()
