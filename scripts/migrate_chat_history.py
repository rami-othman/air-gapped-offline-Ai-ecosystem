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


def main() -> None:
    MIGRATIONS_DIR.mkdir(parents=True, exist_ok=True)
    timestamped_output_path = _build_timestamped_output_path()

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
    LATEST_OUTPUT_PATH.write_text(serialized_payload, encoding="utf-8")

    print(f"Loaded interactions: {len(interactions)}")
    print(f"Migrated interactions: {len(migrated_items)}")
    print(f"Skipped missing content: {skipped_missing_content}")
    print(f"Skipped malformed lines: {malformed_lines}")
    print(f"Output saved to: {timestamped_output_path}")
    print(f"Latest output saved to: {LATEST_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
