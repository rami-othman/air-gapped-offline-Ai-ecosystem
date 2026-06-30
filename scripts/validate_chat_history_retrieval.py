import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = PROJECT_ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from full_rag import _build_context_and_sources, ask_llm, retrieve

VALIDATION_DIR = PROJECT_ROOT / "scripts" / "results" / "validation"
LATEST_OUTPUT_PATH = VALIDATION_DIR / "chat_history_validation_latest.json"

DEFAULT_VALIDATION_QUESTIONS = [
    "What did the assistant previously explain about GDPR breach notification?",
    "What did the assistant previously say about incident response steps?",
    "What did we previously discuss about cybersecurity policy requirements?",
    "Summarize the earlier answer about access controls.",
    "What was the prior guidance about compliance obligations?",
]


def _build_timestamped_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return VALIDATION_DIR / f"chat_history_validation_{timestamp}.json"


def _load_questions(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_VALIDATION_QUESTIONS

    input_path = Path(path).expanduser().resolve()
    with input_path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)

    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]

    if isinstance(payload, dict) and isinstance(payload.get("questions"), list):
        return [str(item).strip() for item in payload["questions"] if str(item).strip()]

    raise ValueError("Questions file must be a JSON list or an object with a questions list.")


def _source_document(metadata: dict[str, Any] | None) -> str:
    return str((metadata or {}).get("source_document", "unknown")).strip() or "unknown"


def _is_chat_history_metadata(metadata: dict[str, Any] | None) -> bool:
    return _source_document(metadata).lower() == "chat_history"


def _normalize_helpful(value: Any) -> bool | None | str:
    if value in {True, False}:
        return value
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        if normalized in {"null", "none", ""}:
            return None
    return str(value)


def _normalize_reuse_weight(value: Any) -> float | str | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _chat_history_metadata_details(metadatas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for metadata in metadatas:
        if not _is_chat_history_metadata(metadata):
            continue

        details.append(
            {
                "interaction_id": metadata.get("interaction_id"),
                "helpful": _normalize_helpful(metadata.get("helpful")),
                "reuse_weight": _normalize_reuse_weight(metadata.get("reuse_weight")),
            }
        )

    return details


def validate_question(question: str, top_k: int | None = None) -> dict[str, Any]:
    retrieval_start = time.perf_counter()
    docs, metadatas = retrieve(question, top_k=top_k, include_chat_history=True)
    retrieval_time_sec = time.perf_counter() - retrieval_start

    retrieved_sources = [_source_document(metadata) for metadata in metadatas]
    retrieved_sources = list(dict.fromkeys(retrieved_sources))
    chat_history_hit = "chat_history" in {source.lower() for source in retrieved_sources}
    chat_history_metadata = _chat_history_metadata_details(metadatas)

    if not docs:
        return {
            "question": question,
            "chat_history_hit": False,
            "retrieved_sources": [],
            "chat_history_metadata": [],
            "answer": "No relevant context found in the vector database.",
            "retrieval_time_sec": retrieval_time_sec,
            "generation_time_sec": 0.0,
            "total_time_sec": retrieval_time_sec,
            "status": "fail",
        }

    context, sources = _build_context_and_sources(docs, metadatas)

    generation_start = time.perf_counter()
    answer = ask_llm(context, question)
    generation_time_sec = time.perf_counter() - generation_start
    total_time_sec = retrieval_time_sec + generation_time_sec

    return {
        "question": question,
        "chat_history_hit": chat_history_hit,
        "retrieved_sources": sources,
        "chat_history_metadata": chat_history_metadata,
        "answer": answer,
        "retrieval_time_sec": retrieval_time_sec,
        "generation_time_sec": generation_time_sec,
        "total_time_sec": total_time_sec,
        "status": "success",
    }


def run_validation(questions: list[str], top_k: int | None = None) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for index, question in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question}")
        try:
            result = validate_question(question=question, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            result = {
                "question": question,
                "chat_history_hit": False,
                "retrieved_sources": [],
                "chat_history_metadata": [],
                "answer": "",
                "retrieval_time_sec": 0.0,
                "generation_time_sec": 0.0,
                "total_time_sec": 0.0,
                "status": "error",
                "error": str(exc),
            }
            print(f"[Error] Validation failed for question {index}: {exc}")
        results.append(result)

    total_questions = len(results)
    chat_history_hits = sum(1 for result in results if result.get("chat_history_hit") is True)
    hit_rate = (chat_history_hits / total_questions) if total_questions else 0.0

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": {
            "total_questions": total_questions,
            "chat_history_hits": chat_history_hits,
            "chat_history_hit_rate": hit_rate,
        },
        "items": results,
    }


def save_validation_result(payload: dict[str, Any]) -> tuple[Path, Path]:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _build_timestamped_output_path()
    serialized_payload = json.dumps(payload, ensure_ascii=False, indent=2)
    output_path.write_text(serialized_payload, encoding="utf-8")
    LATEST_OUTPUT_PATH.write_text(serialized_payload, encoding="utf-8")
    return output_path, LATEST_OUTPUT_PATH


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate that migrated chat history can be retrieved by the RAG pipeline.",
    )
    parser.add_argument(
        "--questions-file",
        default=None,
        help="Optional JSON list of questions, or an object with a questions list.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional retrieval top_k override.",
    )
    args = parser.parse_args()

    questions = _load_questions(args.questions_file)
    payload = run_validation(questions=questions, top_k=args.top_k)
    output_path, latest_output_path = save_validation_result(payload)

    stats = payload["stats"]
    print("")
    print("Chat history retrieval validation summary")
    print(f"Total questions tested: {stats['total_questions']}")
    print(f"Chat history hits: {stats['chat_history_hits']}")
    print(f"Chat history hit rate: {stats['chat_history_hit_rate']:.0%}")
    print(f"Output saved to: {output_path}")
    print(f"Latest output saved to: {latest_output_path}")


if __name__ == "__main__":
    main()
