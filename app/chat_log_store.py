import json
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

CHAT_LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "chat_logs.jsonl"
CHAT_LOG_FIELDS = (
    "id",
    "timestamp",
    "question",
    "answer",
    "retrieved_sources",
    "helpful",
)


def _new_interaction_id() -> str:
    return f"chat_{uuid.uuid4().hex}"


def _normalize_sources(sources: list[str] | None) -> list[str]:
    if not isinstance(sources, list):
        return []
    return [str(source) for source in sources]


def _normalize_helpful(helpful: Any) -> bool | None:
    if helpful in {True, False}:
        return helpful
    if helpful is None:
        return None
    if isinstance(helpful, str):
        normalized = helpful.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _build_legacy_interaction_id(
    *,
    timestamp: str,
    question: str,
    answer: str,
    line_number: int,
) -> str:
    base = f"{timestamp}|{question}|{answer}|{line_number}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:20]
    return f"chat_legacy_{digest}"


def build_interaction_payload(question: str, answer: str, sources: list[str]) -> dict:
    return {
        "id": _new_interaction_id(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "retrieved_sources": _normalize_sources(sources),
        "helpful": None,
    }


def normalize_interaction_payload(payload: dict, line_number: int) -> dict:
    timestamp = str(payload.get("timestamp", "")).strip()
    question = str(payload.get("question", "")).strip()
    answer = str(payload.get("answer", "")).strip()
    interaction_id = str(payload.get("id", "")).strip()
    if not interaction_id:
        interaction_id = _build_legacy_interaction_id(
            timestamp=timestamp,
            question=question,
            answer=answer,
            line_number=line_number,
        )

    return {
        "id": interaction_id,
        "timestamp": timestamp,
        "question": question,
        "answer": answer,
        "retrieved_sources": _normalize_sources(payload.get("retrieved_sources", [])),
        "helpful": _normalize_helpful(payload.get("helpful")),
    }


def load_raw_chat_logs(log_path: Path | None = None) -> tuple[list[dict], int]:
    target_path = log_path or CHAT_LOG_PATH
    if not target_path.exists():
        return [], 0

    rows: list[dict] = []
    malformed_lines = 0
    with target_path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                malformed_lines += 1
                continue

            if isinstance(item, dict):
                rows.append(item)
            else:
                malformed_lines += 1

    return rows, malformed_lines


def load_normalized_chat_logs(log_path: Path | None = None) -> tuple[list[dict], int]:
    rows, malformed_lines = load_raw_chat_logs(log_path=log_path)
    normalized_rows = [
        normalize_interaction_payload(item, line_number=index)
        for index, item in enumerate(rows, start=1)
    ]
    return normalized_rows, malformed_lines


def compute_reuse_weight(helpful: bool | None) -> float:
    if helpful is True:
        return 1.0
    if helpful is False:
        return 0.2
    return 0.5


def append_interaction(payload: dict) -> None:
    CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHAT_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def save_interaction(question: str, answer: str, sources: list[str]) -> dict:
    payload = build_interaction_payload(
        question=question,
        answer=answer,
        sources=sources,
    )
    append_interaction(payload)
    return payload


def update_interaction_helpful(interaction_id: str, helpful: bool) -> bool:
    if helpful not in {True, False}:
        raise ValueError("helpful must be true or false.")

    if not CHAT_LOG_PATH.exists():
        return False

    updated = False
    rewritten_lines: list[str] = []

    with CHAT_LOG_PATH.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            stripped = line.strip()
            if not stripped:
                rewritten_lines.append(line if line.endswith("\n") else f"{line}\n")
                continue

            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                rewritten_lines.append(line if line.endswith("\n") else f"{line}\n")
                continue

            if (
                not updated
                and isinstance(payload, dict)
                and payload.get("id") == interaction_id
            ):
                payload["helpful"] = helpful
                updated = True

            rewritten_lines.append(json.dumps(payload, ensure_ascii=False) + "\n")

    if not updated:
        return False

    with CHAT_LOG_PATH.open("w", encoding="utf-8") as log_file:
        log_file.writelines(rewritten_lines)

    return True
