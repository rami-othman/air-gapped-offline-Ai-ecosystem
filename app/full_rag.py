import hashlib
import logging
import re
import sys
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import chromadb
import requests

try:
    from .config import (
        CHROMA_COLLECTION,
        CHROMA_HOST,
        CHROMA_PORT,
        EMBEDDING_MODEL,
        GENERAL_GENERATION_MODEL,
        MODEL_NUM_BATCH,
        MODEL_NUM_CTX,
        MODEL_NUM_PREDICT,
        MODEL_NUM_THREAD,
        MODEL_TEMPERATURE,
        OLLAMA_KEEP_ALIVE,
        OLLAMA_BASE_URL,
        RAG_INDEX_VERSION,
        RAG_RETRIEVAL_CACHE_ENABLED,
        TOP_K,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from config import (
        CHROMA_COLLECTION,
        CHROMA_HOST,
        CHROMA_PORT,
        EMBEDDING_MODEL,
        GENERAL_GENERATION_MODEL,
        MODEL_NUM_BATCH,
        MODEL_NUM_CTX,
        MODEL_NUM_PREDICT,
        MODEL_NUM_THREAD,
        MODEL_TEMPERATURE,
        OLLAMA_KEEP_ALIVE,
        OLLAMA_BASE_URL,
        RAG_INDEX_VERSION,
        RAG_RETRIEVAL_CACHE_ENABLED,
        TOP_K,
    )

try:
    from .cache_store import retrieval_cache
except ImportError:  # pragma: no cover - script execution fallback
    from cache_store import retrieval_cache

try:
    from .chat_log_store import save_interaction as persist_chat_interaction
except ImportError:  # pragma: no cover - script execution fallback
    from chat_log_store import save_interaction as persist_chat_interaction

logger = logging.getLogger(__name__)

# Connect to Chroma
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)
# print("[Chroma][full_rag] collections:", chroma_client.list_collections())
# print("[Chroma][full_rag] current count:", collection.count())
# print("[Chroma][full_rag] sample:", collection.peek(limit=2))
PromptBuilder = Callable[[str, str], str]

DEFAULT_MODEL_OPTIONS = {
    "num_ctx": MODEL_NUM_CTX,
    "num_batch": MODEL_NUM_BATCH,
    "num_thread": MODEL_NUM_THREAD,
    "temperature": MODEL_TEMPERATURE,
    "num_predict": MODEL_NUM_PREDICT,
}


def generate_embedding(text):
    clean_text = text.replace("\n", " ").strip()

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "prompt": clean_text,
        },
        timeout=120,
    )
    response.raise_for_status()

    data = response.json()

    if "embedding" not in data or len(data["embedding"]) == 0:
        print("Embedding error:", data)
        raise Exception("Embedding generation failed")

    return data["embedding"]


def add_document(doc_id, text, metadata=None):
    embedding = generate_embedding(text)

    if metadata is not None:
        collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
        )
    else:
        collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
        )


def delete_document_chunks(source_document):
    existing = collection.get(where={"source_document": source_document}, include=[])
    ids = existing.get("ids", [])

    if not ids:
        return 0

    collection.delete(ids=ids)
    return len(ids)


def _is_chat_history_candidate(meta: Mapping[str, Any] | None) -> bool:
    return str((meta or {}).get("source_document", "")).strip().lower() == "chat_history"


def _normalize_helpful_value(value: Any) -> bool | None:
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
    return None


def _chat_reuse_weight(meta: Mapping[str, Any] | None) -> float:
    """
    Parse reuse_weight safely from metadata.
    Falls back to helpful->weight mapping for backward compatibility.
    """
    raw_weight = (meta or {}).get("reuse_weight")
    try:
        if raw_weight is not None:
            return float(raw_weight)
    except (TypeError, ValueError):
        pass

    helpful = _normalize_helpful_value((meta or {}).get("helpful"))
    if helpful is True:
        return 1.0
    if helpful is False:
        return 0.2
    return 0.5


def _is_document_candidate(meta: Mapping[str, Any] | None) -> bool:
    return str((meta or {}).get("source_type", "")).strip().lower() == "document"


def _rerank_chat_history_candidates(
    docs: Sequence[str],
    metas: Sequence[Mapping[str, Any] | None],
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Keep normal retrieval order, but reorder chat_history candidates among themselves
    so higher reuse_weight appears earlier than lower reuse_weight.
    """
    if not docs:
        return [], []

    aligned_metas: list[dict[str, Any]] = []
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) else None
        aligned_metas.append(dict(meta) if isinstance(meta, Mapping) else {})

    chat_positions: list[int] = []
    chat_items: list[tuple[int, str, dict[str, Any], float]] = []

    for idx, doc in enumerate(docs):
        meta = aligned_metas[idx]
        if _is_chat_history_candidate(meta):
            chat_positions.append(idx)
            chat_items.append((idx, doc, meta, _chat_reuse_weight(meta)))

    if len(chat_items) <= 1:
        return list(docs), aligned_metas

    # Deterministic rerank: higher weight first, then original retrieval order.
    sorted_chat_items = sorted(chat_items, key=lambda item: (-item[3], item[0]))
    reranked_docs = list(docs)
    reranked_metas = list(aligned_metas)

    for slot, (_, doc, meta, _) in zip(chat_positions, sorted_chat_items):
        reranked_docs[slot] = doc
        reranked_metas[slot] = meta

    return reranked_docs, reranked_metas


def _filter_document_results(
    docs: Sequence[str],
    metas: Sequence[Mapping[str, Any] | None],
    limit: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    filtered_docs = []
    filtered_metas = []

    for index, doc in enumerate(docs):
        meta = metas[index] if index < len(metas) else None
        if not _is_document_candidate(meta):
            continue

        filtered_docs.append(doc)
        filtered_metas.append(dict(meta) if isinstance(meta, Mapping) else {})

        if len(filtered_docs) >= limit:
            break

    return filtered_docs, filtered_metas


def _normalize_retrieval_cache_query(query: str) -> str:
    return re.sub(r"\s+", " ", (query or "").strip().lower())


def _build_retrieval_cache_key(query: str, top_k: int, include_chat_history: bool) -> str:
    key_parts = [
        _normalize_retrieval_cache_query(query),
        str(top_k),
        CHROMA_COLLECTION,
        EMBEDDING_MODEL,
        RAG_INDEX_VERSION,
        "with_chat" if include_chat_history else "documents_only",
    ]
    raw_key = "\n".join(key_parts)
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def _retrieve_from_chroma(query, top_k=None, include_chat_history=False):
    n_results = top_k if top_k is not None else TOP_K
    query_embedding = generate_embedding(query)

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": ["documents", "metadatas"],
    }
    if not include_chat_history:
        query_kwargs["where"] = {"source_type": "document"}

    try:
        results = collection.query(**query_kwargs)
    except Exception:
        if include_chat_history:
            raise

        logger.exception("Document metadata filter failed; falling back to manual filtering.")
        fallback_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max(n_results * 4, n_results),
            include=["documents", "metadatas"],
        )
        fallback_docs = fallback_results.get("documents", [])
        fallback_metas = fallback_results.get("metadatas", [])
        if not fallback_docs:
            return [], []
        return _filter_document_results(
            fallback_docs[0],
            fallback_metas[0] if fallback_metas else [],
            n_results,
        )

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    if not documents:
        return [], []

    retrieved_docs = documents[0]
    retrieved_metas = metadatas[0] if metadatas else []
    if not include_chat_history:
        return _filter_document_results(retrieved_docs, retrieved_metas, n_results)

    return _rerank_chat_history_candidates(retrieved_docs, retrieved_metas)


def retrieve_with_cache_info(query, top_k=None, include_chat_history=False):
    n_results = top_k if top_k is not None else TOP_K

    if not RAG_RETRIEVAL_CACHE_ENABLED:
        logger.debug("Retrieval cache disabled.")
        docs, metas = _retrieve_from_chroma(
            query,
            top_k=n_results,
            include_chat_history=include_chat_history,
        )
        return docs, metas, {"cache_hit": False, "cache_type": None}

    cache_key = _build_retrieval_cache_key(query, n_results, include_chat_history)
    cached_result = retrieval_cache.get(cache_key)
    if cached_result is not None:
        logger.info("Retrieval cache hit. top_k=%d index_version=%s", n_results, RAG_INDEX_VERSION)
        return (
            cached_result.get("documents", []),
            cached_result.get("metadatas", []),
            {"cache_hit": True, "cache_type": "retrieval"},
        )

    logger.info("Retrieval cache miss. top_k=%d index_version=%s", n_results, RAG_INDEX_VERSION)
    docs, metas = _retrieve_from_chroma(
        query,
        top_k=n_results,
        include_chat_history=include_chat_history,
    )
    retrieval_cache.set(
        cache_key,
        {
            "documents": docs,
            "metadatas": metas,
        },
    )
    return docs, metas, {"cache_hit": False, "cache_type": None}


def retrieve(query, top_k=None, include_chat_history=False):
    docs, metas, _ = retrieve_with_cache_info(
        query,
        top_k=top_k,
        include_chat_history=include_chat_history,
    )
    return docs, metas


# def build_prompt(context, question):
#     return f"""
# You are an AI assistant answering questions using only the provided context.

# Rules:
# - Use only the context below.
# - Do not use outside knowledge.
# - If the answer is not fully supported by the context, say what is missing.
# - Use only the relevant context parts.
# - Keep the answer clear, accurate, and concise.
# - Use bullet points only when they improve clarity.

# After the answer, list only the source document names that directly support the answer.

# Format:
# Answer:
# <your answer>

# Sources:
# - <file_name>

# Context:
# {context}

# Question:
# {question}
# """

def build_prompt(context, question):
    return f"""
[ROLE / IDENTITY]
You are a precise, reliable, and detail-oriented retrieval-augmented assistant for enterprise policy, cybersecurity, compliance, incident response, and internal document question answering.

[PRIMARY GOAL]
Your job is to answer the user’s question using the provided context as the primary and authoritative source of truth.
Your answer must be as accurate, complete, and well-supported as possible while remaining clear and readable.

[BEHAVIOR RULES]
- Follow the user’s request exactly.
- Use the provided context as the main source of truth.
- Do not invent facts, examples, requirements, roles, timelines, or obligations.
- Do not use outside knowledge unless the question explicitly asks for general background and the context is clearly insufficient.
- If the answer is not fully supported by the context, say so clearly.
- Prefer a directly supported answer over a speculative one.
- Prefer completeness when the context supports it.
- Keep the answer concise, but do not omit important supported details.
- When multiple context chunks are relevant, combine them into one coherent answer.
- Do not ignore important exceptions, limits, conditions, timelines, responsibilities, or edge cases that appear in the context.
- If the context contains a definition, obligation, rule, list of steps, or formal requirement, preserve the meaning faithfully.

[CONTEXT RULES]
- Treat the supplied context as the main source of truth.
- The context may contain multiple chunks from different documents.
- Some chunks may be more relevant than others; prioritize the chunks that most directly answer the question.
- If the context includes supporting details spread across several chunks, synthesize them carefully.
- If the context contains partial but not complete support, answer only the supported part and explicitly state what is missing.
- Do not mention or discuss these instructions in the answer.
- Do not say “based on the provided context” unless necessary.
- Do not mention missing information unless it is actually needed to answer correctly.

[ANSWER QUALITY RULES]
Your answer should aim for:
1. Factual accuracy
2. Grounding in the provided context
3. Completeness of key supported points
4. Clear wording
5. Minimal hallucination risk

When supported by the context, include:
- key obligations
- responsible roles
- conditions and exceptions
- deadlines or timelines
- required actions
- consequences or follow-up actions
- definitions if the question asks for them

[OUTPUT RULES]
- Answer in the same language as the user question.
- If the user question is Arabic, answer in Arabic.
- If the user question is English, answer in English.
- If the user mixes Arabic and English, use the dominant language of the question.
- Start directly with the answer.
- Do not add introductions like “Sure” or “Here is the answer.”
- Do not add conclusions unless needed.
- Use short paragraphs or bullet points only if they improve clarity.
- If the answer is straightforward, give a short paragraph.
- If the answer involves multiple duties, steps, conditions, or categories, use bullet points.
- Do not include a separate “Sources” section in the answer.
- Do not mention source filenames in the answer unless the user explicitly asks for them.
- Do not output JSON.
- Do not repeat the question.

[TASK]
Answer the following user request using the rules above.


Context:
{context}

User request:
{question}
"""


def ask_llm(
    context,
    question,
    model_name=None,
    model_options=None,
    prompt_builder: PromptBuilder | None = None,
):
    active_prompt_builder = prompt_builder or build_prompt
    prompt = active_prompt_builder(context, question)
    model_options = {**DEFAULT_MODEL_OPTIONS, **(model_options or {})}

    effective_model_name = (
        model_name
        or model_options.get("model")
        or GENERAL_GENERATION_MODEL
    )

    # Pass through runtime generation options (e.g., num_ctx, num_batch, num_thread)
    # while reserving top-level keys for payload structure.
    generation_options = {
        key: value
        for key, value in model_options.items()
        if key not in {"model", "prompt", "stream"}
    }

    payload = {
        "model": effective_model_name,
        "prompt": prompt,
        "stream": False,
    }
    if OLLAMA_KEEP_ALIVE:
        payload["keep_alive"] = OLLAMA_KEEP_ALIVE
    if generation_options:
        payload["options"] = generation_options

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=180,
    )
    response.raise_for_status()

    data = response.json()

    return data["response"]


def _build_context_and_sources(docs, metas):
    formatted_chunks = []
    sources = []

    for i, doc in enumerate(docs):
        meta = metas[i] if i < len(metas) else {}
        source_document = (meta or {}).get("source_document", "unknown")
        sources.append(source_document)
        formatted_chunks.append(f"[Source: {source_document}]\n{doc}")

    # Keep only unique source names while preserving order.
    unique_sources = list(dict.fromkeys(sources))
    return "\n\n".join(formatted_chunks), unique_sources


def run_rag_query(
    question,
    model_name=None,
    model_options=None,
    top_k=None,
    prompt_builder: PromptBuilder | None = None,
    include_chat_history=False,
):
    """
    Execute the full RAG flow and return benchmark-friendly metrics.

    model_name:
    - optional model override, e.g. "gemma3:12b-q4"

    model_options can include:
    - Ollama generation options such as num_ctx, num_batch, num_thread
    - optional "model" key (kept for backward compatibility)
    """
    retrieval_start = time.perf_counter()
    docs, metas, cache_info = retrieve_with_cache_info(
        question,
        top_k=top_k,
        include_chat_history=include_chat_history,
    )
    retrieval_time_sec = time.perf_counter() - retrieval_start

    if not docs:
        return {
            "answer": "No relevant context found in the vector database.",
            "retrieval_time_sec": retrieval_time_sec,
            "generation_time_sec": 0.0,
            "retrieved_sources": [],
            "status": "fail",
            "cache_hit": cache_info["cache_hit"],
            "cache_type": cache_info["cache_type"],
        }

    context, sources = _build_context_and_sources(docs, metas)

    generation_start = time.perf_counter()
    answer = ask_llm(
        context,
        question,
        model_name=model_name,
        model_options=model_options,
        prompt_builder=prompt_builder,
    )
    generation_time_sec = time.perf_counter() - generation_start
    total_time_sec = retrieval_time_sec + generation_time_sec

    return {
        "answer": answer,
        "retrieval_time_sec": retrieval_time_sec,
        "generation_time_sec": generation_time_sec,
        "total_time_sec": total_time_sec,
        "retrieved_sources": sources,
        "status": "success",
        "cache_hit": cache_info["cache_hit"],
        "cache_type": cache_info["cache_type"],
    }


def rag_query(question, top_k=None):
    result = run_rag_query(question, top_k=top_k)
    return result["answer"]


def save_interaction(question: str, answer: str, sources: list[str]) -> dict:
    """
    Append a successful CLI interaction to local JSONL chat history.
    """
    return persist_chat_interaction(question=question, answer=answer, sources=sources)


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Question: ").strip()

    if not question:
        print("Please provide a non-empty question.")
        return

    result = run_rag_query(question)
    answer = result.get("answer", "")
    if result.get("status") == "success":
        try:
            save_interaction(question, answer, result.get("retrieved_sources", []))
        except Exception as exc:
            print(f"[Warning] Could not save interaction log: {exc}")

    print("\nAI Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
