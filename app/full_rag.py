import sys
import time
import json
from datetime import datetime
from pathlib import Path

import chromadb
import requests

try:
    from .config import (
        CHROMA_COLLECTION,
        CHROMA_HOST,
        CHROMA_PORT,
        EMBEDDING_MODEL,
        GENERAL_GENERATION_MODEL,
        OLLAMA_BASE_URL,
        TOP_K,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from config import (
        CHROMA_COLLECTION,
        CHROMA_HOST,
        CHROMA_PORT,
        EMBEDDING_MODEL,
        GENERAL_GENERATION_MODEL,
        OLLAMA_BASE_URL,
        TOP_K,
    )

# Connect to Chroma
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)
# print("[Chroma][full_rag] collections:", chroma_client.list_collections())
# print("[Chroma][full_rag] current count:", collection.count())
# print("[Chroma][full_rag] sample:", collection.peek(limit=2))

CHAT_LOG_PATH = Path(__file__).resolve().parents[1] / "data" / "chat_logs.jsonl"


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


def retrieve(query, top_k=None):
    n_results = top_k if top_k is not None else TOP_K
    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"],
    )

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    if not documents:
        return [], []

    return documents[0], metadatas[0] if metadatas else []


def build_prompt(context, question):
    return f"""
You are an AI assistant answering questions using only the provided context.

Rules:
- Use only the context below.
- Do not use outside knowledge.
- If the answer is not fully supported by the context, say what is missing.
- Use only the relevant context parts.
- Keep the answer clear, accurate, and concise.
- Use bullet points only when they improve clarity.

After the answer, list only the source document names that directly support the answer.

Format:
Answer:
<your answer>

Sources:
- <file_name>

Context:
{context}

Question:
{question}
"""


def ask_llm(context, question, model_name=None, model_options=None):
    prompt = build_prompt(context, question)
    
    default_options = {
    "num_ctx": 4096,
    "num_batch": 64,
    "num_thread": 8,
    "temperature": 0.2,
    }
    model_options = {**default_options, **(model_options or {})}

    # model_options = model_options or {}
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


def run_rag_query(question, model_name=None, model_options=None, top_k=None):
    """
    Execute the full RAG flow and return benchmark-friendly metrics.

    model_name:
    - optional model override, e.g. "gemma3:12b-q4"

    model_options can include:
    - Ollama generation options such as num_ctx, num_batch, num_thread
    - optional "model" key (kept for backward compatibility)
    """
    retrieval_start = time.perf_counter()
    docs, metas = retrieve(question, top_k=top_k)
    retrieval_time_sec = time.perf_counter() - retrieval_start

    if not docs:
        return {
            "answer": "No relevant context found in the vector database.",
            "retrieval_time_sec": retrieval_time_sec,
            "generation_time_sec": 0.0,
            "retrieved_sources": [],
            "status": "fail",
        }

    context, sources = _build_context_and_sources(docs, metas)

    generation_start = time.perf_counter()
    answer = ask_llm(
        context,
        question,
        model_name=model_name,
        model_options=model_options,
    )
    generation_time_sec = time.perf_counter() - generation_start

    return {
        "answer": answer,
        "retrieval_time_sec": retrieval_time_sec,
        "generation_time_sec": generation_time_sec,
        "retrieved_sources": sources,
        "status": "success",
    }


def rag_query(question, top_k=None):
    result = run_rag_query(question, top_k=top_k)
    return result["answer"]


def save_interaction(question: str, answer: str, sources: list[str]) -> None:
    """
    Append a successful CLI interaction to local JSONL chat history.
    """
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "retrieved_sources": sources,
    }

    CHAT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHAT_LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")


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
