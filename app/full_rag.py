import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

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
PromptBuilder = Callable[[str, str], str]

DEFAULT_MODEL_OPTIONS = {
    "num_ctx": 4096,
    "num_batch": 64,
    "num_thread": 16,
    "temperature": 0.2,
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
- Output in English.
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
After the answer, list only the source document names that directly support the answer.

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
