import sys
import time

import chromadb
import requests

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


def generate_embedding(text):
    clean_text = text.replace("\n", " ").strip()

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "prompt": clean_text,
        },
    )

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


def retrieve(query):
    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )

    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])
    if not documents:
        return [], []

    return documents[0], metadatas[0] if metadatas else []


def build_prompt(context, question):
    return f"""
You are an AI assistant specialized in answering questions from documents.

IMPORTANT RULES:
- Use ONLY the provided context.
- Do NOT use external knowledge.
- Read ALL context chunks carefully before answering.
- Combine information from ALL relevant chunks.
- Do NOT give a partial answer if more details exist.
- Extract ALL relevant points explicitly.

ANSWER STYLE:
- Provide a COMPLETE and detailed answer.
- Use bullet points when appropriate.
- Include ALL key details found in the context.

SOURCE REQUIREMENT:
- After the answer, list the source_document names used.
- Only include sources that contributed to the answer.
- Format:
  Sources:
  - <file_name>

If the context is insufficient, say what is missing.

---------------------
CONTEXT:
{context}
---------------------

QUESTION:
{question}

ANSWER:
"""


def ask_llm(context, question):
    prompt = build_prompt(context, question)

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": GENERAL_GENERATION_MODEL,
            "prompt": prompt,
            "stream": False,
        },
    )

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


def run_rag_query(question):
    """
    Execute the full RAG flow and return benchmark-friendly metrics.
    """
    retrieval_start = time.perf_counter()
    docs, metas = retrieve(question)
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
    answer = ask_llm(context, question)
    generation_time_sec = time.perf_counter() - generation_start

    return {
        "answer": answer,
        "retrieval_time_sec": retrieval_time_sec,
        "generation_time_sec": generation_time_sec,
        "retrieved_sources": sources,
        "status": "success",
    }


def rag_query(question):
    result = run_rag_query(question)
    return result["answer"]


def main():
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = input("Question: ").strip()

    if not question:
        print("Please provide a non-empty question.")
        return

    answer = rag_query(question)
    print("\nAI Answer:\n")
    print(answer)


if __name__ == "__main__":
    main()
