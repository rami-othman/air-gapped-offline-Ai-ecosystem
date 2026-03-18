import sys

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
        collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
        )
    else:
        collection.add(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
        )


def retrieve(query):
    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
    )

    documents = results.get("documents", [])
    if not documents:
        return []

    return documents[0]


def build_prompt(context, question):
    return f"""
You are an AI assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
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


def rag_query(question):
    docs = retrieve(question)
    if not docs:
        return "No relevant context found in the vector database."

    context = "\n".join(docs)
    answer = ask_llm(context, question)
    return answer


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
