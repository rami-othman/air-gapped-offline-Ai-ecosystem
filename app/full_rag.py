import requests
import chromadb

# الاتصال بـ Chroma
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

collection = chroma_client.get_or_create_collection(name="documents")


# -----------------------------
# Embedding generation
# -----------------------------
def generate_embedding(text):

    clean_text = text.replace("\n", " ").strip()

    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": clean_text
        }
    )

    data = response.json()

    if "embedding" not in data or len(data["embedding"]) == 0:
        print("Embedding error:", data)
        raise Exception("Embedding generation failed")

    return data["embedding"]


# -----------------------------
# Add document to Vector DB
# -----------------------------
def add_document(doc_id, text):

    embedding = generate_embedding(text)

    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding]
    )


# -----------------------------
# Retrieve relevant documents
# -----------------------------
def retrieve(query):

    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    return results["documents"][0]


# -----------------------------
# Ask LLM
# -----------------------------
def ask_llm(context, question):

    prompt = f"""
You are an AI assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma-prod",
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    return data["response"]


# -----------------------------
# Full RAG pipeline
# -----------------------------
def rag_query(question):

    docs = retrieve(question)

    context = "\n".join(docs)

    answer = ask_llm(context, question)

    return answer


# -----------------------------
# Example documents
# -----------------------------
doc1 = """
Artificial Intelligence (AI) is a field of computer science focused on creating systems
that can perform tasks requiring human intelligence such as learning, reasoning, and problem solving.
"""

doc2 = """
Machine Learning is a subset of Artificial Intelligence that allows computers to learn
from data without being explicitly programmed.
"""

add_document("doc1", doc1)
add_document("doc2", doc2)


# -----------------------------
# Ask question
# -----------------------------
question = "What is machine learning?"

answer = rag_query(question)

print("\nAI Answer:\n")
print(answer)