import requests
import chromadb

client = chromadb.HttpClient(host="localhost", port=8000)
collection = client.get_collection(name="documents")

question = "What is AI?"

# embedding للسؤال
response = requests.post(
    "http://localhost:11434/api/embeddings",
    json={
        "model": "bge-prod",
        "prompt": question
    }
)

query_embedding = response.json()["embedding"]

# البحث في Chroma
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1
)

context = results["documents"][0][0]

# إرسال السؤال + السياق إلى LLM
prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}
"""

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "gemma-prod",
        "prompt": prompt,
        "stream": False
    }
)

print(response.json()["response"])