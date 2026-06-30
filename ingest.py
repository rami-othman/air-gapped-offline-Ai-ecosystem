import requests
import chromadb

# الاتصال بـ Chroma
client = chromadb.HttpClient(host="localhost", port=8000)

collection = client.get_or_create_collection(name="documents")

text = """
Artificial Intelligence (AI) is a field of computer science focused on creating systems
that can perform tasks requiring human intelligence such as learning, reasoning, and problem solving.
"""

# طلب embedding من Ollama
response = requests.post(
    "http://localhost:11434/api/embeddings",
    json={
        "model": "nomic-embed-text",
        "prompt": text
    }
)

embedding = response.json()["embedding"]

# تخزين في Chroma
collection.add(
    documents=[text],
    embeddings=[embedding],
    ids=["doc1"]
)

print("Document stored successfully.")