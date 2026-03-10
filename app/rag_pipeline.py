import requests
import chromadb

# connect to Chroma
chroma_client = chromadb.HttpClient(host="localhost", port=8000)

collection = chroma_client.get_or_create_collection(name="documents")


def generate_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "bge-prod",
            "prompt": text
        }
    )

    return response.json()["embedding"]


def add_document(doc_id, text):
    embedding = generate_embedding(text)

    collection.add(
        ids=[doc_id],
        documents=[text],
        embeddings=[embedding]
    )


def search(query):
    query_embedding = generate_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )

    return results


# test document
doc = """
Artificial Intelligence is a field of computer science that focuses on building intelligent systems.
"""

add_document("doc1", doc)

results = search("What is artificial intelligence?")

print(results)