import argparse
import json
import time
from pathlib import Path

import chromadb
import requests

from chunker import chunk_text
from config import (
    CHROMA_COLLECTION,
    CHROMA_HOST,
    CHROMA_PORT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCS_DIR,
    EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
)
from document_loader import load_pdf

TOPIC_KEYWORDS = {
    "employee": "employee handbook",
    "cyber": "cyber security policy",
    "incident": "incident response plan",
    "gdpr": "gdpr",
    "nist": "nist security framework",
}


def slugify(value):
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def infer_topic(path):
    lower_name = Path(path).name.lower()
    for key, topic in TOPIC_KEYWORDS.items():
        if key in lower_name:
            return topic
    return "general policy"


def embed_text(model, text):
    clean_text = text.replace("\n", " ").strip()
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": model, "prompt": clean_text},
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()

    if "embedding" not in data or not data["embedding"]:
        raise RuntimeError(f"Embedding failed for model {model}: {data}")

    return data["embedding"]


def load_eval_dataset(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


def build_collection_for_model(client, model, docs_dir, max_chunks=None):
    model_slug = slugify(model)
    collection_name = f"{CHROMA_COLLECTION}_eval_{model_slug}"

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=collection_name)

    chunk_counter = 0
    pdf_paths = sorted(Path(docs_dir).glob("*.pdf"))

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for pdf_path in pdf_paths:
        text = load_pdf(str(pdf_path))
        chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        source_document = pdf_path.name
        section_topic = infer_topic(pdf_path)
        doc_slug = slugify(pdf_path.stem)

        for i, chunk in enumerate(chunks):
            if max_chunks is not None and chunk_counter >= max_chunks:
                break

            chunk_id = f"{model_slug}_{doc_slug}_{i:04d}"
            embedding = embed_text(model, chunk)

            ids.append(chunk_id)
            documents.append(chunk)
            embeddings.append(embedding)
            metadatas.append(
                {
                    "source_document": source_document,
                    "section_topic": section_topic,
                    "chunk_id": chunk_id,
                }
            )

            chunk_counter += 1

            if len(ids) >= 16:
                collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                ids, documents, embeddings, metadatas = [], [], [], []

        if max_chunks is not None and chunk_counter >= max_chunks:
            break

    if ids:
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    return collection, chunk_counter, collection_name


def evaluate_model(collection, model, eval_rows, top_k):
    doc_hits = 0
    topic_hits = 0

    for row in eval_rows:
        query_embedding = embed_text(model, row["query"])
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]

        expected_doc = row["expected_document"]
        expected_topic = row["expected_section_topic"].lower()

        if any((meta or {}).get("source_document") == expected_doc for meta in metas):
            doc_hits += 1

        topic_match = False
        for meta, doc in zip(metas, docs):
            meta_topic = (meta or {}).get("section_topic", "").lower()
            doc_text = (doc or "").lower()
            if expected_topic and (expected_topic in meta_topic or meta_topic in expected_topic):
                topic_match = True
                break
            if expected_topic and expected_topic in doc_text:
                topic_match = True
                break

        if topic_match:
            topic_hits += 1

    total = len(eval_rows)
    return {
        "queries": total,
        "doc_hits": doc_hits,
        "topic_hits": topic_hits,
        "doc_hit_rate": round(doc_hits / total, 4) if total else 0.0,
        "topic_hit_rate": round(topic_hits / total, 4) if total else 0.0,
    }


def dedupe_keep_order(items):
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare embedding models on retrieval quality")
    parser.add_argument("--dataset", default="data/eval/retrieval_eval.jsonl")
    parser.add_argument("--docs-dir", default=DOCS_DIR)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[EMBEDDING_MODEL, "bge-m3"],
        help="Embedding model names available in Ollama",
    )
    parser.add_argument(
        "--output",
        default="data/eval/embedding_comparison_results.json",
        help="JSON output path",
    )
    args = parser.parse_args()

    models = dedupe_keep_order(args.models)
    eval_rows = load_eval_dataset(args.dataset)
    client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)

    all_results = []
    for model in models:
        start = time.time()
        print(f"\n[Model] {model}")

        try:
            collection, indexed_chunks, collection_name = build_collection_for_model(
                client=client,
                model=model,
                docs_dir=args.docs_dir,
                max_chunks=args.max_chunks,
            )
            metrics = evaluate_model(
                collection=collection,
                model=model,
                eval_rows=eval_rows,
                top_k=args.top_k,
            )
        except Exception as exc:
            print(f"Failed model {model}: {exc}")
            all_results.append(
                {
                    "model": model,
                    "error": str(exc),
                }
            )
            continue

        elapsed = round(time.time() - start, 2)
        result = {
            "model": model,
            "collection": collection_name,
            "indexed_chunks": indexed_chunks,
            "top_k": args.top_k,
            "elapsed_seconds": elapsed,
            **metrics,
        }
        all_results.append(result)

        print(
            f"doc_hit_rate={result['doc_hit_rate']:.2%} "
            f"topic_hit_rate={result['topic_hit_rate']:.2%} "
            f"chunks={indexed_chunks} time={elapsed}s"
        )

    successful = [r for r in all_results if "error" not in r]
    ranked = sorted(
        successful,
        key=lambda r: (r["doc_hit_rate"], r["topic_hit_rate"], -r["elapsed_seconds"]),
        reverse=True,
    )

    print("\n=== Ranking (best first) ===")
    for i, item in enumerate(ranked, start=1):
        print(
            f"{i}. {item['model']} | doc_hit={item['doc_hit_rate']:.2%} "
            f"topic_hit={item['topic_hit_rate']:.2%} time={item['elapsed_seconds']}s"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
