from pathlib import Path

try:
    from .cache_store import response_cache, retrieval_cache
    from .chunker import chunk_text
    from .config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR
    from .document_loader import load_pdf
    from .full_rag import add_document, delete_document_chunks
except ImportError:  # pragma: no cover - script execution fallback
    from cache_store import response_cache, retrieval_cache
    from chunker import chunk_text
    from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR
    from document_loader import load_pdf
    from full_rag import add_document, delete_document_chunks

# print("[Chroma][ingest_documents] collections:", chroma_client.list_collections())
# print("[Chroma][ingest_documents] current count:", collection.count())
# print("[Chroma][ingest_documents] sample:", collection.peek(limit=2))


def slugify_filename(path):
    stem = Path(path).stem.lower()
    safe = []
    for char in stem:
        if char.isalnum():
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def ingest_pdf(path):
    text = load_pdf(path)
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    doc_slug = slugify_filename(path)
    file_name = Path(path).name
    source_document = file_name

    # Keep ingestion idempotent for reruns and document edits.
    delete_document_chunks(source_document)

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_slug}_chunk_{i:04d}"
        metadata = {
            "source_document": source_document,
            "file_name": file_name,
            "chunk_id": chunk_id,
        }
        add_document(chunk_id, chunk, metadata=metadata)

    return len(chunks)


def ingest_directory(dir_path=DOCS_DIR):
    docs_path = Path(dir_path)
    if not docs_path.exists():
        print(f"Directory not found: {docs_path}")
        return 0, 0

    pdf_paths = sorted(docs_path.glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDF files found in: {docs_path}")
        return 0, 0

    total_chunks = 0
    for pdf_path in pdf_paths:
        chunk_count = ingest_pdf(pdf_path)
        total_chunks += chunk_count
        print(f"Ingested {pdf_path.name}: {chunk_count} chunks")

    retrieval_cache.clear()
    response_cache.clear()

    return len(pdf_paths), total_chunks


if __name__ == "__main__":
    docs_count, chunks_count = ingest_directory()
    print(f"Done. Documents: {docs_count}, Chunks: {chunks_count}")
