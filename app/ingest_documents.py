from pathlib import Path

from chunker import chunk_text
from config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR
from document_loader import load_pdf
from full_rag import add_document


TOPIC_KEYWORDS = {
    "employee": "employee handbook",
    "cyber": "cyber security policy",
    "incident": "incident response plan",
    "gdpr": "gdpr",
    "nist": "nist security framework",
}


def slugify_filename(path):
    stem = Path(path).stem.lower()
    safe = []
    for char in stem:
        if char.isalnum():
            safe.append(char)
        else:
            safe.append("_")
    return "".join(safe).strip("_")


def infer_topic(path):
    lower_name = Path(path).name.lower()
    for key, topic in TOPIC_KEYWORDS.items():
        if key in lower_name:
            return topic
    return "general policy"


def ingest_pdf(path):
    text = load_pdf(path)
    chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)

    doc_slug = slugify_filename(path)
    source_document = Path(path).name
    section_topic = infer_topic(path)

    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_slug}_chunk_{i:04d}"
        metadata = {
            "source_document": source_document,
            "section_topic": section_topic,
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

    return len(pdf_paths), total_chunks


if __name__ == "__main__":
    docs_count, chunks_count = ingest_directory()
    print(f"Done. Documents: {docs_count}, Chunks: {chunks_count}")
