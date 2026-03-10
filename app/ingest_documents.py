from document_loader import load_pdf
from chunker import chunk_text
from full_rag import add_document


def ingest_pdf(path):

    text = load_pdf(path)

    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):

        doc_id = f"pdf_chunk_{i}"

        add_document(doc_id, chunk)


if __name__ == "__main__":

    ingest_pdf("data/sample.pdf")