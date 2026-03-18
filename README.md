# Local RAG System with Ollama + ChromaDB

This repository contains a local Retrieval-Augmented Generation (RAG) baseline for policy/security documents.

## Week 2 Status

Week 2 functional goals are in place:

- [x] Clean main RAG flow in `app/full_rag.py`
- [x] Unified model/config settings in `app/config.py`
- [x] Improved chunking with overlap in `app/chunker.py`
- [x] Real document ingestion from `data/docs/`
- [x] Retrieval evaluation dataset in `data/eval/retrieval_eval.jsonl`
- [x] Embedding comparison script in `app/eval_embeddings.py`

## Main Files

- `app/config.py` - central settings (models, Chroma, chunking, docs path)
- `app/ingest_documents.py` - ingest all PDFs from `data/docs/`
- `app/full_rag.py` - ask questions against indexed documents
- `app/eval_embeddings.py` - compare embedding models on retrieval dataset
- `data/eval/retrieval_eval.jsonl` - Week 2 eval queries and expected labels

## Prerequisites

- Ollama running on `http://localhost:11434`
- ChromaDB running on `localhost:8000`
- Python environment with required packages (`chromadb`, `requests`, `pypdf`)

## Week 2 Run Commands

1. Ingest real documents:

```bash
python app/ingest_documents.py
```

2. Ask a question (CLI argument):

```bash
python app/full_rag.py "What are breach notification obligations under GDPR?"
```

3. Ask a question (interactive mode):

```bash
python app/full_rag.py
```

4. Run embedding comparison (full dataset):

```bash
python app/eval_embeddings.py --models nomic-embed-text bge-m3 --top-k 3 --output data/eval/embedding_comparison_results.json
```

5. Optional smoke run (faster):

```bash
python app/eval_embeddings.py --models nomic-embed-text bge-m3 --max-chunks 8 --top-k 3 --output data/eval/embedding_comparison_results_smoke.json
```

## Outputs

- Ingestion writes vectors/chunks into the configured Chroma collection.
- Query command prints the generated answer.
- Embedding evaluation writes JSON metrics under `data/eval/`.

## Notes

- `query.py` and `ingest.py` are older prototype scripts; use `app/` scripts for Week 2 workflow.
