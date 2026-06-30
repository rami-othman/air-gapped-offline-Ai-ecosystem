# Context Pipeline (CLI)

This pipeline lets you save RAG CLI interactions, export them, and re-import them into Chroma for retrieval.

## 1) Generate chat logs from CLI RAG

Run your RAG CLI and ask questions:

```bash
python app/full_rag.py
```

Successful interactions are appended to:

- `data/chat_logs.jsonl`

Each row includes:

- `timestamp`
- `question`
- `answer`
- `retrieved_sources`

## 2) Export logs to JSON and CSV

```bash
python scripts/context_pipeline/export_chat_logs.py
```

Exports:

- `data/exports/chat_logs.json`
- `data/exports/chat_logs.csv`

## 3) Import logs back into the vector DB

```bash
python scripts/context_pipeline/import_chat_logs.py
```

Import behavior:

- Prefers `data/exports/chat_logs.json`
- Falls back to `data/chat_logs.jsonl`
- Converts each entry to a retrieval document:
  - `Q: ...`
  - `A: ...`
- Ingests via your existing RAG function `add_document(...)`
- Adds metadata like:
  - `source = chat_history`
  - `source_document = chat_history`
  - `timestamp`
  - `question`

## 4) Verify retrieval of chat history

After import, ask follow-up questions in `app/full_rag.py` and check whether returned sources include `chat_history`.
