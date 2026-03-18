def chunk_text(text, chunk_size=500, overlap=100):
    if not text:
        return []

    clean_text = " ".join(text.split()).strip()
    if not clean_text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must be 0 or greater")

    step = chunk_size - overlap
    if step <= 0:
        step = 1

    chunks = []
    start = 0

    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(clean_text):
            break

        start += step

    return chunks
