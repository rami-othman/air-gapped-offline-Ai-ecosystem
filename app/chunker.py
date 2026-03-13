def chunk_text(text, chunk_size=500):

    chunks = []

    start = 0

    while start < len(text):

        chunk = text[start:start + chunk_size]

        chunks.append(chunk)

        start += chunk_size

    return chunks