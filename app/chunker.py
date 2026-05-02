import re


SENTENCE_END_RE = re.compile(r"(?<=[.?!؛؟])\s+")


def _text_length(text):
    return len((text or "").strip())


def _append_text(left, right, separator):
    right = (right or "").strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left}{separator}{right}".strip()


def _unit_text(units):
    text = ""
    for unit in units:
        text = _append_text(text, unit["text"], unit.get("separator", "\n\n"))
    return text.strip()


def _unit_length(units):
    return len(_unit_text(units))


def _hard_split_text(text, chunk_size):
    parts = []
    remaining = (text or "").strip()

    while len(remaining) > chunk_size:
        split_at = remaining.rfind(" ", 0, chunk_size + 1)
        if split_at < max(1, int(chunk_size * 0.5)):
            split_at = chunk_size

        part = remaining[:split_at].strip()
        if part:
            parts.append(part)
        remaining = remaining[split_at:].strip()

    if remaining:
        parts.append(remaining)

    return parts


def _split_sentences(text):
    clean_text = " ".join((text or "").split())
    if not clean_text:
        return []
    return [part.strip() for part in SENTENCE_END_RE.split(clean_text) if part.strip()]


def _make_unit(text, page_number, separator):
    return {
        "text": text.strip(),
        "page_start": page_number,
        "page_end": page_number,
        "separator": separator,
    }


def _split_large_text(text, page_number, chunk_size, separator):
    units = []
    sentence_parts = _split_sentences(text)

    if len(sentence_parts) <= 1:
        sentence_parts = [text.strip()]

    for sentence in sentence_parts:
        if _text_length(sentence) <= chunk_size:
            units.append(_make_unit(sentence, page_number, separator))
            continue

        for part in _hard_split_text(sentence, chunk_size):
            units.append(_make_unit(part, page_number, " "))

    return units


def _page_units(page, chunk_size):
    page_number = page["page_number"]
    text = (page.get("text") or "").strip()
    if not text:
        return []

    units = []
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]

    for paragraph in paragraphs:
        if _text_length(paragraph) <= chunk_size:
            units.append(_make_unit(paragraph, page_number, "\n\n"))
            continue

        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        for line in lines:
            if _text_length(line) <= chunk_size:
                units.append(_make_unit(line, page_number, "\n"))
            else:
                units.extend(_split_large_text(line, page_number, chunk_size, " "))

    return units


def _overlap_units(units, overlap):
    if overlap <= 0 or not units:
        return []

    selected = []
    selected_length = 0

    for unit in reversed(units):
        unit_length = _text_length(unit["text"])
        if selected and selected_length + unit_length > overlap:
            break

        if not selected and unit_length > overlap:
            text = unit["text"]
            start = max(0, len(text) - overlap)
            boundary = text.find(" ", start)
            if boundary != -1 and boundary < len(text) - 1:
                text = text[boundary + 1 :]
            else:
                text = text[start:]

            return [
                {
                    **unit,
                    "text": text.strip(),
                    "separator": " ",
                }
            ]

        selected.insert(0, unit)
        selected_length += unit_length

    return selected


def _chunk_units(units, chunk_size, overlap):
    if not units:
        return []

    min_chunk_size = min(max(120, int(chunk_size * 0.25)), max(1, chunk_size))
    max_soft_size = max(chunk_size, int(chunk_size * 1.15))

    chunk_groups = []
    current = []

    for unit in units:
        candidate = [*current, unit]
        candidate_length = _unit_length(candidate)

        if (
            not current
            or candidate_length <= chunk_size
            or (_unit_length(current) < min_chunk_size and candidate_length <= max_soft_size)
        ):
            current = candidate
            continue

        chunk_groups.append(current)

        current = _overlap_units(current, overlap)
        if current and _unit_length([*current, unit]) > max_soft_size:
            current = []
        current.append(unit)

    if current:
        chunk_groups.append(current)

    return chunk_groups


def chunk_text(pages, chunk_size=900, overlap=180):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must be 0 or greater")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if not pages:
        return []

    if isinstance(pages, str):
        pages = [{"text": pages, "page_number": 1}]

    units = []
    for page in pages:
        if "page_number" not in page:
            raise ValueError("each page must include page_number")
        units.extend(_page_units(page, chunk_size))

    chunks = []
    for index, chunk_units in enumerate(_chunk_units(units, chunk_size, overlap)):
        text = _unit_text(chunk_units)
        if not text:
            continue

        chunks.append(
            {
                "text": text,
                "page_start": min(unit["page_start"] for unit in chunk_units),
                "page_end": max(unit["page_end"] for unit in chunk_units),
                "chunk_index": index,
            }
        )

    return chunks
