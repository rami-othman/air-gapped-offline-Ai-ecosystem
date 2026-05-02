import re

from pypdf import PdfReader


def _normalize_page_text(text):
    """
    Normalize PDF extraction whitespace without changing letters or punctuation.
    This keeps Arabic text intact while making chunk boundaries more predictable.
    """
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"[ \t\f\v]+", " ", normalized)

    lines = [line.strip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)

    return normalized.strip()


def load_pdf(path):
    reader = PdfReader(path)
    pages = []

    for page_index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append(
            {
                "text": _normalize_page_text(page_text),
                "page_number": page_index,
            }
        )

    return pages
