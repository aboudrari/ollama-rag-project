import re


def split_text(text, source, chunk_size=100, chunk_overlap=20):
    """Split text into overlapping chunks while preserving source metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    # Remove the BOM and normalize whitespace.
    text = text.replace("\ufeff", "").strip()
    text = re.sub(r"\s+", " ", text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer ending at a word boundary when possible.
        if end < len(text) and text[end] != " ":
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "source": source
            })

        if end >= len(text):
            break

        # Move forward while retaining the requested character overlap.
        next_start = max(start + 1, end - chunk_overlap)

        # Avoid starting in the middle of a word.
        if next_start > 0 and text[next_start - 1] != " ":
            next_space = text.find(" ", next_start)

            if next_space != -1 and next_space < end:
                next_start = next_space + 1

        start = next_start

    return chunks
    