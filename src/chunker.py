
import re


def split_text(
    text,
    source,
    chunk_size=1000,
    chunk_overlap=200,
):
    """Split text into overlapping chunks with source metadata and unique IDs."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be >= 0 and < chunk_size"
        )

    # Remove BOM and normalize whitespace.
    text = text.replace("\ufeff", "").strip()
    text = re.sub(r"\s+", " ", text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Prefer sentence boundaries.
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
            )

            # Fall back to a word boundary.
            if boundary <= start:
                boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary + 1

        chunk_text = text[start:end].strip()

        if chunk_text:
            chunk_id = f"{source}::chunk_{len(chunks)}"

            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "source": source,
            })

        if end >= len(text):
            break

        next_start = max(
            start + 1,
            end - chunk_overlap,
        )

        # Avoid starting in the middle of a word.
        next_space = text.find(" ", next_start, end)

        if next_space != -1:
            next_start = next_space + 1

        start = next_start

    return chunks
