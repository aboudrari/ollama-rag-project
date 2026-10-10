
import re


def split_text(text, source, chunk_size=100, chunk_overlap=20):
    """Split text into chunks while preserving source metadata."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    # Remove BOM and split the document into paragraphs.
    paragraphs = text.replace("\ufeff", "").split("\n\n")

    paragraphs = [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]

    # Split paragraphs into manageable text segments.
    segments = []

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            segments.append((paragraph, "\n\n"))
            continue

        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) > chunk_size:
                words = sentence.split()
                piece = ""

                for word in words:
                    candidate = f"{piece} {word}".strip()

                    if len(candidate) <= chunk_size:
                        piece = candidate
                    else:
                        if piece:
                            segments.append((piece, " "))
                        piece = word

                if piece:
                    segments.append((piece, " "))
            else:
                segments.append((sentence, " "))

    # Combine segments into chunks.
    chunks = []
    current_chunk = ""

    for segment, separator in segments:
        if not current_chunk:
            current_chunk = segment
            continue

        candidate = current_chunk + separator + segment

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            chunks.append({
                "text": current_chunk,
                "source": source
            })

            # Keep some words from the previous chunk as overlap.
            overlap = ""

            if chunk_overlap > 0:
                words = current_chunk.split()
                suffix = []

                for word in reversed(words):
                    candidate_overlap = " ".join([word] + suffix)

                    if len(candidate_overlap) > chunk_overlap:
                        break

                    suffix.insert(0, word)

                overlap = " ".join(suffix)

            current_chunk = segment

            while (
                overlap
                and len(overlap + separator + current_chunk) > chunk_size
            ):
                overlap_words = overlap.split()
                overlap = " ".join(overlap_words[1:])

            if overlap:
                current_chunk = overlap + separator + current_chunk

    # Save the last chunk.
    if current_chunk:
        chunks.append({
            "text": current_chunk,
            "source": source
        })

    return chunks