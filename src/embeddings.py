
import ollama


def embed_texts(texts, model="nomic-embed-text", batch_size=16):
    """Generate embeddings in smaller batches."""

    if not texts:
        return []

    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        response = ollama.embed(
            model=model,
            input=batch,
        )

        all_embeddings.extend(response.embeddings)

        print(
            f"Generated embeddings: "
            f"{len(all_embeddings)}/{len(texts)}"
        )

    return all_embeddings
