import ollama


def embed_texts(texts, model="nomic-embed-text"):
    """Generate embeddings for a list of texts using Ollama."""

    if not texts:
        return []

    response = ollama.embed(
        model=model,
        input=texts
    )

    return response.embeddings