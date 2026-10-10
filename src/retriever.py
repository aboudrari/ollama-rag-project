import math
import ollama

def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two vectors."""

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def retrieve_relevant_chunks(
    question,
    chunks,
    embeddings,
    top_k=3,
    threshold=0.6,
):
    """Retrieve the most relevant chunks for a question."""

    response = ollama.embed(
        model="nomic-embed-text",
        input=question,
    )
    question_embedding = response.embeddings[0]

    indexed_similarities = []

    for index, embedding in enumerate(embeddings):
        similarity = cosine_similarity(
            question_embedding,
            embedding,
        )

        if similarity >= threshold:
            indexed_similarities.append((similarity, index))

    indexed_similarities.sort(reverse=True)

    best_matches = indexed_similarities[:top_k]

    return [chunks[index] for similarity, index in best_matches]