
import sys
from pathlib import Path

# Make the src directory available for imports.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from main import load_documents
from embeddings import embed_texts
from retriever import cosine_similarity
from evaluation import EVALUATION_DATASET
import ollama


def debug_retriever(top_n=5):
    """Display the most relevant chunks for every evaluation question."""

    documents_dir = PROJECT_ROOT / "data" / "documents"

    chunks = load_documents(documents_dir)

    if not chunks:
        raise ValueError("No document chunks were loaded.")

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    for index, case in enumerate(EVALUATION_DATASET, start=1):
        question = case["question"]

        response = ollama.embed(
            model="nomic-embed-text",
            input=question,
        )

        question_embedding = response.embeddings[0]

        scored_chunks = []

        for chunk, embedding in zip(chunks, embeddings):
            similarity = cosine_similarity(
                question_embedding,
                embedding,
            )

            scored_chunks.append((similarity, chunk))

        scored_chunks.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        print("\n" + "=" * 80)
        print(f"QUESTION {index}: {question}")
        print("=" * 80)

        for rank, (similarity, chunk) in enumerate(
            scored_chunks[:top_n],
            start=1,
        ):
            print(f"\nRank: {rank}")
            print(f"Similarity: {similarity:.4f}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Source: {chunk['source']}")
            print(f"Text: {chunk['text'][:700]}")
            print("-" * 80)


if __name__ == "__main__":
    debug_retriever()
