import sys
from pathlib import Path

# Make the src directory available for imports.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from main import load_documents
from embeddings import embed_texts
from retriever import cosine_similarity
import ollama


DEBUG_QUESTIONS = [
    {
        "question": (
            "How many midterm exams are held at minimum "
            "each semester?"
        ),
        "expected_chunk_id": "ciu_exam_regulations.pdf::chunk_19",
    },
    {
        "question": (
            "What is the minimum passing letter grade "
            "listed in the regulations?"
        ),
        "expected_chunk_id": "ciu_exam_regulations.pdf::chunk_38",
    },
]


def debug_retriever(top_n=5):
    documents_dir = PROJECT_ROOT / "data" / "documents"

    chunks = load_documents(documents_dir)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    for case in DEBUG_QUESTIONS:
        question = case["question"]
        expected_id = case["expected_chunk_id"]

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

        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print(f"Expected chunk: {expected_id}")
        print("=" * 70)

        expected_rank = None

        for rank, (similarity, chunk) in enumerate(
            scored_chunks,
            start=1,
        ):
            if chunk["chunk_id"] == expected_id:
                expected_rank = rank

            if rank <= top_n:
                print(f"\nRank: {rank}")
                print(f"Similarity: {similarity:.4f}")
                print(f"Chunk ID: {chunk['chunk_id']}")
                print(f"Text: {chunk['text'][:350]}")

        if expected_rank is not None:
            expected_similarity = scored_chunks[
                expected_rank - 1
            ][0]

            print(f"\nExpected chunk rank: {expected_rank}")
            print(
                "Expected chunk similarity: "
                f"{expected_similarity:.4f}"
            )


if __name__ == "__main__":
    debug_retriever()