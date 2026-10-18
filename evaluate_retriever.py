import sys
from pathlib import Path

# Make the src directory available for imports.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from main import load_documents
from embeddings import embed_texts
from retriever import retrieve_relevant_chunks
from evaluation import (
    EVALUATION_DATASET,
    precision_at_k,
    recall_at_k,
)


def evaluate_top_k(
    top_k_values=(2, 3, 5),
    threshold=0.6,
):
    """Compare retrieval quality across different top_k values."""

    documents_dir = PROJECT_ROOT / "data" / "documents"

    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {documents_dir}"
        )

    # Load and chunk documents once.
    chunks = load_documents(documents_dir)

    if not chunks:
        raise ValueError("No document chunks were loaded.")

    # Generate document embeddings once.
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    print("\n" + "=" * 60)
    print("TOP-K EVALUATION")
    print(f"Threshold: {threshold}")
    print(f"Number of chunks: {len(chunks)}")
    print("=" * 60)

    # Compare each top_k value using the same documents and embeddings.
    for top_k in top_k_values:
        if top_k <= 0:
            raise ValueError("Every top_k value must be greater than 0.")

        precision_scores = []
        recall_scores = []

        print(f"\n\n{'-' * 60}")
        print(f"RESULTS FOR top_k = {top_k}")
        print("-" * 60)

        for case in EVALUATION_DATASET:
            question = case["question"]
            relevant_ids = set(case["relevant_chunk_ids"])

            matches = retrieve_relevant_chunks(
                question=question,
                chunks=chunks,
                embeddings=embeddings,
                top_k=top_k,
                threshold=threshold,
            )

            retrieved_ids = [
                chunk["chunk_id"] for chunk in matches
            ]

            precision = precision_at_k(
                retrieved_items=retrieved_ids,
                relevant_items=relevant_ids,
                k=top_k,
            )

            recall = recall_at_k(
                retrieved_items=retrieved_ids,
                relevant_items=relevant_ids,
                k=top_k,
            )

            precision_scores.append(precision)
            recall_scores.append(recall)

            print(f"\nQuestion: {question}")
            print(f"Retrieved IDs: {retrieved_ids}")
            print(f"Relevant IDs:  {sorted(relevant_ids)}")
            print(f"Precision@{top_k}: {precision:.3f}")
            print(f"Recall@{top_k}:    {recall:.3f}")

        mean_precision = sum(precision_scores) / len(precision_scores)
        mean_recall = sum(recall_scores) / len(recall_scores)

        print(f"\nMean Precision@{top_k}: {mean_precision:.3f}")
        print(f"Mean Recall@{top_k}:    {mean_recall:.3f}")


if __name__ == "__main__":
    evaluate_top_k()