
import sys
from pathlib import Path

# Make the src directory available for imports.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from main import load_documents
from embeddings import embed_texts
from rag import rag
from evaluation import EVALUATION_DATASET


def evaluate_rag(top_k_values=(2, 3, 5), threshold=0.6):
    """Compare generated RAG answers across different top_k values."""

    documents_dir = PROJECT_ROOT / "data" / "documents"

    if not documents_dir.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {documents_dir}"
        )

    # Load and chunk documents once.
    chunks = load_documents(documents_dir)

    if not chunks:
        raise ValueError("No document chunks were loaded.")

    # Generate document embeddings once and reuse them.
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    print("\n" + "=" * 70)
    print("END-TO-END RAG EVALUATION")
    print(f"Threshold: {threshold}")
    print(f"Documents chunks: {len(chunks)}")
    print("=" * 70)

    for top_k in top_k_values:
        if top_k <= 0:
            raise ValueError("Every top_k value must be greater than 0.")

        print(f"\n\n{'#' * 70}")
        print(f"TESTING top_k = {top_k}")
        print("#" * 70)

        for index, case in enumerate(EVALUATION_DATASET, start=1):
            question = case["question"]

            print(f"\n{'-' * 70}")
            print(f"Question {index}: {question}")
            print("-" * 70)

            result = rag(
                question=question,
                chunks=chunks,
                embeddings=embeddings,
                top_k=top_k,
                threshold=threshold,
            )

            print("Expected relevant chunk IDs:",
                  case["relevant_chunk_ids"])
            print("Generated answer:", result["answer"])
            print("Sources:", result["sources"])


if __name__ == "__main__":
    evaluate_rag()
