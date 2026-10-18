
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from main import load_documents
from embeddings import embed_texts
from rag import rag


def main():
    documents_dir = PROJECT_ROOT / "data" / "documents"

    chunks = load_documents(documents_dir)
    embeddings = embed_texts([chunk["text"] for chunk in chunks])

    question = (
        "Which letter grade is the lowest passing grade "
        "on the regular letter-grade scale?"
    )

    for top_k in (3, 5):
        print(f"\n{'=' * 60}")
        print(f"Testing top_k={top_k}")
        print("=" * 60)

        result = rag(
            question=question,
            chunks=chunks,
            embeddings=embeddings,
            top_k=top_k,
            threshold=0.6,
        )

        print("Answer:", result["answer"])
        print("Sources:", result["sources"])


if __name__ == "__main__":
    main()
