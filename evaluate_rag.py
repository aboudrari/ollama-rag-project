
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from main import load_documents
from embeddings import embed_texts
from rag import rag
from evaluation import EVALUATION_DATASET


OUT_OF_DOCUMENT_QUESTIONS = [
    "Who is the president of the United States?",
    "What is the university's founding year?",
    "What is the tuition fee for international students?",
]


def evaluate_rag(top_k_values=(2, 3, 5), threshold=0.6):
    documents_dir = PROJECT_ROOT / "data" / "documents"

    print("\nLoading documents...")
    chunks = load_documents(documents_dir)

    print(f"Loaded {len(chunks)} chunks.")

    print("\nGenerating embeddings...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    total_passed = 0
    total_failed = 0

    for top_k in top_k_values:
        print("\n" + "=" * 60)
        print(f"RAG EVALUATION - top_k={top_k}")
        print("=" * 60)

        for case in EVALUATION_DATASET:
            result = rag(
                question=case["question"],
                chunks=chunks,
                embeddings=embeddings,
                top_k=top_k,
                threshold=threshold,
            )

            answer = result["answer"]

            print(f"\nQuestion: {case['question']}")
            print(f"Answer: {answer}")
            print(f"Sources: {result['sources']}")

    print("\n" + "=" * 60)
    print("OUT-OF-DOCUMENT QUESTIONS")
    print("=" * 60)

    for question in OUT_OF_DOCUMENT_QUESTIONS:
        result = rag(
            question=question,
            chunks=chunks,
            embeddings=embeddings,
            top_k=2,
            threshold=threshold,
        )

        passed = result["answer"].strip().lower().rstrip(".!? ") == "i don't know"

        if passed:
            total_passed += 1
        else:
            total_failed += 1

        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print("Status:", "PASS" if passed else "FAIL")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Out-of-document tests passed: {total_passed}/3")
    print(f"Out-of-document tests failed: {total_failed}/3")


if __name__ == "__main__":
    evaluate_rag()
