
import sys
from pathlib import Path

# Make the src directory available for imports.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from chunker import split_text
from embeddings import embed_texts
from rag import rag


def main():
    # Load the document.
    document_path = (
        PROJECT_ROOT / "data" / "documents" / "student_handbook.txt"
    )
    text = document_path.read_text(encoding="utf-8")

    # Split the document into chunks.
    chunks = split_text(
        text,
        source=document_path.name,
        chunk_size=100,
        chunk_overlap=20,
    )

    if not chunks:
        print("No text was found in the document.")
        return

    # Generate embeddings for all document chunks.
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    print("RAG application is ready!")
    print(f"Loaded document: {document_path.name}")
    print(f"Number of chunks: {len(chunks)}")
    print("\nAsk a question, or type 'exit' to quit.")

    # Interactive question-answering loop.
    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        result = rag(
            question=question,
            chunks=chunks,
            embeddings=embeddings,
        )

        print("\nAnswer:", result["answer"])
        print("Sources:", result["sources"])


if __name__ == "__main__":
    main()