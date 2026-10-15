import sys
from pathlib import Path

# Make the src directory available for imports.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from chunker import split_text
from document_loader import load_pdf
from embeddings import embed_texts
from rag import rag
from retriever import retrieve_relevant_chunks



def load_documents(documents_dir):
    """Load all supported TXT and PDF documents."""

    chunks = []

    document_paths = sorted(
        path
        for path in documents_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".txt", ".pdf"}
    )

    for document_path in document_paths:
        print(f"Loading: {document_path.name}")

        if document_path.suffix.lower() == ".pdf":
            text = load_pdf(document_path)
        else:
            text = document_path.read_text(encoding="utf-8")

        document_chunks = split_text(
            text,
            source=document_path.name,
            chunk_size=500,
            chunk_overlap=100,
        )

        chunks.extend(document_chunks)

        print(f"  Chunks created: {len(document_chunks)}")

    return chunks


def inspect_retrieval(chunks, embeddings):
    """Inspect retrieval results for evaluation questions."""

    questions = [
        "How many weeks does an education year consist of?",
        "How many working days must an education year contain at minimum?",
        "What is the language of instruction at the university?",
        "How many half-semesters are included in an education year?",
    ]

    for question in questions:
        print("\n" + "=" * 70)
        print(f"Question: {question}")
        print("=" * 70)

        matches = retrieve_relevant_chunks(
            question=question,
            chunks=chunks,
            embeddings=embeddings,
            top_k=3,
            threshold=0.6,
        )

        if not matches:
            print("No chunks retrieved.")
            continue

        for rank, chunk in enumerate(matches, start=1):
            print(f"\nResult {rank}")
            print("Chunk ID:", chunk["chunk_id"])
            print("Source:", chunk["source"])
            print("Text:", chunk["text"])


def main():
    documents_dir = PROJECT_ROOT / "data" / "documents"

    if not documents_dir.exists():
        print("Documents directory was not found.")
        return

    # Load and chunk every supported document.
    chunks = load_documents(documents_dir)

    if not chunks:
        print("No text was found in the documents.")
        return

    # Generate embeddings for all chunks.
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts)

    sources = sorted({chunk["source"] for chunk in chunks})

    print("\nRAG application is ready!")
    print(f"Documents loaded: {len(sources)}")
    print(f"Total chunks: {len(chunks)}")

    # Temporarily inspect retrieval results for evaluation preparation.
    inspect_retrieval(chunks, embeddings)

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