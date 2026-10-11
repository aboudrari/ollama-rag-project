
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from main import load_documents
from embeddings import embed_texts
from rag import rag


@pytest.fixture(scope="module")
def rag_data():
    """Load documents and generate embeddings once per test run."""

    documents_dir = PROJECT_ROOT / "data" / "documents"

    chunks = load_documents(documents_dir)
    embeddings = embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    return chunks, embeddings


@pytest.mark.parametrize(
    ("question", "expected_keyword"),
    [
        (
            "How many weeks does an education year consist of?",
            "16",
        ),
        (
            "How many working days must an education year contain at minimum?",
            "80",
        ),
        (
            "What is the language of instruction at the university?",
            "english",
        ),
        (
            "How many half-semesters are included in an education year?",
            "two",
        ),
    ],
)
def test_answers_from_documents(rag_data, question, expected_keyword):
    chunks, embeddings = rag_data

    result = rag(
        question=question,
        chunks=chunks,
        embeddings=embeddings,
    )

    assert expected_keyword in result["answer"].lower()
    assert result["sources"]


def test_unknown_question(rag_data):
    chunks, embeddings = rag_data

    result = rag(
        question="What is the university's tuition fee?",
        chunks=chunks,
        embeddings=embeddings,
    )

    assert result["answer"].strip().lower() == "i don't know."
    assert result["sources"]
