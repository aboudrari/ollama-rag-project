
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
from retriever import retrieve_relevant_chunks


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
            "2",
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


def test_retriever_finds_the_correct_answer_chunk(rag_data):
    chunks, embeddings = rag_data

    matches = retrieve_relevant_chunks(
        question="How many weeks does an education year consist of?",
        chunks=chunks,
        embeddings=embeddings,
        top_k=3,
        threshold=0.6,
    )

    assert matches, "The retriever returned no matching chunks."

    assert any(
        "16 weeks" in chunk["text"].lower()
        for chunk in matches
    ), "The retrieved chunks did not contain the expected answer."


def test_retriever_finds_tuition_related_context(rag_data):
    chunks, embeddings = rag_data

    matches = retrieve_relevant_chunks(
        question="What is the university's tuition fee?",
        chunks=chunks,
        embeddings=embeddings,
        top_k=3,
        threshold=0.6,
    )

    assert matches, "The retriever returned no matching chunks."

    assert any(
        "tuition fee" in chunk["text"].lower()
        for chunk in matches
    ), "No retrieved chunk mentioned tuition fees."


def test_correct_answer_chunk_is_ranked_first(rag_data):
    chunks, embeddings = rag_data

    matches = retrieve_relevant_chunks(
        question="How many weeks does an education year consist of?",
        chunks=chunks,
        embeddings=embeddings,
        top_k=3,
        threshold=0.6,
    )

    assert matches, "The retriever returned no matching chunks."

    assert "16 weeks" in matches[0]["text"].lower(), (
        "The most relevant chunk was not ranked first."
    )


def test_top_k_one_returns_only_the_best_match(rag_data):
    chunks, embeddings = rag_data

    matches = retrieve_relevant_chunks(
        question="How many weeks does an education year consist of?",
        chunks=chunks,
        embeddings=embeddings,
        top_k=1,
        threshold=0.6,
    )

    assert len(matches) <= 1, (
        "The retriever returned more than one chunk."
    )

    assert matches, "The retriever returned no matching chunks."

    assert "16 weeks" in matches[0]["text"].lower(), (
        "The best matching chunk did not contain the expected answer."
    )

def test_higher_threshold_does_not_increase_results(rag_data):
    chunks, embeddings = rag_data

    question = "How many weeks does an education year consist of?"

    lower_threshold_matches = retrieve_relevant_chunks(
        question=question,
        chunks=chunks,
        embeddings=embeddings,
        top_k=len(chunks),
        threshold=0.6,
    )

    higher_threshold_matches = retrieve_relevant_chunks(
        question=question,
        chunks=chunks,
        embeddings=embeddings,
        top_k=len(chunks),
        threshold=0.9,
    )

    assert len(higher_threshold_matches) <= len(lower_threshold_matches), (
        "Increasing the threshold should not increase the number of results."
    )


def test_very_high_threshold_returns_no_matches(rag_data):
    chunks, embeddings = rag_data

    matches = retrieve_relevant_chunks(
        question="How many weeks does an education year consist of?",
        chunks=chunks,
        embeddings=embeddings,
        top_k=3,
        threshold=1.1,
    )

    assert matches == [], (
        "The retriever should return no matches when the threshold "
        "is higher than the maximum possible cosine similarity."
    )


def test_rag_returns_unknown_when_no_chunks_are_retrieved(rag_data):
    chunks, embeddings = rag_data

    result = rag(
        question="How many weeks does an education year consist of?",
        chunks=chunks,
        embeddings=embeddings,
        threshold=1.1,
    )

    assert result["answer"].strip() == "I don't know."
    assert result["sources"] == []
