
EVALUATION_DATASET = [
    {
        "question": "How many weeks does an education year consist of?",
        "relevant_chunk_ids": [
            "ciu_exam_regulations.pdf::chunk_1",
        ],
    },
    {
        "question": (
            "How many working days must an education year "
            "contain at minimum?"
        ),
        "relevant_chunk_ids": [
            "ciu_exam_regulations.pdf::chunk_1",
        ],
    },
    {
        "question": "What is the language of instruction at the university?",
        "relevant_chunk_ids": [
            "ciu_exam_regulations.pdf::chunk_2",
            "ciu_exam_regulations.pdf::chunk_3",
        ],
    },
    {
        "question": (
            "How many half-semesters are included in an education year?"
        ),
        "relevant_chunk_ids": [
            "ciu_exam_regulations.pdf::chunk_1",
        ],
    },
]


def precision_at_k(retrieved_items, relevant_items, k):
    """Calculate Precision@K for retrieved items."""

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    relevant_items = set(relevant_items)
    retrieved_at_k = retrieved_items[:k]

    relevant_count = sum(
        item in relevant_items
        for item in retrieved_at_k
    )

    return relevant_count / k


def recall_at_k(retrieved_items, relevant_items, k):
    """Calculate Recall@K for retrieved items."""

    if k <= 0:
        raise ValueError("k must be greater than 0.")

    relevant_items = set(relevant_items)

    if not relevant_items:
        return 0.0

    retrieved_at_k = set(retrieved_items[:k])
    relevant_count = len(retrieved_at_k & relevant_items)

    return relevant_count / len(relevant_items)
