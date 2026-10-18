
EVALUATION_DATASET = [
    {
        "question": (
            "What is the minimum attendance requirement for students?"
        ),
        "relevant_chunk_ids": [
            "ciu_exam_regulations.pdf::chunk_33",
        ],
    },
    {
        "question": (
            "What happens when a student's absenteeism reaches 30%?"
        ),
        "relevant_chunk_ids": [
            "ciu_exam_regulations.pdf::chunk_34",
        ],
    },
    {
        "question": (
            "How many midterm exams are held at minimum each semester?"
        ),
        "relevant_chunk_ids": [
            "ciu_exam_regulations.pdf::chunk_19",
        ],
    },
    {
        "question": "What is the minimum duration of a midterm exam?",
        "relevant_chunk_ids": [
            "ciu_exam_regulations.pdf::chunk_27",
        ],
    },
    {
        "question": (
            "How late can a student arrive for a midterm exam "
            "and still be admitted?"
        ),
        "relevant_chunk_ids": [
            "ciu_exam_regulations.pdf::chunk_28",
        ],
    },
    {
        "question": "What is the minimum CGPA required for graduation?",
        "relevant_chunk_ids": [
            "ciu_exam_regulations.pdf::chunk_91",
        ],
    },
    {
        "question": (
            "How many days does a student have to object to an announced grade?"
        ),
        "relevant_chunk_ids": [
            "ciu_exam_regulations.pdf::chunk_49",
        ],
    },
    {
    "question": (
        "Which letter grade is the lowest passing grade "
        "on the regular letter-grade scale?"
    ),
    "relevant_chunk_ids": [
        "ciu_exam_regulations.pdf::chunk_37",
        "ciu_exam_regulations.pdf::chunk_38",
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
