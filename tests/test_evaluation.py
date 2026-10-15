
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from evaluation import precision_at_k, recall_at_k



def test_precision_at_k():
    retrieved = ["A", "B", "C"]
    relevant = {"A", "C", "D"}

    assert precision_at_k(retrieved, relevant, 3) == pytest.approx(
        2 / 3
    )


def test_recall_at_k():
    retrieved = ["A", "B", "C"]
    relevant = {"A", "C", "D"}

    assert recall_at_k(retrieved, relevant, 3) == pytest.approx(
        2 / 3
    )


def test_recall_is_zero_when_no_relevant_items_exist():
    assert recall_at_k(["A", "B"], set(), 2) == 0.0


def test_invalid_k_raises_error():
    with pytest.raises(ValueError):
        precision_at_k(["A"], {"A"}, 0)

    with pytest.raises(ValueError):
        recall_at_k(["A"], {"A"}, 0)
