import math

import pytest

from farm_advisory_assistant.evaluation.metrics import (
    average_precision,
    evaluate,
    f1_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Worked example used by most tests.
# The search returned a..e. Relevant chunks: b (grade 2), d (grade 1) and x (grade 1, never retrieved).
# So the hits are at rank 2 and rank 4, and there are 3 relevant chunks in total.
RETRIEVED = ["a", "b", "c", "d", "e"]
RELEVANT = {"b": 2, "d": 1, "x": 1}


@pytest.mark.parametrize("k, expected", [(1, 0.0), (2, 1 / 2), (3, 1 / 3), (5, 2 / 5)])
def test_precision_at_k(k, expected):
    assert precision_at_k(RETRIEVED, RELEVANT, k) == pytest.approx(expected)


@pytest.mark.parametrize("k, expected", [(1, 0.0), (3, 1 / 3), (5, 2 / 3)])
def test_recall_at_k(k, expected):
    assert recall_at_k(RETRIEVED, RELEVANT, k) == pytest.approx(expected)


def test_f1_at_k():
    # P@5 = 2/5 and R@5 = 2/3, so F1 = 2 * (2/5) * (2/3) / (2/5 + 2/3) = (8/15) / (16/15) = 0.5
    assert f1_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(0.5)


def test_reciprocal_rank():
    # The first relevant chunk (b) is at rank 2.
    assert reciprocal_rank(RETRIEVED, RELEVANT, 5) == pytest.approx(1 / 2)
    # In the top 1 there is no relevant chunk at all.
    assert reciprocal_rank(RETRIEVED, RELEVANT, 1) == 0.0


def test_average_precision():
    # Hits at rank 2 (P@2 = 1/2) and rank 4 (P@4 = 2/4), divided by 3 relevant chunks: (0.5 + 0.5) / 3
    assert average_precision(RETRIEVED, RELEVANT, 5) == pytest.approx(1 / 3)


def test_ndcg_at_k():
    # DCG: grade 2 at rank 2 and grade 1 at rank 4.
    dcg = 2 / math.log2(3) + 1 / math.log2(5)
    # Ideal order puts the grades 2, 1, 1 at ranks 1, 2, 3.
    ideal = 2 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(dcg / ideal)


def test_ndcg_is_one_for_perfect_ranking():
    assert ndcg_at_k(["b", "d", "x"], RELEVANT, 3) == pytest.approx(1.0)


def test_ndcg_rewards_higher_grade_first():
    better = ndcg_at_k(["b", "d"], {"b": 2, "d": 1}, 2)
    worse = ndcg_at_k(["d", "b"], {"b": 2, "d": 1}, 2)
    assert better > worse


def test_no_relevant_chunk_retrieved():
    retrieved = ["a", "c", "e"]
    for metric in (precision_at_k, recall_at_k, f1_at_k, reciprocal_rank, average_precision, ndcg_at_k):
        assert metric(retrieved, RELEVANT, 3) == 0.0


def test_no_results_at_all():
    for metric in (precision_at_k, recall_at_k, f1_at_k, reciprocal_rank, average_precision, ndcg_at_k):
        assert metric([], RELEVANT, 3) == 0.0


def test_question_without_relevant_chunks_does_not_divide_by_zero():
    for metric in (recall_at_k, f1_at_k, average_precision, ndcg_at_k):
        assert metric(RETRIEVED, {}, 5) == 0.0


def test_evaluate_averages_over_questions():
    runs = [
        (["b", "a"], {"b": 1}),  # first relevant at rank 1: RR = 1, AP = 1
        (["a", "b"], {"b": 1}),  # first relevant at rank 2: RR = 1/2, AP = 1/2
    ]

    scores = evaluate(runs, k=2)

    assert scores["MRR"] == pytest.approx(0.75)
    assert scores["MAP"] == pytest.approx(0.75)
    assert scores["Precision"] == pytest.approx(0.5)
    assert scores["Recall"] == pytest.approx(1.0)
