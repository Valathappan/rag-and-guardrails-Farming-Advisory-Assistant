"""Search metrics, written in plain Python.

Every function takes the ranked chunk ids the search returned and a dict of
relevance grades ({chunk_id: grade}). Chunks missing from the dict have grade 0.
Any grade above 0 counts as relevant, except in NDCG, which uses the grades.
"""

import math


def is_relevant(chunk_id: str, relevant: dict[str, int]) -> bool:
    return relevant.get(chunk_id, 0) > 0


def count_relevant(relevant: dict[str, int]) -> int:
    return sum(1 for grade in relevant.values() if grade > 0)


def precision_at_k(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """Relevant results in the top K, divided by K."""
    hits = sum(1 for chunk_id in retrieved[:k] if is_relevant(chunk_id, relevant))
    return hits / k


def recall_at_k(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """Relevant results in the top K, divided by all relevant chunks."""
    total = count_relevant(relevant)
    if total == 0:
        return 0.0
    hits = sum(1 for chunk_id in retrieved[:k] if is_relevant(chunk_id, relevant))
    return hits / total


def f1_at_k(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """Harmonic mean of Precision@K and Recall@K."""
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """1 / rank of the first relevant result in the top K, or 0 if there is none."""
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if is_relevant(chunk_id, relevant):
            return 1 / rank
    return 0.0


def average_precision(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """Sum of Precision@i at every rank i in the top K that holds a relevant result,
    divided by all relevant chunks."""
    total = count_relevant(relevant)
    if total == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if is_relevant(chunk_id, relevant):
            hits += 1
            precision_sum += hits / rank
    return precision_sum / total


def dcg(grades: list[int]) -> float:
    """Discounted cumulative gain: each grade divided by log2(rank + 1)."""
    return sum(grade / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1))


def ndcg_at_k(retrieved: list[str], relevant: dict[str, int], k: int) -> float:
    """DCG of the top K divided by the DCG of the perfect top K."""
    actual = dcg([relevant.get(chunk_id, 0) for chunk_id in retrieved[:k]])
    ideal = dcg(sorted(relevant.values(), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return actual / ideal


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_question(retrieved: list[str], relevant: dict[str, int], k: int) -> dict[str, float]:
    """All six metrics for one question. Averaged over questions, RR becomes MRR and AP becomes MAP."""
    return {
        "Precision": precision_at_k(retrieved, relevant, k),
        "Recall": recall_at_k(retrieved, relevant, k),
        "F1": f1_at_k(retrieved, relevant, k),
        "MRR": reciprocal_rank(retrieved, relevant, k),
        "MAP": average_precision(retrieved, relevant, k),
        "NDCG": ndcg_at_k(retrieved, relevant, k),
    }


def evaluate(runs: list[tuple[list[str], dict[str, int]]], k: int) -> dict[str, float]:
    """Average every metric over all questions. `runs` holds (retrieved ids, relevance grades) per question."""
    scores = [evaluate_question(retrieved, relevant, k) for retrieved, relevant in runs]
    return {name: mean([score[name] for score in scores]) for name in scores[0]} if scores else {}
