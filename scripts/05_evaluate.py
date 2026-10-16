"""Search evaluation: test questions -> search pipeline -> Precision, Recall, F1, MRR, MAP, NDCG."""

import sys
from pathlib import Path

# Make the farm_advisory_assistant package in src/ importable without installing the project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json

from rich.console import Console
from rich.table import Table

from farm_advisory_assistant.config import settings
from farm_advisory_assistant.evaluation.metrics import evaluate, evaluate_question
from farm_advisory_assistant.rag.query_rewriter import rewrite_query
from farm_advisory_assistant.rag.retriever import retrieve
from farm_advisory_assistant.rag.vector_store import get_collection

console = Console()

K_VALUES = [1, 3, 5, 10]
WORST_COUNT = 3


def load_eval_set(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def find_unknown_ids(eval_set: list[dict]) -> list[str]:
    """Labelled chunk ids that are not in ChromaDB, e.g. after a change to CHUNK_SIZE or CHUNK_OVERLAP."""
    labelled = sorted({chunk_id for item in eval_set for chunk_id in item["relevant"]})
    found = set(get_collection().get(ids=labelled, include=[])["ids"])
    return [chunk_id for chunk_id in labelled if chunk_id not in found]


def run_search(eval_set: list[dict], use_rewrite: bool) -> list[dict]:
    results = []
    for number, item in enumerate(eval_set, start=1):
        query = rewrite_query(item["question"]) if use_rewrite else item["question"]
        chunks = retrieve(query, max(K_VALUES))
        console.print(f"  {number:>2}. {item['question']}")
        if use_rewrite:
            console.print(f"      [dim]query: {query}[/dim]")
        results.append({**item, "query": query, "retrieved": [chunk.id for chunk in chunks]})
    return results


def metrics_table(results: list[dict]) -> Table:
    runs = [(result["retrieved"], result["relevant"]) for result in results]
    scores = {k: evaluate(runs, k) for k in K_VALUES}

    table = Table(show_lines=True)
    table.add_column("Metric")
    for k in K_VALUES:
        table.add_column(f"K = {k}", justify="right")
    for metric in scores[K_VALUES[0]]:
        table.add_row(metric, *[f"{scores[k][metric]:.3f}" for k in K_VALUES])
    return table


def worst_table(results: list[dict], k: int) -> Table:
    scored = [(evaluate_question(r["retrieved"], r["relevant"], k), r) for r in results]
    scored.sort(key=lambda pair: (pair[0]["NDCG"], pair[0]["MAP"]))

    table = Table(show_lines=True)
    table.add_column("Question")
    table.add_column(f"NDCG@{k}", justify="right")
    table.add_column(f"MAP@{k}", justify="right")
    table.add_column(f"Top {k} results (✓ = relevant)")
    for score, result in scored[:WORST_COUNT]:
        hits = [
            f"{'✓' if result['relevant'].get(chunk_id, 0) > 0 else '✗'} {chunk_id}"
            for chunk_id in result["retrieved"][:k]
        ]
        table.add_row(result["question"], f"{score['NDCG']:.3f}", f"{score['MAP']:.3f}", "\n".join(hits))
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rewrite", action="store_true", help="search with the raw question, skipping query reformation")
    args = parser.parse_args()

    console.rule("Step 1: Load the test set")
    eval_set = load_eval_set(settings.eval_set)
    labelled = sum(len(item["relevant"]) for item in eval_set)
    console.print(f"  {len(eval_set)} questions, {labelled} labelled chunks from {settings.eval_set.name}")
    unknown = find_unknown_ids(eval_set)
    if unknown:
        console.print(f"[red]  {len(unknown)} labelled chunk ids are not in ChromaDB, for example {unknown[0]}.[/red]")
        console.print("[red]  Run scripts/01_ingest.py with the chunk settings used for labelling, or label again.[/red]")
        return

    mode = "raw question (no query reformation)" if args.no_rewrite else "reformed query"
    console.rule(f"Step 2: Search every question, top {max(K_VALUES)}, using the {mode}")
    results = run_search(eval_set, use_rewrite=not args.no_rewrite)

    console.rule("Step 3: Metrics, averaged over all questions")
    console.print(metrics_table(results))

    worst_k = 5
    console.rule(f"Step 4: The {WORST_COUNT} questions with the worst NDCG@{worst_k}")
    console.print(worst_table(results, worst_k))


if __name__ == "__main__":
    main()
