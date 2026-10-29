import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict, cast

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli.inverted_index_class import load_movies
from cli.lib.hybrid_search import HybridSearch
from cli.lib.semantic_search import MovieDocument


class EvaluationCase(TypedDict):
    query: str
    relevant_docs: list[str]


class GoldenDataset(TypedDict):
    test_cases: list[EvaluationCase]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "golden_dataset.json"
    )
    with dataset_path.open(encoding="utf-8") as dataset_file:
        golden_dataset = cast(GoldenDataset, json.load(dataset_file))

    documents = cast(list[MovieDocument], load_movies())
    search = HybridSearch(documents)

    print(f"k={limit}\n")
    for test_case in golden_dataset["test_cases"]:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]
        results = search.rrf_search(query, k=60, limit=limit)
        retrieved_titles = [result["title"] for result in results]
        relevant_titles = set(relevant_docs)
        relevant_results = sum(
            title in relevant_titles for title in retrieved_titles
        )
        precision = relevant_results / len(results) if results else 0.0
        recall = (
            relevant_results / len(relevant_docs) if relevant_docs else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant: {', '.join(relevant_docs)}")
        print()


if __name__ == "__main__":
    main()
