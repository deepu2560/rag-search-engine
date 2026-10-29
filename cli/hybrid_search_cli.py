import argparse
import logging
import sys
import time
from pathlib import Path
from typing import cast

from sentence_transformers import CrossEncoder

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli.inverted_index_class import load_movies
from cli.lib.hybrid_search import HybridSearch, normalize_scores
from cli.lib.llm import (
    create_openrouter_client,
    evaluate_results,
    expand_query,
    rank_documents_batch,
    rewrite_query,
    score_document_relevance,
    spell_correct_query,
)
from cli.lib.semantic_search import MovieDocument

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser(
        "normalize", help="Min-max normalize a list of scores"
    )
    normalize_parser.add_argument("scores", type=float, nargs="*")
    weighted_search_parser = subparsers.add_parser(
        "weighted-search", help="Search using weighted hybrid scores"
    )
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5)
    weighted_search_parser.add_argument("--limit", type=int, default=5)
    rrf_search_parser = subparsers.add_parser(
        "rrf-search", help="Search using reciprocal rank fusion"
    )
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("-k", type=int, default=60)
    rrf_search_parser.add_argument("--limit", type=int, default=5)
    rrf_search_parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Use an LLM to rate the final results from 0 to 3",
    )
    rrf_search_parser.add_argument(
        "--enhance",
        type=str,
        choices=["spell", "rewrite", "expand"],
        help="Query enhancement method",
    )
    rrf_search_parser.add_argument(
        "--rerank-method",
        type=str,
        choices=["individual", "batch", "cross_encoder"],
        help="Re-ranking method",
    )

    args = parser.parse_args()

    match args.command:
        case "normalize":
            for score in normalize_scores(args.scores):
                print(f"* {score:.4f}")
        case "weighted-search":
            documents = cast(list[MovieDocument], load_movies())
            search = HybridSearch(documents)
            results = search.weighted_search(
                args.query, args.alpha, args.limit
            )

            for index, result in enumerate(results, start=1):
                print(f"{index}. {result['title']}")
                print(f"  Hybrid Score: {result['hybrid_score']:.3f}")
                print(
                    f"  BM25: {result['bm25_score']:.3f}, "
                    f"Semantic: {result['semantic_score']:.3f}"
                )
                print(f"  {result['document']}...")
        case "rrf-search":
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("DEBUG: %(message)s"))
                logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            query = args.query
            logger.debug("Original query: %r", query)
            if args.enhance is not None:
                if args.enhance == "spell":
                    enhanced_query = spell_correct_query(query)
                elif args.enhance == "rewrite":
                    enhanced_query = rewrite_query(query)
                else:
                    enhanced_query = expand_query(query)
                print(
                    f"Enhanced query ({args.enhance}): '{query}' -> "
                    f"'{enhanced_query}'\n"
                )
                query = enhanced_query

            logger.debug("Query after enhancements (%s): %r", args.enhance or "none", query)

            documents = cast(list[MovieDocument], load_movies())
            search = HybridSearch(documents)
            search_limit = (
                args.limit * 5
                if args.rerank_method is not None
                else args.limit
            )
            results = search.rrf_search(query, args.k, search_limit)
            logger.debug("Results after RRF search (%d): %s", len(results), results)

            rerank_scores: dict[int, float] = {}
            cross_encoder_scores: dict[int, float] = {}
            if args.rerank_method is not None:
                print(
                    f"Re-ranking top {args.limit} results using "
                    f"{args.rerank_method} method..."
                )
                if args.rerank_method == "individual":
                    client = create_openrouter_client()
                    for index, result in enumerate(results):
                        rerank_scores[result["id"]] = score_document_relevance(
                            query, result, client
                        )
                        if index < len(results) - 1:
                            time.sleep(3)

                    # Python's sort is stable, so documents with the same LLM
                    # score retain their original RRF ordering.
                    results.sort(
                        key=lambda result: rerank_scores[result["id"]],
                        reverse=True,
                    )
                elif args.rerank_method == "batch":
                    client = create_openrouter_client()
                    ranked_ids = rank_documents_batch(query, results, client)
                    rerank_ranks = {
                        document_id: rank
                        for rank, document_id in enumerate(ranked_ids, start=1)
                    }
                    results.sort(
                        key=lambda result: rerank_ranks.get(
                            result["id"], float("inf")
                        )
                    )
                else:
                    pairs = []
                    for result in results:
                        pairs.append(
                            [
                                query,
                                f"{result.get('title', '')} - "
                                f"{result.get('document', '')}",
                            ]
                        )

                    cross_encoder = CrossEncoder(
                        "cross-encoder/ms-marco-TinyBERT-L2-v2"
                    )
                    scores = cross_encoder.predict(pairs)
                    cross_encoder_scores = {
                        result["id"]: float(score)
                        for result, score in zip(results, scores)
                    }
                    results.sort(
                        key=lambda result: cross_encoder_scores[result["id"]],
                        reverse=True,
                    )
                results = results[: args.limit]
                print(
                    f"Reciprocal Rank Fusion Results for '{query}' "
                    f"(k={args.k}):\n"
                )

            logger.debug(
                "Final results after re-ranking (%s): %s",
                args.rerank_method or "none",
                [
                    {
                        **result,
                        **({"rerank_score": rerank_scores[result["id"]]}
                           if args.rerank_method == "individual" else {}),
                        **({"rerank_rank": rerank_ranks.get(result["id"])}
                           if args.rerank_method == "batch" else {}),
                        **({"cross_encoder_score": cross_encoder_scores[result["id"]]}
                           if args.rerank_method == "cross_encoder" else {}),
                    }
                    for result in results
                ],
            )

            for index, result in enumerate(results, start=1):
                print(f"{index}. {result['title']}")
                if args.rerank_method == "individual":
                    print(
                        "   Re-rank Score: "
                        f"{rerank_scores[result['id']]:.3f}/10"
                    )
                    prefix = "   "
                elif args.rerank_method == "batch":
                    print(
                        f"   Re-rank Rank: {rerank_ranks[result['id']]}"
                    )
                    prefix = "   "
                elif args.rerank_method == "cross_encoder":
                    print(
                        "   Cross Encoder Score: "
                        f"{cross_encoder_scores[result['id']]:.3f}"
                    )
                    prefix = "   "
                else:
                    prefix = "  "
                print(f"{prefix}RRF Score: {result['rrf_score']:.3f}")
                print(
                    f"{prefix}BM25 Rank: {result['bm25_rank']}, "
                    f"Semantic Rank: {result['semantic_rank']}"
                )
                print(f"{prefix}{result['document']}...")
                if index < len(results):
                    print()
            if args.evaluate:
                evaluation_scores = evaluate_results(query, results)
                print()
                for index, (result, score) in enumerate(
                    zip(results, evaluation_scores), start=1
                ):
                    print(f"{index}. {result['title']}: {score}/3")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
