import argparse
import json
import sys
from pathlib import Path
from typing import cast

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cli.inverted_index_class import load_movies
from cli.lib.semantic_search import (
    ChunkedSemanticSearch,
    MovieDocument,
    SemanticSearch,
    embed_query_text,
    embed_text,
    semantic_chunk_text,
    verify_embeddings,
    verify_model,
)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 0) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero")
    if overlap < 0:
        raise ValueError("Overlap must not be negative")
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than the chunk size")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for index in range(0, len(words), step):
        chunk_words = words[index:index + chunk_size]
        chunks.append(" ".join(chunk_words))
        if index + chunk_size >= len(words):
            break

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify", help="Verify the semantic search model")
    embed_text_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for text"
    )
    embed_text_parser.add_argument("text", type=str, help="Text to embed")
    embed_query_parser = subparsers.add_parser(
        "embed_query", help="Generate an embedding for a search query"
    )
    embed_query_parser.add_argument("query", type=str, help="Query to embed")
    subparsers.add_parser(
        "verify_embeddings", help="Verify the movie embeddings cache"
    )
    search_parser = subparsers.add_parser(
        "search", help="Search movies by semantic similarity"
    )
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )
    chunk_parser = subparsers.add_parser(
        "chunk", help="Split text into fixed-size word chunks"
    )
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument(
        "--chunk-size", type=int, default=200,
        help="Number of words per chunk"
    )
    chunk_parser.add_argument(
        "--overlap", type=int, default=0,
        help="Number of words shared between consecutive chunks"
    )
    semantic_chunk_parser = subparsers.add_parser(
        "semantic_chunk", help="Split text into sentence-aware chunks"
    )
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument(
        "--max-chunk-size", type=int, default=4,
        help="Maximum number of sentences per chunk"
    )
    semantic_chunk_parser.add_argument(
        "--overlap", type=int, default=0,
        help="Number of sentences shared between consecutive chunks"
    )
    subparsers.add_parser(
        "embed_chunks", help="Load or build embeddings for movie chunks"
    )
    search_chunked_parser = subparsers.add_parser(
        "search_chunked", help="Search movies using chunk embeddings"
    )
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "embed_query":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "search":
            semantic_search = SemanticSearch()
            with open("data/movies.json", "r", encoding="utf-8") as file:
                documents = json.load(file)["movies"]
            semantic_search.load_or_create_embeddings(documents)
            results = semantic_search.search(args.query, args.limit)

            for rank, result in enumerate(results, start=1):
                print(
                    f"{rank}. {result['title']} "
                    f"(score: {result['score']:.4f})"
                )
                print(f"  {result['description']}\n")
        case "chunk":
            chunks = chunk_text(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for number, chunk in enumerate(chunks, start=1):
                print(f"{number}. {chunk}")
        case "semantic_chunk":
            chunks = semantic_chunk_text(
                args.text, args.max_chunk_size, args.overlap
            )
            print(f"Semantically chunking {len(args.text)} characters")
            for number, chunk in enumerate(chunks, start=1):
                print(f"{number}. {chunk}")
        case "embed_chunks":
            with open("data/movies.json", "r", encoding="utf-8") as file:
                documents = cast(
                    list[MovieDocument], json.load(file)["movies"]
                )
            chunked_search = ChunkedSemanticSearch()
            embeddings = chunked_search.load_or_create_chunk_embeddings(
                documents
            )
            print(f"Generated {len(embeddings)} chunked embeddings")
        case "search_chunked":
            documents = cast(list[MovieDocument], load_movies())
            chunked_search = ChunkedSemanticSearch()
            chunked_search.load_or_create_chunk_embeddings(documents)
            results = chunked_search.search_chunks(args.query, args.limit)

            for index, result in enumerate(results, start=1):
                print(
                    f"\n{index}. {result['title']} "
                    f"(score: {result['score']:.4f})"
                )
                print(f"   {result['document']}...")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
