import argparse
import math
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.search_helpers import load_stopwords, tokenize_term, tokenize_text

if __package__:
    from .inverted_index_class import BM25_B, BM25_K1, InvertedIndex
else:
    from cli.inverted_index_class import BM25_B, BM25_K1, InvertedIndex


def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()

    print("Inverted index built successfully")


def search_command(query: str) -> None:
    inverted_index = InvertedIndex()

    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Error: search index not found. Run the build command first.")
        return

    document_ids: set[int] = set()
    for token in tokenize_text(query, load_stopwords()):
        matching_terms = (
            indexed_term
            for indexed_term in inverted_index.index
            if indexed_term.startswith(token)
        )
        for matching_term in matching_terms:
            document_ids.update(inverted_index.get_documents(matching_term))

    for doc_id in sorted(document_ids)[:5]:
        movie = inverted_index.docmap[doc_id]
        print(f"{movie['title']} (ID: {doc_id})")


def tf_command(doc_id: int, term: str) -> None:
    inverted_index = InvertedIndex()

    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Error: search index not found. Run the build command first.")
        return

    token = tokenize_term(term)
    print(inverted_index.get_tf(doc_id, token))


def idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    token = tokenize_term(term)
    total_doc_count = len(inverted_index.docmap)
    term_match_doc_count = len(inverted_index.get_documents(token))

    return math.log(
        (total_doc_count + 1) / (term_match_doc_count + 1)
    )


def tfidf_command(doc_id: int, term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    token = tokenize_term(term)
    tf = inverted_index.get_tf(doc_id, token)
    total_doc_count = len(inverted_index.docmap)
    term_match_doc_count = len(inverted_index.get_documents(token))
    idf = math.log(
        (total_doc_count + 1) / (term_match_doc_count + 1)
    )

    return tf * idf


def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    token = tokenize_term(term)
    return inverted_index.get_bm25_idf(token)


def bm25_tf_command(
    doc_id: int,
    term: str,
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()

    token = tokenize_term(term)
    return inverted_index.get_bm25_tf(doc_id, token, k1, b)


def bm25_search_command(query: str, limit: int = 5) -> None:
    inverted_index = InvertedIndex()
    inverted_index.load()

    results = inverted_index.bm25_search(query, limit)
    for rank, (doc_id, score) in enumerate(results, start=1):
        movie = inverted_index.docmap[doc_id]
        print(f"{rank}. ({doc_id}) {movie['title']} - Score: {score:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    subparsers.add_parser("build", help="Build and save the inverted index")
    tf_parser = subparsers.add_parser("tf", help="Get a term's frequency in a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Single search term")
    idf_parser = subparsers.add_parser(
        "idf", help="Get the inverse document frequency of a term"
    )
    idf_parser.add_argument("term", type=str, help="Single search term")
    tfidf_parser = subparsers.add_parser(
        "tfidf", help="Get the TF-IDF score of a term in a document"
    )
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Single search term")
    bm25_idf_parser = subparsers.add_parser(
        "bm25idf", help="Get BM25 IDF score for a given term"
    )
    bm25_idf_parser.add_argument(
        "term", type=str, help="Term to get BM25 IDF score for"
    )
    bm25_tf_parser = subparsers.add_parser(
        "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument(
        "term", type=str, help="Term to get BM25 TF score for"
    )
    bm25_tf_parser.add_argument(
        "k1", type=float, nargs="?", default=BM25_K1,
        help="Tunable BM25 K1 parameter"
    )
    bm25_tf_parser.add_argument(
        "b", type=float, nargs="?", default=BM25_B,
        help="Tunable BM25 b parameter"
    )
    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "--limit", type=int, default=5, help="Maximum number of results"
    )

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            search_command(args.query)
        case "build":
            build_command()
        case "tf":
            tf_command(args.doc_id, args.term)
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(
                f"TF-IDF score of '{args.term}' in document "
                f"'{args.doc_id}': {tf_idf:.2f}"
            )
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(
                f"BM25 TF score of '{args.term}' in document "
                f"'{args.doc_id}': {bm25tf:.2f}"
            )
        case "bm25search":
            bm25_search_command(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
