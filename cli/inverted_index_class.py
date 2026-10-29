import json
import math
import os
import pickle
from collections import Counter
from typing import TypedDict, cast

from helpers.search_helpers import load_stopwords, tokenize_text

BM25_B = 0.75
BM25_K1 = 1.5
CACHE_DIR = "cache"


class Movie(TypedDict):
    id: int
    title: str
    description: str


def load_movies() -> list[Movie]:
    with open("data/movies.json", "r", encoding="utf-8") as file:
        return cast(list[Movie], json.load(file)["movies"])


class InvertedIndex:
    def __init__(self):
        # token -> set of document IDs
        self.index: dict[str, set[int]] = {}

        # document ID -> complete movie object
        self.docmap: dict[int, Movie] = {}

        # document ID -> token counts
        self.term_frequencies: dict[int, Counter[str]] = {}

        # document ID -> number of tokens
        self.doc_lengths: dict[int, int] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(
        self, doc_id: int, text: str, stopwords: set[str] | None = None
    ) -> None:
        if stopwords is None:
            stopwords = load_stopwords()
        tokens = tokenize_text(text, stopwords)
        self.term_frequencies[doc_id] = Counter()
        self.doc_lengths[doc_id] = len(tokens)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()

            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term: str) -> list[int]:
        document_ids = self.index.get(term, set())

        return sorted(document_ids)

    def get_tf(self, doc_id: int, term: str) -> int:
        term_frequency = self.term_frequencies.get(doc_id)

        if term_frequency is None:
            return 0

        return term_frequency.get(term, 0)

    def get_bm25_idf(self, term: str) -> float:
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.get_documents(term))

        return math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0

        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_tf(
        self,
        doc_id: int,
        term: str,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> float:
        tf = self.get_tf(doc_id, term)
        if tf == 0:
            return 0.0

        avg_doc_length = self.__get_avg_doc_length()
        doc_length = self.doc_lengths.get(doc_id, 0)
        length_normalization = 1 - b + b * (doc_length / avg_doc_length)

        return (tf * (k1 + 1)) / (tf + k1 * length_normalization)

    def bm25(self, doc_id: int, term: str) -> float:
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)

        return bm25_tf * bm25_idf

    def bm25_search(self, query: str, limit: int) -> list[tuple[int, float]]:
        tokens = tokenize_text(query, load_stopwords())
        scores: dict[int, float] = {}

        for doc_id in self.docmap:
            scores[doc_id] = sum(
                self.bm25(doc_id, token) for token in tokens
            )

        ranked_documents = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )
        return ranked_documents[:limit]

    def build(self) -> None:
        movies = load_movies()
        stopwords = load_stopwords()

        for movie in movies:
            doc_id = movie["id"]

            # Store complete movie
            self.docmap[doc_id] = movie

            # Searchable movie content
            text = f"{movie['title']} {movie['description']}"

            # Add searchable content to inverted index
            self.__add_document(doc_id, text, stopwords)

    def save(self) -> None:
        # Create cache directory if it doesn't exist
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Save inverted index
        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)

        # Save document map
        with open("cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)

        # Save term frequencies
        with open("cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_frequencies, file)

        # Save document lengths
        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self) -> None:
        with open(self.index_path, "rb") as file:
            self.index = cast(dict[str, set[int]], pickle.load(file))

        with open("cache/docmap.pkl", "rb") as file:
            self.docmap = cast(dict[int, Movie], pickle.load(file))

        with open("cache/term_frequencies.pkl", "rb") as file:
            self.term_frequencies = cast(
                dict[int, Counter[str]], pickle.load(file)
            )

        with open(self.doc_lengths_path, "rb") as file:
            self.doc_lengths = cast(dict[int, int], pickle.load(file))
