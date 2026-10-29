import string
from collections.abc import Collection

from nltk.stem import PorterStemmer


def load_stopwords() -> set[str]:
    punctuation_table = str.maketrans("", "", string.punctuation)

    with open("data/stopwords.txt", "r", encoding="utf-8") as file:
        return {
            word.strip().lower().translate(punctuation_table)
            for word in file
            if word.strip()
        }


def tokenize_text(
    input_string: str, stopwords: Collection[str] | None = None
) -> list[str]:
    if stopwords is None:
        stopwords = []

    punctuation_table = str.maketrans("", "", string.punctuation)
    stemmer = PorterStemmer()

    normalized_text = input_string.lower().translate(punctuation_table)
    words = normalized_text.split()
    if len(stopwords):
        words = [word for word in words if word not in stopwords]
    return [stemmer.stem(word_to_stem) for word_to_stem in words]


def tokenize_term(term: str) -> str:
    tokens = tokenize_text(term)

    if len(tokens) != 1:
        raise ValueError("Term must tokenize to exactly one token")

    return tokens[0]
