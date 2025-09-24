"""Reusable text preprocessing and modeling pipeline components.

This module centralises the logic required to build an NLP pipeline that
combines TF-IDF, Word2Vec and simple meta-features while avoiding data
leakage.  The exposed ``build_text_classification_pipeline`` helper returns
an sklearn ``Pipeline`` ready to be used with cross-validation.
"""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable, List, Optional, Sequence, Set

import numpy as np
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "DEFAULT_ARTIFACTS",
    "normalize_historic_text",
    "preprocess_text",
    "HistoricTextPreprocessor",
    "MetaFeaturesExtractor",
    "Word2VecVectorizer",
    "build_text_feature_union",
    "build_text_classification_pipeline",
]

# ---------------------------------------------------------------------------
# Text normalisation utilities
# ---------------------------------------------------------------------------

_RE_DEHYPHEN = re.compile(r"-\s*\n\s*")
_RE_WS = re.compile(r"\s+")
_RE_OCR_NOISE = re.compile(r"[\\^?:*\[\]\(\)\|;]")
_SOFT_HYPHEN = "\u00AD"

DEFAULT_ARTIFACTS: Set[str] = {
    "■",
    "•",
    "¬",
    "©",
    "®",
    "€",
    "£",
    "§",
    "¶",
    "‡",
    "†",
    "÷",
    "¿",
    "¡",
    "~",
    "_",
    "{",
    "}",
    "<",
    ">",
    "”",
    "“",
    "‘",
    "’",
}


def _ensure_text_list(texts: Sequence[str]) -> List[str]:
    return ["" if text is None else str(text) for text in texts]


def normalize_historic_text(
    text: str,
    *,
    remove_artifacts: bool = True,
    artifacts: Optional[Iterable[str]] = None,
    dehyphenate: bool = True,
    collapse_spaces: bool = True,
    map_long_s_to_s: bool = True,
    correct_common_ocr_errors: bool = True,
) -> str:
    """Aggressively normalise OCR-heavy historic texts.

    The routine performs multiple steps including unicode normalisation,
    OCR noise removal, artefact stripping and collapsing of whitespace.
    """

    if not isinstance(text, str):
        text = "" if text is None else str(text)

    normalised = unicodedata.normalize("NFKC", text)

    if correct_common_ocr_errors:
        normalised = _RE_OCR_NOISE.sub(" ", normalised)

    if dehyphenate:
        normalised = _RE_DEHYPHEN.sub("", normalised)
        normalised = normalised.replace(_SOFT_HYPHEN, "")

    if correct_common_ocr_errors:
        normalised = re.sub(r"\bmucl?as\b", "muchas", normalised)
        normalised = re.sub(r"\bdefalir\b", "de salir", normalised)
        normalised = re.sub(r"\bpefquifa\b", "pesquisa", normalised)
        normalised = re.sub(r"\bfolre\^", "sobre", normalised)
        normalised = re.sub(r"(\w)c(\w)", r"\1e\2", normalised)
        normalised = re.sub(r"([a-z0-9])([A-ZÁÉÍÓÚ])", r"\1 \2", normalised)

    if remove_artifacts:
        arts = set(artifacts) if artifacts is not None else DEFAULT_ARTIFACTS
        if arts:
            pattern = "[" + "".join(re.escape(ch) for ch in sorted(arts)) + "]"
            normalised = re.sub(pattern, "", normalised)

    if map_long_s_to_s:
        normalised = normalised.replace("ſ", "s").replace("ƨ", "s")

    if collapse_spaces:
        normalised = _RE_WS.sub(" ", normalised).strip()

    return normalised


def preprocess_text(text: str) -> str:
    """Light-weight preprocessing tailored for historical Spanish texts."""

    cleaned = normalize_historic_text(text)
    cleaned = cleaned.lower()
    cleaned = re.sub(r"[^a-záéíóúñüç\s.,;:!?¿¡-]", " ", cleaned)
    tokens = cleaned.split()
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------


class HistoricTextPreprocessor(BaseEstimator, TransformerMixin):
    """Sklearn transformer that applies :func:`preprocess_text` to each sample."""

    def __init__(self):
        self._fn = preprocess_text

    def fit(self, X, y=None):  # type: ignore[override]
        _ = self
        return self

    def transform(self, X, y=None):  # type: ignore[override]
        texts = _ensure_text_list(X)
        processed = [self._fn(text) for text in texts]
        return np.asarray(processed, dtype=object)


class MetaFeaturesExtractor(BaseEstimator, TransformerMixin):
    """Compute simple length and lexical statistics as dense features."""

    def fit(self, X, y=None):  # type: ignore[override]
        _ = X, y
        return self

    def transform(self, X, y=None):  # type: ignore[override]
        texts = _ensure_text_list(X)
        features: List[List[float]] = []
        for text in texts:
            tokens = text.split()
            char_count = float(len(text))
            word_count = float(len(tokens))

            if tokens:
                avg_word_length = float(np.mean([len(token) for token in tokens]))
                unique_ratio = float(len(set(tokens)) / word_count)
                long_words_ratio = float(sum(1 for token in tokens if len(token) > 6) / word_count)
            else:
                avg_word_length = 0.0
                unique_ratio = 0.0
                long_words_ratio = 0.0

            punctuation_density = (
                float(sum(1 for char in text if char in ".,;:!?¿¡")) / char_count
                if char_count
                else 0.0
            )
            digit_ratio = (
                float(sum(1 for char in text if char.isdigit()) / char_count)
                if char_count
                else 0.0
            )

            features.append(
                [
                    avg_word_length,
                    unique_ratio,
                    punctuation_density,
                    math.log1p(char_count),
                    math.log1p(word_count),
                    long_words_ratio,
                    digit_ratio,
                ]
            )

        dense = np.asarray(features, dtype=np.float32)
        return csr_matrix(dense)


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    """Train a Word2Vec model and produce average document vectors."""

    def __init__(
        self,
        *,
        vector_size: int = 150,
        window: int = 8,
        min_count: int = 3,
        workers: int = 1,
        sg: int = 1,
        epochs: int = 15,
        random_state: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.epochs = epochs
        self.random_state = random_state

    def fit(self, X, y=None):  # type: ignore[override]
        texts = _ensure_text_list(X)
        sentences = [text.split() for text in texts]
        self.model_ = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs,
            seed=self.random_state,
        )
        return self

    def transform(self, X, y=None):  # type: ignore[override]
        check_is_fitted(self, "model_")
        texts = _ensure_text_list(X)
        vectors = [self._document_vector(text) for text in texts]
        dense = np.asarray(vectors, dtype=np.float32)
        return csr_matrix(dense)

    def _document_vector(self, text: str) -> np.ndarray:
        tokens = [token for token in text.split() if token in self.model_.wv]
        if not tokens:
            return np.zeros(self.model_.vector_size, dtype=np.float32)
        embeddings = [self.model_.wv[token] for token in tokens]
        return np.mean(embeddings, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Pipeline factory helpers
# ---------------------------------------------------------------------------


def build_text_feature_union(
    tfidf_params: Optional[dict] = None,
    w2v_params: Optional[dict] = None,
) -> FeatureUnion:
    tfidf_defaults = dict(
        max_features=30000,
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.75,
        sublinear_tf=True,
        analyzer="word",
    )
    if tfidf_params:
        tfidf_defaults.update(tfidf_params)

    w2v_defaults = dict(
        vector_size=150,
        window=8,
        min_count=3,
        workers=1,
        sg=1,
        epochs=15,
        random_state=42,
    )
    if w2v_params:
        w2v_defaults.update(w2v_params)

    return FeatureUnion(
        transformer_list=[
            ("tfidf", TfidfVectorizer(**tfidf_defaults)),
            ("word2vec", Word2VecVectorizer(**w2v_defaults)),
            ("meta", MetaFeaturesExtractor()),
        ]
    )


def build_text_classification_pipeline(
    *,
    tfidf_params: Optional[dict] = None,
    w2v_params: Optional[dict] = None,
    classifier: Optional[LogisticRegression] = None,
    random_state: int = 42,
) -> Pipeline:
    """Create a full sklearn Pipeline ready for cross-validation."""

    features = build_text_feature_union(tfidf_params=tfidf_params, w2v_params=w2v_params)
    if classifier is None:
        classifier = LogisticRegression(
            max_iter=1500,
            solver="saga",
            penalty="l2",
            random_state=random_state,
            n_jobs=-1,
        )
    return Pipeline(
        steps=[
            ("preprocess", HistoricTextPreprocessor()),
            ("features", features),
            ("classifier", classifier),
        ]
    )
