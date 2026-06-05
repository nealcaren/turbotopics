from __future__ import annotations

from typing import Any, Sequence
import numpy
import numpy.typing

from ._topica import (
    LDA as LDA,
    DMR as DMR,
    LabeledLDA as LabeledLDA,
    SAGE as SAGE,
    CTM as CTM,
    STM as STM,
    HDP as HDP,
    DTM as DTM,
    SupervisedLDA as SupervisedLDA,
    PT as PT,
    GSDMM as GSDMM,
    PA as PA,
    HLDA as HLDA,
    SeededLDA as SeededLDA,
    KeyATM as KeyATM,
    Corpus as Corpus,
    tokenize as tokenize,
    DEFAULT_TOKEN_REGEX as DEFAULT_TOKEN_REGEX,
    __version__ as __version__,
)
from . import stm as stm
from . import keyatm as keyatm
from .embedding import EmbeddingLDA as EmbeddingLDA, embedding_seeds as embedding_seeds

def one_hot(
    values: Sequence[object],
    *,
    drop_first: bool = True,
    prefix: str = "",
) -> tuple[numpy.typing.NDArray[numpy.float64], list[str]]:
    """One-hot encode a categorical covariate into (matrix, names) for DMR.fit."""
    ...

def coherence(
    topics: Any,
    texts: Sequence[Sequence[str]],
    *,
    coherence_type: str = "c_v",
    topn: int = 10,
    window_size: int | None = None,
    epsilon: float = 1e-12,
) -> numpy.typing.NDArray[numpy.float64]:
    """Per-topic coherence (u_mass / c_uci / c_npmi / c_v) of a model or list of
    word lists against a reference corpus `texts`. Returns shape (num_topics,)."""
    ...

def topic_diversity(topics: Any, topn: int = 25) -> float:
    """Fraction of unique words across all topics' top-`topn` words."""
    ...


def exclusivity(model_or_phi: Any, *, n: int = 10) -> numpy.typing.NDArray[numpy.float64]:
    """Per-topic exclusivity of the top-n words, shape (num_topics,). Pair with
    per-topic coherence for the coherence-vs-exclusivity quality plot."""
    ...


def word_intrusion(
    model_or_phi: Any,
    vocabulary: Sequence[str] | None = None,
    *,
    n_words: int = 5,
    seed: int = 0,
) -> list[dict]:
    """Word-intrusion test (Chang et al. 2009): per topic, top words + one
    intruder. Dict keys: topic, words (shuffled), intruder, intruder_index."""
    ...


def document_intrusion(
    model_or_theta: Any,
    texts: Sequence[str] | None = None,
    *,
    n_docs: int = 3,
    seed: int = 0,
) -> list[dict]:
    """Document-intrusion test: per topic, top docs + one low-share intruder.
    Dict keys: topic, doc_indices (shuffled), intruder_index, texts (if given)."""
    ...


# General, model-agnostic post-hoc analyses (also in topica.diagnostics).
def frex(topic_word: Any, vocabulary: Sequence[str], *, w: float = 0.5, n: int = 10) -> list:
    """FREX (frequency-exclusivity) top words per topic."""
    ...


def label_topics(topic_word: Any, vocabulary: Sequence[str], *, n: int = 10) -> list[dict]:
    """Per-topic word lists with keys prob / frex / lift / score."""
    ...


def topic_correlation(doc_topic: Any, *, threshold: float = 0.05) -> Any:
    """Topic-correlation network (.cor, .adjacency, .edges)."""
    ...


def find_thoughts(doc_topic: Any, texts: Sequence[str] | None = None, *, topic: int, n: int = 3) -> list:
    """The n documents most associated with a topic."""
    ...


def search_k(docs: Any, ks: Sequence[int], *, held_out: Any = None, **kwargs: Any) -> list[dict]:
    """Fit an LDA per K; report coherence, exclusivity, and (optional) perplexity."""
    ...


def relevance(
    topic_word: Any,
    vocabulary: Sequence[str],
    *,
    topic: int | None = None,
    lam: float = 0.6,
    n: int = 10,
    term_frequency: Any = None,
) -> list:
    """LDAvis word relevance (Sievert & Shirley 2014)."""
    ...


def prepare_pyldavis(model: Any, docs: Any, **kwargs: Any) -> Any:
    """Build the LDAvis intertopic-distance view (pyLDAvis PreparedData or inputs)."""
    ...


def check_residuals(model: Any, docs: Any, *, tol: float = 0.01) -> Any:
    """Taddy (2012) residual-dispersion test for whether K is too small."""
    ...


def align_topics(a: Any, b: Any, *, metric: str = "cosine") -> list:
    """One-to-one topic matching across two fits (Hungarian)."""
    ...


def topic_stability(runs: Any, *, topn: int = 10, metric: str = "cosine") -> float:
    """Term-centric topic stability across fits (Greene et al. 2014)."""
    ...


__all__ = [
    "LDA",
    "DMR",
    "LabeledLDA",
    "SAGE",
    "CTM",
    "STM",
    "HDP",
    "DTM",
    "SupervisedLDA",
    "Corpus",
    "tokenize",
    "one_hot",
    "stm",
    "coherence",
    "topic_diversity",
    "exclusivity",
    "word_intrusion",
    "document_intrusion",
    "frex",
    "label_topics",
    "topic_correlation",
    "find_thoughts",
    "search_k",
    "relevance",
    "prepare_pyldavis",
    "check_residuals",
    "align_topics",
    "topic_stability",
    "DEFAULT_TOKEN_REGEX",
    "__version__",
]
