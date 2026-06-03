from __future__ import annotations

from typing import Any, Sequence
import numpy
import numpy.typing

from ._turbotopics import (
    LDA as LDA,
    DMR as DMR,
    LabeledLDA as LabeledLDA,
    SAGE as SAGE,
    CTM as CTM,
    STM as STM,
    HDP as HDP,
    DTM as DTM,
    SupervisedLDA as SupervisedLDA,
    Corpus as Corpus,
    tokenize as tokenize,
    DEFAULT_TOKEN_REGEX as DEFAULT_TOKEN_REGEX,
    __version__ as __version__,
)
from . import stm as stm

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
    "DEFAULT_TOKEN_REGEX",
    "__version__",
]
