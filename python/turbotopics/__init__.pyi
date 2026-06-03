from __future__ import annotations

from typing import Sequence
import numpy
import numpy.typing

from ._turbotopics import (
    LDA as LDA,
    DMR as DMR,
    LabeledLDA as LabeledLDA,
    SAGE as SAGE,
    CTM as CTM,
    STM as STM,
    Corpus as Corpus,
    tokenize as tokenize,
    DEFAULT_TOKEN_REGEX as DEFAULT_TOKEN_REGEX,
    __version__ as __version__,
)

def one_hot(
    values: Sequence[object],
    *,
    drop_first: bool = True,
    prefix: str = "",
) -> tuple[numpy.typing.NDArray[numpy.float64], list[str]]:
    """One-hot encode a categorical covariate into (matrix, names) for DMR.fit."""
    ...


__all__ = [
    "LDA",
    "DMR",
    "LabeledLDA",
    "SAGE",
    "CTM",
    "STM",
    "Corpus",
    "tokenize",
    "one_hot",
    "DEFAULT_TOKEN_REGEX",
    "__version__",
]
