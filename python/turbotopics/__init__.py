"""turbotopics: fast SparseLDA topic modeling (MALLET's algorithm) in Rust.

The heavy lifting lives in the compiled extension ``turbotopics._turbotopics``;
this module just re-exports its public surface so ``import turbotopics`` works
and editors/type-checkers see a stable namespace.
"""

from ._turbotopics import (
    LDA,
    DMR,
    LabeledLDA,
    SAGE,
    CTM,
    STM,
    HDP,
    DTM,
    SupervisedLDA,
    Corpus,
    tokenize,
    DEFAULT_TOKEN_REGEX,
    __version__,
)


def one_hot(values, *, drop_first=True, prefix=""):
    """One-hot encode a categorical covariate for use as DMR features.

    Given a sequence of category labels (one per document), returns
    ``(matrix, names)`` where ``matrix`` is a ``(num_docs, num_categories)``
    float array of 0/1 indicators and ``names`` are the corresponding column
    names. With ``drop_first=True`` (default) the first category (sorted) is
    omitted as the reference level, which avoids collinearity with the DMR
    intercept. Pass the result straight to ``DMR.fit(docs, matrix,
    feature_names=names)``; combine multiple covariates with
    ``numpy.hstack``.
    """
    import numpy as np

    values = list(values)
    categories = sorted(set(values))
    if drop_first and categories:
        categories = categories[1:]
    index = {c: j for j, c in enumerate(categories)}
    matrix = np.zeros((len(values), len(categories)), dtype=np.float64)
    for i, v in enumerate(values):
        j = index.get(v)
        if j is not None:
            matrix[i, j] = 1.0
    names = [f"{prefix}{c}" for c in categories]
    return matrix, names


from . import stm  # noqa: E402  (stm imports names defined above)

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
    "DEFAULT_TOKEN_REGEX",
    "__version__",
]
