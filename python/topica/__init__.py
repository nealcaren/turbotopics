"""topica: fast SparseLDA topic modeling (MALLET's algorithm) in Rust.

The heavy lifting lives in the compiled extension ``topica._topica``;
this module just re-exports its public surface so ``import topica`` works
and editors/type-checkers see a stable namespace.
"""

from ._topica import (
    LDA,
    DMR,
    LabeledLDA,
    SAGE,
    CTM,
    STM,
    HDP,
    DTM,
    SupervisedLDA,
    PT,
    GSDMM,
    SeededLDA,
    KeyATM,
    PA,
    HLDA,
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


def summary(model, topn=8):
    """A human-readable overview of a fitted model (à la tomotopy's ``summary``).

    Returns a multi-line string: the model's repr, its key scalar attributes
    (num_topics, concentrations, etc.), the vocabulary size, and the top words of
    each topic. Pass to ``print``. For models whose ``top_words`` needs extra
    arguments (``DTM`` by time, ``SAGE`` by group) the per-topic word lists are
    omitted.
    """
    lines = [repr(model)]
    for attr in ("num_topics", "num_times", "num_groups", "alpha", "gamma",
                 "sigma2", "bound"):
        try:
            value = getattr(model, attr)
        except Exception:
            continue
        if not callable(value):
            lines.append(f"  {attr}: {value}")
    try:
        lines.append(f"  vocab_size: {len(model.vocabulary)}")
    except Exception:
        pass
    try:
        tops = model.top_words(topn)
        if isinstance(tops, list) and tops and isinstance(tops[0], list):
            for i, words in enumerate(tops):
                lines.append(f"  topic {i}: " + " ".join(w for w, _ in words))
    except Exception:
        pass
    return "\n".join(lines)


from . import stm  # noqa: E402  (stm imports names defined above)
from . import keyatm  # noqa: E402  (keyATM-specific workflow helpers)
from . import diagnostics  # noqa: E402
from . import phrases  # noqa: E402
from .coherence import (  # noqa: E402
    coherence,
    topic_diversity,
    exclusivity,
    word_intrusion,
    document_intrusion,
)
from .diagnostics import (  # noqa: E402  general, model-agnostic post-hoc analyses
    frex,
    label_topics,
    topic_table,
    topic_correlation,
    find_thoughts,
    find_thoughts_html,
    quality_frontier,
    bootstrap_stability,
    search_k,
    plot_search_k,
    plot_topic_discovery,
    relevance,
    prepare_pyldavis,
    check_residuals,
    align_topics,
    topic_stability,
)
from .keywords import fighting_words, top_fighting_words  # noqa: E402
from .embedding import EmbeddingLDA, embedding_seeds  # noqa: E402
from .preprocess import split_documents  # noqa: E402
from .phrases import learn_phrases, apply_phrases, Phrases  # noqa: E402
from .frames import from_dataframe, align  # noqa: E402
from .formulas import design_matrix  # noqa: E402

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
    "PT",
    "GSDMM",
    "SeededLDA",
    "KeyATM",
    "PA",
    "HLDA",
    "Corpus",
    "tokenize",
    "one_hot",
    "stm",
    "keyatm",
    "phrases",
    "coherence",
    "topic_diversity",
    "exclusivity",
    "word_intrusion",
    "document_intrusion",
    "frex",
    "label_topics",
    "topic_table",
    "topic_correlation",
    "find_thoughts",
    "search_k",
    "plot_search_k",
    "plot_topic_discovery",
    "relevance",
    "prepare_pyldavis",
    "check_residuals",
    "align_topics",
    "topic_stability",
    "find_thoughts_html",
    "quality_frontier",
    "bootstrap_stability",
    "fighting_words",
    "top_fighting_words",
    "EmbeddingLDA",
    "embedding_seeds",
    "split_documents",
    "from_dataframe",
    "align",
    "design_matrix",
    "summary",
    "diagnostics",
    "keywords",
    "preprocess",
    "learn_phrases",
    "apply_phrases",
    "Phrases",
    "DEFAULT_TOKEN_REGEX",
    "__version__",
]
