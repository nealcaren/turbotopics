"""DataFrame-aware corpus construction and metadata alignment.

Topic models drop documents that become empty after tokenization and vocabulary
pruning. When document-level covariates live in a separate array, that silently
misaligns them with the surviving documents, which quietly corrupts any STM
prevalence regression. These helpers keep text and metadata bound together.
"""

from __future__ import annotations

from . import Corpus, tokenize


def _is_polars(obj) -> bool:
    """True for a Polars DataFrame/Series, without importing Polars (it is an
    optional dependency)."""
    return type(obj).__module__.split(".", 1)[0] == "polars"


def from_dataframe(
    df,
    *,
    text_col,
    metadata_cols=None,
    tokenizer=None,
    stopwords=None,
    min_length=1,
    min_doc_freq=1,
    max_doc_fraction=1.0,
    min_cf=0,
    rm_top=0,
):
    """Build a :class:`Corpus` from a pandas or Polars DataFrame, keeping
    per-document metadata aligned to the documents that survive pruning.

    ``df[text_col]`` is tokenized (with ``tokenizer`` if given, otherwise
    :func:`topica.tokenize`), a :class:`Corpus` is built with the usual pruning
    options, and the surviving rows of ``metadata_cols`` (default: every column
    except ``text_col``) are attached as ``corpus.metadata`` — a DataFrame of the
    same kind you passed in (pandas in, pandas out; Polars in, Polars out),
    aligned one-to-one with the corpus documents, in the same row order. Feed
    that metadata straight to an STM prevalence design with no manual alignment.

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame
        One row per document.
    text_col : str
        Column holding the document text.
    metadata_cols : sequence[str], optional
        Columns to carry as aligned metadata. Defaults to all columns except
        ``text_col``.
    tokenizer : callable, optional
        ``str -> list[str]``. Defaults to :func:`topica.tokenize` with the
        ``stopwords`` and ``min_length`` arguments below.
    """
    texts = list(df[text_col])  # pandas Series and Polars Series both iterate to values
    if tokenizer is None:
        sw = list(stopwords) if stopwords is not None else None
        docs = [
            tokenize(t if isinstance(t, str) else "", stopwords=sw, min_length=min_length)
            for t in texts
        ]
    else:
        docs = [tokenizer(t if isinstance(t, str) else "") for t in texts]

    corpus = Corpus.from_documents(
        docs,
        min_doc_freq=min_doc_freq,
        max_doc_fraction=max_doc_fraction,
        min_cf=min_cf,
        rm_top=rm_top,
    )

    cols = (
        list(metadata_cols)
        if metadata_cols is not None
        else [c for c in df.columns if c != text_col]
    )
    idx = corpus.kept_indices
    if _is_polars(df):
        corpus.metadata = df[list(idx)].select(cols)  # row-select then column-select
    else:
        corpus.metadata = df.iloc[idx][cols].reset_index(drop=True)
    return corpus


def align(x, corpus):
    """Realign an external covariate array, DataFrame, Series, or list to the
    documents a :class:`Corpus` kept after pruning. Accepts pandas and Polars
    DataFrames/Series, numpy arrays, and plain lists.

    Use it when your covariates were built against the original documents and
    the corpus dropped some during pruning::

        corpus = topica.Corpus.from_documents(docs, min_doc_freq=5)
        X = topica.align(X, corpus)          # now aligned to corpus rows
        model.fit(corpus, X, prevalence_names=names)
    """
    idx = corpus.kept_indices
    if hasattr(x, "iloc"):  # pandas DataFrame / Series
        return x.iloc[idx].reset_index(drop=True)
    if _is_polars(x):  # polars DataFrame / Series: positional row selection
        return x[list(idx)]
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return x[idx]
    except ImportError:
        pass
    return [x[i] for i in idx]
