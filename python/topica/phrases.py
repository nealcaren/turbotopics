"""Phrase / n-gram (collocation) extraction for pre-processing topic models.

Detects multiword expressions — collocations such as "new york" or
"machine learning" — in tokenized documents and merges them into single
compound tokens (e.g. ``"new_york"``) before fitting a topic model. The
approach follows gensim's ``Phrases`` / ``Phraser`` interface but depends only
on the Python standard library and numpy.

Workflow
--------
1. Learn bigrams from raw tokens::

       from topica.phrases import learn_phrases, apply_phrases

       phrases1 = learn_phrases(docs, min_count=5, threshold=10.0)
       docs1    = apply_phrases(docs, phrases1)

2. Learn trigrams by composing two passes::

       phrases2 = learn_phrases(docs1, min_count=5, threshold=10.0)
       docs2    = apply_phrases(docs1, phrases2)

   ``docs2`` now contains tokens like ``"new_york_city"`` wherever all three
   words co-occurred frequently enough.

Scoring
-------
Two scoring methods are supported via the ``scoring`` keyword:

``"default"`` (gensim's original ``original_scorer``, Mikolov et al. 2013)::

    score = (count(ab) - min_count) * V / (count(a) * count(b))

where ``V`` is the vocabulary size (number of distinct tokens). Scores have no
fixed range; on real corpora (large ``V``) ``threshold`` in ``[5, 100]`` is
typical, matching gensim. Small vocabularies want a smaller threshold.

``"npmi"`` (normalized pointwise mutual information)::

    score = ln( P(ab) / (P(a) * P(b)) ) / -ln(P(ab))

where ``P(x) = count(x) / total_tokens``. Scores lie in ``[-1, 1]``; a
sensible ``threshold`` is around ``0.5`` for moderately frequent collocations
(the default of ``10.0`` is too high for npmi — pass ``threshold=0.5``
explicitly when using this scorer).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Frequency counting helpers
# ---------------------------------------------------------------------------

def _count_frequencies(
    docs: List[List[str]],
) -> Tuple[Dict[str, int], Dict[Tuple[str, str], int], int]:
    """Scan `docs` once and return unigram counts, bigram counts, total tokens.

    Returns
    -------
    unigram_counts : dict[str, int]
    bigram_counts  : dict[(str, str), int]
    total_tokens   : int   (sum of document lengths)
    """
    unigram_counts: Dict[str, int] = defaultdict(int)
    bigram_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    total_tokens = 0

    for doc in docs:
        for tok in doc:
            unigram_counts[tok] += 1
            total_tokens += 1
        for i in range(len(doc) - 1):
            bigram_counts[(doc[i], doc[i + 1])] += 1

    return dict(unigram_counts), dict(bigram_counts), total_tokens


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------

def _score_default(
    a: str,
    b: str,
    count_a: int,
    count_b: int,
    count_ab: int,
    vocab_size: int,
    min_count: float,
) -> float:
    """Gensim's original bigram score (Mikolov et al. 2013), matching gensim's
    ``original_scorer``:

    score = (count(ab) - min_count) * V / (count(a) * count(b))

    where V is the vocabulary size (number of distinct tokens). V scales every
    pair's score by the same constant, so the *ranking* is unaffected; the
    constant just sets where a given ``threshold`` falls. On real corpora V is
    large (thousands), which is why gensim's default ``threshold=10`` is
    sensible; small vocabularies want a correspondingly smaller threshold.
    """
    denom = count_a * count_b
    if denom == 0:
        return float("-inf")
    return (count_ab - min_count) * vocab_size / denom


def _score_npmi(
    a: str,
    b: str,
    count_a: int,
    count_b: int,
    count_ab: int,
    total_tokens: int,
) -> float:
    """Normalized Pointwise Mutual Information (NPMI).

    score = ln(P(ab) / (P(a) * P(b))) / -ln(P(ab))

    where P(x) = count(x) / total_tokens.

    Range: [-1, 1]. Pairs that never co-occur → -1; pairs that always appear
    together → 1. A threshold of ~0.5 is reasonable for collocations.
    """
    if total_tokens == 0 or count_a == 0 or count_b == 0 or count_ab == 0:
        return float("-inf")
    p_a = count_a / total_tokens
    p_b = count_b / total_tokens
    p_ab = count_ab / total_tokens
    # Avoid domain errors from rounding.
    if p_ab <= 0.0 or p_a <= 0.0 or p_b <= 0.0:
        return float("-inf")
    pmi = math.log(p_ab / (p_a * p_b))
    denom = -math.log(p_ab)
    if denom == 0.0:
        return 1.0
    return pmi / denom


# ---------------------------------------------------------------------------
# Public data class
# ---------------------------------------------------------------------------

@dataclass
class Phrases:
    """A learned collocation model.

    Attributes
    ----------
    bigrams : dict[tuple[str,str], float]
        Maps each accepted ``(a, b)`` pair to its score.
    min_count : int
        Minimum co-occurrence count used during training.
    threshold : float
        Minimum score used during training.
    scoring : str
        Scoring method used: ``"default"`` or ``"npmi"``.
    delimiter : str
        The character used to join tokens when transforming documents.

    Notes
    -----
    To build trigrams, learn a second :class:`Phrases` on the output of
    :meth:`transform` from the first model::

        p1   = learn_phrases(docs)
        bi   = p1.transform(docs)
        p2   = learn_phrases(bi)
        tri  = p2.transform(bi)
    """

    bigrams: Dict[Tuple[str, str], float] = field(default_factory=dict)
    min_count: int = 5
    threshold: float = 10.0
    scoring: str = "default"
    delimiter: str = "_"

    # -----------------------------------------------------------------------

    def transform(self, docs: List[List[str]]) -> List[List[str]]:
        """Merge detected adjacent bigrams in each document.

        Applies a greedy left-to-right scan: when an adjacent pair ``(a, b)``
        is a known collocation the two tokens are replaced with
        ``"a{delimiter}b"`` and the scan advances past both (non-overlapping).
        Single tokens that are not part of a detected bigram are kept as-is.

        Parameters
        ----------
        docs : list[list[str]]
            Tokenized documents.

        Returns
        -------
        list[list[str]]
            New documents with collocations merged.
        """
        return apply_phrases(docs, self)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def learn_phrases(
    docs: List[List[str]],
    *,
    min_count: int = 5,
    threshold: float = 10.0,
    scoring: str = "default",
    delimiter: str = "_",
) -> Phrases:
    """Learn a collocation (phrase) model from tokenized documents.

    Counts unigram and adjacent-bigram frequencies across all documents; scores
    each candidate bigram and keeps those meeting both the ``min_count`` and
    ``threshold`` criteria.

    Parameters
    ----------
    docs : list[list[str]]
        Tokenized documents — each document is a list of string tokens.
    min_count : int
        A bigram must appear at least this many times to be considered.
        Default ``5``.
    threshold : float
        Minimum score for a bigram to be kept.  For ``scoring="default"`` a
        value of ``10.0`` is a reasonable starting point; for
        ``scoring="npmi"`` use a value in ``[-1, 1]`` (e.g. ``0.5``).
        Default ``10.0``.
    scoring : ``"default"`` or ``"npmi"``
        Which association measure to use (see module docstring for formulas).
        Default ``"default"``.
    delimiter : str
        Character used to join tokens when transforming documents.
        Default ``"_"``.

    Returns
    -------
    Phrases
        A fitted :class:`Phrases` object whose :meth:`~Phrases.transform`
        method merges detected collocations in new documents.

    Examples
    --------
    Bigrams::

        p = learn_phrases(docs, min_count=5, threshold=10.0)
        docs_bi = p.transform(docs)

    Trigrams via composition::

        p2   = learn_phrases(docs_bi, min_count=5, threshold=10.0)
        docs_tri = p2.transform(docs_bi)
    """
    scoring = scoring.lower()
    if scoring not in ("default", "npmi"):
        raise ValueError(f"scoring must be 'default' or 'npmi', got {scoring!r}")

    unigram_counts, bigram_counts, total_tokens = _count_frequencies(docs)

    accepted: Dict[Tuple[str, str], float] = {}

    for (a, b), count_ab in bigram_counts.items():
        if count_ab < min_count:
            continue
        count_a = unigram_counts.get(a, 0)
        count_b = unigram_counts.get(b, 0)

        if scoring == "default":
            sc = _score_default(a, b, count_a, count_b, count_ab, len(unigram_counts), min_count)
        else:  # npmi
            sc = _score_npmi(a, b, count_a, count_b, count_ab, total_tokens)

        if sc >= threshold:
            accepted[(a, b)] = sc

    return Phrases(
        bigrams=accepted,
        min_count=min_count,
        threshold=threshold,
        scoring=scoring,
        delimiter=delimiter,
    )


def apply_phrases(
    docs: List[List[str]],
    phrases: Phrases,
) -> List[List[str]]:
    """Apply a :class:`Phrases` model to tokenized documents.

    Performs a greedy left-to-right scan of each document: whenever an adjacent
    pair ``(tok[i], tok[i+1])`` is a known collocation the pair is merged into
    ``"tok[i]{delimiter}tok[i+1]"`` and the cursor advances by 2 (so the merged
    token cannot overlap with the next merge). Tokens not involved in a
    collocation are passed through unchanged.

    Parameters
    ----------
    docs : list[list[str]]
        Tokenized documents.
    phrases : Phrases
        A fitted :class:`Phrases` model (from :func:`learn_phrases`).

    Returns
    -------
    list[list[str]]
        New documents with collocations merged.
    """
    bigrams = phrases.bigrams
    delim = phrases.delimiter
    result = []
    for doc in docs:
        out: List[str] = []
        i = 0
        while i < len(doc):
            if i < len(doc) - 1 and (doc[i], doc[i + 1]) in bigrams:
                out.append(doc[i] + delim + doc[i + 1])
                i += 2
            else:
                out.append(doc[i])
                i += 1
        result.append(out)
    return result


def export_phrases(phrases: Phrases) -> List[Tuple[str, float]]:
    """Return a sorted list of ``(phrase_string, score)`` pairs for inspection.

    The phrase string uses the model's delimiter (e.g. ``"new_york"``). Results
    are sorted by descending score.

    Parameters
    ----------
    phrases : Phrases
        A fitted :class:`Phrases` model.

    Returns
    -------
    list[tuple[str, float]]
        ``[(phrase_str, score), ...]`` sorted by descending score.
    """
    delim = phrases.delimiter
    return sorted(
        ((a + delim + b, sc) for (a, b), sc in phrases.bigrams.items()),
        key=lambda x: x[1],
        reverse=True,
    )


def add_ngrams(docs, ngram_range=(1, 2), min_df=1, sep="_"):
    """Expand pre-tokenized documents with contiguous n-grams.

    The mechanical, exhaustive counterpart to :func:`learn_phrases`: rather than
    keeping only statistically significant collocations, it emits *every*
    contiguous n-gram, mirroring scikit-learn's
    ``CountVectorizer(ngram_range=..., min_df=...)``. For each document and each
    ``n`` in ``range(min_n, max_n + 1)`` it adds the joined n-grams (e.g.
    ``"machine_learning"``), then drops terms occurring in fewer than ``min_df``
    documents. Use it before fitting an embedding model so its class-based TF-IDF
    topic words can include bigrams.

    Parameters
    ----------
    docs : list of token lists.
    ngram_range : ``(min_n, max_n)``. ``(1, 2)`` keeps unigrams and adds bigrams;
        ``(2, 2)`` is bigrams only.
    min_df : drop terms appearing in fewer than this many documents (an integer
        document-frequency cut, as in scikit-learn). ``1`` keeps everything.
    sep : the string joining the words of an n-gram.

    Returns
    -------
    New token lists (one per input document; an emptied document stays as an empty
    list, so the result stays aligned with any per-document embeddings).
    """
    lo, hi = ngram_range
    if lo < 1 or hi < lo:
        raise ValueError("ngram_range must be (min_n, max_n) with 1 <= min_n <= max_n")

    expanded = []
    for d in docs:
        d = list(d)
        toks = []
        for n in range(lo, hi + 1):
            if n == 1:
                toks.extend(d)
            else:
                toks.extend(sep.join(d[i:i + n]) for i in range(len(d) - n + 1))
        expanded.append(toks)

    if min_df <= 1:
        return expanded

    from collections import Counter

    df = Counter()
    for toks in expanded:
        df.update(set(toks))
    keep = {t for t, c in df.items() if c >= min_df}
    return [[t for t in toks if t in keep] for toks in expanded]


__all__ = [
    "Phrases",
    "learn_phrases",
    "apply_phrases",
    "export_phrases",
    "add_ngrams",
]
