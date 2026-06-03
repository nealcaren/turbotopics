"""Topic coherence and diversity diagnostics.

Windowed PMI-based coherence measures (Röder, Both & Hinneburg, *Exploring the
Space of Topic Coherence Measures*, WSDM 2015) alongside UMass (Mimno et al.
2011) and topic diversity (Dieng, Ruiz & Blei 2020), exposed through a single
gensim-style ``coherence_type=`` switch:

- ``"u_mass"``  — document co-occurrence, intrinsic; range roughly ``(-inf, 0]``.
- ``"c_uci"``   — pairwise PMI over a sliding window (Newman et al. 2010).
- ``"c_npmi"``  — pairwise normalized PMI; range ``[-1, 1]``.
- ``"c_v"``     — the indirect-cosine/NPMI measure that correlates best with human
  judgements in Röder et al.; range roughly ``[0, 1]``.

Every measure scores each topic's top words against a *reference corpus* of
tokenized documents. By default that is your training corpus, but — as with
gensim's :class:`CoherenceModel` — you can pass any external reference (e.g. a
Wikipedia dump) via ``texts`` for a more human-aligned signal. ``topic_diversity``
reports the fraction of unique words across all topics' top-N, the standard
companion to coherence in modern topic-model papers.

These are pure-Python/numpy and work with any model here: pass a fitted model
(its top words are read automatically) or an explicit list of word lists.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

# Default sliding-window widths, following gensim's conventions.
_DEFAULT_WINDOW = {"c_v": 110, "c_uci": 10, "c_npmi": 10, "u_mass": None}
_VALID = ("u_mass", "c_uci", "c_npmi", "c_v")


# ---------------------------------------------------------------------------
# Topic extraction
# ---------------------------------------------------------------------------

def _extract_topics(topics, topn):
    """Normalize `topics` to a list of word lists, truncated to `topn`.

    Accepts a fitted model (its ``top_words(topn)`` is read), a list of word
    lists, or a list of ``(word, prob)`` lists.
    """
    if hasattr(topics, "top_words") and not isinstance(topics, (list, tuple)):
        rows = topics.top_words(topn)
        return [[w for w, _ in row] for row in rows]
    out = []
    for row in topics:
        words = [item[0] if isinstance(item, (tuple, list)) else item for item in row]
        out.append(words[:topn] if topn else list(words))
    return out


# ---------------------------------------------------------------------------
# Co-occurrence accumulation (restricted to the relevant words)
# ---------------------------------------------------------------------------

def _doc_occurrence(texts, vocab):
    """Document frequencies and pairwise document co-occurrence over the
    relevant words (for UMass). Returns (occ[R], co[R,R])."""
    r = len(vocab)
    occ = np.zeros(r)
    co = np.zeros((r, r))
    for doc in texts:
        present = {vocab[w] for w in doc if w in vocab}
        pl = list(present)
        for a in pl:
            occ[a] += 1.0
        for x in range(len(pl)):
            for y in range(x + 1, len(pl)):
                a, b = pl[x], pl[y]
                co[a, b] += 1.0
                co[b, a] += 1.0
    return occ, co


def _window_occurrence(texts, vocab, window):
    """Boolean sliding-window word and pairwise co-occurrence counts over the
    relevant words. Returns (occ[R], co[R,R], n_windows).

    A window of width `window` slides one token at a time; a document shorter
    than the window contributes a single window spanning the whole document.
    Counting is incremental (O(1) per step) and restricted to relevant words,
    which are sparse, so the per-window work is tiny.
    """
    r = len(vocab)
    occ = np.zeros(r)
    co = np.zeros((r, r))
    n_windows = 0

    def emit(present):
        for a in present:
            occ[a] += 1.0
        for x in range(len(present)):
            for y in range(x + 1, len(present)):
                a, b = present[x], present[y]
                co[a, b] += 1.0
                co[b, a] += 1.0

    for doc in texts:
        ids = [vocab.get(w, -1) for w in doc]
        length = len(ids)
        if length == 0:
            continue
        w = window if (window and window > 0) else length
        if length <= w:
            present = list({i for i in ids if i >= 0})
            emit(present)
            n_windows += 1
            continue
        cnt = defaultdict(int)
        for p in range(w):
            if ids[p] >= 0:
                cnt[ids[p]] += 1
        emit([k for k, v in cnt.items() if v > 0])
        n_windows += 1
        for s in range(1, length - w + 1):
            out_i = ids[s - 1]
            in_i = ids[s + w - 1]
            if out_i >= 0:
                cnt[out_i] -= 1
                if cnt[out_i] == 0:
                    del cnt[out_i]
            if in_i >= 0:
                cnt[in_i] += 1
            emit([k for k, v in cnt.items() if v > 0])
            n_windows += 1
    return occ, co, n_windows


# ---------------------------------------------------------------------------
# Per-topic scoring
# ---------------------------------------------------------------------------

def _idx(topic, vocab):
    return [vocab[w] for w in topic if w in vocab]


def _score_umass(topic, vocab, occ, co, eps):
    idx = _idx(topic, vocab)
    if len(idx) < 2:
        return float("nan")
    total = 0.0
    n = 0
    for i in range(1, len(idx)):
        for j in range(i):
            a, b = idx[i], idx[j]  # a follows b in the ranked list
            denom = occ[b] if occ[b] > 0 else eps
            total += math.log((co[a, b] + 1.0) / denom)
            n += 1
    return total / n if n else float("nan")


def _pair_npmi(pi, pj, pij, eps):
    pi = max(pi, eps)
    pj = max(pj, eps)
    if pij <= 0.0:
        pij = eps
    if pij >= 1.0:
        return 1.0
    return math.log(pij / (pi * pj)) / (-math.log(pij))


def _score_uci(topic, vocab, p, co, nw, eps):
    idx = _idx(topic, vocab)
    if len(idx) < 2:
        return float("nan")
    total = 0.0
    n = 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a, b = idx[i], idx[j]
            pij = co[a, b] / nw
            total += math.log((pij + eps) / (max(p[a], eps) * max(p[b], eps)))
            n += 1
    return total / n if n else float("nan")


def _score_npmi(topic, vocab, p, co, nw, eps):
    idx = _idx(topic, vocab)
    if len(idx) < 2:
        return float("nan")
    total = 0.0
    n = 0
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a, b = idx[i], idx[j]
            total += _pair_npmi(p[a], p[b], co[a, b] / nw, eps)
            n += 1
    return total / n if n else float("nan")


def _score_cv(topic, vocab, p, co, nw, eps):
    idx = _idx(topic, vocab)
    n = len(idx)
    if n < 2:
        return float("nan")
    # NPMI matrix over the topic's words (diagonal = 1, since P(w,w) = P(w)).
    m = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                m[i, j] = 1.0
            else:
                a, b = idx[i], idx[j]
                m[i, j] = _pair_npmi(p[a], p[b], co[a, b] / nw, eps)
    # Indirect cosine: each word's context vector (its row) vs. the set vector
    # (column sums), averaged (Röder et al. 2015, "C_v").
    set_vec = m.sum(axis=0)
    sn = np.linalg.norm(set_vec)
    sims = []
    for i in range(n):
        rn = np.linalg.norm(m[i])
        sims.append(float(m[i] @ set_vec / (rn * sn)) if rn > 0 and sn > 0 else 0.0)
    return float(np.mean(sims))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def coherence(topics, texts, *, coherence_type="c_v", topn=10, window_size=None, epsilon=1e-12):
    """Per-topic coherence against a reference corpus.

    Parameters
    ----------
    topics : a fitted model, or a list of topics (each a list of words, or of
        ``(word, prob)`` pairs).
    texts : list of tokenized documents (``list[list[str]]``) — the reference
        corpus. Pass your training documents, or an external corpus.
    coherence_type : one of ``"u_mass"``, ``"c_uci"``, ``"c_npmi"``, ``"c_v"``
        (default ``"c_v"``).
    topn : number of top words per topic to score (default 10).
    window_size : sliding-window width for the windowed measures; ``None`` uses
        the per-measure default (110 for ``c_v``, 10 for ``c_uci``/``c_npmi``).
        Ignored by ``u_mass``.

    Returns
    -------
    numpy.ndarray of shape ``(num_topics,)`` — the coherence of each topic.
    Take ``.mean()`` for the overall model score.
    """
    ct = coherence_type.lower()
    if ct not in _VALID:
        raise ValueError(f"coherence_type must be one of {_VALID}, got {coherence_type!r}")
    tops = _extract_topics(topics, topn)
    texts = [list(d) for d in texts]
    relevant = sorted({w for t in tops for w in t})
    vocab = {w: i for i, w in enumerate(relevant)}

    if ct == "u_mass":
        occ, co = _doc_occurrence(texts, vocab)
        return np.array([_score_umass(t, vocab, occ, co, epsilon) for t in tops])

    win = window_size if window_size is not None else _DEFAULT_WINDOW[ct]
    occ, co, nw = _window_occurrence(texts, vocab, win)
    if nw == 0:
        return np.full(len(tops), float("nan"))
    p = occ / nw
    scorer = {"c_uci": _score_uci, "c_npmi": _score_npmi, "c_v": _score_cv}[ct]
    return np.array([scorer(t, vocab, p, co, nw, epsilon) for t in tops])


def topic_diversity(topics, topn=25):
    """Fraction of unique words across all topics' top-`topn` words (Dieng,
    Ruiz & Blei 2020). 1.0 means every top word is unique to its topic; low
    values indicate topics that recycle the same words.

    `topics` is a fitted model or a list of word lists.
    """
    tops = _extract_topics(topics, topn)
    seen = set()
    total = 0
    for t in tops:
        for w in t[:topn]:
            seen.add(w)
            total += 1
    return len(seen) / total if total else float("nan")
