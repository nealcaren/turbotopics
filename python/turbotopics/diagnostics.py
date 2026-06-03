"""General post-hoc topic-model diagnostics.

Interpretation, labeling, comparison, and visualization helpers that operate on
any fitted model's topic-word (φ) and document-topic (θ) arrays — independent of
how the model was fit (LDA, DMR, CTM, STM, HDP, …). The structural / covariate
pieces (``estimate_effect``, ``posterior_theta_samples``, ``spline``,
``interaction``) live in :mod:`turbotopics.stm`; coherence, diversity,
exclusivity, and the intrusion tests live in :mod:`turbotopics.coherence`.

- :func:`frex` / :func:`label_topics` — prob / FREX / lift / score topic words
  (≈ ``stm::labelTopics``).
- :func:`topic_correlation` — topic-correlation network (≈ ``stm::topicCorr``).
- :func:`find_thoughts` — representative documents per topic (≈ ``stm::findThoughts``).
- :func:`search_k` — fit across topic counts, report quality (≈ ``stm::searchK``).
- :func:`relevance` / :func:`prepare_pyldavis` — LDAvis relevance + export.
- :func:`check_residuals` — Taddy (2012) residual-dispersion test for K.
- :func:`align_topics` / :func:`topic_stability` — match and score topics across fits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .coherence import _as_topic_word

# ---------------------------------------------------------------------------
# labelTopics: prob / FREX / lift / score
# ---------------------------------------------------------------------------

def _ecdf_ranks(x: np.ndarray) -> np.ndarray:
    """Empirical-CDF rank of each value within `x` (ties share the high rank)."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks / len(x)


def frex(topic_word, vocabulary, *, w=0.5, n=10):
    """FREX (FRequency–EXclusivity) top words per topic.

    For each topic, words are scored by the weighted harmonic mean of the ECDF
    rank of their probability (frequency) and the ECDF rank of their exclusivity
    ``φ_{t,v} / Σ_k φ_{k,v}`` — the same combination stm uses. ``w`` weights
    frequency vs exclusivity. Returns a list (per topic) of ``(word, frex)``.
    """
    phi = np.asarray(topic_word, dtype=np.float64)
    K, V = phi.shape
    col = phi.sum(axis=0)
    col[col == 0] = 1.0
    excl = phi / col  # exclusivity per (topic, word)

    results = []
    for t in range(K):
        f_rank = _ecdf_ranks(phi[t])
        e_rank = _ecdf_ranks(excl[t])
        with np.errstate(divide="ignore", invalid="ignore"):
            score = 1.0 / (w / f_rank + (1.0 - w) / e_rank)
        idx = np.argsort(score)[::-1][:n]
        results.append([(vocabulary[i], float(score[i])) for i in idx])
    return results


def label_topics(topic_word, vocabulary, *, n=10):
    """stm-style topic labels: prob, FREX, lift, and score word lists per topic.

    Returns a list (per topic) of dicts with keys ``prob``, ``frex``, ``lift``,
    ``score``, each a list of ``(word, value)`` pairs.
    """
    phi = np.asarray(topic_word, dtype=np.float64)
    K, V = phi.shape
    marginal = phi.mean(axis=0)
    marginal_safe = np.where(marginal > 0, marginal, 1e-12)
    log_phi = np.log(np.clip(phi, 1e-12, None))
    mean_log = log_phi.mean(axis=0)

    frex_words = frex(topic_word, vocabulary, n=n)
    out = []
    for t in range(K):
        prob_idx = np.argsort(phi[t])[::-1][:n]
        lift = phi[t] / marginal_safe
        lift_idx = np.argsort(lift)[::-1][:n]
        score = phi[t] * (log_phi[t] - mean_log)
        score_idx = np.argsort(score)[::-1][:n]
        out.append({
            "prob": [(vocabulary[i], float(phi[t, i])) for i in prob_idx],
            "frex": frex_words[t],
            "lift": [(vocabulary[i], float(lift[i])) for i in lift_idx],
            "score": [(vocabulary[i], float(score[i])) for i in score_idx],
        })
    return out


# ---------------------------------------------------------------------------
# topicCorr: topic-correlation network
# ---------------------------------------------------------------------------

@dataclass
class TopicCorrelation:
    cor: np.ndarray
    adjacency: np.ndarray
    edges: list[tuple[int, int, float]] = field(default_factory=list)


def topic_correlation(doc_topic, *, threshold=0.05):
    """Topic-correlation network (≈ stm's ``topicCorr`` "simple" method).

    Correlates topic proportions across documents; topic pairs whose correlation
    exceeds ``threshold`` become network edges. Returns a
    :class:`TopicCorrelation` with the correlation matrix, a 0/1 adjacency
    matrix (zero diagonal), and the edge list.
    """
    theta = np.asarray(doc_topic, dtype=np.float64)
    cor = np.corrcoef(theta.T)
    cor = np.nan_to_num(cor)
    K = cor.shape[0]
    adj = (cor > threshold).astype(int)
    np.fill_diagonal(adj, 0)
    edges = [
        (i, j, float(cor[i, j]))
        for i in range(K)
        for j in range(i + 1, K)
        if cor[i, j] > threshold
    ]
    return TopicCorrelation(cor=cor, adjacency=adj, edges=edges)


# ---------------------------------------------------------------------------
# findThoughts: representative documents per topic
# ---------------------------------------------------------------------------

def find_thoughts(doc_topic, texts=None, *, topic, n=3):
    """The `n` documents most associated with `topic` (≈ stm's ``findThoughts``).

    Returns a list of ``(doc_index, proportion, text)`` sorted by descending
    topic proportion; ``text`` is ``None`` when ``texts`` is not supplied.
    """
    theta = np.asarray(doc_topic, dtype=np.float64)
    if topic < 0 or topic >= theta.shape[1]:
        raise ValueError(f"topic {topic} out of range (num_topics={theta.shape[1]})")
    idx = np.argsort(theta[:, topic])[::-1][:n]
    out = []
    for i in idx:
        text = texts[i] if texts is not None else None
        out.append((int(i), float(theta[i, topic]), text))
    return out


# ---------------------------------------------------------------------------
# searchK: fit across topic counts, report quality
# ---------------------------------------------------------------------------

def search_k(
    docs,
    ks,
    *,
    held_out=None,
    iterations=500,
    num_samples=3,
    sample_interval=10,
    seed=42,
    coherence_n=10,
):
    """Fit an :class:`~turbotopics.LDA` for each K and report quality metrics.

    Returns a list of dicts (one per K) with ``k``, ``coherence`` (mean UMass),
    ``exclusivity`` (mean top-word exclusivity), and — when ``held_out`` is
    provided — ``perplexity`` (held-out). Mirrors the semantic-coherence /
    exclusivity trade-off plot from stm's ``searchK``.
    """
    from . import LDA  # local import to avoid a cycle at module load

    rows = []
    for k in ks:
        model = LDA(num_topics=k, seed=seed)
        model.fit(docs, iterations=iterations, num_samples=num_samples,
                  sample_interval=sample_interval)
        coh = float(np.mean(model.coherence(coherence_n)))
        excl = _mean_exclusivity(model.topic_word, coherence_n)
        row = {"k": k, "coherence": coh, "exclusivity": excl}
        if held_out is not None:
            row["perplexity"] = float(model.perplexity(held_out, seed=seed))
        rows.append(row)
    return rows


def _mean_exclusivity(topic_word, n: int) -> float:
    from .coherence import exclusivity
    return float(np.mean(exclusivity(topic_word, n=n)))


# ---------------------------------------------------------------------------
# LDAvis relevance + pyLDAvis export
# ---------------------------------------------------------------------------

def relevance(topic_word, vocabulary, *, topic=None, lam=0.6, n=10, term_frequency=None):
    """LDAvis *relevance* of words to topics (Sievert & Shirley 2014):

    ``relevance(w | t) = λ·log p(w|t) + (1-λ)·log[p(w|t) / p(w)]``

    λ=1 ranks by probability; λ=0 by lift (exclusivity); the LDAvis default 0.6
    balances them. ``p(w)`` is the corpus word marginal — pass ``term_frequency``
    (word counts in `vocabulary` order) for the empirical marginal, else the
    topic-averaged φ is used. Returns ``(word, relevance)`` lists per topic, or
    for one ``topic``.
    """
    phi = np.asarray(topic_word, dtype=np.float64)
    k, _ = phi.shape
    if term_frequency is not None:
        tf = np.asarray(term_frequency, dtype=np.float64)
        pw = tf / tf.sum()
    else:
        pw = phi.mean(axis=0)
    pw = np.clip(pw, 1e-12, None)
    log_phi = np.log(np.clip(phi, 1e-12, None))
    rel = lam * log_phi + (1.0 - lam) * (log_phi - np.log(pw))  # (K, V)

    def top(t):
        idx = np.argsort(rel[t])[::-1][:n]
        return [(vocabulary[i], float(rel[t, i])) for i in idx]

    if topic is not None:
        if topic < 0 or topic >= k:
            raise ValueError(f"topic {topic} out of range (num_topics={k})")
        return top(topic)
    return [top(t) for t in range(k)]


@dataclass
class PyLDAvisInputs:
    """The five arrays ``pyLDAvis.prepare`` needs, for when pyLDAvis is not
    installed. ``pyLDAvis.prepare(*inputs.unpack())`` reconstructs the view."""

    topic_term_dists: np.ndarray
    doc_topic_dists: np.ndarray
    doc_lengths: np.ndarray
    vocab: list
    term_frequency: np.ndarray

    def unpack(self):
        return (self.topic_term_dists, self.doc_topic_dists, self.doc_lengths,
                self.vocab, self.term_frequency)


def prepare_pyldavis(model, docs, **kwargs):
    """Build the LDAvis intertopic-distance visualization for a fitted model.

    `docs` are the tokenized training documents (``list[list[str]]``), used for
    document lengths and term frequencies. If ``pyLDAvis`` is installed this
    returns its ``PreparedData`` (pass to ``pyLDAvis.display`` / ``save_html``);
    otherwise it returns a :class:`PyLDAvisInputs` you can feed to
    ``pyLDAvis.prepare`` later. Extra ``kwargs`` go to ``pyLDAvis.prepare``
    (e.g. ``sort_topics=False``).
    """
    phi = np.asarray(model.topic_word, dtype=np.float64)
    theta = np.asarray(model.doc_topic, dtype=np.float64)
    vocab = list(model.vocabulary)
    if len(docs) != theta.shape[0]:
        raise ValueError(
            f"docs has {len(docs)} entries but doc_topic has {theta.shape[0]} rows; "
            "pass the same documents used to fit the model"
        )
    vindex = {w: i for i, w in enumerate(vocab)}
    tf = np.zeros(len(vocab))
    doc_lengths = np.zeros(len(docs), dtype=np.int64)
    for d, doc in enumerate(docs):
        for w in doc:
            i = vindex.get(w)
            if i is not None:
                tf[i] += 1.0
                doc_lengths[d] += 1
    inputs = PyLDAvisInputs(phi, theta, doc_lengths, vocab, tf)
    try:
        import pyLDAvis
    except ImportError:
        return inputs
    return pyLDAvis.prepare(phi, theta, doc_lengths, vocab, tf, **kwargs)


# ---------------------------------------------------------------------------
# checkResiduals: residual-dispersion test for K selection (Taddy 2012)
# ---------------------------------------------------------------------------

def _gammq(a, x):
    """Regularized upper incomplete gamma Q(a, x) (Numerical Recipes)."""
    import math
    if x < 0 or a <= 0:
        return float("nan")
    if x == 0.0:
        return 1.0  # Q(a, 0) = 1
    if x < a + 1.0:  # series for the lower P, then complement
        ap = a
        s = 1.0 / a
        d = s
        for _ in range(500):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # continued fraction for the upper Q
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def _chisq_sf(x, df):
    """Upper-tail (survival) probability of a chi-square with `df` df."""
    if df <= 0:
        return float("nan")
    return _gammq(df / 2.0, x / 2.0)


@dataclass
class ResidualCheck:
    """Result of :func:`check_residuals`: multinomial residual dispersion."""

    dispersion: float
    pvalue: float
    df: float


def check_residuals(model, docs, *, tol=0.01):
    """Residual-dispersion test for whether K is too small (Taddy 2012), a faithful
    port of R ``stm``'s ``checkResiduals``.

    Under a correctly specified model the multinomial residuals have dispersion
    ``σ² = 1``. A dispersion well above 1 (small p-value) is evidence the latent
    topics cannot absorb the overdispersion — i.e. K is too low. Run it alongside
    :func:`search_k`. `docs` are the tokenized training documents aligned to
    ``model.doc_topic``'s rows.

    Returns a :class:`ResidualCheck` with ``dispersion`` (σ²), ``pvalue`` (χ²
    test of σ²=1 vs σ²>1), and ``df``.
    """
    phi = np.asarray(model.topic_word, dtype=np.float64)
    theta = np.asarray(model.doc_topic, dtype=np.float64)
    vocab = list(model.vocabulary)
    k, v = phi.shape
    n = theta.shape[0]
    if len(docs) != n:
        raise ValueError(
            f"docs has {len(docs)} entries but doc_topic has {n} rows; "
            "pass the same documents used to fit the model"
        )
    vindex = {w: i for i, w in enumerate(vocab)}

    d_stat = 0.0
    nhat = 0
    for d in range(n):
        q = np.clip(theta[d] @ phi, 1e-12, 1.0 - 1e-12)  # (V,) model word probs
        x = np.zeros(v)
        m = 0.0
        for w in docs[d]:
            i = vindex.get(w)
            if i is not None:
                x[i] += 1.0
                m += 1.0
        if m == 0:
            continue
        nhat += int(np.sum(q * m > tol))
        first = np.sum((x * x - 2.0 * x * q * m) / (m * q * (1.0 - q)))
        second = np.sum(m * q / (1.0 - q))
        d_stat += float(first + second)

    n_params = n * (k - 1) + k * (v - 1)
    df = nhat - v - n_params
    dispersion = d_stat / df if df > 0 else float("nan")
    pvalue = _chisq_sf(d_stat, df) if df > 0 else float("nan")
    return ResidualCheck(dispersion=float(dispersion), pvalue=float(pvalue), df=float(df))


# ---------------------------------------------------------------------------
# Topic alignment + stability (exploits determinism)
# ---------------------------------------------------------------------------

def _hungarian(cost):
    """Optimal min-cost assignment (Hungarian / Kuhn-Munkres). Returns a list of
    ``(row, col)`` pairs. Rectangular costs are padded to square."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    size = max(n, m)
    big = (cost.max() * size + 1.0) if cost.size else 1.0
    c = np.full((size, size), big)
    c[:n, :m] = cost
    inf = float("inf")
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = -1
            for j in range(1, size + 1):
                if not used[j]:
                    cur = c[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out = []
    for j in range(1, size + 1):
        if p[j] != 0 and p[j] - 1 < n and j - 1 < m:
            out.append((p[j] - 1, j - 1))
    return sorted(out)


def align_topics(a, b, *, metric="cosine"):
    """Match the topics of two fits one-to-one by minimal total distance
    (Hungarian on the cross-fit topic-word distance matrix). Use it to compare
    runs across seeds, across K, or train vs. resample — your fits are
    deterministic, so the matching is reproducible.

    `a`, `b` are fitted models or K×V topic-word arrays (same vocabulary order).
    `metric` is ``"cosine"`` or ``"js"`` (Jensen-Shannon). Returns a list of
    ``(topic_a, topic_b, distance)`` sorted by ``topic_a``.
    """
    A = _as_topic_word(a)
    B = _as_topic_word(b)
    if A.shape[1] != B.shape[1]:
        raise ValueError("the two fits must share a vocabulary (same V)")
    if metric == "cosine":
        an = A / np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)
        bn = B / np.clip(np.linalg.norm(B, axis=1, keepdims=True), 1e-12, None)
        dist = 1.0 - an @ bn.T
    elif metric == "js":
        dist = np.zeros((A.shape[0], B.shape[0]))
        for i in range(A.shape[0]):
            pi = A[i]
            for j in range(B.shape[0]):
                qj = B[j]
                mm = 0.5 * (pi + qj)
                dist[i, j] = 0.5 * _kl(pi, mm) + 0.5 * _kl(qj, mm)
    else:
        raise ValueError("metric must be 'cosine' or 'js'")
    return [(i, j, float(dist[i, j])) for (i, j) in _hungarian(dist)]


def _kl(p, q):
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    return float(np.sum(p * np.log(p / q)))


def topic_stability(runs, *, topn=10, metric="cosine"):
    """Term-centric stability of topics across multiple fits (Greene, O'Callaghan
    & Cunningham 2014): a "how robust is this K?" score.

    `runs` is a list of fitted models or topic-word arrays over the *same*
    vocabulary (e.g. fits at different seeds, or on bootstrap resamples). Each
    later run's topics are matched to the first run's, and stability is the mean
    Jaccard overlap of their top-`topn` words. Returns a float in ``[0, 1]``;
    higher means more reproducible topics.
    """
    mats = [_as_topic_word(r) for r in runs]
    if len(mats) < 2:
        raise ValueError("need at least two runs to measure stability")
    ref = mats[0]
    k = ref.shape[0]
    ref_top = [set(np.argsort(ref[t])[::-1][:topn]) for t in range(k)]
    scores = []
    for mat in mats[1:]:
        for i, j, _ in align_topics(ref, mat, metric=metric):
            other = set(np.argsort(mat[j])[::-1][:topn])
            union = ref_top[i] | other
            scores.append(len(ref_top[i] & other) / len(union) if union else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


__all__ = [
    "frex",
    "label_topics",
    "topic_correlation",
    "TopicCorrelation",
    "find_thoughts",
    "search_k",
    "relevance",
    "prepare_pyldavis",
    "PyLDAvisInputs",
    "check_residuals",
    "ResidualCheck",
    "align_topics",
    "topic_stability",
]
