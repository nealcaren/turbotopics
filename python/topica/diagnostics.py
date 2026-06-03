"""General post-hoc topic-model diagnostics.

Interpretation, labeling, comparison, and visualization helpers that operate on
any fitted model's topic-word (φ) and document-topic (θ) arrays — independent of
how the model was fit (LDA, DMR, CTM, STM, HDP, …). The structural / covariate
pieces (``estimate_effect``, ``posterior_theta_samples``, ``spline``,
``interaction``) live in :mod:`topica.stm`; coherence, diversity,
exclusivity, and the intrusion tests live in :mod:`topica.coherence`.

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

import html as _html
import re
from dataclasses import dataclass, field

import numpy as np

from .coherence import _as_topic_word, _as_doc_topic

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


def topic_table(model, *, n=7):
    """A publication-ready topic table: one row per topic with its prevalence and
    its top probability and FREX words.

    Returns a list of dicts with ``topic``, ``prevalence`` (mean θ), ``prob`` (the
    top-`n` highest-probability words), and ``frex`` (the top-`n` FREX words —
    usually the better label). Hand it to ``pandas.DataFrame`` for the table that
    goes in a results section.

    `model` is any fitted model exposing ``topic_word``, ``doc_topic``, and
    ``vocabulary``.
    """
    phi = _as_topic_word(model)
    prevalence = _as_doc_topic(model).mean(axis=0)
    vocab = list(model.vocabulary)
    labels = label_topics(phi, vocab, n=n)
    return [
        {
            "topic": t,
            "prevalence": float(prevalence[t]),
            "prob": [w for w, _ in labels[t]["prob"]],
            "frex": [w for w, _ in labels[t]["frex"]],
        }
        for t in range(len(labels))
    ]


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
    model="lda",
    prevalence=None,
    held_out=None,
    iterations=500,
    em_iters=30,
    num_samples=3,
    sample_interval=10,
    seed=42,
    coherence_n=10,
):
    """Fit a model for each K and report quality metrics (stm's ``searchK``).

    With ``model="lda"`` (default) fits an :class:`~topica.LDA` per K. With
    ``model="stm"`` fits an :class:`~topica.STM` per K — pass ``prevalence``
    (a covariate design matrix) to scan K for the model you'll actually report.

    Returns a list of dicts (one per K) with ``k``, ``coherence`` (mean UMass),
    ``exclusivity`` (mean top-word exclusivity), and — for ``model="lda"`` with
    ``held_out`` — ``perplexity`` (held-out). The coherence/exclusivity trade-off
    is the signal: there is rarely a single best K, so read it alongside
    interpretability (see the K guide).
    """
    from . import LDA, STM  # local import to avoid a cycle at module load

    if model not in ("lda", "stm"):
        raise ValueError("model must be 'lda' or 'stm'")

    rows = []
    for k in ks:
        if model == "stm":
            m = STM(num_topics=k, seed=seed)
            m.fit(docs, prevalence, em_iters=em_iters)
        else:
            m = LDA(num_topics=k, seed=seed)
            m.fit(docs, iterations=iterations, num_samples=num_samples,
                  sample_interval=sample_interval)
        row = {
            "k": k,
            "coherence": float(np.mean(m.coherence(coherence_n))),
            "exclusivity": _mean_exclusivity(m.topic_word, coherence_n),
        }
        if model == "lda" and held_out is not None:
            row["perplexity"] = float(m.perplexity(held_out, seed=seed))
        rows.append(row)
    return rows


def plot_search_k(rows, *, metrics=("coherence", "exclusivity"), ax=None):
    """Plot :func:`search_k` results: each metric against the number of topics.

    Researchers read this curve to choose `K`: coherence and exclusivity usually
    trade off, so the goal is a knee, not a maximum. Each metric gets its own
    y-axis (they live on different scales). ``rows`` is the list returned by
    :func:`search_k`; ``metrics`` selects which of its keys to draw (any of
    ``"coherence"``, ``"exclusivity"``, ``"perplexity"``). Returns the primary
    matplotlib ``Axes``. Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised via message
        raise ImportError(
            "plot_search_k needs matplotlib (pip install matplotlib)."
        ) from e

    rows = sorted(rows, key=lambda r: r["k"])
    ks = [r["k"] for r in rows]
    metrics = [m for m in metrics if any(m in r for r in rows)]
    if not metrics:
        raise ValueError("none of the requested metrics are present in rows")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    lines = []
    for i, metric in enumerate(metrics):
        a = ax if i == 0 else ax.twinx()
        if i >= 2:  # offset a third axis so it doesn't overlap the second
            a.spines["right"].set_position(("axes", 1.0 + 0.18 * (i - 1)))
        color = f"C{i}"
        vals = [r.get(metric, float("nan")) for r in rows]
        (line,) = a.plot(ks, vals, marker="o", color=color, label=metric)
        a.set_ylabel(metric, color=color)
        a.tick_params(axis="y", labelcolor=color)
        lines.append(line)

    ax.set_xlabel("number of topics (K)")
    ax.set_xticks(ks)
    ax.legend(lines, [li.get_label() for li in lines], loc="best")
    ax.figure.tight_layout()
    return ax


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


# ---------------------------------------------------------------------------
# Qualitative validation: highlighted close-reading export
# ---------------------------------------------------------------------------

def find_thoughts_html(
    model,
    texts,
    *,
    topics=None,
    n_docs=3,
    n_words=8,
    max_chars=400,
    markdown=False,
):
    """Render each topic's most representative documents for close reading, with
    the topic's top words **highlighted** in the document text.

    Distant reading (top words) is only half of topic validation; the other half
    is reading the actual documents a topic loads on. This builds a self-contained
    HTML snippet (or Markdown) you can ``display`` in a notebook: per topic, its
    top words followed by its `n_docs` highest-θ documents, each truncated to
    ``max_chars`` with the topic's words marked.

    `model` is any fitted model exposing ``topic_word``, ``doc_topic`` and
    ``vocabulary``; `texts` are the original document strings, aligned to the
    rows of ``doc_topic``. Returns a string (HTML unless ``markdown=True``).
    """
    phi = _as_topic_word(model)
    theta = _as_doc_topic(model)
    vocab = list(model.vocabulary)
    if len(texts) != theta.shape[0]:
        raise ValueError("texts must be aligned with the model's documents")
    K = phi.shape[0]
    topics = range(K) if topics is None else topics

    blocks = []
    for t in topics:
        top_ids = np.argsort(phi[t])[::-1][:n_words]
        words = [vocab[i] for i in top_ids]
        docs = np.argsort(theta[:, t])[::-1][:n_docs]
        if markdown:
            blocks.append(_thoughts_md(t, words, docs, theta, texts, max_chars))
        else:
            blocks.append(_thoughts_html(t, words, docs, theta, texts, max_chars))
    if markdown:
        return "\n\n".join(blocks)
    return "<div class=\"tt-thoughts\">\n" + "\n".join(blocks) + "\n</div>"


def _keyword_pattern(words):
    # Match the readable surface form of each top word (phrase tokens use "_").
    surfaces = sorted({w.replace("_", " ") for w in words}, key=len, reverse=True)
    surfaces = [re.escape(s) for s in surfaces if s]
    if not surfaces:
        return None
    return re.compile(r"\b(" + "|".join(surfaces) + r")\b", re.IGNORECASE)


def _truncate(text, max_chars):
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + " …"


def _thoughts_html(t, words, docs, theta, texts, max_chars):
    pat = _keyword_pattern(words)
    head = (f"<h4>Topic {t}</h4>\n<p><em>"
            + ", ".join(_html.escape(w.replace('_', ' ')) for w in words)
            + "</em></p>\n<ul>")
    items = []
    for d in docs:
        body = _html.escape(_truncate(str(texts[d]), max_chars))
        if pat is not None:
            body = pat.sub(lambda m: f"<mark>{m.group(0)}</mark>", body)
        items.append(f"<li><small>doc {int(d)} (θ={theta[d, t]:.2f})</small><br>{body}</li>")
    return head + "\n" + "\n".join(items) + "\n</ul>"


def _thoughts_md(t, words, docs, theta, texts, max_chars):
    pat = _keyword_pattern(words)
    lines = [f"### Topic {t}",
             "*" + ", ".join(w.replace("_", " ") for w in words) + "*", ""]
    for d in docs:
        body = _truncate(str(texts[d]), max_chars)
        if pat is not None:
            body = pat.sub(lambda m: f"**{m.group(0)}**", body)
        lines.append(f"- **doc {int(d)}** (θ={theta[d, t]:.2f}): {body}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model-quality frontier + bootstrap stability
# ---------------------------------------------------------------------------

def quality_frontier(model, *, n=10, texts=None, coherence_type="u_mass", plot=False):
    """Per-topic coherence, exclusivity, and prevalence — the data behind stm's
    classic coherence-vs-exclusivity quality plot.

    Returns a dict of equal-length arrays: ``topic``, ``coherence``,
    ``exclusivity``, ``prevalence`` (mean θ). By default coherence is the fast
    per-topic UMass score; pass ``texts`` and a windowed ``coherence_type`` (e.g.
    ``"c_v"``) for the human-aligned measure. Feed the dict straight to pandas /
    matplotlib; with ``plot=True`` (and matplotlib installed) a labeled scatter
    ``Figure`` is returned alongside the dict as ``(data, fig)``.
    """
    from .coherence import coherence as _coherence, exclusivity as _exclusivity

    phi = _as_topic_word(model)
    theta = _as_doc_topic(model)
    K = phi.shape[0]
    if texts is not None and coherence_type != "u_mass":
        coh = np.asarray(_coherence(model, texts, coherence_type=coherence_type, topn=n))
    else:
        coh = np.asarray(model.coherence(n))
    data = {
        "topic": np.arange(K),
        "coherence": coh,
        "exclusivity": _exclusivity(phi, n=n),
        "prevalence": theta.mean(axis=0),
    }
    if not plot:
        return data
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("plot=True requires matplotlib") from exc
    fig, ax = plt.subplots()
    ax.scatter(data["coherence"], data["exclusivity"],
               s=300 * data["prevalence"] + 20)
    for t in range(K):
        ax.annotate(str(t), (data["coherence"][t], data["exclusivity"][t]))
    ax.set_xlabel("Semantic coherence")
    ax.set_ylabel("Exclusivity")
    ax.set_title("Topic quality (size ∝ prevalence)")
    return data, fig


def bootstrap_stability(
    docs,
    *,
    k,
    n_boot=20,
    topn=10,
    seed=0,
    model_factory=None,
    **fit_kwargs,
):
    """Flag fragile topics by refitting on bootstrap resamples of the corpus.

    The standard defense against "topic modeling is a fishing expedition": fit a
    reference model on the full corpus, then refit on `n_boot` resamples of the
    documents (drawn with replacement). Each bootstrap model's topics are matched
    to the reference's by top-word overlap, and a reference topic's **stability**
    is the mean Jaccard overlap of its top-`topn` words with its matched bootstrap
    topic. Topics that dissolve under resampling score low.

    Matching is on the top words as *strings*, so it is correct even though each
    resample is fit as a fresh corpus with its own vocabulary indexing.

    Parameters
    ----------
    docs : the corpus (``list[list[str]]`` or a ``Corpus``).
    k : number of topics.
    n_boot : number of bootstrap resamples.
    model_factory : ``callable(seed) -> unfitted model``. Defaults to
        ``LDA(num_topics=k, seed=seed)``. Use it to bootstrap any model.
    fit_kwargs : forwarded to each model's ``fit`` (e.g. ``iterations=500``).

    Returns
    -------
    dict with ``topic`` (indices), ``stability`` (per-topic mean Jaccard in
    ``[0, 1]``), ``mean`` (overall), and ``reference`` (the full-corpus model).
    """
    from . import LDA  # local import to avoid a cycle at module load

    if hasattr(docs, "docs") and hasattr(docs, "id_to_word"):
        raise TypeError("bootstrap_stability needs a list of token lists, not a Corpus")
    docs = [list(d) for d in docs]
    D = len(docs)
    if D < 2:
        raise ValueError("need at least two documents to resample")
    factory = model_factory or (lambda s: LDA(num_topics=k, seed=s))

    def top_word_sets(model):
        phi = _as_topic_word(model)
        vocab = list(model.vocabulary)
        return [set(vocab[i] for i in np.argsort(phi[t])[::-1][:topn])
                for t in range(phi.shape[0])]

    ref = factory(seed)
    ref.fit(docs, **fit_kwargs)
    ref_sets = top_word_sets(ref)
    K = len(ref_sets)

    rng = np.random.RandomState(seed)
    per_topic = [[] for _ in range(K)]
    for b in range(n_boot):
        pick = rng.randint(0, D, size=D)
        sample = [docs[i] for i in pick]
        m = factory(seed + b + 1)
        m.fit(sample, **fit_kwargs)
        boot_sets = top_word_sets(m)
        # Match bootstrap topics to reference topics by top-word Jaccard, then
        # record each reference topic's overlap with its match.
        cost = np.empty((K, len(boot_sets)))
        for i, rs in enumerate(ref_sets):
            for j, bs in enumerate(boot_sets):
                union = rs | bs
                cost[i, j] = 1.0 - (len(rs & bs) / len(union) if union else 0.0)
        for i, j in _hungarian(cost):
            union = ref_sets[i] | boot_sets[j]
            per_topic[i].append(len(ref_sets[i] & boot_sets[j]) / len(union) if union else 0.0)

    stability = np.array([float(np.mean(s)) if s else float("nan") for s in per_topic])
    return {
        "topic": np.arange(K),
        "stability": stability,
        "mean": float(np.nanmean(stability)),
        "reference": ref,
    }


__all__ = [
    "frex",
    "label_topics",
    "topic_correlation",
    "TopicCorrelation",
    "find_thoughts",
    "find_thoughts_html",
    "topic_table",
    "quality_frontier",
    "bootstrap_stability",
    "search_k",
    "relevance",
    "prepare_pyldavis",
    "PyLDAvisInputs",
    "check_residuals",
    "ResidualCheck",
    "align_topics",
    "topic_stability",
]
