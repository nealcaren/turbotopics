"""STM-style analysis toolkit on top of turbotopics's Gibbs topic models.

These are post-hoc analyses of a fitted model's outputs (the topic-word matrix
``topic_word`` = φ and the document-topic matrix ``doc_topic`` = θ), mirroring
the user-facing functions of the R ``stm`` package:

- :func:`estimate_effect` — regress topic proportions on document covariates
  (≈ ``stm::estimateEffect``).
- :func:`label_topics` / :func:`frex` — prob / FREX / lift / score topic words
  (≈ ``stm::labelTopics``).
- :func:`topic_correlation` — topic-correlation network (≈ ``stm::topicCorr``).
- :func:`find_thoughts` — representative documents per topic
  (≈ ``stm::findThoughts``).
- :func:`search_k` — fit across topic counts and report quality
  (≈ ``stm::searchK``).

Everything operates on numpy arrays, so it works with any model here (LDA, DMR,
LabeledLDA). :func:`estimate_effect` does ordinary OLS on a point estimate of θ,
or — given posterior draws from :func:`posterior_theta_samples` (an STM/CTM
variational posterior) — the **method of composition**, pooling per-draw
regressions by Rubin's rules so the standard errors propagate topic-estimation
uncertainty, exactly as R ``stm``'s ``estimateEffect`` does. Nonlinear and
interaction terms are built with :func:`spline` and :func:`interaction`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# estimateEffect: covariate -> topic-proportion regression
# ---------------------------------------------------------------------------

@dataclass
class TopicEffect:
    """OLS of one topic's proportion on the covariates."""

    topic: int
    feature_names: list[str]
    coef: np.ndarray
    se: np.ndarray
    z: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    r_squared: float

    def as_dict(self) -> dict:
        return {
            "topic": self.topic,
            **{
                f"{name}": {
                    "coef": float(self.coef[j]),
                    "se": float(self.se[j]),
                    "z": float(self.z[j]),
                    "ci": (float(self.ci_low[j]), float(self.ci_high[j])),
                }
                for j, name in enumerate(self.feature_names)
            },
            "r_squared": self.r_squared,
        }


def _ols(y, X, hat, XtX_inv, dof):
    """One OLS fit. Returns (beta, cov, r2)."""
    beta = hat @ y
    resid = y - X @ beta
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * XtX_inv
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else 0.0
    return beta, cov, r2


def estimate_effect(
    doc_topic,
    X,
    *,
    feature_names=None,
    topics=None,
    add_intercept=True,
    ci=0.95,
):
    """Regress each topic's proportion on document covariates.

    Pass a point estimate of θ for an ordinary OLS, or a *stack of posterior
    draws* of θ for the **method of composition** — the uncertainty-propagating
    procedure R ``stm`` uses (Treier & Jackman 2008). With draws, each one is
    regressed and the results are pooled by Rubin's rules, so the reported
    standard errors include the topic-estimation uncertainty, not just OLS
    sampling error. Get draws with :func:`posterior_theta_samples`.

    Parameters
    ----------
    doc_topic : array
        Either ``(num_docs, num_topics)`` — a point θ (``model.doc_topic``) for
        plain OLS — or ``(nsims, num_docs, num_topics)`` — posterior θ draws for
        method-of-composition pooling.
    X : array (num_docs, p)
        Document covariates (design matrix); build nonlinear/interaction terms
        with :func:`spline` / :func:`interaction`. An intercept is prepended when
        ``add_intercept`` is True.
    feature_names : list[str], optional
        Column names for ``X``. Defaults to ``feature_0 ...``.
    topics : sequence[int], optional
        Restrict to these topics. Defaults to all.
    ci : float
        Confidence level for the (normal-approximation) intervals.

    Returns
    -------
    list[TopicEffect]
        One regression per topic. ``[e.as_dict() for e in result]`` builds a table.
    """
    from math import sqrt

    theta = np.asarray(doc_topic, dtype=np.float64)
    pooled = theta.ndim == 3
    if pooled:
        nsims, n, num_topics = theta.shape
    elif theta.ndim == 2:
        n, num_topics = theta.shape
    else:
        raise ValueError("doc_topic must be 2-D (num_docs, K) or 3-D (nsims, num_docs, K)")

    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] != n:
        raise ValueError(f"X has {X.shape[0]} rows but doc_topic has {n} documents")

    names = list(feature_names) if feature_names is not None else [
        f"feature_{i}" for i in range(X.shape[1])
    ]
    if len(names) != X.shape[1]:
        raise ValueError("feature_names length must match X columns")
    if add_intercept:
        X = np.hstack([np.ones((n, 1)), X])
        names = ["intercept"] + names

    p = X.shape[1]
    XtX_inv = np.linalg.pinv(X.T @ X)
    hat = XtX_inv @ X.T  # (p, n)
    dof = max(n - p, 1)
    z_crit = _normal_ppf(0.5 + ci / 2.0)  # normal-approx critical value (no scipy)

    topic_list = range(num_topics) if topics is None else list(topics)
    out: list[TopicEffect] = []
    for t in topic_list:
        if t < 0 or t >= num_topics:
            raise ValueError(f"topic {t} out of range (num_topics={num_topics})")
        if pooled:
            betas = np.empty((nsims, p))
            within = np.zeros((p, p))
            r2s = np.empty(nsims)
            for m in range(nsims):
                b, cov_m, r2_m = _ols(theta[m, :, t], X, hat, XtX_inv, dof)
                betas[m] = b
                within += cov_m
                r2s[m] = r2_m
            within /= nsims
            beta = betas.mean(axis=0)
            # Rubin's rules: total var = within + (1 + 1/M) * between.
            between = np.cov(betas, rowvar=False) if nsims > 1 else np.zeros((p, p))
            between = np.atleast_2d(between)
            total = within + (1.0 + 1.0 / nsims) * between
            se = np.sqrt(np.clip(np.diag(total), 0.0, None))
            r2 = float(r2s.mean())
        else:
            beta, cov, r2 = _ols(theta[:, t], X, hat, XtX_inv, dof)
            se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            zvals = np.where(se > 0, beta / se, 0.0)
        out.append(
            TopicEffect(
                topic=t,
                feature_names=names,
                coef=beta,
                se=se,
                z=zvals,
                ci_low=beta - z_crit * se,
                ci_high=beta + z_crit * se,
                r_squared=r2,
            )
        )
    return out


def posterior_theta_samples(model, nsims=25, seed=0):
    """Draw `nsims` samples of the document-topic matrix θ from a fitted
    :class:`STM`/:class:`CTM`'s variational posterior.

    Each document's logistic-normal posterior is ``η_d ~ N(λ_d, ν_d)`` (from
    ``model.eta_mean`` / ``model.eta_cov``); a draw of η is mapped through the
    softmax (with the reference category fixed at 0) to a θ row. Feed the result
    to :func:`estimate_effect` for method-of-composition uncertainty.

    Returns an array of shape ``(nsims, num_docs, num_topics)``.
    """
    lam = np.asarray(model.eta_mean, dtype=np.float64)  # (D, K-1)
    cov = np.asarray(model.eta_cov, dtype=np.float64)   # (D, K-1, K-1)
    d, km1 = lam.shape
    k = km1 + 1
    rng = np.random.default_rng(seed)
    out = np.empty((nsims, d, k))
    eye = np.eye(km1)
    for di in range(d):
        c = 0.5 * (cov[di] + cov[di].T)  # symmetrize
        try:
            chol = np.linalg.cholesky(c + 1e-10 * eye)
        except np.linalg.LinAlgError:
            w, v = np.linalg.eigh(c)
            chol = v @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))
        z = rng.standard_normal((nsims, km1))
        eta = lam[di] + z @ chol.T                       # (nsims, K-1)
        full = np.hstack([eta, np.zeros((nsims, 1))])    # reference category = 0
        full -= full.max(axis=1, keepdims=True)
        e = np.exp(full)
        out[:, di, :] = e / e.sum(axis=1, keepdims=True)
    return out


def spline(x, df=4, knots=None):
    """Restricted (natural) cubic-spline basis for a covariate — the building
    block for nonlinear prevalence terms like R ``stm``'s ``~ s(day)``.

    Uses Harrell's restricted-cubic-spline parameterization: `df+1` knots (at
    evenly spaced quantiles of `x` unless `knots` is given) yield `df` basis
    columns whose first is the linear term. ``np.column_stack`` the result into
    your design matrix and extend ``feature_names`` with the returned names.

    Returns ``(basis (n, df), names)``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if df < 2:
        raise ValueError("spline df must be >= 2")
    if knots is None:
        knots = np.quantile(x, np.linspace(0.0, 1.0, df + 1))
    t = np.asarray(knots, dtype=np.float64)
    k = len(t)
    if k < 3:
        raise ValueError("need at least 3 knots (df >= 2)")
    denom = (t[-1] - t[0]) ** 2

    def cube(u):
        return np.clip(u, 0.0, None) ** 3

    cols = [x]
    for j in range(k - 2):
        term = (
            cube(x - t[j])
            - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
            + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
        ) / denom
        cols.append(term)
    basis = np.column_stack(cols)
    names = ["spline_lin"] + [f"spline_{j + 1}" for j in range(basis.shape[1] - 1)]
    return basis, names


def interaction(a, b, name="interaction"):
    """Interaction columns between two covariate blocks (all pairwise products of
    their columns) — for terms like R ``stm``'s ``~ treatment * party``.

    `a`, `b` are 1-D or 2-D arrays with the same number of rows. Returns
    ``(products (n, ncols), names)``; ``np.column_stack`` into your design matrix.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    if a.shape[0] != b.shape[0]:
        raise ValueError("a and b must have the same number of rows")
    cols = []
    names = []
    multi = a.shape[1] > 1 or b.shape[1] > 1
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            cols.append(a[:, i] * b[:, j])
            names.append(f"{name}_{i}x{j}" if multi else name)
    return np.column_stack(cols), names


def _normal_ppf(q: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation)."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        r = (-2 * np.log(q)) ** 0.5
        return (((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / (
            (((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    if q > phigh:
        r = (-2 * np.log(1 - q)) ** 0.5
        return -(((((c[0]*r+c[1])*r+c[2])*r+c[3])*r+c[4])*r+c[5]) / (
            (((d[0]*r+d[1])*r+d[2])*r+d[3])*r+1)
    r = q - 0.5
    s = r * r
    return (((((a[0]*s+a[1])*s+a[2])*s+a[3])*s+a[4])*s+a[5])*r / (
        ((((b[0]*s+b[1])*s+b[2])*s+b[3])*s+b[4])*s+1)


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
    phi = np.asarray(topic_word, dtype=np.float64)
    K, V = phi.shape
    col = phi.sum(axis=0)
    col[col == 0] = 1.0
    excl = phi / col
    vals = []
    for t in range(K):
        top = np.argsort(phi[t])[::-1][:n]
        vals.append(float(excl[t, top].mean()))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# LDAvis relevance + pyLDAvis export
# ---------------------------------------------------------------------------

def _as_topic_word(obj):
    """Accept a fitted model (use its ``topic_word``) or a K×V array."""
    if hasattr(obj, "topic_word") and not isinstance(obj, np.ndarray):
        return np.asarray(obj.topic_word, dtype=np.float64)
    return np.asarray(obj, dtype=np.float64)


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
    "estimate_effect",
    "TopicEffect",
    "posterior_theta_samples",
    "spline",
    "interaction",
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
