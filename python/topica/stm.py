"""STM-style analysis toolkit on top of topica's Gibbs topic models.

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

from dataclasses import dataclass

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


def _link_inv(eta, link):
    if link == "logit":
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -700, 700)))
    if link == "log":
        return np.exp(np.clip(eta, -700, 700))
    return eta


def _sandwich(X, bread, score_resid, groups, n, p):
    """Robust covariance ``bread · meat · bread``. With `groups` (a list of index
    arrays) the cluster-robust CR1 estimator; otherwise heteroskedasticity-robust
    HC1. `score_resid` is the estimating-equation residual (y−μ)."""
    if groups is None:
        meat = X.T @ (X * (score_resid ** 2)[:, None])
        cov = bread @ meat @ bread
        cov *= n / max(n - p, 1)                       # HC1 small-sample factor
    else:
        g_count = len(groups)
        meat = np.zeros((p, p))
        for g in groups:
            s = X[g].T @ score_resid[g]
            meat += np.outer(s, s)
        cov = bread @ meat @ bread
        if g_count > 1:                                # CR1 small-sample factor
            cov *= (g_count / (g_count - 1)) * ((n - 1) / max(n - p, 1))
    return cov


def _glm_irls(y, X, link, *, iters=50, tol=1e-9):
    """Iteratively reweighted least squares for a quasi-likelihood GLM (binomial
    for ``logit``, Poisson for ``log``). Returns (beta, final IRLS weights)."""
    p = X.shape[1]
    beta = np.zeros(p)
    W = np.ones(X.shape[0])
    for _ in range(iters):
        eta = X @ beta
        mu = _link_inv(eta, link)
        if link == "logit":
            mu = np.clip(mu, 1e-8, 1 - 1e-8)
            gprime = 1.0 / (mu * (1.0 - mu))
            W = mu * (1.0 - mu)                        # 1 / (g'(μ)² · V(μ))
        else:  # log / quasi-Poisson
            mu = np.clip(mu, 1e-8, None)
            gprime = 1.0 / mu
            W = mu
        z = eta + (y - mu) * gprime                    # working response
        new = np.linalg.pinv(X.T @ (X * W[:, None])) @ (X.T @ (W * z))
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return beta, np.clip(W, 1e-12, None)


def _fit_one(y, X, *, link, groups, hat, XtX_inv, dof):
    """Fit one topic's regression. ``link`` is identity (OLS) / logit (fractional
    logit) / log (quasi-Poisson); ``groups`` (or None) selects cluster-robust vs
    classical/robust covariance. Returns (beta, cov, r2)."""
    n, p = X.shape
    if link == "identity" and groups is None:
        return _ols(y, X, hat, XtX_inv, dof)           # legacy classical OLS
    if link == "identity":
        beta = hat @ y
        cov = _sandwich(X, XtX_inv, y - X @ beta, groups, n, p)
        mu = X @ beta
    else:
        beta, W = _glm_irls(y, X, link)
        bread = np.linalg.pinv(X.T @ (X * W[:, None]))
        mu = _link_inv(X @ beta, link)
        cov = _sandwich(X, bread, y - mu, groups, n, p)  # GLM ψ_i = X_i(y_i−μ_i)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - mu) ** 2).sum()) / tss if tss > 0 else 0.0
    return beta, cov, r2


def estimate_effect(
    doc_topic,
    X,
    *,
    feature_names=None,
    topics=None,
    add_intercept=True,
    ci=0.95,
    cluster=None,
    link="identity",
):
    """Regress each topic's proportion on document covariates.

    Pass a point estimate of θ for an ordinary OLS, or a *stack of posterior
    draws* of θ for the **method of composition** — the uncertainty-propagating
    procedure R ``stm`` uses (Treier & Jackman 2008). With draws, each one is
    regressed and the results are pooled by Rubin's rules, so the reported
    standard errors include the topic-estimation uncertainty, not just OLS
    sampling error. Get draws with :func:`posterior_theta_samples`.

    For paper-grade inference two extras matter:

    - ``cluster`` — a length-``num_docs`` array of group labels (e.g. speaker,
      user, outlet). Text data is almost always nested, and ignoring it
      understates uncertainty. Supplying it switches the standard errors to the
      **cluster-robust** (CR1) sandwich estimator. (With posterior draws, each
      draw is clustered and the per-draw covariances are then Rubin-pooled.)
    - ``link`` — ``"identity"`` (default OLS), ``"logit"`` (fractional logit, via
      binomial quasi-likelihood), or ``"log"`` (quasi-Poisson). Because topic
      proportions live in ``[0, 1]``, the logit link keeps fitted values in
      bounds where OLS can wander outside them (Papke & Wooldridge). Non-identity
      links report heteroskedasticity- or cluster-robust standard errors.

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

    if link not in ("identity", "logit", "log"):
        raise ValueError("link must be 'identity', 'logit', or 'log'")
    groups = None
    if cluster is not None:
        cluster = np.asarray(cluster)
        if cluster.shape[0] != n:
            raise ValueError("cluster must have one label per document")
        groups = [np.where(cluster == g)[0] for g in np.unique(cluster)]

    p = X.shape[1]
    XtX_inv = np.linalg.pinv(X.T @ X)
    hat = XtX_inv @ X.T  # (p, n)
    dof = max(n - p, 1)
    z_crit = _normal_ppf(0.5 + ci / 2.0)  # normal-approx critical value (no scipy)

    topic_list = list(range(num_topics)) if topics is None else list(topics)
    for t in topic_list:
        if t < 0 or t >= num_topics:
            raise ValueError(f"topic {t} out of range (num_topics={num_topics})")

    # Fast path: plain OLS (identity link, no clustering) is just shared-hat
    # matrix algebra, so solve for every requested topic — and every posterior
    # draw — with a few batched matmuls instead of a Python loop per (topic, sim).
    fast = link == "identity" and groups is None
    if fast and pooled:
        Yt = theta[:, :, topic_list]                          # (M, n, T)
        B = np.einsum("pn,snt->spt", hat, Yt)                 # (M, p, T)
        R = Yt - np.einsum("np,spt->snt", X, B)
        ss = np.einsum("snt,snt->st", R, R)                   # (M, T)
        within_scale = (ss / dof).mean(axis=0)                # (T,)
        beta_mean = B.mean(axis=0)                            # (p, T)
        tss = ((Yt - Yt.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)  # (M, T)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2_all = np.where(tss > 0, 1.0 - ss / tss, 0.0).mean(axis=0)  # (T,)
    elif fast:
        Y = theta[:, topic_list]                              # (n, T)
        B = hat @ Y                                           # (p, T)
        R = Y - X @ B
        ss = np.einsum("nt,nt->t", R, R)                      # (T,)
        SE = np.sqrt(np.clip(np.diag(XtX_inv)[:, None] * (ss / dof)[None, :], 0.0, None))
        tss = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)         # (T,)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2_all = np.where(tss > 0, 1.0 - ss / tss, 0.0)

    out: list[TopicEffect] = []
    for i, t in enumerate(topic_list):
        if fast and pooled:
            # Rubin's rules: total var = within + (1 + 1/M) * between.
            between = np.cov(B[:, :, i], rowvar=False) if nsims > 1 else np.zeros((p, p))
            total = within_scale[i] * XtX_inv + (1.0 + 1.0 / nsims) * np.atleast_2d(between)
            beta, se, r2 = beta_mean[:, i], np.sqrt(np.clip(np.diag(total), 0.0, None)), float(r2_all[i])
        elif fast:
            beta, se, r2 = B[:, i], SE[:, i], float(r2_all[i])
        elif pooled:
            betas = np.empty((nsims, p))
            within = np.zeros((p, p))
            r2s = np.empty(nsims)
            for m in range(nsims):
                b, cov_m, r2_m = _fit_one(theta[m, :, t], X, link=link, groups=groups,
                                          hat=hat, XtX_inv=XtX_inv, dof=dof)
                betas[m] = b
                within += cov_m
                r2s[m] = r2_m
            within /= nsims
            beta = betas.mean(axis=0)
            between = np.cov(betas, rowvar=False) if nsims > 1 else np.zeros((p, p))
            total = within + (1.0 + 1.0 / nsims) * np.atleast_2d(between)
            se = np.sqrt(np.clip(np.diag(total), 0.0, None))
            r2 = float(r2s.mean())
        else:
            beta, cov, r2 = _fit_one(theta[:, t], X, link=link, groups=groups,
                                     hat=hat, XtX_inv=XtX_inv, dof=dof)
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
    eye = np.eye(km1)

    # Cholesky factors for every document at once. Cholesky is all-or-nothing on
    # a batch, so only fall back to per-doc eigh for the docs that aren't PD —
    # the common (all-PD) case stays a single batched LAPACK call.
    csym = 0.5 * (cov + cov.transpose(0, 2, 1)) + 1e-10 * eye
    try:
        chol = np.linalg.cholesky(csym)                  # (D, K-1, K-1)
    except np.linalg.LinAlgError:
        chol = np.empty_like(csym)
        for di in range(d):
            try:
                chol[di] = np.linalg.cholesky(csym[di])
            except np.linalg.LinAlgError:
                w, v = np.linalg.eigh(csym[di])
                chol[di] = v @ np.diag(np.sqrt(np.clip(w, 1e-12, None)))

    # Draw in document order (matches the old per-doc loop's RNG stream), then
    # η = λ + Z·Lᵀ for all docs/sims via one batched matmul.
    z = rng.standard_normal((d, nsims, km1))
    eta = lam[:, None, :] + z @ chol.transpose(0, 2, 1)  # (D, nsims, K-1)
    full = np.concatenate([eta, np.zeros((d, nsims, 1))], axis=2)  # ref cat = 0
    full -= full.max(axis=2, keepdims=True)
    e = np.exp(full)
    theta = e / e.sum(axis=2, keepdims=True)             # (D, nsims, K)
    return theta.transpose(1, 0, 2).copy()               # (nsims, D, K)


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
# Back-compatibility: the general post-hoc diagnostics were moved to
# ``topica.diagnostics`` (they apply to any model, not just STM) and are
# also exported at the package top level. They are re-exported here so existing
# ``topica.stm.<name>`` calls keep working.
# ---------------------------------------------------------------------------
from .diagnostics import (  # noqa: E402,F401
    frex,
    label_topics,
    topic_correlation,
    TopicCorrelation,
    find_thoughts,
    search_k,
    relevance,
    prepare_pyldavis,
    PyLDAvisInputs,
    check_residuals,
    ResidualCheck,
    align_topics,
    topic_stability,
)


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
