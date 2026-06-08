"""Honest dynamic prevalence: topic prevalence as an estimable time series.

Most "topics over time" tooling models time as a *smooth mean function* — a
spline of date on prevalence (STM), or a random-walk drift of topic content
(DTM). This module does something different: it treats the per-period topic
prevalence as the state of an estimable **vector autoregression** (VAR), so you
can ask the questions time-series methods are built for — persistence, lead-lag
(Granger) structure between topics, impulse response, and forecasting.

The catch that this module exists to handle: each period's prevalence is
*measured with error*. You do not observe the period's true topic mix; you
estimate it from a finite, noisy sample of documents whose topic proportions are
themselves uncertain. Feeding those point estimates into an ordinary VAR (the
common "fit topics, then run a VAR on the proportions" two-step) is a textbook
errors-in-variables mistake: it attenuates persistence and manufactures
cross-topic Granger links that are not there — and the false-discovery rate
*grows* with the number of periods. See ``period_prevalence`` for how the
measurement covariance ``R_t`` is built, and ``fit_prevalence_var`` for the
state-space estimator that propagates it.

Two public entry points:

``period_prevalence(model, timestamps, ...)``
    Turn a fitted model + per-document timestamps into per-period prevalence in
    isometric log-ratio (ILR) space, with an honest measurement covariance
    ``R_t`` per period. The reusable measurement layer.

``fit_prevalence_var(prevalence, ...)``
    Fit a VAR(1) to that prevalence by a linear-Gaussian state-space EM that
    uses ``R_t`` as a known, time-varying observation covariance. Exposes
    ``granger_test``, ``forecast`` and ``impulse_response``.

Everything here is post-hoc and model-neutral: any model exposing ``doc_topic``
works, and uncertainty is drawn through topica's existing method-of-composition
machinery (``topica.effects.composition_theta``).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

__all__ = [
    "period_prevalence",
    "period_prevalence_from_counts",
    "period_prevalence_panel_from_counts",
    "fit_prevalence_var",
    "fit_multilevel_var",
    "PeriodPrevalence",
    "PrevalenceVAR",
    "PanelPrevalence",
    "MultilevelVAR",
]


# --------------------------------------------------------------------------- #
# Isometric log-ratio (ILR) geometry
#
# Prevalence lives on the K-simplex; a VAR lives in R^(K-1). The ILR transform
# is the right bridge: an isometry onto R^(K-1) with an orthonormal basis, so it
# has no arbitrary reference category (unlike the additive log-ratio) and is
# full-rank (unlike the centered log-ratio). We use the canonical Egozcue (2003)
# basis.
# --------------------------------------------------------------------------- #
def _ilr_basis(K: int) -> np.ndarray:
    """Canonical ILR contrast basis, shape (K, K-1) with orthonormal columns.

    Each column sums to zero (orthogonal to the constant direction), so the
    log-ratio coordinates are invariant to the closure/normalization of the
    composition.
    """
    V = np.zeros((K, K - 1))
    for i in range(1, K):                       # column i-1 contrasts {1..i} vs {i+1}
        norm = math.sqrt(i / (i + 1.0))
        V[:i, i - 1] = norm / i
        V[i, i - 1] = -norm
    return V


def _ilr(p: np.ndarray, V: np.ndarray) -> np.ndarray:
    """ILR coordinates of a (strictly positive) composition ``p``."""
    return V.T @ np.log(p)                       # mean-subtraction cancels: V_j ⊥ 1


def _ilr_inv(eta: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Inverse ILR: back to the simplex."""
    z = V @ eta
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _smooth(p: np.ndarray, delta: float) -> np.ndarray:
    """Additive smoothing so a period/draw with an unused topic stays positive."""
    return (p + delta) / (1.0 + len(p) * delta)


# --------------------------------------------------------------------------- #
# The measurement layer: per-period prevalence with honest covariance R_t
# --------------------------------------------------------------------------- #
@dataclass
class PeriodPrevalence:
    """Per-period topic prevalence in ILR space, with measurement covariance.

    Attributes
    ----------
    labels : list
        Sorted distinct period labels (length ``T``).
    eta : np.ndarray, shape (T, K-1)
        ILR-coordinate prevalence estimate per period (the VAR's observations).
    R : np.ndarray, shape (T, K-1, K-1)
        Measurement covariance of each ``eta[t]``, combining finite-document
        sampling error and topic-estimation uncertainty (Rubin's total
        variance). This is what makes downstream inference honest.
    n_docs : np.ndarray, shape (T,)
        Number of documents in each period.
    basis : np.ndarray, shape (K, K-1)
        The ILR basis ``V`` (kept so ``prevalence`` can map back to the simplex).
    num_topics : int
    """

    labels: list
    eta: np.ndarray
    R: np.ndarray
    n_docs: np.ndarray
    basis: np.ndarray
    num_topics: int

    @property
    def num_periods(self) -> int:
        return len(self.labels)

    def prevalence(self) -> np.ndarray:
        """Back-transform ``eta`` to simplex proportions, shape (T, K)."""
        return np.vstack([_ilr_inv(self.eta[t], self.basis)
                          for t in range(self.num_periods)])

    def reliability(self) -> np.ndarray:
        """Crude per-period signal-to-noise: trace(Cov(eta)) / (that + trace(R_t)).

        1.0 = measurement error negligible; ->0 = prevalence is mostly noise.
        A quick read on which periods the two-step would most distort.
        """
        if self.num_periods > 1:
            between = float(np.trace(np.atleast_2d(np.cov(self.eta, rowvar=False))))
        else:
            between = 0.0
        noise = np.array([np.trace(self.R[t]) for t in range(self.num_periods)])
        denom = between + noise
        return np.where(denom > 0, between / denom, 0.0)


def period_prevalence(
    model,
    timestamps: Sequence,
    *,
    corpus=None,
    nsims: int = 25,
    smoothing: float = 1e-4,
    min_docs: int = 1,
    seed: int = 0,
) -> PeriodPrevalence:
    """Extract per-period prevalence and its measurement covariance ``R_t``.

    Parameters
    ----------
    model
        A fitted topica model exposing ``doc_topic`` (LDA, STM, CTM, KeyATM, …).
    timestamps
        One label per document (same order/length as ``model.doc_topic``).
        Sorted distinct values define the periods; values may be numbers,
        strings, or dates, and need not be contiguous.
    corpus
        The fitted ``Corpus`` — needed for document lengths when the model's
        uncertainty family is Dirichlet (Gibbs models). Logistic-normal models
        (STM, CTM) carry their own posterior and do not need it.
    nsims
        Number of method-of-composition draws used to quantify topic-estimation
        uncertainty. ``nsims=1`` falls back to sampling error only.
    smoothing
        Additive smoothing applied to period/draw means before the log-ratio,
        so an unused topic does not send a coordinate to -inf.
    min_docs
        Periods with fewer documents are dropped (with a warning).
    seed
        Seed for the composition draws.

    Returns
    -------
    PeriodPrevalence

    Notes
    -----
    ``R_t`` is assembled by Rubin's total-variance rule across composition draws:

        R_t = mean_s W_t^(s)  +  (1 + 1/S) * Cov_s( eta_t^(s) )

    where, for draw ``s``, ``eta_t^(s)`` is the ILR of that period's mean
    proportions and ``W_t^(s)`` is the finite-document sampling covariance of
    that mean, delta-method-propagated through the ILR map:

        W_t^(s) = J · ( Cov_docs(theta) / N_t ) · J^T ,   J = V^T · diag(1/p̄)

    The first term is the error you would have even if theta were known exactly
    (a finite sample of documents); the second is the error from theta being
    estimated. Both shrink as documents accumulate — ``W`` like 1/N_t, ``Cov_s``
    as the posterior sharpens — which is exactly the behavior the two-step
    ignores by plugging in a point estimate.
    """
    doc_topic = np.asarray(model.doc_topic, dtype=np.float64)
    n_docs_total, K = doc_topic.shape
    if len(timestamps) != n_docs_total:
        raise ValueError(
            f"timestamps has length {len(timestamps)} but the model has "
            f"{n_docs_total} documents")
    if K < 2:
        raise ValueError("need at least 2 topics for a prevalence VAR")

    # group document indices by period
    labels = sorted(set(timestamps))
    by_period = {lab: [] for lab in labels}
    for d, lab in enumerate(timestamps):
        by_period[lab].append(d)
    kept = [lab for lab in labels if len(by_period[lab]) >= min_docs]
    if len(kept) < len(labels):
        dropped = [lab for lab in labels if lab not in kept]
        warnings.warn(f"dropping {len(dropped)} period(s) with < {min_docs} "
                      f"documents: {dropped}")
    labels = kept
    if len(labels) < 2:
        raise ValueError("need at least 2 periods after the min_docs filter")

    # method-of-composition draws of theta: (S, n_docs, K). Captures the
    # topic-estimation uncertainty. Falls back to the point estimate if the
    # model family carries no posterior (embedding/clustering models).
    draws = _composition_draws(model, corpus, nsims, seed)
    S = draws.shape[0]

    V = _ilr_basis(K)
    d = K - 1
    eta = np.zeros((len(labels), d))
    R = np.zeros((len(labels), d, d))
    n_docs = np.zeros(len(labels), dtype=int)

    for t, lab in enumerate(labels):
        idx = by_period[lab]
        N_t = len(idx)
        n_docs[t] = N_t
        eta_s = np.zeros((S, d))
        W = np.zeros((d, d))
        for s in range(S):
            theta = draws[s][idx]                     # (N_t, K)
            pbar = _smooth(theta.mean(axis=0), smoothing)
            eta_s[s] = _ilr(pbar, V)
            if N_t > 1:
                cov_docs = np.cov(theta, rowvar=False)  # (K, K)
                J = V.T / pbar                          # (K-1, K) == V^T diag(1/pbar)
                W += J @ (cov_docs / N_t) @ J.T
        W /= S                                          # mean within-draw sampling cov
        eta[t] = eta_s.mean(axis=0)
        if S > 1:
            between = np.cov(eta_s, rowvar=False)       # (d, d)
            R[t] = W + (1.0 + 1.0 / S) * between
        else:
            R[t] = W
        R[t] = 0.5 * (R[t] + R[t].T) + np.eye(d) * 1e-12

    return PeriodPrevalence(labels=labels, eta=eta, R=R, n_docs=n_docs,
                            basis=V, num_topics=K)


def period_prevalence_from_counts(counts, labels=None, *, smoothing: float = 0.5
                                  ) -> PeriodPrevalence:
    """Build a :class:`PeriodPrevalence` from per-period category COUNTS.

    For models that assign each document to a single topic/category (BERTopic,
    GSDMM, any clustering) or for pre-aggregated data, the per-period prevalence
    is a multinomial proportion and its honest measurement covariance is the
    multinomial sampling covariance, delta-method-propagated to ILR space:

        p_t = counts_t / N_t
        R_t = J ( diag(p) - p p^T ) / N_t J^T ,   J = V^T diag(1/p)

    This captures the finite-document sampling error you get even with hard
    assignments. It does NOT include topic-estimation uncertainty (hard
    assignments carry none); use :func:`period_prevalence` with a probabilistic
    model when you need that layer too.

    Parameters
    ----------
    counts : array (T, K)
        Nonnegative article/document counts per category per period.
    labels : sequence, optional
        Period labels (length T); defaults to ``range(T)``.
    smoothing : float
        Additive count smoothing (default 0.5, Jeffreys), keeps empty cells off
        the simplex boundary.
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] < 2:
        raise ValueError("counts must be (T, K) with K >= 2 categories")
    T, K = counts.shape
    labels = list(range(T)) if labels is None else list(labels)
    V = _ilr_basis(K)
    d = K - 1
    eta = np.zeros((T, d))
    R = np.zeros((T, d, d))
    n_docs = counts.sum(axis=1).astype(int)
    for t in range(T):
        N = counts[t].sum()
        p = (counts[t] + smoothing) / (N + K * smoothing)
        eta[t] = _ilr(p, V)
        J = V.T / p                                  # (K-1, K)
        multinom = (np.diag(p) - np.outer(p, p)) / max(N, 1.0)
        R[t] = J @ multinom @ J.T
        R[t] = 0.5 * (R[t] + R[t].T) + np.eye(d) * 1e-12
    return PeriodPrevalence(labels=labels, eta=eta, R=R, n_docs=n_docs,
                            basis=V, num_topics=K)


def _composition_draws(model, corpus, nsims, seed) -> np.ndarray:
    """Draws of theta via topica's existing method-of-composition machinery.

    Returns shape (S, n_docs, K). Degrades to a single point-estimate "draw"
    (S=1) for models with no posterior, with a warning — R_t then reflects
    sampling error only.
    """
    doc_topic = np.asarray(model.doc_topic, dtype=np.float64)
    if nsims is None or nsims <= 1:
        return doc_topic[None, :, :]
    try:
        from . import effects
    except Exception:                                   # pragma: no cover
        return doc_topic[None, :, :]
    family = None
    try:
        family = effects.model_family(model)
    except Exception:
        pass
    if family == "none":
        warnings.warn("model has no posterior over theta (embedding/clustering "
                      "family); R_t will reflect document-sampling error only. "
                      "Consider a generative model (LDA/STM/CTM/keyATM) for the "
                      "full honest covariance.")
        return doc_topic[None, :, :]
    # composition_theta auto-dispatches logistic-normal vs Dirichlet draws.
    for kwargs in ({"corpus": corpus, "nsims": nsims, "seed": seed},
                   {"corpus": corpus, "nsims": nsims},
                   {"nsims": nsims, "seed": seed},
                   {"nsims": nsims}):
        try:
            draws = effects.composition_theta(model, **kwargs)
            return np.asarray(draws, dtype=np.float64)
        except TypeError:
            continue
        except Exception:
            break
    warnings.warn("could not draw composition samples; falling back to the "
                  "point estimate (sampling error only).")
    return doc_topic[None, :, :]


# --------------------------------------------------------------------------- #
# The estimator: state-space VAR(1) that consumes R_t honestly
#
# Ported verbatim (and generalized to d = K-1 dimensions) from the validated
# proof-of-concept; the EM M-step for Q uses smoothed state covariances, which
# is what removes the measurement variance from the innovation estimate. Getting
# that sign wrong silently inflates persistence — the synthetic recovery test is
# the guard.
# --------------------------------------------------------------------------- #
@dataclass
class PrevalenceVAR:
    """A fitted VAR(1) on period prevalence, with measurement error accounted.

    state:  m_t = c + A m_{t-1} + eps_t,  eps ~ N(0, Q)
    obs:    eta_t = m_t + u_t,            u   ~ N(0, R_t)   (R_t known)
    """

    A: np.ndarray
    c: np.ndarray
    Q: np.ndarray
    loglik: float
    labels: list
    eta: np.ndarray
    R: np.ndarray
    basis: np.ndarray

    @property
    def num_topics(self) -> int:
        return self.basis.shape[0]

    def stationary(self) -> bool:
        """Is the fitted VAR stationary (companion eigenvalues inside unit circle)?"""
        return bool(np.max(np.abs(np.linalg.eigvals(self.A))) < 1.0)

    def granger_test(self, cause: int, effect: int, *, method: str = "bootstrap",
                     n_boot: int = 199, seed: int = 0) -> dict:
        """Test that ILR coordinate ``cause`` does not Granger-cause ``effect``
        (i.e. A[effect, cause] = 0).

        method="bootstrap" (default, recommended): a PARAMETRIC-BOOTSTRAP null.
        The naive likelihood-ratio-vs-chi2(1) test (method="lr_chi2") is badly
        over-sized for this finite-sample state-space EM -- ~30% false positives
        at d=4 (see granger_calibration). The bootstrap instead fits the
        restricted (no-link) model, simulates ``n_boot`` datasets from it through
        the SAME measurement covariances R_t, refits both models on each, and
        reports the fraction of null LRs at least as large as the observed one.
        Exact in size by construction; the cost is ``n_boot`` warm-started refits.

        Returns the LR statistic, the (calibrated) p-value, the unrestricted
        coefficient, and the method used.
        """
        d = self.A.shape[0]
        if not (0 <= cause < d and 0 <= effect < d):
            raise ValueError(f"cause/effect must be coordinates in [0, {d})")
        drop = {(effect, cause)}
        FIT_ITERS = 60

        def _lr(y):                              # ONE fitting procedure, used for
            _, _, _, lf = _fit_ssm(y, self.R, drop=set(), n_iter=FIT_ITERS)
            _, _, _, lr = _fit_ssm(y, self.R, drop=drop, n_iter=FIT_ITERS)
            return 2.0 * max(lf - lr, 0.0)

        lr_obs = _lr(self.eta)
        if method == "lr_chi2":
            p = math.erfc(math.sqrt(lr_obs / 2.0)) if lr_obs > 0 else 1.0
        elif method == "bootstrap":
            # H0 params for the simulator: a well-converged restricted fit.
            A_r, c_r, Q_r, _ = _fit_ssm(self.eta, self.R, drop=drop, n_iter=100)
            rng = np.random.default_rng(seed)
            ge = 0
            for _ in range(n_boot):              # identical _lr() on every draw ->
                yb = _simulate_null(A_r, c_r, Q_r, self.R, self.eta[0], rng)
                if _lr(yb) >= lr_obs:            # valid bootstrap by construction
                    ge += 1
            p = (1.0 + ge) / (n_boot + 1.0)
        else:
            raise ValueError("method must be 'bootstrap' or 'lr_chi2'")
        return {"cause": cause, "effect": effect, "lr_stat": lr_obs, "p_value": p,
                "coef": float(self.A[effect, cause]), "method": method}

    def forecast(self, horizon: int) -> np.ndarray:
        """Point forecast of ILR prevalence for the next ``horizon`` periods,
        shape (horizon, K-1). Use ``ilr``-inverse via a PeriodPrevalence if you
        want simplex proportions."""
        m = self.eta[-1].copy()
        out = np.zeros((horizon, self.A.shape[0]))
        for h in range(horizon):
            m = self.c + self.A @ m
            out[h] = m
        return out

    def impulse_response(self, shock_topic: int, horizon: int = 10) -> np.ndarray:
        """Response of all coordinates to a unit ILR shock in ``shock_topic``,
        shape (horizon+1, K-1). Row 0 is the impact period."""
        d = self.A.shape[0]
        resp = np.zeros((horizon + 1, d))
        e = np.zeros(d); e[shock_topic] = 1.0
        resp[0] = e
        for h in range(1, horizon + 1):
            resp[h] = self.A @ resp[h - 1]
        return resp


def fit_prevalence_var(prevalence: PeriodPrevalence, *, n_iter: int = 100
                       ) -> PrevalenceVAR:
    """Fit a VAR(1) to per-period prevalence by state-space EM with known R_t.

    Parameters
    ----------
    prevalence
        Output of :func:`period_prevalence`.
    n_iter
        EM iterations.

    Returns
    -------
    PrevalenceVAR
    """
    eta, R = prevalence.eta, prevalence.R
    A, c, Q, loglik = _fit_ssm(eta, R, drop=set(), n_iter=n_iter)
    return PrevalenceVAR(A=A, c=c, Q=Q, loglik=loglik, labels=prevalence.labels,
                         eta=eta, R=R, basis=prevalence.basis)


def _psd(M, floor=1e-10):
    """Nearest PSD matrix by clipping eigenvalues (the EM Q-update is not
    guaranteed PSD in finite samples)."""
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    return (V * np.clip(w, floor, None)) @ V.T


def _chol_psd(M):
    """A factor L with L L^T a PSD version of M (robust to indefiniteness)."""
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    return V * np.sqrt(np.clip(w, 1e-12, None))


# --------------------------------------------------------------------------- #
# Multilevel HDPM: paper(group)-specific intercepts + one shared latent VAR.
#
#   eta_{p,t} = alpha_p + m_t + u_{p,t},   u ~ N(0, R_{p,t})       (observation)
#   m_t       = c + A m_{t-1} + eps_t,     eps ~ N(0, Q)           (shared state)
#   alpha_p   ~ N(0, Sigma_alpha)                                  (random effect)
#
# This separates *which papers contributed* (alpha_p, time-invariant composition)
# from *when the discourse moved* (m_t), defending the dynamics against the
# compositional confound of pooling. Per-paper-per-period data is sparse, so the
# honest R_{p,t} layer is essential here -- this is where it earns its keep.
#
# Estimation is a nested EM. Given alpha_p, the papers present in period t give
# Gaussian measurements of the same m_t, which combine EXACTLY into one effective
# observation (precision-weighted), so the standard Kalman smoother applies. The
# M-step then refreshes (A, c, Q) from the smoothed state and updates each
# alpha_p by a shrinkage (random-effects) posterior. Identification of the
# additive level is handled by the alpha_p ~ N(0, Sigma_alpha) prior.
# --------------------------------------------------------------------------- #
@dataclass
class PanelPrevalence:
    """Per-paper, per-period prevalence in ILR space with measurement covariance.

    eta : (P, T, d) ; R : (P, T, d, d) ; present : (P, T) bool mask of which
    (paper, period) cells are observed.
    """
    period_labels: list
    paper_labels: list
    eta: np.ndarray
    R: np.ndarray
    present: np.ndarray
    basis: np.ndarray
    num_topics: int


@dataclass
class MultilevelVAR:
    """A fitted multilevel HDPM: shared VAR(1) + paper intercepts."""
    A: np.ndarray
    c: np.ndarray
    Q: np.ndarray
    alpha: np.ndarray            # (P, d) paper intercepts
    Sigma_alpha: np.ndarray      # (d, d) random-effect covariance
    m: np.ndarray                # (T, d) smoothed common state
    loglik: float
    period_labels: list
    paper_labels: list
    basis: np.ndarray

    def stationary(self) -> bool:
        return bool(np.max(np.abs(np.linalg.eigvals(self.A))) < 1.0)


def period_prevalence_panel_from_counts(counts, paper_labels=None,
                                        period_labels=None, *, smoothing=0.5):
    """Build a :class:`PanelPrevalence` from per-paper, per-period COUNTS.

    counts : (P, T, K) nonnegative counts. A (paper, period) cell with zero
    documents is marked not-present. Measurement covariance per cell is the
    multinomial sampling covariance, delta-mapped to ILR (as in
    :func:`period_prevalence_from_counts`).
    """
    counts = np.asarray(counts, dtype=np.float64)
    P, T, K = counts.shape
    V = _ilr_basis(K)
    d = K - 1
    eta = np.zeros((P, T, d))
    R = np.zeros((P, T, d, d))
    present = np.zeros((P, T), dtype=bool)
    for pp in range(P):
        for t in range(T):
            N = counts[pp, t].sum()
            if N <= 0:
                continue
            present[pp, t] = True
            p = (counts[pp, t] + smoothing) / (N + K * smoothing)
            eta[pp, t] = _ilr(p, V)
            J = V.T / p
            R[pp, t] = J @ ((np.diag(p) - np.outer(p, p)) / N) @ J.T
            R[pp, t] = 0.5 * (R[pp, t] + R[pp, t].T) + np.eye(d) * 1e-12
    return PanelPrevalence(
        period_labels=list(period_labels) if period_labels is not None else list(range(T)),
        paper_labels=list(paper_labels) if paper_labels is not None else list(range(P)),
        eta=eta, R=R, present=present, basis=V, num_topics=K)


def fit_multilevel_var(panel: PanelPrevalence, *, n_iter: int = 40,
                       inner_iter: int = 15) -> MultilevelVAR:
    """Fit the multilevel HDPM (shared VAR + paper random intercepts) by EM."""
    P, T, d = panel.eta.shape
    eta, R, present = panel.eta, panel.R, panel.present
    Rinv = np.zeros_like(R)
    for pp in range(P):
        for t in range(T):
            if present[pp, t]:
                Rinv[pp, t] = np.linalg.inv(R[pp, t] + np.eye(d) * 1e-10)

    alpha = np.zeros((P, d))
    Sig_a = np.eye(d) * 0.5
    A = np.eye(d) * 0.5
    c = np.zeros(d)
    Q = np.eye(d) * 0.1
    m_s = np.zeros((T, d))
    loglik = -np.inf

    for _ in range(n_iter):
        # ---- E-step part 1: combine papers -> effective obs (yhat_t, Rtil_t) ----
        yhat = np.zeros((T, d))
        Rtil = np.zeros((T, d, d))
        for t in range(T):
            Lam = np.zeros((d, d))
            b = np.zeros(d)
            for pp in range(P):
                if present[pp, t]:
                    Lam += Rinv[pp, t]
                    b += Rinv[pp, t] @ (eta[pp, t] - alpha[pp])
            if np.trace(Lam) <= 0:
                Rtil[t] = np.eye(d) * 1e6
            else:
                Rtil[t] = np.linalg.inv(Lam + np.eye(d) * 1e-10)
                yhat[t] = Rtil[t] @ b
        # ---- M-step part 1: refresh (A, c, Q) on the effective series ----
        A, c, Q, loglik = _fit_ssm(yhat, Rtil, n_iter=inner_iter, init=(A, c, Q))
        # smoothed common state (for the alpha update)
        var0 = float(np.mean(np.var(yhat, axis=0)))
        mu0 = yhat[0].copy()
        P0 = np.eye(d) * (var0 * 10.0 + 1.0)
        m_s, P_smooth, _, _ = _smoother(yhat, Rtil, A, c, Q, mu0, P0)
        # ---- M-step part 2: update paper intercepts (shrinkage / random effect) ----
        Sig_a_inv = np.linalg.inv(Sig_a + np.eye(d) * 1e-8)
        new_alpha = np.zeros((P, d))
        Sig_acc = np.zeros((d, d))
        for pp in range(P):
            Lam_p = Sig_a_inv.copy()
            bp = np.zeros(d)
            for t in range(T):
                if present[pp, t]:
                    Reff = np.linalg.inv(R[pp, t] + P_smooth[t] + np.eye(d) * 1e-10)
                    Lam_p += Reff
                    bp += Reff @ (eta[pp, t] - m_s[t])
            cov_p = np.linalg.inv(Lam_p)
            new_alpha[pp] = cov_p @ bp
            Sig_acc += cov_p + np.outer(new_alpha[pp], new_alpha[pp])
        alpha = new_alpha
        Sig_a = _psd(Sig_acc / P, floor=1e-6)

    return MultilevelVAR(A=A, c=c, Q=Q, alpha=alpha, Sigma_alpha=Sig_a, m=m_s,
                         loglik=loglik, period_labels=panel.period_labels,
                         paper_labels=panel.paper_labels, basis=panel.basis)


def _simulate_null(A, c, Q, R, y0, rng):
    """Simulate an observed ILR-prevalence series under a given VAR(1) state law
    (A, c, Q) seen through the actual per-period measurement covariances R_t.
    Used to build the parametric-bootstrap null in granger_test."""
    T, d = R.shape[0], A.shape[0]
    LQ = _chol_psd(Q)
    m = np.zeros((T, d))
    m[0] = y0
    y = np.zeros((T, d))
    for t in range(T):
        if t > 0:
            m[t] = c + A @ m[t - 1] + LQ @ rng.standard_normal(d)
        y[t] = m[t] + _chol_psd(R[t]) @ rng.standard_normal(d)
    return y


# ----- linear-Gaussian state-space machinery (Kalman/RTS/EM) ----------------- #
def _smoother(y, R, A, c, Q, mu0, P0):
    T, d = y.shape
    I = np.eye(d)
    mf = np.zeros((T, d)); Pf = np.zeros((T, d, d))
    mp = np.zeros((T, d)); Pp = np.zeros((T, d, d))
    loglik = 0.0
    K_last = None
    for t in range(T):
        if t == 0:
            mp[t], Pp[t] = mu0, P0
        else:
            mp[t] = c + A @ mf[t - 1]
            Pp[t] = A @ Pf[t - 1] @ A.T + Q
        S = Pp[t] + R[t] + np.eye(d) * 1e-9      # ridge: keep S well-conditioned
        v = y[t] - mp[t]
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        loglik += -0.5 * (d * math.log(2 * math.pi)
                          + np.linalg.slogdet(S)[1] + v @ Sinv @ v)
        Kg = Pp[t] @ Sinv
        mf[t] = mp[t] + Kg @ v
        Pf[t] = (I - Kg) @ Pp[t]
        if t == T - 1:
            K_last = Kg
    ms = np.zeros((T, d)); Ps = np.zeros((T, d, d)); J = np.zeros((T, d, d))
    ms[-1], Ps[-1] = mf[-1], Pf[-1]
    for t in range(T - 2, -1, -1):
        Ppt1 = Pp[t + 1] + np.eye(d) * 1e-9
        try:
            J[t] = np.linalg.solve(Ppt1, (A @ Pf[t].T)).T
        except np.linalg.LinAlgError:
            J[t] = (A @ Pf[t].T @ np.linalg.pinv(Ppt1)).T
        ms[t] = mf[t] + J[t] @ (ms[t + 1] - mp[t + 1])
        Ps[t] = Pf[t] + J[t] @ (Ps[t + 1] - Pp[t + 1]) @ J[t].T
    Plag = np.zeros((T, d, d))
    Plag[-1] = (I - K_last) @ A @ Pf[-2]
    for t in range(T - 2, 0, -1):
        Plag[t] = Pf[t] @ J[t - 1].T + J[t] @ (Plag[t + 1] - A @ Pf[t]) @ J[t - 1].T
    return ms, Ps, Plag, loglik


def _fit_ssm(y, R, *, drop=None, n_iter: int = 100, init=None):
    """EM for the state-space VAR(1). ``drop`` = set of (row, predictor_col)
    transition coefficients pinned to 0; predictor cols 0..d-1 are lags, col d
    is the intercept. ``init`` = optional (A, c, Q) warm start (used by the
    bootstrap null so refits converge in few iterations). Returns
    (A, c, Q, loglik)."""
    drop = drop or set()
    T, d = y.shape
    if init is not None:
        A = init[0].copy()
        c = init[1].copy()
        Q = init[2].copy()
    else:
        # init from a shrunk OLS on the (noisy) observations
        Y = y[1:]
        X = np.hstack([y[:-1], np.ones((T - 1, 1))])
        B0 = np.linalg.lstsq(X, Y, rcond=None)[0].T
        A = B0[:, :d] * 0.5
        c = B0[:, d].copy()
        Q = np.eye(d) * float(np.mean(np.var(y, axis=0))) * 0.5 + np.eye(d) * 1e-6
    for (i, j) in drop:            # honor the constraint on the warm start
        if j < d:
            A[i, j] = 0.0
    mu0 = y[0].copy()
    P0 = np.eye(d) * (float(np.mean(np.var(y, axis=0))) * 10.0 + 1.0)
    allowed = [[j for j in range(d + 1) if (i, j) not in drop] for i in range(d)]
    for _ in range(n_iter):
        ms, Ps, Plag, _ = _smoother(y, R, A, c, Q, mu0, P0)
        Sxx = np.zeros((d + 1, d + 1))
        Syx = np.zeros((d, d + 1))
        Syy = np.zeros((d, d))
        for t in range(1, T):
            mtm1 = ms[t - 1]
            Sxx[:d, :d] += Ps[t - 1] + np.outer(mtm1, mtm1)
            Sxx[:d, d] += mtm1
            Sxx[d, :d] += mtm1
            Sxx[d, d] += 1.0
            Syx[:, :d] += Plag[t] + np.outer(ms[t], mtm1)
            Syx[:, d] += ms[t]
            Syy += Ps[t] + np.outer(ms[t], ms[t])
        B = np.zeros((d, d + 1))
        for i in range(d):
            cols = allowed[i]
            B[i, cols] = Syx[i, cols] @ np.linalg.inv(Sxx[np.ix_(cols, cols)])
        A, c = B[:, :d], B[:, d]
        Q = _psd((Syy - B @ Syx.T) / (T - 1), floor=1e-9)
    *_, loglik = _smoother(y, R, A, c, Q, mu0, P0)
    return A, c, Q, loglik
