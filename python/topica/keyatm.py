"""keyATM-specific workflow helpers, mirroring the R ``keyATM`` package.

The model-agnostic analyses already live in :mod:`topica.diagnostics` and work
on a fitted :class:`~topica.KeyATM`'s numpy outputs, so they cover most of the R
workflow directly:

- ``keyATM::top_words``       -> :meth:`topica.KeyATM.top_words`
- ``keyATM::top_docs``        -> :func:`topica.find_thoughts`
- ``keyATM::semantic_coherence`` -> :meth:`topica.KeyATM.coherence`
- ``keyATM::plot_modelfit``   -> :attr:`topica.KeyATM.log_likelihood_history`
- ``keyATM::covariates_info`` -> :attr:`topica.KeyATM.feature_effects` / ``feature_names``
- ``estimateEffect``-style    -> :func:`topica.stm.estimate_effect`

This module adds the keyATM-flavored pieces that operate on the keywords and the
document-topic matrix:

- :func:`top_topics`        ~ ``keyATM::top_topics``       (top topics per document)
- :func:`by_strata`         ~ ``keyATM::by_strata_DocTopic`` (covariate-stratified prevalence)
- :func:`visualize_keywords` ~ ``keyATM::visualize_keywords`` (keyword corpus frequencies)
- :func:`refine_keywords`   ~ ``keyATM::refine_keywords``   (drop too-rare keywords)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


def _theta_and_names(model_or_theta, topic_names=None):
    """Accept either a fitted model (with ``doc_topic``/``topic_names``) or a raw
    theta array, returning ``(theta, names)``."""
    if hasattr(model_or_theta, "doc_topic"):
        theta = np.asarray(model_or_theta.doc_topic, dtype=np.float64)
        if topic_names is None:
            topic_names = list(getattr(model_or_theta, "topic_names", []))
    else:
        theta = np.asarray(model_or_theta, dtype=np.float64)
    if theta.ndim != 2:
        raise ValueError("doc_topic must be 2-D (num_docs, num_topics)")
    k = theta.shape[1]
    if not topic_names:
        topic_names = [f"topic_{t}" for t in range(k)]
    if len(topic_names) != k:
        raise ValueError(f"topic_names has {len(topic_names)} entries but theta has {k} topics")
    return theta, list(topic_names)


def top_topics(model_or_theta, *, n=2, topic_names=None):
    """The ``n`` most prevalent topics in each document (≈ ``keyATM::top_topics``).

    Returns a list (one per document) of ``(topic_name, proportion)`` pairs,
    sorted by descending document-topic proportion. Pass a fitted
    :class:`~topica.KeyATM` (topic names are taken from it) or a raw ``theta``
    array.
    """
    theta, names = _theta_and_names(model_or_theta, topic_names)
    if n < 1:
        raise ValueError("n must be >= 1")
    n = min(n, theta.shape[1])
    out = []
    for row in theta:
        idx = np.argsort(row)[::-1][:n]
        out.append([(names[t], float(row[t])) for t in idx])
    return out


@dataclass
class StrataPrevalence:
    """Mean topic prevalence within one covariate stratum, with intervals."""

    stratum: object
    n: int
    topic_names: list
    mean: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray

    def as_dict(self) -> dict:
        return {
            "stratum": self.stratum,
            "n": self.n,
            **{
                name: {
                    "mean": float(self.mean[t]),
                    "ci": (float(self.ci_low[t]), float(self.ci_high[t])),
                }
                for t, name in enumerate(self.topic_names)
            },
        }


def by_strata(model_or_theta, strata, *, ci=0.95, topic_names=None, corpus=None, nsims=None, seed=0):
    """Mean topic prevalence within each level of a document covariate
    (≈ ``keyATM::by_strata_DocTopic``).

    Splits documents by their value in ``strata`` (one label per document) and,
    for each level, reports the mean of each topic's proportion with a
    normal-approximation confidence interval on that mean. This is keyATM's
    descriptive answer to "how does topic prevalence differ across groups".

    With ``nsims`` (and a fitted **model** as the first argument), the interval is
    widened by the **method of composition**: the model's θ posterior is drawn for
    you (logistic-normal for STM/CTM, Dirichlet for the Gibbs models — pass
    ``corpus=`` so document lengths are available) and the per-stratum means are
    pooled by Rubin's rules, so the topic-estimation uncertainty is propagated, not
    just the across-document spread. For a regression with the same propagation use
    :func:`topica.stm.estimate_effect`.

    Returns a list of :class:`StrataPrevalence`, one per unique stratum (sorted).
    ``[s.as_dict() for s in result]`` builds a table.
    """
    from .stm import _normal_ppf

    z = _normal_ppf(0.5 + ci / 2.0)
    strata = np.asarray(strata)

    if nsims:
        from .effects import composition_theta

        if not hasattr(model_or_theta, "doc_topic"):
            raise ValueError("by_strata(..., nsims=) needs a fitted model to draw theta")
        names = list(getattr(model_or_theta, "topic_names", [])) or None
        draws = composition_theta(model_or_theta, corpus, nsims=nsims, seed=seed)
        m, d, k = draws.shape
        names = names or [f"topic_{t}" for t in range(k)]
        if strata.shape[0] != d:
            raise ValueError("strata must have one label per document")
        out = []
        for level in sorted(np.unique(strata), key=lambda v: str(v)):
            mask = strata == level
            n = int(mask.sum())
            sub = draws[:, mask, :]                       # (M, n, K)
            per_draw = sub.mean(axis=1)                   # (M, K)
            estimate = per_draw.mean(axis=0)
            between = per_draw.var(axis=0, ddof=1) if m > 1 else np.zeros(k)
            within = sub.var(axis=1, ddof=1).mean(axis=0) / n if n > 1 else np.zeros(k)
            se = np.sqrt(np.clip(within + (1.0 + 1.0 / m) * between, 0.0, None))
            out.append(
                StrataPrevalence(
                    stratum=level.item() if hasattr(level, "item") else level,
                    n=n,
                    topic_names=names,
                    mean=estimate,
                    ci_low=np.clip(estimate - z * se, 0.0, 1.0),
                    ci_high=np.clip(estimate + z * se, 0.0, 1.0),
                )
            )
        return out

    theta, names = _theta_and_names(model_or_theta, topic_names)
    if strata.shape[0] != theta.shape[0]:
        raise ValueError("strata must have one label per document")

    out = []
    for level in sorted(np.unique(strata), key=lambda v: str(v)):
        rows = theta[strata == level]
        n = rows.shape[0]
        mean = rows.mean(axis=0)
        # Standard error of the mean per topic (0 when a single document).
        se = rows.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
        out.append(
            StrataPrevalence(
                stratum=level.item() if hasattr(level, "item") else level,
                n=int(n),
                topic_names=names,
                mean=mean,
                ci_low=np.clip(mean - z * se, 0.0, 1.0),
                ci_high=np.clip(mean + z * se, 0.0, 1.0),
            )
        )
    return out


def _corpus_counts(docs):
    """(per-word corpus count, per-word document frequency, total tokens)."""
    counts = Counter()
    doc_freq = Counter()
    total = 0
    for d in docs:
        counts.update(d)
        doc_freq.update(set(d))
        total += len(d)
    return counts, doc_freq, total


def visualize_keywords(docs, keywords):
    """Corpus frequency of each keyword (≈ ``keyATM::visualize_keywords``).

    For every keyword in every set, reports how common it is in ``docs`` so you
    can catch keywords that are too rare to anchor a topic or so frequent they
    dominate it — the diagnostic keyATM asks you to run *before* fitting.

    Returns a dict mapping each keyword-set name to a list of dicts
    ``{"keyword", "count", "proportion", "doc_freq"}`` sorted by descending
    proportion, where ``proportion`` is the keyword's share of all corpus tokens
    and ``doc_freq`` is the number of documents containing it.
    """
    counts, doc_freq, total = _corpus_counts(docs)
    total = max(total, 1)
    out = {}
    for name, words in keywords.items():
        rows = [
            {
                "keyword": w,
                "count": int(counts.get(w, 0)),
                "proportion": counts.get(w, 0) / total,
                "doc_freq": int(doc_freq.get(w, 0)),
            }
            for w in words
        ]
        rows.sort(key=lambda r: r["proportion"], reverse=True)
        out[name] = rows
    return out


def time_prevalence_ci(model, timestamps, *, ci=0.95, normalize=True):
    """Per-period topic prevalence with credible intervals from the dynamic keyATM posterior.

    For a dynamic :class:`~topica.KeyATM` (fit with ``timestamps=`` and
    ``keep_theta_draws=True``), this computes per-period prevalence uncertainty
    directly from the retained MCMC ``theta_draws``. For each posterior draw and
    each time period, the per-draw average of theta over the documents in that
    period is computed, giving a (S, T, K) array of per-draw period-level
    prevalences. The point estimate is the posterior mean over draws; ``ci_low``
    and ``ci_high`` are the empirical (1-ci)/2 and (1+ci)/2 quantiles; ``sd`` is
    the posterior standard deviation.

    The periods are ordered to match ``model.time_labels`` exactly, so the result
    aligns with ``model.time_prevalence``.

    Parameters
    ----------
    model
        A fitted dynamic :class:`~topica.KeyATM` with non-empty ``time_labels``
        and non-``None`` ``theta_draws``. Refit with ``keep_theta_draws=True``
        (the default) if draws are absent.
    timestamps
        One value per document — the same array passed to ``fit``.
    ci
        Credible interval coverage (default 0.95 gives a 95 percent interval).
    normalize
        When ``True`` (default), each per-draw per-period prevalence row is
        normalized to sum to 1 before computing the summary statistics.

    Returns
    -------
    dict with keys:
        - ``labels``: list of period labels (equals ``model.time_labels``)
        - ``mean``: ndarray shape (T, K), posterior mean prevalence per period
        - ``ci_low``: ndarray shape (T, K), lower credible bound
        - ``ci_high``: ndarray shape (T, K), upper credible bound
        - ``sd``: ndarray shape (T, K), posterior standard deviation
    """
    time_labels = list(getattr(model, "time_labels", []))
    if not time_labels:
        raise ValueError(
            "model does not have time_labels: this helper requires a dynamic KeyATM "
            "(fit with timestamps= and num_states=)"
        )
    theta_draws = getattr(model, "theta_draws", None)
    if theta_draws is None:
        raise ValueError(
            "model.theta_draws is None: refit with keep_theta_draws=True "
            "(the default) to enable per-period posterior credible intervals"
        )

    draws = np.asarray(theta_draws, dtype=np.float64)  # (S, D, K)
    if draws.ndim != 3:
        raise ValueError(
            f"theta_draws must be 3-D (num_draws, num_docs, num_topics); got shape {draws.shape}"
        )
    s, d, k = draws.shape

    timestamps = np.asarray(timestamps)
    if timestamps.shape[0] != d:
        raise ValueError(
            f"timestamps has length {timestamps.shape[0]} but theta_draws has {d} documents; "
            "timestamps must be one value per document"
        )

    # Build a map from each distinct timestamp string to its index in time_labels.
    # time_labels is Rust-sorted (the authoritative order); we must not rely on
    # Python's str-sort matching it, so we build the index explicitly.
    label_to_idx = {lbl: i for i, lbl in enumerate(time_labels)}
    t_count = len(time_labels)

    # Convert each timestamp to its string form and look up the period index.
    ts_str = np.asarray([str(v) for v in timestamps])
    period_idx = np.array([label_to_idx[s] for s in ts_str], dtype=np.intp)

    # per_draw_period[s, t, k] = mean theta over documents in period t, draw s.
    per_draw = np.zeros((s, t_count, k), dtype=np.float64)
    for t_idx in range(t_count):
        mask = period_idx == t_idx
        if not mask.any():
            continue
        # draws[:, mask, :] has shape (S, n_period, K); mean over documents.
        per_draw[:, t_idx, :] = draws[:, mask, :].mean(axis=1)

    if normalize:
        row_sums = per_draw.sum(axis=2, keepdims=True)
        # Avoid division by zero for empty periods (row_sum == 0 stays 0).
        nz = row_sums > 0
        per_draw = np.where(nz, per_draw / np.where(nz, row_sums, 1.0), 0.0)

    alpha_lo = (1.0 - ci) / 2.0
    alpha_hi = 1.0 - alpha_lo

    mean = per_draw.mean(axis=0)                          # (T, K)
    ci_low = np.quantile(per_draw, alpha_lo, axis=0)      # (T, K)
    ci_high = np.quantile(per_draw, alpha_hi, axis=0)     # (T, K)
    sd = per_draw.std(axis=0, ddof=1) if s > 1 else np.zeros_like(mean)

    return {
        "labels": time_labels,
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "sd": sd,
    }


def refine_keywords(docs, keywords, *, min_count=2, min_doc_freq=1, verbose=False):
    """Drop keywords too rare to anchor a topic (≈ ``keyATM::refine_keywords``).

    Removes any keyword whose corpus count is below ``min_count`` or whose
    document frequency is below ``min_doc_freq`` (so out-of-vocabulary keywords,
    with count 0, always go). Keyword sets that end up empty are dropped, since
    a keyword topic needs at least one surviving keyword.

    Returns ``(refined, dropped)`` where ``refined`` is the cleaned keyword dict
    and ``dropped`` maps each set name to the list of removed keywords. Set
    ``verbose=True`` to print a short report.
    """
    counts, doc_freq, _ = _corpus_counts(docs)
    refined, dropped = {}, {}
    for name, words in keywords.items():
        keep, drop = [], []
        for w in words:
            if counts.get(w, 0) >= min_count and doc_freq.get(w, 0) >= min_doc_freq:
                keep.append(w)
            else:
                drop.append(w)
        if drop:
            dropped[name] = drop
        if keep:
            refined[name] = keep
        if verbose and drop:
            print(f"  {name}: dropped {drop} (below threshold)")
    if verbose:
        gone = [n for n in keywords if n not in refined]
        if gone:
            print(f"  removed empty keyword sets: {gone}")
    return refined, dropped
