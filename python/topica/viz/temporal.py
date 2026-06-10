"""Topics over time -- small-multiple prevalence trajectories with CI ribbons.

The canonical dynamic-topic figure, and the design's deliberate replacement for the
streamgraph (which destroys single-series readability). One small panel per topic,
each a prevalence-over-time line.

For a dynamic :class:`~topica.KeyATM` (fit with ``timestamps=`` and retaining
``theta_draws``), the CI ribbon comes from the model's own HMM posterior: for each
retained MCMC draw, per-period prevalences are averaged over the documents in that
period, giving a (num_draws, T, K) posterior. This is the same posterior that
keyATM's ``plot_timetrend`` uses, so the bands reflect genuine HMM uncertainty
rather than an approximation.

For all other models (or when ``theta_draws`` are absent), passing ``corpus=`` and
``nsims=`` produces a method-of-composition ribbon via the generic :func:`by_strata`
path: the model's theta posterior is drawn per document, stratified by timestamp,
and the per-stratum means are pooled by Rubin's rules. Both paths respect the same
``ci`` level and land in the same ``ci_low`` / ``ci_high`` columns of
:meth:`TopicsOverTime.to_frame`.
"""

from __future__ import annotations

import numpy as np

from .base import Panel
from .capability import capabilities


def _numeric_axis(labels):
    """Return float x-positions if the timestamps are numeric, else integer ranks
    (with the original labels kept for ticks)."""
    try:
        return np.asarray([float(v) for v in labels], dtype=np.float64)
    except (TypeError, ValueError):
        return np.arange(len(labels), dtype=np.float64)


def _is_dynamic_keyatm(model):
    """Return True when model is a dynamic KeyATM with retained theta draws.

    Detection criteria: the model has a non-empty ``time_labels`` list AND
    ``theta_draws`` is not None. Both are defined only on a KeyATM fit with
    ``timestamps=``; non-dynamic KeyATM has an empty ``time_labels``.
    """
    return (
        bool(getattr(model, "time_labels", [])) and
        getattr(model, "theta_draws", None) is not None
    )


class TopicsOverTime(Panel):
    """Per-topic prevalence trajectories as small multiples.

    For a dynamic :class:`~topica.KeyATM` with retained ``theta_draws``, the CI
    ribbon is derived from the model's own HMM posterior via
    :func:`topica.keyatm.time_prevalence_ci`. For all other models, passing
    ``corpus=`` and ``nsims=`` produces a method-of-composition ribbon via
    :func:`topica.keyatm.by_strata`. See the module docstring for the difference.
    """

    title = "Topics over time"

    def __init__(self, model, timestamps, *, corpus=None, nsims=None, ci=0.95,
                 normalize=True, seed=0, ncols=4):
        from ..analysis import topic_labels, topics_over_time

        self.cap = capabilities(model)
        self._model_ref = model
        self.ncols = int(ncols)
        ot = topics_over_time(model, timestamps, normalize=normalize)
        self._times = ot["labels"]
        self._x = _numeric_axis(self._times)
        self._prev = np.asarray(ot["prevalence"], dtype=np.float64)  # (T, K)
        self._low = self._high = None
        self.has_ci = False

        if _is_dynamic_keyatm(model):
            # Dynamic KeyATM: derive CI ribbon from the model's own HMM posterior
            # theta draws. This is the same posterior keyATM's plot_timetrend uses,
            # aligned with time_labels exactly. No corpus or nsims required.
            from ..keyatm import time_prevalence_ci

            result = time_prevalence_ci(model, timestamps, ci=ci, normalize=normalize)
            self._low = result["ci_low"]
            self._high = result["ci_high"]
            # Center the ribbon on the posterior mean, not the point-estimate
            # time_prevalence, so that mean/ci_low/ci_high are mutually consistent.
            self._prev = result["mean"]
            self.has_ci = True
        elif nsims:
            # Generic method-of-composition path: stratify by timestamp and pool
            # theta draws by Rubin's rules.
            from ..keyatm import by_strata

            strata = {s.stratum: s for s in
                      by_strata(model, timestamps, ci=ci, corpus=corpus, nsims=nsims, seed=seed)}
            lo = np.full_like(self._prev, np.nan)
            hi = np.full_like(self._prev, np.nan)
            for i, t in enumerate(self._times):
                s = strata.get(t)
                if s is not None:
                    lo[i] = np.asarray(s.ci_low)
                    hi[i] = np.asarray(s.ci_high)
            self._low, self._high = lo, hi
            self.has_ci = True

        k = self._prev.shape[1]
        labels = topic_labels(model)
        self._labels = [labels[t] if t < len(labels) else f"topic_{t}" for t in range(k)]

    def to_frame(self):
        import pandas as pd

        rows = []
        for ti, t in enumerate(self._times):
            for k in range(self._prev.shape[1]):
                row = {"time": t, "topic": k, "label": self._labels[k],
                       "prevalence": float(self._prev[ti, k])}
                if self.has_ci:
                    row["ci_low"] = float(self._low[ti, k])
                    row["ci_high"] = float(self._high[ti, k])
                rows.append(row)
        return pd.DataFrame(rows)

    def _grid(self):
        k = self._prev.shape[1]
        ncols = min(self.ncols, k)
        nrows = int(np.ceil(k / ncols))
        return nrows, ncols

    def _figsize(self):
        nrows, ncols = self._grid()
        return (max(6.0, 2.4 * ncols), max(2.2, 1.7 * nrows))

    def _draw(self, fig, *, shared_y=True):
        # Shared y by default: a free y-axis per panel makes a 0.01->0.02 move look
        # as dramatic as a 0.2->0.4 one. Pass shared_y=False to zoom each topic.
        k = self._prev.shape[1]
        nrows, ncols = self._grid()
        axes = fig.subplots(nrows, ncols, sharex=True, sharey=shared_y,
                            squeeze=False)
        ymax = float(self._prev.max()) if not self.has_ci else \
            float(np.nanmax(np.where(np.isnan(self._high), self._prev, self._high)))
        for t in range(k):
            ax = axes[t // ncols][t % ncols]
            ax.plot(self._x, self._prev[:, t], "-o", color="#4C72B0", ms=3, lw=1.5)
            if self.has_ci:
                lo, hi = self._low[:, t], self._high[:, t]
                ok = ~np.isnan(lo)
                ax.fill_between(self._x[ok], lo[ok], hi[ok], color="#4C72B0", alpha=0.2)
            ax.set_title(f"{t}: {self._labels[t]}", fontsize=8)
            ax.tick_params(labelsize=6)
            if not shared_y:
                ax.set_ylim(0, None)
            else:
                ax.set_ylim(0, ymax * 1.05)
        for j in range(k, nrows * ncols):  # blank the unused cells
            axes[j // ncols][j % ncols].axis("off")
        # Restore integer-rank ticks to their string labels when non-numeric.
        if not np.array_equal(self._x, _numeric_axis(self._times)) or \
                any(not isinstance(v, (int, float, np.integer, np.floating)) for v in self._times):
            for ax in axes[-1]:
                ax.set_xticks(self._x)
                ax.set_xticklabels([str(v) for v in self._times], rotation=45, fontsize=6)
        if self.has_ci:
            # Label the CI source so the reader knows which posterior was used.
            ci_label = " with HMM posterior CIs" if _is_dynamic_keyatm(self._model_ref) \
                else " with composition CIs"
        else:
            ci_label = ""
        fig.suptitle(f"{self.title}{ci_label}", fontsize=10)
