"""Panel: predicted topic prevalence at covariate values, with simulation CIs.

Mirrors the R ``stm`` ``plot.estimateEffect`` output:

- **forest** (``at`` / ``contrast`` mode) — one row per topic, point + interval.
- **curve** (``continuous`` mode) — one line + shaded band per topic.

Uncertainty is labeled for what it is: a logistic-normal posterior gives genuine
Bayesian CIs; the Dirichlet-conditional approximation is stated as such; and for
models with no theta posterior only point estimates are drawn.
"""

from __future__ import annotations

from .base import Panel
from .capability import capabilities

_POSTERIOR_LABEL = {
    "logistic_normal": "logistic-normal posterior (method of composition)",
    "dirichlet": "Dirichlet-conditional, within-document (method of composition)",
    "none": "no posterior — point estimate only",
}


class PrevalencePlot(Panel):
    """Predicted topic prevalence at covariate values, with simulation-based CIs.

    Produced by :func:`topica.viz.predicted_prevalence_plot`. Wraps the result
    of :func:`topica.predicted_prevalence` and renders it as either a forest
    plot (``at`` / ``contrast`` mode) or a curve-and-band plot (``continuous``
    mode).
    """

    title = "Predicted topic prevalence"

    def __init__(self, model, *, results, ci_level=0.95):
        """
        Parameters
        ----------
        model : fitted topica model
            Used only to read the capability descriptor for labeling.
        results : list[PredictedPrevalence]
            The output of :func:`topica.predicted_prevalence`.
        ci_level : float
            The CI level used when computing ``results`` (for axis labeling).
        """
        self.cap = capabilities(model)
        self.ci_level = ci_level
        self._results = results
        self._mode = results[0].mode if results else "at"
        self.uncertainty = self.cap.theta_posterior

    def to_frame(self):
        """Return a tidy DataFrame with all topics concatenated."""
        import pandas as pd

        frames = [r.to_frame() for r in self._results]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _figsize(self):
        k = len(self._results)
        if self._mode in ("at", "contrast"):
            return (6.5, max(2.5, 0.5 * k + 1.5))
        # continuous: one subplot per topic
        ncols = min(k, 3)
        nrows = (k + ncols - 1) // ncols
        return (4.5 * ncols, 3.5 * nrows)

    def _draw(self, fig, *, sort=True):
        import numpy as np

        if self._mode in ("at", "contrast"):
            self._draw_forest(fig, sort=sort)
        else:
            self._draw_curves(fig)

    def _draw_forest(self, fig, *, sort=True):
        """Forest / point-interval plot: one row per topic."""
        import numpy as np

        df = self.to_frame()
        if sort and self._mode == "contrast":
            df = df.sort_values("estimate")
        elif sort and "estimate" in df.columns:
            df = df.sort_values("estimate")

        k = len(self._results)
        ax = fig.subplots()
        color = "#4C72B0"
        y = np.arange(k)

        for i, r in enumerate(
            self._results if not sort else sorted(
                self._results, key=lambda r: float(r.estimate[0])
            )
        ):
            est = float(r.estimate[0])
            lo = float(r.ci_low[0])
            hi = float(r.ci_high[0])
            has_ci = self.uncertainty != "none"
            if has_ci:
                ax.plot([lo, hi], [i, i], color=color, lw=2.2, alpha=0.85,
                        solid_capstyle="round", zorder=2)
                ax.plot(est, i, "o", color=color, ms=6, zorder=3)
            else:
                ax.plot(est, i, "o", mfc="white", mec=color, ms=6, zorder=3)

        ax.axvline(0.0, color="0.4", lw=1.0, zorder=1)
        ax.set_yticks(y)
        labels_sorted = [
            r.topic_name for r in (
                sorted(self._results, key=lambda r: float(r.estimate[0]))
                if sort else self._results
            )
        ]
        ax.set_yticklabels(
            [f"{r.topic}: {r.topic_name}" for r in (
                sorted(self._results, key=lambda r: float(r.estimate[0]))
                if sort else self._results
            )],
            fontsize=8,
        )
        pct = int(round(self.ci_level * 100))
        unc = _POSTERIOR_LABEL.get(self.uncertainty, self.uncertainty)
        mode_label = "difference" if self._mode == "contrast" else "predicted prevalence"
        ax.set_xlabel(mode_label)
        ax.set_title(
            f"{self.title}\n{pct}% CI — {unc}" if self.uncertainty != "none"
            else f"{self.title}\n(no posterior — point estimates only)",
            fontsize=9,
        )

    def _draw_curves(self, fig):
        """Curve + shaded band: one subplot per topic."""
        import numpy as np

        k = len(self._results)
        ncols = min(k, 3)
        nrows = (k + ncols - 1) // ncols
        axes = fig.subplots(nrows, ncols, squeeze=False)
        color = "#4C72B0"
        cov = self._results[0].covariate or "covariate"

        for idx, r in enumerate(self._results):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            # Extract x values from grid (list of dicts)
            if r.grid and isinstance(r.grid[0], dict):
                xs = np.array([g.get(cov, i) for i, g in enumerate(r.grid)],
                              dtype=float)
            else:
                xs = np.arange(len(r.estimate))

            ax.fill_between(xs, r.ci_low, r.ci_high, alpha=0.25, color=color)
            ax.plot(xs, r.estimate, color=color, lw=2.0)
            ax.set_title(f"{r.topic}: {r.topic_name}", fontsize=8)
            ax.set_xlabel(cov, fontsize=8)
            ax.set_ylabel("prevalence", fontsize=8)
            ax.tick_params(labelsize=7)

        # Hide unused axes.
        for idx in range(k, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        pct = int(round(self.ci_level * 100))
        unc = _POSTERIOR_LABEL.get(self.uncertainty, self.uncertainty)
        fig.suptitle(
            f"{self.title} — {cov}\n{pct}% CI — {unc}",
            fontsize=9,
        )
