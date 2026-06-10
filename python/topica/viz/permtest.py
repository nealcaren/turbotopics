"""Panel: observed-vs-null plot for :func:`topica.permutation_test`.

Mirrors R ``stm``'s ``plot.STMpermute``: for each topic a density/histogram of
the permutation null is drawn alongside a vertical line at the observed effect,
so the reader can immediately see whether the observed value sits in the tail or
the bulk of the null distribution.
"""

from __future__ import annotations

from .base import Panel


class PermutationTestPlot(Panel):
    """Observed effect vs permutation null distribution, per topic.

    Produced by :func:`topica.viz.permutation_test_plot`. Each subplot shows
    the null histogram for one topic, with the observed effect as a vertical
    line and the two-sided p-value in the title.
    """

    title = "Permutation test: observed vs null"

    def __init__(self, results, *, covariate_name=None):
        """
        Parameters
        ----------
        results : list[PermutationResult]
            The output of :func:`topica.permutation_test`.
        covariate_name : str, optional
            Label for the effect axis (e.g. the covariate name).
        """
        self._results = results
        self._covariate_name = covariate_name or "covariate effect"

    def to_frame(self):
        """Return a DataFrame with the per-topic observed effects and p-values."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "to_frame() needs pandas; install it with pip install pandas"
            ) from exc
        rows = [r.as_dict() for r in self._results]
        return pd.DataFrame(rows)

    def _figsize(self):
        k = len(self._results)
        ncols = min(k, 3)
        nrows = (k + ncols - 1) // ncols
        return (4.5 * ncols, 3.5 * nrows)

    def _draw(self, fig, **kwargs):
        import numpy as np

        results = self._results
        k = len(results)
        if k == 0:
            return
        ncols = min(k, 3)
        nrows = (k + ncols - 1) // ncols
        axes = fig.subplots(nrows, ncols, squeeze=False)

        color_null = "#4C72B0"
        color_obs = "#C44E52"

        for idx, r in enumerate(results):
            row, col = divmod(idx, ncols)
            ax = axes[row][col]
            null = r.null
            obs = r.observed

            if len(null) > 0:
                ax.hist(null, bins=max(10, len(null) // 5), color=color_null,
                        alpha=0.6, edgecolor="white", linewidth=0.5)
            ax.axvline(obs, color=color_obs, lw=2.0, label=f"observed ({obs:.3f})")
            ax.axvline(0.0, color="0.5", lw=1.0, ls="--")
            pval_str = f"p = {r.pvalue:.3f}" if not np.isnan(r.pvalue) else "p = NA"
            ax.set_title(
                f"topic {r.topic}: {r.topic_name}\n{pval_str}",
                fontsize=8,
            )
            ax.set_xlabel(self._covariate_name, fontsize=8)
            ax.set_ylabel("count", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="upper left")

        # Hide unused subplots.
        for idx in range(k, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row][col].set_visible(False)

        fig.suptitle(self.title, fontsize=9, y=1.02)
