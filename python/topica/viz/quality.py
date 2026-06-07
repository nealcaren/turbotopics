"""Panel 1: "why this K" -- the coherence/exclusivity frontier and search-K curves.

Both reuse the existing diagnostics (``quality_frontier``, ``search_k``); the panel
adds the ``.to_frame()`` data export and a clean publication figure.
"""

from __future__ import annotations

from .base import Panel
from .capability import capabilities


class CoherenceFrontier(Panel):
    """Per-topic coherence vs exclusivity -- the figure authors use to defend
    dropping weak topics. Size encodes prevalence (by area)."""

    title = "Topic quality"

    def __init__(self, model, texts=None, *, n=10, coherence_type=None):
        from ..analysis import topic_labels
        from ..validation import quality_frontier

        self.cap = capabilities(model)
        ref = texts
        if ref is not None and len(ref) and isinstance(ref[0], str):
            ref = [t.split() for t in ref]
        self.metric = coherence_type or ("c_v" if ref is not None else "u_mass")
        self._q = quality_frontier(
            model, n=n, texts=ref,
            coherence_type=self.metric if ref is not None else "u_mass",
        )
        self._labels = topic_labels(model)

    def to_frame(self):
        import pandas as pd

        q = self._q
        topics = [int(t) for t in q["topic"]]
        return pd.DataFrame({
            "topic": topics,
            "label": [self._labels[t] if t < len(self._labels) else f"topic_{t}" for t in topics],
            "coherence": q["coherence"],
            "exclusivity": q["exclusivity"],
            "prevalence": q["prevalence"],
        })

    def _figsize(self):
        return (6.0, 5.0)

    def _draw(self, fig):
        import numpy as np

        df = self.to_frame()
        ax = fig.subplots()
        pmax = max(df["prevalence"].max(), 1e-9)
        size_of = lambda frac: 40 + 360 * frac
        sizes = size_of(df["prevalence"] / pmax)
        ax.scatter(df["coherence"], df["exclusivity"], s=sizes,
                   color="#4C72B0", alpha=0.75, edgecolor="white", linewidth=0.6)
        # A size legend so "size ∝ prevalence" is readable, not just asserted.
        handles = [ax.scatter([], [], s=size_of(f), color="#4C72B0", alpha=0.6,
                              edgecolor="white", linewidth=0.6, label=f"{f * pmax:.2g}")
                   for f in (0.25, 0.5, 1.0)]
        ax.legend(handles=handles, title="prevalence", fontsize=6, title_fontsize=7,
                  loc="lower right", labelspacing=1.4, borderpad=0.9, frameon=True,
                  handletextpad=1.0)
        for _, r in df.iterrows():
            ax.annotate(str(int(r["topic"])), (r["coherence"], r["exclusivity"]),
                        fontsize=7, ha="center", va="center", color="white")
        ax.set_xlabel(f"Semantic coherence ({self.metric})")
        ax.set_ylabel("Exclusivity")
        ax.ticklabel_format(useOffset=False, style="plain")
        # Exclusivity often saturates near 1; auto-zoom then makes a 0.001 spread
        # look dramatic. Give headroom and flag it when the spread is tiny.
        ex = np.asarray(df["exclusivity"], dtype=float)
        lo, hi = float(ex.min()), float(ex.max())
        note = ""
        if hi - lo < 0.02 and hi > 0.9:
            pad = max((hi - lo), 1e-3)
            ax.set_ylim(lo - 3 * pad, min(1.0 + pad, 1.001))
            note = f"  (exclusivity all near 1: {lo:.3f}–{hi:.3f})"
        ax.set_title(self.title + " (size ∝ prevalence)" + note, fontsize=10)


class SearchK(Panel):
    """Coherence / exclusivity (and held-out perplexity, when present) across K --
    the canonical "choosing K" curve. Pass the rows from ``topica.search_k``."""

    title = "Choosing K"

    def __init__(self, rows):
        self._rows = list(rows)
        if not self._rows:
            raise ValueError("search_k returned no rows")

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(self._rows)

    def _metrics(self, df):
        """The metric rows to facet: (column, label, marker), perplexity only if present."""
        rows = [("coherence", "coherence (higher better)", "-o"),
                ("exclusivity", "exclusivity (higher better)", "-s")]
        if "perplexity" in df.columns and df["perplexity"].notna().any():
            rows.append(("perplexity", "held-out perplexity (lower better)", "-^"))
        return rows

    def _figsize(self):
        n = len(self._metrics(self.to_frame()))
        return (6.0, 1.6 * n + 0.6)

    def _draw(self, fig):
        # Faceted small multiples sharing the K axis, one metric per panel: a triple
        # twin-y axis can't be read (three arbitrary scales, one gridline set).
        df = self.to_frame().sort_values("k")
        metrics = self._metrics(df)
        axes = fig.subplots(len(metrics), 1, sharex=True, squeeze=False)[:, 0]
        for ax, (col, label, marker) in zip(axes, metrics):
            ax.plot(df["k"], df[col], marker, color="#4C72B0")
            ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("K (number of topics)")
        metric = self._rows[0].get("coherence_metric", "u_mass")
        axes[0].set_title(f"{self.title} (coherence = {metric})", fontsize=10)
