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
        sizes = 40 + 360 * (df["prevalence"] / max(df["prevalence"].max(), 1e-9))
        ax.scatter(df["coherence"], df["exclusivity"], s=sizes,
                   color="#4C72B0", alpha=0.75, edgecolor="white", linewidth=0.6)
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

    def _figsize(self):
        return (6.5, 4.5)

    def _draw(self, fig):
        df = self.to_frame().sort_values("k")
        ax = fig.subplots()
        ax.plot(df["k"], df["coherence"], "-o", color="#4C72B0", label="coherence")
        ax.set_xlabel("K (number of topics)")
        ax.set_ylabel("coherence", color="#4C72B0")
        ax.tick_params(axis="y", labelcolor="#4C72B0")
        ax2 = ax.twinx()
        ax2.plot(df["k"], df["exclusivity"], "-s", color="#C44E52", label="exclusivity")
        ax2.set_ylabel("exclusivity", color="#C44E52")
        ax2.tick_params(axis="y", labelcolor="#C44E52")
        if "perplexity" in df.columns and df["perplexity"].notna().any():
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("axes", 1.12))
            ax3.plot(df["k"], df["perplexity"], "-^", color="#55A868", label="perplexity")
            ax3.set_ylabel("held-out perplexity", color="#55A868")
            ax3.tick_params(axis="y", labelcolor="#55A868")
        metric = self._rows[0].get("coherence_metric", "u_mass")
        ax.set_title(f"{self.title} (coherence = {metric}; higher coherence/exclusivity, lower perplexity)",
                     fontsize=10)
