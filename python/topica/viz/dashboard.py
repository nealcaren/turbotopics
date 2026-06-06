"""``dashboard()`` -- the convenience composite.

It introspects the capability descriptor and the arguments you pass to assemble
the applicable panels: the topic-similarity heatmap and a term barchart always,
the coherence frontier when ``texts`` is given, and the covariate effect plot when
a design (``formula``/``data`` or ``X``) is given. ``.to_png()`` stacks the static
panels; ``.to_html()`` writes a self-contained report (the interactive linked
term/heatmap browser plus the static panels embedded as images); ``.to_frame()``
returns the numbers behind every panel.
"""

from __future__ import annotations

import base64
import io

from .base import _require
from .capability import capabilities
from .effect import EffectPlot
from .quality import CoherenceFrontier
from .terms import TermBarchart, TopicSimilarity, term_topic_browser


class Dashboard:
    def __init__(self, model, texts=None, *, corpus=None, formula=None, data=None,
                 X=None, topic=0, mode="prob"):
        self.model = model
        self.cap = capabilities(model)
        self.panels = {}
        self.panels["similarity"] = TopicSimilarity(model)
        self.panels["terms"] = TermBarchart(model, topic=topic, mode=mode, texts=texts)
        if texts is not None:
            self.panels["frontier"] = CoherenceFrontier(model, texts)
        if (formula is not None and data is not None) or X is not None:
            try:
                self.panels["effect"] = EffectPlot(
                    model, corpus, formula=formula, data=data, X=X
                )
            except Exception:  # a model/design that can't support the panel: skip it
                pass

    def to_frame(self):
        """A dict of ``{panel_name: DataFrame}``."""
        return {name: p.to_frame() for name, p in self.panels.items()}

    def _png_bytes(self, panel, **kw):
        fig = panel._figure(**kw)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        _require("matplotlib.pyplot", "viz").close(fig)
        return buf.getvalue()

    def to_png(self, path: str | None = None, *, dpi: int = 150):
        """Stack the static panels vertically into one matplotlib figure."""
        plt = _require("matplotlib.pyplot", "viz")
        import matplotlib.image as mpimg

        order = [k for k in ("effect", "frontier", "similarity", "terms") if k in self.panels]
        imgs = [mpimg.imread(io.BytesIO(self._png_bytes(self.panels[k]))) for k in order]
        heights = [im.shape[0] for im in imgs]
        widths = [im.shape[1] for im in imgs]
        w = max(widths)
        fig = plt.figure(figsize=(w / 150, sum(heights) / 150))
        gs = fig.add_gridspec(len(imgs), 1, height_ratios=heights, hspace=0.04)
        for i, im in enumerate(imgs):
            ax = fig.add_subplot(gs[i])
            ax.imshow(im)
            ax.axis("off")
        if path is not None:
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return fig

    def to_html(self, path: str | None = None, *, title=None, mode="prob", n=10):
        """A self-contained HTML report: the interactive term/heatmap browser plus
        the static panels embedded as images."""
        chart = term_topic_browser(self.model, n=n, mode=mode)
        chart_html = chart.to_html()
        blocks = []
        for name in ("effect", "frontier"):
            if name in self.panels:
                b64 = base64.b64encode(self._png_bytes(self.panels[name])).decode("ascii")
                blocks.append(f'<h2>{name}</h2><img src="data:image/png;base64,{b64}">')
        title = title or f"topica report — {self.cap.name}"
        html = (
            f"<!doctype html><html><head><meta charset='utf-8'><title>{title}</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;}"
            "img{max-width:100%;}</style></head><body>"
            f"<h1>{title}</h1>{chart_html}{''.join(blocks)}</body></html>"
        )
        if path is not None:
            with open(path, "w") as fh:
                fh.write(html)
        return html

    def __repr__(self):
        return f"Dashboard({self.cap.name}, panels={list(self.panels)})"


def dashboard(model, texts=None, **kwargs) -> Dashboard:
    """Assemble the applicable panels for a model into one report."""
    return Dashboard(model, texts, **kwargs)
