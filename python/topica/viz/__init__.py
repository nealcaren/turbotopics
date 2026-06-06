"""topica.viz -- an honest, manuscript-first visualization toolkit.

Each view is a :class:`~topica.viz.base.Panel`: it reads a model's analysis
surface plus a per-model capability descriptor and exposes three renderers --
``.to_frame()`` (the numbers, always), ``.to_png()`` (matplotlib, for papers), and
``.to_html()`` (Altair, for the few views interaction genuinely helps). The
statistics and labels switch on the descriptor, so a c-TF-IDF ``topic_word`` is
never mislabeled a probability and a confidence interval is never drawn where
there is no posterior to draw it from.

    import topica.viz as viz
    viz.coherence_frontier(model, texts).to_png("quality.png")
    viz.effect_plot(model, corpus, formula="~ year", data=meta).to_png("effects.pdf")
    viz.term_barchart(model, topic=3, mode="frex").to_frame()
"""

from __future__ import annotations

from .base import Panel
from .capability import Capabilities, capabilities
from .quality import CoherenceFrontier, SearchK
from .effect import EffectPlot
from .terms import TermBarchart, TopicSimilarity, term_topic_browser
from .dashboard import Dashboard, dashboard


def coherence_frontier(model, texts=None, *, n=10, coherence_type=None) -> CoherenceFrontier:
    """Per-topic coherence vs exclusivity (size = prevalence)."""
    return CoherenceFrontier(model, texts, n=n, coherence_type=coherence_type)


def search_k(rows) -> SearchK:
    """Coherence / exclusivity / perplexity across K. Pass ``topica.search_k(...)`` rows."""
    return SearchK(rows)


def effect_plot(model, corpus=None, **kwargs) -> EffectPlot:
    """One covariate's effect on each topic's prevalence, with honest CIs."""
    return EffectPlot(model, corpus, **kwargs)


def term_barchart(model, *, topic, mode="prob", n=10, texts=None, error_bars=False, **kwargs) -> TermBarchart:
    """Top words of one topic as a weighted bar chart (mode = prob/frex/lift/...)."""
    return TermBarchart(model, topic=topic, mode=mode, n=n, texts=texts,
                        error_bars=error_bars, **kwargs)


def topic_similarity(model, **kwargs) -> TopicSimilarity:
    """Seriated K x K topic-similarity heatmap (the honest pyLDAvis overview)."""
    return TopicSimilarity(model, **kwargs)


__all__ = [
    "Panel",
    "Capabilities",
    "capabilities",
    "CoherenceFrontier",
    "SearchK",
    "EffectPlot",
    "TermBarchart",
    "TopicSimilarity",
    "coherence_frontier",
    "search_k",
    "effect_plot",
    "term_barchart",
    "topic_similarity",
    "term_topic_browser",
    "Dashboard",
    "dashboard",
]
