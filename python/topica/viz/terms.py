"""Panel 3: the honest pyLDAvis replacement.

A **term barchart** (top words with named weighting modes, gated by the model's
capability descriptor) and a **seriated topic-similarity heatmap** (K x K, ordered
by hierarchical clustering) -- all pairwise relationships at full fidelity, with no
spurious 2-D metric plane. The interactive build links the two: click a topic in
the heatmap and the barchart follows.
"""

from __future__ import annotations

import numpy as np

from .base import Panel, _require
from .capability import capabilities

_DARK = "#4C72B0"
_LIGHT = "#C7D3E8"


def _topic_word(model):
    from ..coherence import _as_topic_word
    return _as_topic_word(model)


def _scored_words(model, topic, mode, n, cap):
    """Return ``[(word, weight), ...]`` for one topic under a weighting `mode`."""
    from ..validation import frex, relevance, label_topics

    if mode not in cap.word_modes:
        raise ValueError(
            f"mode={mode!r} is not valid for {cap.name} (its topic_word is "
            f"{cap.word_weight_label}); available modes: {cap.word_modes}"
        )
    phi = _topic_word(model)
    vocab = list(model.vocabulary)
    if mode == "prob":
        idx = np.argsort(phi[topic])[::-1][:n]
        return [(vocab[i], float(phi[topic, i])) for i in idx]
    if mode == "frex":
        return frex(model, n=n)[topic]
    if mode == "relevance":
        return relevance(model, topic=topic, n=n)
    if mode == "lift":
        marg = np.clip(phi.mean(axis=0), 1e-12, None)
        lift = phi[topic] / marg
        idx = np.argsort(lift)[::-1][:n]
        return [(vocab[i], float(lift[i])) for i in idx]
    if mode == "score":
        return label_topics(model, n=n)[topic]["score"]
    raise ValueError(f"unknown mode {mode!r}")


class TermBarchart(Panel):
    """Top words of one topic, weighted by a named mode, with the corpus-frequency
    overlay. Optional inclusion-probability error bars (a bootstrap, so off by
    default)."""

    title = "Top words"

    def __init__(self, model, *, topic, mode="prob", n=10, texts=None,
                 error_bars=False, n_boot=100, seed=0):
        self.cap = capabilities(model)
        self.topic = int(topic)
        self.mode = mode
        self.n = n
        self._words = _scored_words(model, self.topic, mode, n, self.cap)
        # Corpus-frequency overlay: the prevalence-weighted marginal
        # p(w) = sum_t p(t) p(w|t). Only meaningful in "prob" mode -- frex / lift /
        # relevance live on a different (non-probability) scale, so an overlay of
        # p(w) ~ 0.02 against frex in [0, 1] would be a meaningless sliver.
        self._overlay = None
        if mode == "prob":
            phi = _topic_word(model)
            vocab = {w: i for i, w in enumerate(model.vocabulary)}
            theta = np.asarray(model.doc_topic, dtype=np.float64)
            pw = theta.mean(axis=0) @ phi
            self._overlay = [float(pw[vocab[w]]) if w in vocab else 0.0 for w, _ in self._words]
        self._inclusion = None
        if error_bars:
            from ..effects import standard_errors

            tw = standard_errors(model, texts, of="top_words", method="bootstrap",
                                 n_boot=n_boot, topn=n, seed=seed)
            inc = {w: (p, lo, hi) for (w, p, lo, hi) in tw[self.topic].words}
            self._inclusion = [inc.get(w) for w, _ in self._words]

    def to_frame(self):
        import pandas as pd

        rows = []
        for i, (w, val) in enumerate(self._words):
            row = {"topic": self.topic, "rank": i, "word": w, "weight": val}
            if self._overlay is not None:
                row["corpus_weight"] = self._overlay[i]
            if self._inclusion is not None and self._inclusion[i] is not None:
                p, lo, hi = self._inclusion[i]
                row.update(inclusion_prob=p, inclusion_low=lo, inclusion_high=hi)
            rows.append(row)
        return pd.DataFrame(rows)

    def _figsize(self):
        return (5.5, max(2.2, 0.34 * len(self._words) + 1.0))

    def _draw(self, fig):
        words = [w for w, _ in self._words][::-1]
        vals = [v for _, v in self._words][::-1]
        y = np.arange(len(words))
        ax = fig.subplots()
        ax.barh(y, vals, color=_DARK, height=0.7, label="in topic", zorder=2)
        if self._overlay is not None:
            ax.barh(y, self._overlay[::-1], color=_LIGHT, height=0.7,
                    label="corpus overall", zorder=1)
            ax.legend(fontsize=7, loc="lower right")
        ax.set_yticks(y)
        ax.set_yticklabels(words, fontsize=8)
        label = self.cap.word_weight_label if self.mode == "prob" else self.mode
        ax.set_xlabel(label)
        ax.set_title(f"Topic {self.topic} — {self.mode}", fontsize=10)


def _topic_distance(model, cap):
    """K x K topic distance: sqrt-Jensen-Shannon for probability topic_word,
    cosine distance for c-TF-IDF (which is not a distribution)."""
    phi = _topic_word(model).astype(np.float64)
    k = phi.shape[0]
    if cap.prob_simplex_words:
        p = phi / np.clip(phi.sum(axis=1, keepdims=True), 1e-12, None)
        # Vectorized over the inner pair: JS(p,q) = H(m) - (H(p)+H(q))/2, one numpy
        # pass per row (O(K) Python iterations, not O(K^2) with a per-element KL).
        ent = lambda a: -(a * np.log(np.clip(a, 1e-12, None))).sum(axis=-1)
        h = ent(p)  # (K,) per-topic entropy
        d = np.zeros((k, k))
        for i in range(k):
            m = 0.5 * (p[i] + p)            # (K, V)
            js = ent(m) - 0.5 * (h[i] + h)  # (K,)
            d[i] = np.sqrt(np.clip(js, 0.0, None))
        d = 0.5 * (d + d.T)                 # symmetrize away float drift
        np.fill_diagonal(d, 0.0)
        return d, "sqrt Jensen-Shannon"
    norm = phi / np.clip(np.linalg.norm(phi, axis=1, keepdims=True), 1e-12, None)
    d = 1.0 - norm @ norm.T
    np.fill_diagonal(d, 0.0)
    return np.clip(d, 0.0, None), "cosine distance"


def _seriate(dist):
    """Hierarchical-clustering leaf order for a distance matrix (identity if scipy
    is unavailable or K is tiny)."""
    k = dist.shape[0]
    if k < 3:
        return list(range(k)), None
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
        from scipy.spatial.distance import squareform
    except ImportError:
        return list(range(k)), None
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    z = optimal_leaf_ordering(z, condensed)
    return list(leaves_list(z)), z


class TopicSimilarity(Panel):
    """A seriated K x K topic-similarity heatmap -- every pairwise relationship at
    full fidelity, ordered by hierarchical clustering, no spurious metric plane."""

    title = "Topic similarity"

    def __init__(self, model):
        from ..analysis import topic_labels

        self.cap = capabilities(model)
        self._dist, self.metric = _topic_distance(model, self.cap)
        self._order, self._linkage = _seriate(self._dist)
        self._labels = topic_labels(model)

    def to_frame(self):
        import pandas as pd

        order = self._order
        sim = 1.0 - self._dist
        labels = [f'{t}: {self._labels[t] if t < len(self._labels) else t}' for t in order]
        return pd.DataFrame(sim[np.ix_(order, order)], index=labels, columns=labels)

    def _figsize(self):
        return (max(4.0, 0.4 * len(self._order) + 1.5),) * 2

    def _draw(self, fig):
        order = self._order
        k = len(order)
        sim = (1.0 - self._dist)[np.ix_(order, order)]
        ax = fig.subplots()
        im = ax.imshow(sim, cmap="viridis", vmin=sim.min(), vmax=1.0)
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_xticklabels([str(t) for t in order], fontsize=7, rotation=90)
        ax.set_yticklabels([str(t) for t in order], fontsize=7)
        ax.set_title(f"{self.title} (1 − {self.metric}, seriated)", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="similarity")


def term_topic_browser(model, *, n=10, mode="prob"):
    """An interactive (Altair) linked term browser + topic-similarity heatmap:
    click a topic in the heatmap and the term barchart follows. Returns an Altair
    chart; ``.save("page.html")`` writes a self-contained page. Needs
    ``topica[viz-interactive]``."""
    alt = _require("altair", "viz-interactive")
    import pandas as pd

    cap = capabilities(model)
    sim_panel = TopicSimilarity(model)
    order = sim_panel._order
    sim = 1.0 - sim_panel._dist

    sim_rows = [
        {"i": int(a), "j": int(b), "similarity": float(sim[a, b]),
         "i_ord": order.index(a), "j_ord": order.index(b)}
        for a in order for b in order
    ]
    sim_df = pd.DataFrame(sim_rows)

    term_rows = []
    for t in range(_topic_word(model).shape[0]):
        for rank, (w, val) in enumerate(_scored_words(model, t, mode, n, cap)):
            term_rows.append({"topic": int(t), "word": w, "weight": float(val), "rank": rank})
    term_df = pd.DataFrame(term_rows)

    sel = alt.selection_point(fields=["i"], value=int(order[0]), on="click", empty=False)
    heat = (
        alt.Chart(sim_df, title="Topic similarity (click a row)")
        .mark_rect()
        .encode(
            x=alt.X("i_ord:O", title="topic", axis=alt.Axis(labelExpr="")),
            y=alt.Y("j_ord:O", title="topic", axis=alt.Axis(labelExpr="")),
            color=alt.Color("similarity:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["i", "j", alt.Tooltip("similarity:Q", format=".2f")],
            opacity=alt.condition(sel, alt.value(1.0), alt.value(0.65)),
        )
        .add_params(sel)
        .properties(width=320, height=320)
    )
    bars = (
        alt.Chart(term_df, title=f"Top words ({mode})")
        .mark_bar(color=_DARK)
        .encode(
            x=alt.X("weight:Q", title=cap.word_weight_label if mode == "prob" else mode),
            y=alt.Y("word:N", sort=alt.EncodingSortField(field="weight", order="descending"),
                    title=None),
            tooltip=["word", alt.Tooltip("weight:Q", format=".4f")],
        )
        .transform_filter(sel)
        .properties(width=260, height=320)
    )
    return alt.hconcat(heat, bars).resolve_scale(y="independent").properties(
        title=f"{cap.name}: topics and their words"
    )
