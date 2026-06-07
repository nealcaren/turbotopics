"""The document inspector -- read one document the way the model read it.

Three views of a single document, for a count / soft-θ model:

- its **θ mixture** (which topics the document loads on), as a bar;
- its **words shaded by topic attribution** -- each token colored by the topic it
  is most attributed to, ``argmax_t p(t | w, d)`` with
  ``p(t | w, d) ∝ θ_d[t] · φ[t, w]`` (no per-token assignments needed, so it works
  for any model that exposes θ and φ);
- its **neighbors** -- the documents most associated with this document's dominant
  topic (``find_thoughts``).

It is refused for hard/degenerate-θ cluster models, where a per-token mixed-
membership attribution is meaningless.
"""

from __future__ import annotations

import numpy as np

from .base import Panel
from .capability import capabilities
from .terms import _topic_word

# Okabe-Ito (colorblind-safe) for the document's leading topics; everything else
# is grayed so the few topics that matter stand out.
_OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#D55E00",
              "#0072B2", "#CC79A7", "#F0E442", "#000000"]
_OTHER = "#9AA0A6"   # a topic outside the leading set
_OOV = "#D7DBDF"     # a token not in the model vocabulary


class DocumentInspector(Panel):
    """One document: its θ mixture, its words shaded by topic, its neighbors."""

    title = "Document inspector"

    def __init__(self, model, texts, *, doc, top_topics=6, n_neighbors=3,
                 max_tokens=400):
        from ..analysis import topic_labels
        from ..validation import find_thoughts

        self.cap = capabilities(model)
        if not self.cap.soft_theta:
            raise ValueError(
                f"{self.cap.name} has degenerate (hard cluster) θ; a per-token "
                "topic attribution is not meaningful. Use the document map instead."
            )
        self.doc = int(doc)
        theta = np.asarray(model.doc_topic, dtype=np.float64)
        if not 0 <= self.doc < theta.shape[0]:
            raise ValueError(f"doc {self.doc} out of range (num_docs={theta.shape[0]})")
        self._theta = theta[self.doc]
        self._labels = topic_labels(model)

        # The document's leading topics get a color; the rest share gray.
        order = np.argsort(self._theta)[::-1]
        self._top = [int(t) for t in order[:top_topics] if self._theta[t] > 0]
        self._color = {t: _OKABE_ITO[i % len(_OKABE_ITO)] for i, t in enumerate(self._top)}
        self.dominant = int(order[0])

        # Per-token attribution: argmax_t theta_d[t] * phi[t, w].
        phi = _topic_word(model).astype(np.float64)
        vocab = {w: i for i, w in enumerate(model.vocabulary)}
        raw = texts[self.doc]
        tokens = raw.split() if isinstance(raw, str) else list(raw)
        self._truncated = len(tokens) > max_tokens
        tokens = tokens[:max_tokens]
        self._tokens = []
        for tok in tokens:
            wi = vocab.get(tok, vocab.get(tok.lower()))
            if wi is None:
                self._tokens.append((tok, -1, np.nan))
                continue
            post = self._theta * phi[:, wi]
            t = int(np.argmax(post))
            p = float(post[t] / post.sum()) if post.sum() > 0 else np.nan
            self._tokens.append((tok, t, p))

        self._neighbors = find_thoughts(model, texts, topic=self.dominant, n=n_neighbors)

    # --- data ---------------------------------------------------------------
    def to_frame(self):
        """Per-token attribution: position, word, in_vocab, dominant_topic,
        p(topic | word, doc)."""
        import pandas as pd

        rows = [{"pos": i, "word": w, "in_vocab": t >= 0,
                 "dominant_topic": (t if t >= 0 else None), "p_topic": p}
                for i, (w, t, p) in enumerate(self._tokens)]
        return pd.DataFrame(rows)

    @property
    def theta(self):
        """The document's topic proportions (a length-K array)."""
        return self._theta

    @property
    def neighbors(self):
        """The dominant topic's representative documents: ``[(idx, prop, text)]``."""
        return self._neighbors

    # --- figure -------------------------------------------------------------
    def _figsize(self):
        n_lines = max(1, len(self._tokens) // 14 + 1)
        return (8.0, 2.2 + 0.16 * n_lines + 0.18 * len(self._neighbors))

    def _label(self, t):
        return self._labels[t] if t < len(self._labels) else f"topic_{t}"

    def _color_for(self, t):
        if t < 0:
            return _OOV
        return self._color.get(t, _OTHER)

    def _draw(self, fig):
        gs = fig.add_gridspec(3, 1, height_ratios=[len(self._top) * 0.5 + 1.0, 3.2, 1.6])
        self._draw_theta(fig.add_subplot(gs[0]))
        self._draw_text(fig.add_subplot(gs[1]))
        self._draw_neighbors(fig.add_subplot(gs[2]))
        fig.suptitle(f"{self.title} — document {self.doc}", fontsize=11)

    def _draw_theta(self, ax):
        top = self._top
        y = np.arange(len(top))
        ax.barh(y, [self._theta[t] for t in top],
                color=[self._color[t] for t in top], height=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{t}: {self._label(t)}" for t in top], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("topic proportion (θ)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title("Topic mixture", fontsize=9, loc="left")

    def _draw_text(self, ax):
        ax.set_axis_off()
        ax.set_title("Words shaded by attributed topic", fontsize=9, loc="left")
        # Deterministic monospace flow layout (no renderer dependency): a fixed
        # character grid keeps it stable across backends and subfigures.
        wrap = 92
        x_unit, line_h = 0.985 / wrap, 0.052
        col = row = 0
        for tok, t, _ in self._tokens:
            length = len(tok) + 1
            if col + length > wrap and col > 0:
                col, row = 0, row + 1
            ax.text(col * x_unit, 1.0 - row * line_h, tok, transform=ax.transAxes,
                    fontsize=7.5, family="monospace", va="top", ha="left",
                    color=self._color_for(t))
            col += length
        if self._truncated:
            ax.text(0.0, 1.0 - (row + 1.4) * line_h, "… (truncated)", transform=ax.transAxes,
                    fontsize=7, style="italic", color="0.5", va="top")

    def _draw_neighbors(self, ax):
        ax.set_axis_off()
        ax.set_title(f"Neighbors — most on topic {self.dominant}: {self._label(self.dominant)}",
                     fontsize=9, loc="left")
        line_h = 1.0 / max(len(self._neighbors), 1)
        for i, (idx, prop, text) in enumerate(self._neighbors):
            snippet = "" if text is None else (text if len(str(text)) <= 90 else str(text)[:89] + "…")
            ax.text(0.0, 1.0 - (i + 0.5) * line_h,
                    f"[{idx}] θ={prop:.2f}  {snippet}", transform=ax.transAxes,
                    fontsize=7.5, va="center", ha="left", color="0.2")
