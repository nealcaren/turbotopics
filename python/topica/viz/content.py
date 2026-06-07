"""The content-covariate view -- how a topic is *worded* across groups.

For an STM/SAGE content model, the topic-word distribution is covariate-conditional:
the same topic is phrased differently by, say, party or decade. The design's
guardrail is to surface that per-group distribution rather than silently collapse it
to a reference snapshot. This panel takes one topic and shows, for the union of each
group's top words, ``p(w | topic, group)`` as a words x groups heatmap -- so a word
emphasized in one group and not another reads off a single row.

It is refused for a model with no content covariate (a plain STM fit on prevalence
only, or any non-content model).
"""

from __future__ import annotations

import numpy as np

from .base import SEQ_CMAP, Panel
from .capability import capabilities


def _per_group_topic_word(model, cap):
    """The (K, G, V) per-group topic-word array and the group names, or a clear
    refusal for a model that carries no content covariate."""
    if not cap.content_covariate:
        raise ValueError(
            f"{cap.name} has no content covariate; the per-group wording view needs "
            "an STM or SAGE content model (fit with content=)."
        )
    # STM exposes topic_word_by_group (and raises if fit without content); SAGE's
    # own topic_word is already the 3-D per-group array.
    if hasattr(model, "topic_word_by_group"):
        try:
            arr = np.asarray(model.topic_word_by_group, dtype=np.float64)
        except RuntimeError as exc:
            raise ValueError(
                f"{cap.name} was fit without a content covariate; refit with content= "
                "to use the per-group wording view."
            ) from exc
    else:
        arr = np.asarray(model.topic_word, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"{cap.name} did not expose a per-group topic-word array; refit with content=."
        )
    return arr, list(model.groups)


class ContentCovariate(Panel):
    """One topic's wording across content-covariate groups, p(w | topic, group)."""

    title = "Topic wording by group"

    def __init__(self, model, *, topic, n=10):
        from ..analysis import topic_labels

        self.cap = capabilities(model)
        arr, self.groups = _per_group_topic_word(model, self.cap)
        self.topic = int(topic)
        if not 0 <= self.topic < arr.shape[0]:
            raise ValueError(f"topic {self.topic} out of range (num_topics={arr.shape[0]})")
        self.n = int(n)
        phi = arr[self.topic]                       # (G, V)
        vocab = list(model.vocabulary)

        # Union of each group's top-n words, ordered by mean prob across groups so
        # the most prominent words sit at the top and per-group emphasis reads down.
        seen, ids = set(), []
        for g in range(phi.shape[0]):
            for i in np.argsort(phi[g])[::-1][: self.n]:
                if int(i) not in seen:
                    seen.add(int(i))
                    ids.append(int(i))
        mean = phi.mean(axis=0)
        ids.sort(key=lambda i: -mean[i])
        self._ids = ids
        self._words = [vocab[i] for i in ids]
        self._mat = phi[:, ids].T                   # (W, G)
        labels = topic_labels(model)
        self.label = labels[self.topic] if self.topic < len(labels) else f"topic_{self.topic}"

    def to_frame(self):
        """Long form: group, word, prob, and whether the word is in that group's
        own top-n."""
        import pandas as pd

        # which (word, group) pairs are in that group's top-n, for the flag
        topn_by_group = {}
        for gi in range(self._mat.shape[1]):
            order = np.argsort(self._mat[:, gi])[::-1][: self.n]
            topn_by_group[gi] = set(int(i) for i in order)
        rows = []
        for wi, word in enumerate(self._words):
            for gi, group in enumerate(self.groups):
                rows.append({
                    "topic": self.topic, "group": group, "word": word,
                    "prob": float(self._mat[wi, gi]),
                    "in_group_top": wi in topn_by_group[gi],
                })
        return pd.DataFrame(rows)

    def matrix(self):
        """The words x groups probability matrix as a wide DataFrame."""
        import pandas as pd

        return pd.DataFrame(self._mat, index=self._words, columns=[str(g) for g in self.groups])

    def _figsize(self):
        w, g = self._mat.shape
        return (max(4.5, 1.1 * g + 2.5), max(2.5, 0.32 * w + 1.2))

    def _draw(self, fig):
        w, g = self._mat.shape
        ax = fig.subplots()
        im = ax.imshow(self._mat, cmap=SEQ_CMAP, aspect="auto", vmin=0.0)
        ax.set_xticks(range(g))
        ax.set_xticklabels([str(x) for x in self.groups], fontsize=8, rotation=30, ha="right")
        ax.set_yticks(range(w))
        ax.set_yticklabels(self._words, fontsize=7)
        ax.set_title(f"Topic {self.topic}: {self.label} — wording by group", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(word | topic, group)")

    def contrast(self, model, group_a, group_b, *, n=10):
        """Convenience wrapper over the model's ``word_contrast`` for this topic:
        words that most distinguish how the topic is worded in ``group_a`` vs
        ``group_b``."""
        return model.word_contrast(self.topic, group_a, group_b, n=n)
