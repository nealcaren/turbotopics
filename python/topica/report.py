"""A model-neutral analysis surface for any fitted topica model.

These helpers read only a fitted model's public attributes — ``topic_word``
(K x V), ``doc_topic`` (D x K), ``topic_names``, ``vocabulary``, ``num_topics``,
and the optional ``labels`` (hard document assignments on the embedding-cluster
models, where ``-1`` marks a noise/outlier document) — so they work uniformly
across LDA, STM, CTM, keyATM, Top2Vec, BERTopic, and the rest. The goal is the
overview a researcher reaches for first: how big each topic is, what it is about,
who its representative documents are, and how its prevalence moves across time or
across groups.

- :func:`topic_info` — one summary row per topic (the headline table).
- :func:`topic_sizes` — hard size and expected mass per topic.
- :func:`topic_labels` / :func:`set_topic_labels` — effective topic labels,
  with a custom override.
- :func:`representative_docs` — each topic's highest-loading documents.
- :func:`topics_over_time` — mean prevalence per distinct timestamp.
- :func:`topics_per_class` — mean prevalence within each group.
"""

from __future__ import annotations

import numpy as np

from . import diagnostics as _diagnostics
from . import effects as _effects


# A custom-label registry keyed by ``id(model)``. PyO3 extension classes may not
# support weakref or attribute assignment, so we do not stash labels on the model
# itself; the caller's process holds the mapping for as long as the model lives.
_LABELS: dict[int, dict[int, str]] = {}


def _doc_topic(model) -> np.ndarray:
    return np.asarray(model.doc_topic, dtype=np.float64)


def _has_labels(model) -> bool:
    """Whether the model carries hard ``labels`` (the clustering models do)."""
    labels = getattr(model, "labels", None)
    return labels is not None and len(labels) > 0


def topic_sizes(model) -> dict:
    """Per-topic size and expected mass for any fitted model.

    The ``size`` is each topic's count of hard document assignments. On a
    clustering model that exposes ``labels`` (Top2Vec / BERTopic) we count those
    assignments directly and report the number of ``-1`` (noise/outlier)
    documents separately under ``"outliers"``; on every other model we take the
    argmax of ``doc_topic`` per document. The ``mass`` is the expected number of
    documents in each topic, ``doc_topic.sum(axis=0)`` — the soft analog of the
    hard count.

    Returns ``{"size": (K,) int array, "mass": (K,) float array,
    "outliers": int}``.
    """
    theta = _doc_topic(model)
    k = theta.shape[1]
    mass = theta.sum(axis=0)
    outliers = 0
    if _has_labels(model):
        labels = np.asarray(list(model.labels), dtype=np.int64)
        outliers = int(np.sum(labels == -1))
        size = np.bincount(labels[labels >= 0], minlength=k)[:k]
    else:
        size = np.bincount(theta.argmax(axis=1), minlength=k)[:k]
    return {"size": size.astype(np.int64), "mass": mass, "outliers": outliers}


def set_topic_labels(model, mapping: dict) -> None:
    """Store custom labels for some or all of a model's topics.

    ``mapping`` is ``{topic_id: label}``; labels merge over (and override)
    ``model.topic_names`` everywhere this module reports a topic. The store is
    keyed by ``id(model)`` rather than set on the model, since the compiled model
    classes may not allow attribute assignment.
    """
    store = _LABELS.setdefault(id(model), {})
    for topic, label in mapping.items():
        store[int(topic)] = str(label)


def topic_labels(model) -> list:
    """The effective per-topic labels: any custom labels set via
    :func:`set_topic_labels` override the model's own ``topic_names``."""
    names = list(getattr(model, "topic_names", []))
    k = int(getattr(model, "num_topics", len(names)))
    if len(names) < k:
        names = names + [f"topic_{t}" for t in range(len(names), k)]
    custom = _LABELS.get(id(model), {})
    for topic, label in custom.items():
        if 0 <= topic < len(names):
            names[topic] = label
    return names


def representative_docs(model, texts, *, topic=None, n=5):
    """The documents that load most heavily on a topic, with their text.

    Wraps :func:`topica.find_thoughts`, returning ``texts`` for the ``n``
    highest-``doc_topic`` documents. With ``topic`` given, returns that topic's
    list; with ``topic=None`` returns ``{topic_id: [texts]}`` for every topic.
    Each list is ordered by descending topic proportion.
    """
    def docs_for(t):
        thoughts = _diagnostics.find_thoughts(model.doc_topic, texts, topic=t, n=n)
        return [text for _, _, text in thoughts]

    if topic is not None:
        return docs_for(topic)
    k = _doc_topic(model).shape[1]
    return {t: docs_for(t) for t in range(k)}


def _top_words(model, t, n):
    """Top-``n`` words for topic ``t`` as a plain list of strings, using the
    model's ``top_words`` method when present and falling back to the raw φ row."""
    method = getattr(model, "top_words", None)
    if callable(method):
        try:
            pairs = method(n, topic=t)
            return [w for w, _ in pairs]
        except Exception:
            pass
    phi = np.asarray(model.topic_word, dtype=np.float64)
    vocab = list(model.vocabulary)
    idx = np.argsort(phi[t])[::-1][:n]
    return [vocab[i] for i in idx]


def topic_info(model, texts=None, *, n=8, labels=None) -> list:
    """One summary row per topic — the headline table for a fitted model.

    Each row is a dict with ``topic`` (id), ``label``, ``size`` (hard
    assignments), ``prevalence`` (mean of the topic's ``doc_topic`` column), and
    ``top_words`` (the top-``n`` words, via ``model.top_words`` when available
    else the raw topic-word row). When ``texts`` is given each row also carries
    ``representative_docs``, its ``n`` highest-loading documents. On a clustering
    model with outliers a final ``topic=-1`` row reports the outlier count and
    carries no words. Rows are sorted by topic id.

    ``labels`` overrides the labels for this table only; otherwise
    :func:`topic_labels` (custom labels over ``topic_names``) is used.
    """
    theta = _doc_topic(model)
    k = theta.shape[1]
    sizes = topic_sizes(model)
    effective = labels if labels is not None else topic_labels(model)
    prevalence = theta.mean(axis=0)

    rows = []
    for t in range(k):
        row = {
            "topic": t,
            "label": effective[t] if t < len(effective) else f"topic_{t}",
            "size": int(sizes["size"][t]),
            "prevalence": float(prevalence[t]),
            "top_words": _top_words(model, t, n),
        }
        if texts is not None:
            row["representative_docs"] = representative_docs(model, texts, topic=t, n=n)
        rows.append(row)

    if sizes["outliers"] > 0:
        outlier_row = {
            "topic": -1,
            "label": "outliers",
            "size": int(sizes["outliers"]),
            "prevalence": 0.0,
            "top_words": [],
        }
        if texts is not None:
            outlier_row["representative_docs"] = []
        rows.append(outlier_row)
    return rows


def topics_over_time(model, timestamps, *, normalize=True) -> dict:
    """Mean topic prevalence at each distinct timestamp value.

    ``timestamps`` is one value per document. For each distinct timestamp we
    average ``doc_topic`` over the documents stamped with it, giving a topic
    prevalence trajectory you can plot directly. With ``normalize=True`` each
    row is rescaled to sum to one (so it reads as a topic share at that time).

    Returns ``{"labels": [sorted distinct timestamps], "prevalence": (T, K)
    array}``.
    """
    theta = _doc_topic(model)
    stamps = np.asarray(list(timestamps))
    if stamps.shape[0] != theta.shape[0]:
        raise ValueError("timestamps must have one value per document")
    levels = sorted(np.unique(stamps), key=lambda v: str(v))
    prevalence = np.zeros((len(levels), theta.shape[1]), dtype=np.float64)
    for i, level in enumerate(levels):
        prevalence[i] = theta[stamps == level].mean(axis=0)
    if normalize:
        totals = prevalence.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        prevalence = prevalence / totals
    labels = [lv.item() if hasattr(lv, "item") else lv for lv in levels]
    return {"labels": labels, "prevalence": prevalence}


def topics_per_class(model, groups, *, ci=0.95):
    """Mean topic prevalence within each level of a grouping variable.

    A thin wrapper over :func:`topica.by_strata` on ``model.doc_topic``:
    ``groups`` is one label per document, and the result is a list of
    per-stratum prevalence records (mean and confidence interval per topic).
    """
    return _effects.by_strata(model.doc_topic, groups, ci=ci)


__all__ = [
    "topic_info",
    "topic_sizes",
    "topic_labels",
    "set_topic_labels",
    "representative_docs",
    "topics_over_time",
    "topics_per_class",
]
