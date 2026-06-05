"""Embedding-guided LDA: anchor topics with pre-trained word embeddings.

The idea is a warm start, not a constraint. We cluster the vocabulary's
embeddings into ``num_topics`` semantic groups, seed each topic with the words
nearest its cluster centroid, and give those seed words a prior boost in that
topic. The Gibbs sampler then runs as ordinary LDA and can override any seed the
text data contradicts, so the embeddings shape where topics form without
dictating what they end up being.

This reuses the validated :class:`~topica.SeededLDA` sampler: an embedding-guided
fit is a seeded fit whose seeds are discovered from an embedding space instead of
typed by hand. The asymmetric topic-word prior, the seeded initialization, and
the (correctly disabled) beta optimization all come from there.

Users bring their own ``embeddings`` (a dense ``V x E`` matrix, e.g. from
``sentence-transformers`` run over the vocabulary). topica only needs the matrix
and the matching vocabulary list; it does the clustering and seeding.

    from sentence_transformers import SentenceTransformer
    import topica

    vocab = sorted({w for d in docs for w in d})
    emb = SentenceTransformer("all-MiniLM-L6-v2").encode(vocab)
    model = topica.EmbeddingLDA(num_topics=10, embeddings=emb, vocabulary=vocab)
    model.fit(docs, iters=1000)
    print(model.top_words(8))
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _kmeans(x: np.ndarray, k: int, *, seed: int, iters: int = 50):
    """k-means++ initialization then Lloyd iterations. Pure numpy (no sklearn),
    deterministic for a fixed ``seed``. Returns ``(labels, centroids)``."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    sq = (x * x).sum(axis=1)  # squared norms, reused in the distance identity

    # k-means++ seeding: pick centers far from those already chosen.
    centers = np.empty((k, x.shape[1]), dtype=x.dtype)
    first = int(rng.integers(n))
    centers[0] = x[first]
    d2 = sq + (centers[0] * centers[0]).sum() - 2.0 * x @ centers[0]
    np.maximum(d2, 0, out=d2)
    for c in range(1, k):
        total = d2.sum()
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        nxt = int(rng.choice(n, p=probs))
        centers[c] = x[nxt]
        dc = sq + (centers[c] * centers[c]).sum() - 2.0 * x @ centers[c]
        np.minimum(d2, np.maximum(dc, 0), out=d2)

    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(iters):
        # (n, k) squared distances via |x|^2 - 2 x.c + |c|^2 (no n*k*d tensor).
        dists = sq[:, None] - 2.0 * x @ centers.T + (centers * centers).sum(axis=1)[None, :]
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for c in range(k):
            members = labels == c
            if members.any():
                centers[c] = x[members].mean(axis=0)
            # An emptied centroid keeps its last position (rare; harmless here).
    return labels, centers


def embedding_seeds(
    embeddings,
    vocabulary: Sequence[str],
    num_topics: int,
    *,
    top_m: int = 20,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Turn word embeddings into per-topic seed-word sets.

    Clusters the (row-normalized) embeddings into ``num_topics`` groups and, for
    each cluster, returns the ``top_m`` member words closest to the centroid by
    cosine similarity. Each word seeds at most one topic (its own cluster), so
    the seed sets are disjoint and the anchors stay distinct. Returns a dict
    ``{"topic_k": [words]}`` ready for :class:`~topica.SeededLDA`; a degenerate
    empty cluster yields an empty (unseeded) topic.
    """
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be a 2-D (V, E) array")
    if len(vocabulary) != x.shape[0]:
        raise ValueError(
            f"vocabulary has {len(vocabulary)} words but embeddings has {x.shape[0]} rows"
        )
    if num_topics < 2:
        raise ValueError("num_topics must be >= 2")
    if num_topics > x.shape[0]:
        raise ValueError("num_topics cannot exceed the vocabulary size")
    if top_m < 1:
        raise ValueError("top_m must be >= 1")

    # Row-normalize so euclidean k-means clusters by direction (cosine).
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    xn = x / norms

    labels, centers = _kmeans(xn, num_topics, seed=seed)
    sims = xn @ centers.T  # (V, K); rank within a cluster by similarity to its centroid

    seeds: dict[str, list[str]] = {}
    for k in range(num_topics):
        members = np.where(labels == k)[0]
        if members.size == 0:
            seeds[f"topic_{k}"] = []
            continue
        order = members[np.argsort(-sims[members, k])][:top_m]
        seeds[f"topic_{k}"] = [str(vocabulary[i]) for i in order]
    return seeds


class EmbeddingLDA:
    """LDA whose topics are anchored by pre-trained word embeddings.

    Clusters the vocabulary embeddings into ``num_topics`` semantic groups, seeds
    each topic with the ``top_m`` words nearest its centroid, and fits a
    :class:`~topica.SeededLDA` so those seeds get a prior boost the sampler can
    still override. All of the fitted-model surface (``topic_word``,
    ``doc_topic``, ``top_words``, ``coherence``, ``vocabulary``, ...) is delegated
    to the underlying SeededLDA.

    Parameters
    ----------
    num_topics : int
        Number of topics (and embedding clusters) to form.
    embeddings : array (V, E)
        Dense word-embedding matrix, one row per vocabulary word.
    vocabulary : sequence of str
        The words, aligned row-for-row with ``embeddings``.
    top_m : int
        How many of each cluster's nearest words to use as seeds.
    weight : float
        Seed strength, passed to SeededLDA: a seed word gets ``weight * 100``
        extra prior pseudocounts in its topic. Higher anchors harder.
    alpha, beta : float
        Document-topic and base topic-word Dirichlet priors.
    seed : int
        Random seed for the k-means clustering and the sampler.
    """

    def __init__(
        self,
        num_topics: int,
        *,
        embeddings,
        vocabulary: Sequence[str],
        top_m: int = 20,
        weight: float = 1.0,
        alpha: float = 0.1,
        beta: float = 0.01,
        seed: int = 42,
    ) -> None:
        from . import SeededLDA

        self.num_topics = num_topics
        self.top_m = top_m
        self.seeds = embedding_seeds(
            embeddings, vocabulary, num_topics, top_m=top_m, seed=seed
        )
        self._model = SeededLDA(
            self.seeds, alpha=alpha, beta=beta, weight=weight, seed=seed
        )

    def fit(self, data, *, iters: int = 1000) -> "EmbeddingLDA":
        """Fit the seeded model on ``data`` (a Corpus or list of token lists)."""
        self._model.fit(data, iters=iters)
        return self

    @property
    def model(self):
        """The underlying fitted :class:`~topica.SeededLDA`."""
        return self._model

    def __getattr__(self, name):
        # Delegate the fitted-model API (topic_word, top_words, ...) to SeededLDA.
        model = self.__dict__.get("_model")
        if model is None:
            raise AttributeError(name)
        return getattr(model, name)

    def __repr__(self) -> str:
        seeded = sum(1 for s in self.seeds.values() if s)
        return (
            f"EmbeddingLDA(num_topics={self.num_topics}, top_m={self.top_m}, "
            f"{seeded} topics seeded)"
        )
