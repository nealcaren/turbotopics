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


def _cluster_words(embeddings, num_topics: int, *, seed: int):
    """Row-normalize the embeddings, k-means into ``num_topics`` groups, and
    return ``(unit_word_vectors, labels, unit_centroids)``. The unit centroids
    are comparable by cosine with any document embedding from the same space."""
    x = np.asarray(embeddings, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("embeddings must be a 2-D (V, E) array")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    xn = x / norms
    labels, centers = _kmeans(xn, num_topics, seed=seed)
    cnorm = np.linalg.norm(centers, axis=1, keepdims=True)
    cnorm[cnorm == 0.0] = 1.0
    return xn, labels, centers / cnorm


def _seeds_from_clusters(xn, labels, centroids, vocabulary, num_topics, top_m):
    sims = xn @ centroids.T  # (V, K) word-to-centroid cosine
    seeds: dict[str, list[str]] = {}
    for k in range(num_topics):
        members = np.where(labels == k)[0]
        if members.size == 0:
            seeds[f"topic_{k}"] = []
            continue
        order = members[np.argsort(-sims[members, k])][:top_m]
        seeds[f"topic_{k}"] = [str(vocabulary[i]) for i in order]
    return seeds


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
    if len(vocabulary) != np.asarray(embeddings).shape[0]:
        raise ValueError(
            f"vocabulary has {len(vocabulary)} words but embeddings has "
            f"{np.asarray(embeddings).shape[0]} rows"
        )
    if num_topics < 2:
        raise ValueError("num_topics must be >= 2")
    if num_topics > len(vocabulary):
        raise ValueError("num_topics cannot exceed the vocabulary size")
    if top_m < 1:
        raise ValueError("top_m must be >= 1")
    xn, labels, centroids = _cluster_words(embeddings, num_topics, seed=seed)
    return _seeds_from_clusters(xn, labels, centroids, vocabulary, num_topics, top_m)


class EmbeddingLDA:
    """LDA whose topics are anchored by pre-trained embeddings, on both sides.

    The vocabulary embeddings define the topics: k-means clusters them into
    ``num_topics`` semantic groups, and each topic is seeded with the ``top_m``
    words nearest its centroid (a prior on the **topic-word** side, via
    :class:`~topica.SeededLDA`). Optionally, at fit time, **document** embeddings
    in the same space bias each document's topic mixture toward the topics its
    own embedding is closest to (a per-document prior on the **document-topic**
    side, ``α_{d,k}``). Both are priors: the Gibbs sampler reconciles them with
    word co-occurrence and can override either.

    Word seeds alone (no ``doc_embeddings``) is the lighter mode; adding document
    embeddings is closer in spirit to embedding-clustering methods, but keeps the
    generative, mixed-membership, override-able model. The fitted-model surface
    (``topic_word``, ``doc_topic``, ``top_words``, ``coherence``, ...) is
    delegated to the underlying SeededLDA.

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
        Seed strength: a seed word gets ``weight * 100`` extra prior pseudocounts
        in its topic. Higher anchors the topic-word side harder.
    doc_anchor : float
        Strength of the document-embedding prior used when ``doc_embeddings`` is
        passed to :meth:`fit`. ``α_{d,k} = alpha + doc_anchor * max(cos, 0)``.
    alpha, beta : float
        Base document-topic and topic-word Dirichlet priors.
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
        doc_anchor: float = 1.0,
        alpha: float = 0.1,
        beta: float = 0.01,
        seed: int = 42,
    ) -> None:
        from . import SeededLDA

        if len(vocabulary) != np.asarray(embeddings).shape[0]:
            raise ValueError("vocabulary length must match the number of embedding rows")
        if num_topics < 2:
            raise ValueError("num_topics must be >= 2")
        if num_topics > len(vocabulary):
            raise ValueError("num_topics cannot exceed the vocabulary size")
        if top_m < 1:
            raise ValueError("top_m must be >= 1")
        if doc_anchor < 0:
            raise ValueError("doc_anchor must be >= 0")

        self.num_topics = num_topics
        self.top_m = top_m
        self.alpha = alpha
        self.doc_anchor = doc_anchor
        # One clustering pass: keep the unit centroids for the document prior.
        xn, labels, self._centroids = _cluster_words(embeddings, num_topics, seed=seed)
        self.seeds = _seeds_from_clusters(xn, labels, self._centroids, vocabulary, num_topics, top_m)
        self._model = SeededLDA(
            self.seeds, alpha=alpha, beta=beta, weight=weight, seed=seed
        )

    def document_topic_prior(self, doc_embeddings) -> np.ndarray:
        """The per-document Dirichlet prior ``α_{d,k}`` implied by document
        embeddings: ``alpha + doc_anchor * max(cos(doc_d, centroid_k), 0)``,
        shape ``(num_docs, num_topics)``. Useful for inspection."""
        de = np.asarray(doc_embeddings, dtype=np.float64)
        if de.ndim != 2 or de.shape[1] != self._centroids.shape[1]:
            raise ValueError(
                "doc_embeddings must be (num_docs, E) with E matching the word embeddings"
            )
        norms = np.linalg.norm(de, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        sim = (de / norms) @ self._centroids.T
        return self.alpha + self.doc_anchor * np.maximum(sim, 0.0)

    def fit(self, data, *, doc_embeddings=None, iters: int = 1000) -> "EmbeddingLDA":
        """Fit on ``data`` (a Corpus or list of token lists). If ``doc_embeddings``
        is given (one row per document, same embedding space as the vocabulary),
        each document's topic mixture is biased toward the topics its embedding is
        nearest, as a prior the sampler can still override."""
        prior = self.document_topic_prior(doc_embeddings) if doc_embeddings is not None else None
        self._model.fit(data, iters=iters, doc_topic_prior=prior)
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
