"""Structural Topic and Sentiment-Discourse model (Chen & Mankad 2024).

STS extends STM with a per-document, per-topic continuous sentiment-discourse
latent driven by covariates. These tests exercise the Python binding end to end:
shapes, topic recovery, the sentiment outputs, the covariate regressions, and the
shared analysis surface.
"""

import numpy as np
import pytest

import topica


A = ["cat", "dog", "pet", "kitten", "puppy", "vet"]
B = ["star", "moon", "sky", "sun", "comet", "orbit"]


def _planted(n=84, seed=0):
    """Two disjoint-vocabulary topics; each document is drawn from one. The
    prevalence covariate is the topic indicator; the sentiment seed is a separate
    3-level signal *independent* of the topics (so prevalence and sentiment are
    not confounded — using the topic indicator for both is unidentifiable)."""
    rng = np.random.default_rng(seed)
    docs, sent_seed, prev, truth = [], [], [], []
    for d in range(n):
        a = d % 2 == 0
        vocab = A if a else B
        docs.append([vocab[int(rng.integers(len(vocab)))] for _ in range(12)])
        sent_seed.append(float(d % 3))  # 3 aggregation groups, orthogonal to topic
        prev.append([0.0 if a else 1.0])
        truth.append(0 if a else 1)
    return docs, sent_seed, prev, np.array(truth)


def _fit(**kw):
    docs, sent_seed, prev, truth = _planted()
    m = topica.STS(num_topics=2, seed=1)
    m.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=kw.get("iters", 25))
    return m, docs, truth


class TestFit:
    def test_shapes(self):
        m, docs, _ = _fit()
        v = len(set(w for d in docs for w in d))
        assert m.topic_word.shape == (2, v)
        assert m.doc_topic.shape == (len(docs), 2)
        assert m.sentiment.shape == (len(docs), 2)
        assert m.prevalence_effects.shape == (2, 1)  # (intercept+indicator) x (K-1)
        assert m.sentiment_effects.shape == (2, 2)    # (intercept+indicator) x K
        np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-6)

    def test_recovers_topics(self):
        m, docs, truth = _fit()
        # Map each topic to a planted block by its actual top words (the corpus
        # assigns vocabulary ids in its own order, so word strings are the anchor).
        top_words = [{w for w, _ in m.top_words(4)[t]} for t in range(2)]
        a_set = set(A)
        topic_for_a = 0 if len(top_words[0] & a_set) >= len(top_words[1] & a_set) else 1
        theta = np.asarray(m.doc_topic)
        dominant = (theta[:, 1] > theta[:, 0]).astype(int)
        expected = np.where(truth == 0, topic_for_a, 1 - topic_for_a)
        assert (dominant == expected).mean() > 0.9

    def test_sentiment_and_bound(self):
        m, _, _ = _fit()
        # Sentiment latents are non-trivial and the bound trajectory is recorded.
        assert np.abs(np.asarray(m.sentiment)).max() > 1e-6
        assert len(m.bound_history) >= 1
        assert np.isfinite(m.bound)

    def test_topic_word_at_levels(self):
        m, _, _ = _fit()
        for level in (-2.0, 0.0, 2.0):
            b = np.asarray(m.topic_word_at(level))
            assert b.shape == m.topic_word.shape
            np.testing.assert_allclose(b.sum(axis=1), 1.0, atol=1e-6)

    def test_deterministic(self):
        docs, sent_seed, prev, _ = _planted()
        m1 = topica.STS(num_topics=2, seed=7)
        m2 = topica.STS(num_topics=2, seed=7)
        m1.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=15)
        m2.fit(docs, sentiment_seed=sent_seed, prevalence=prev, iters=15)
        np.testing.assert_allclose(m1.topic_word, m2.topic_word)
        np.testing.assert_allclose(m1.sentiment, m2.sentiment)


class TestAnalysisSurface:
    def test_flows_into_coherence_and_top_words(self):
        m, docs, _ = _fit()
        cv = topica.coherence(m, docs, coherence_type="c_v", topn=4)
        assert np.asarray(cv).shape == (2,)
        rows = m.top_words(3)
        assert len(rows) == 2 and all(isinstance(w, str) for w, _ in rows[0])

    def test_no_prevalence_still_fits(self):
        docs, sent_seed, _, _ = _planted()
        m = topica.STS(num_topics=2, seed=1)
        m.fit(docs, sentiment_seed=sent_seed, iters=15)  # no prevalence design
        assert m.doc_topic.shape == (len(docs), 2)
        assert m.sentiment.shape == (len(docs), 2)


class TestErrors:
    def test_num_topics_too_small(self):
        with pytest.raises(ValueError):
            topica.STS(num_topics=1)

    def test_seed_length_mismatch(self):
        docs, _, prev, _ = _planted()
        m = topica.STS(num_topics=2)
        with pytest.raises(ValueError, match="sentiment_seed"):
            m.fit(docs, sentiment_seed=[0.0, 1.0], prevalence=prev)

    def test_effects_require_prevalence(self):
        docs, sent_seed, _, _ = _planted()
        m = topica.STS(num_topics=2, seed=1)
        m.fit(docs, sentiment_seed=sent_seed, iters=10)
        with pytest.raises(RuntimeError, match="prevalence"):
            _ = m.prevalence_effects

    def test_getters_require_fit(self):
        m = topica.STS(num_topics=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = m.topic_word
