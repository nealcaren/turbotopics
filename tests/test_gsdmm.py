"""GSDMM (Movie Group Process) short-text clustering: inference of K, output
shapes, one-topic-per-document assignment, determinism, and save/load."""

import numpy as np
import pytest

import topica as tt


def _short_corpus(seed=0, n=150):
    rng = np.random.default_rng(seed)
    blocks = [["cat", "dog", "pet", "vet"],
              ["star", "moon", "sky", "sun"],
              ["tax", "vote", "law", "bill"]]
    docs = []
    for _ in range(n):
        blk = blocks[int(rng.integers(3))]
        docs.append([blk[int(rng.integers(4))] for _ in range(3)])  # 3-token docs
    return docs


class TestGSDMM:
    def test_infers_fewer_than_k_max(self):
        docs = _short_corpus()
        m = tt.GSDMM(num_topics=15, seed=1)
        m.fit(docs, iters=40)
        # Empty clusters die out -> effective K is below the cap.
        assert 0 < m.num_topics <= 15

    def test_output_shapes(self):
        docs = _short_corpus()
        m = tt.GSDMM(num_topics=15, seed=1)
        m.fit(docs, iters=40)
        k = m.num_topics
        assert m.topic_word.shape == (k, len(m.vocabulary))
        assert m.doc_topic.shape == (len(docs), k)
        np.testing.assert_allclose(m.topic_word.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(m.doc_topic.sum(axis=1), 1.0, atol=1e-9)

    def test_hard_assignment(self):
        docs = _short_corpus()
        m = tt.GSDMM(num_topics=15, seed=1)
        m.fit(docs, iters=40)
        dc = m.doc_cluster
        assert dc.shape == (len(docs),)
        assert dc.min() >= 0 and dc.max() < m.num_topics

    def test_recovers_blocks(self):
        docs = _short_corpus()
        m = tt.GSDMM(num_topics=15, seed=1)
        m.fit(docs, iters=60)
        blocks = [{"cat", "dog", "pet", "vet"},
                  {"star", "moon", "sky", "sun"},
                  {"tax", "vote", "law", "bill"}]
        covered = set()
        for t in range(m.num_topics):
            top = {w for w, _ in m.top_words(4, topic=t)}
            for b, blk in enumerate(blocks):
                if len(top & blk) >= 3:    # a cluster cleanly owns a block
                    covered.add(b)
        assert covered == {0, 1, 2}        # all three blocks recovered

    def test_deterministic(self):
        docs = _short_corpus()
        a = tt.GSDMM(num_topics=12, seed=3); a.fit(docs, iters=30)
        b = tt.GSDMM(num_topics=12, seed=3); b.fit(docs, iters=30)
        assert a.num_topics == b.num_topics
        assert np.array_equal(a.topic_word, b.topic_word)
        assert np.array_equal(a.doc_cluster, b.doc_cluster)

    def test_save_load(self, tmp_path):
        docs = _short_corpus()
        m = tt.GSDMM(num_topics=12, seed=1); m.fit(docs, iters=30)
        p = str(tmp_path / "gsdmm.tt"); m.save(p)
        ld = tt.GSDMM.load(p)
        assert ld.num_topics == m.num_topics
        assert np.array_equal(ld.topic_word, m.topic_word)
        assert np.array_equal(ld.doc_cluster, m.doc_cluster)

    def test_bad_params(self):
        with pytest.raises(ValueError):
            tt.GSDMM(num_topics=1)
        with pytest.raises(ValueError):
            tt.GSDMM(num_topics=10, alpha=0.0)
