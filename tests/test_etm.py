"""ETM (Embedded Topic Model) on topica's variational-EM core: it recovers
planted word blocks from embeddings, exposes the standard fitted surface, and
validates its inputs."""

import numpy as np
import pytest

import topica


def _planted(k=3, block=8, e=3, seed=0):
    """K word-blocks; each word's embedding points along its block's axis, each
    document draws from one block. Returns (docs, vocab, word_emb, truth)."""
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    word_emb = np.zeros((k * block, e))
    for w in range(k * block):
        word_emb[w, w // block] = 3.0
        word_emb[w] += rng.normal(0, 0.2, e)
    docs, truth = [], []
    for d in range(120):
        b = d % k
        docs.append([f"b{b}w{int(rng.integers(block))}" for _ in range(10)])
        truth.append(b)
    return docs, vocab, word_emb, np.array(truth)


def test_etm_recovers_planted_blocks():
    docs, vocab, word_emb, truth = _planted()
    m = topica.ETM(num_topics=3, em_iters=50, seed=1)
    m.fit(docs, word_emb, vocab)

    assert m.topic_word.shape == (3, len(vocab))
    assert m.doc_topic.shape == (len(docs), 3)
    assert m.topic_embeddings.shape == (3, word_emb.shape[1])
    assert np.allclose(m.doc_topic.sum(axis=1), 1.0)
    assert m.converged or m.bound < 0  # a real bound was tracked

    # Each topic's top words come from one block, covering all blocks.
    covered = set()
    for t in range(3):
        blocks = {w.split("w")[0] for w, _ in m.top_words(4, topic=t)}
        assert len(blocks) == 1, f"topic {t} mixes blocks: {blocks}"
        covered |= blocks
    assert len(covered) == 3


def test_etm_top_words_all_topics_and_names():
    docs, vocab, word_emb, _ = _planted()
    m = topica.ETM(num_topics=3, em_iters=20, seed=1)
    m.fit(docs, word_emb, vocab)
    allw = m.top_words(5)  # topic=None -> list per topic
    assert len(allw) == 3 and all(len(t) == 5 for t in allw)
    assert m.topic_names == ["topic_0", "topic_1", "topic_2"]
    assert list(m.vocabulary) == vocab


def test_etm_validation():
    docs, vocab, word_emb, _ = _planted()
    with pytest.raises(ValueError):
        topica.ETM(num_topics=1)  # need >= 2 topics
    m = topica.ETM(num_topics=3, seed=1)
    with pytest.raises(ValueError):
        m.fit(docs, word_emb[:-1], vocab)  # embeddings rows != vocab length
    with pytest.raises(RuntimeError):
        topica.ETM(num_topics=3).topic_word  # not fitted
