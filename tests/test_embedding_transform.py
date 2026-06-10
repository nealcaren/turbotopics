"""transform / fit_transform for the embedding models, and BERTopic's c-TF-IDF
options."""

import numpy as np
import pytest

import topica


def _setup(k=3, block=6, e=8, n=120, seed=0):
    rng = np.random.default_rng(seed)
    vocab = [f"b{b}w{i}" for b in range(k) for i in range(block)]
    v = len(vocab)
    word_emb = np.array([[3.0 if d == w // block else 0.0 for d in range(e)] for w in range(v)])
    word_emb += rng.normal(0, 0.2, (v, e))
    docs = [[f"b{d % k}w{int(rng.integers(block))}" for _ in range(10)] for d in range(n)]
    doc_emb = np.array([word_emb[vocab.index(docs[d][0])] + rng.normal(0, 0.3, e) for d in range(n)])
    return docs, vocab, word_emb, doc_emb


def test_etm_transform_roundtrip():
    docs, vocab, word_emb, _ = _setup()
    m = topica.ETM(num_topics=3, seed=1)
    th = m.fit_transform(docs, word_emb, vocab, iters=30)
    assert th.shape == (len(docs), 3)
    assert np.allclose(th.sum(axis=1), 1.0)
    held = m.transform(docs[:5])
    assert held.shape == (5, 3) and np.allclose(held.sum(axis=1), 1.0)
    # fit_transform equals the model's own doc_topic.
    assert np.allclose(th, m.doc_topic)


def test_top2vec_transform():
    docs, vocab, word_emb, doc_emb = _setup()
    m = topica.Top2Vec(min_cluster_size=8, seed=1)
    th = m.fit_transform(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)
    assert th.shape[0] == len(docs)
    new = m.transform(docs[:5], doc_emb[:5])
    assert new.shape == (5, m.num_topics) and np.allclose(new.sum(axis=1), 1.0)


def test_bertopic_transform_and_ctfidf_options():
    docs, _, _, doc_emb = _setup()
    m = topica.BERTopic(min_cluster_size=8, seed=1)
    th = m.fit_transform(docs, doc_emb)
    assert th.shape[0] == len(docs)
    new = m.transform(docs[:5])
    assert new.shape == (5, m.num_topics)
    # c-TF-IDF options run and keep topics block-pure.
    mb = topica.BERTopic(min_cluster_size=8, bm25=True, reduce_frequent=True, seed=1)
    mb.fit(docs, doc_emb)
    for t in range(mb.num_topics):
        blocks = {w.split("w")[0] for w, _ in mb.top_words(4, topic=t)}
        assert len(blocks) == 1
