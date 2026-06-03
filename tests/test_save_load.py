"""Model serialization: every model saves to disk and loads back identically,
and a reloaded LDA can still infer topics for new documents.
"""

import os
import tempfile

import numpy as np
import pytest

import topica

DOCS = [["cat", "dog", "pet"]] * 30 + [["star", "moon", "sky"]] * 30


@pytest.fixture
def tmp(tmp_path):
    return str(tmp_path / "model.tt")


def _fit_all():
    """Return one fitted instance of every model."""
    X = np.array([[0.0]] * 30 + [[1.0]] * 30)
    y = np.array([0.0] * 30 + [1.0] * 30)
    groups = ["a"] * 30 + ["b"] * 30
    times = [0] * 30 + [1] * 30

    lda = topica.LDA(num_topics=2, seed=1); lda.fit(DOCS, iterations=200)
    dmr = topica.DMR(num_topics=2, seed=1); dmr.fit(DOCS, X, feature_names=["g"])
    lab = topica.LabeledLDA(seed=1); lab.fit(DOCS, [["x"]] * 60)
    sage = topica.SAGE(num_topics=2, seed=1); sage.fit(DOCS, groups)
    ctm = topica.CTM(num_topics=2, seed=1); ctm.fit(DOCS, em_iters=20)
    stm = topica.STM(num_topics=2, seed=1); stm.fit(DOCS, X, prevalence_names=["g"], em_iters=20)
    hdp = topica.HDP(seed=1); hdp.fit(DOCS, iters=40)
    dtm = topica.DTM(num_topics=2, seed=1); dtm.fit(DOCS, times, em_iters=8)
    slda = topica.SupervisedLDA(num_topics=2, seed=1); slda.fit(DOCS, y, em_iters=10)
    return {
        "LDA": lda, "DMR": dmr, "LabeledLDA": lab, "SAGE": sage, "CTM": ctm,
        "STM": stm, "HDP": hdp, "DTM": dtm, "SupervisedLDA": slda,
    }


@pytest.mark.parametrize("name", ["LDA", "DMR", "LabeledLDA", "SAGE", "CTM", "STM", "HDP", "DTM", "SupervisedLDA"])
def test_roundtrip(name, tmp):
    m = _fit_all()[name]
    m.save(tmp)
    assert os.path.getsize(tmp) > 0
    loaded = type(m).load(tmp)

    if name == "DTM":
        for t in range(m.num_times):
            assert np.array_equal(m.topic_word(t), loaded.topic_word(t))
    elif name == "SAGE":
        assert np.array_equal(m.topic_word, loaded.topic_word)  # (K, G, V)
    else:
        assert np.array_equal(m.topic_word, loaded.topic_word)
        assert np.array_equal(m.doc_topic, loaded.doc_topic)
    assert list(m.vocabulary) == list(loaded.vocabulary)


def test_lda_transform_after_load(tmp):
    m = topica.LDA(num_topics=2, seed=1)
    m.fit(DOCS, iterations=200)
    before = m.transform([["cat", "dog"], ["star", "moon"]])
    m.save(tmp)
    loaded = topica.LDA.load(tmp)
    after = loaded.transform([["cat", "dog"], ["star", "moon"]])
    assert np.array_equal(before, after)


def test_stm_posterior_survives(tmp):
    m = topica.STM(num_topics=2, seed=1)
    X = np.array([[0.0]] * 30 + [[1.0]] * 30)
    m.fit(DOCS, X, prevalence_names=["g"], em_iters=20)
    m.save(tmp)
    loaded = topica.STM.load(tmp)
    assert np.array_equal(m.eta_mean, loaded.eta_mean)
    assert np.array_equal(m.eta_cov, loaded.eta_cov)
    assert np.array_equal(m.prevalence_effects, loaded.prevalence_effects)


def test_slda_predict_after_load(tmp):
    m = topica.SupervisedLDA(num_topics=2, seed=1)
    y = np.array([0.0] * 30 + [1.0] * 30)
    m.fit(DOCS, y, em_iters=10)
    m.save(tmp)
    loaded = topica.SupervisedLDA.load(tmp)
    assert np.allclose(m.coefficients, loaded.coefficients)
    assert np.allclose(m.predict(DOCS), loaded.predict(DOCS))


def test_save_unfitted_raises(tmp):
    with pytest.raises(RuntimeError):
        topica.LDA(num_topics=2).save(tmp)


def test_load_garbage_raises(tmp_path):
    bad = str(tmp_path / "bad.tt")
    with open(bad, "wb") as f:
        f.write(b"not a model")
    with pytest.raises(ValueError):
        topica.LDA.load(bad)
