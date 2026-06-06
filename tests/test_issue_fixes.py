"""Regression tests for v0.7.0 user-testing fixes."""

import warnings

import numpy as np
import pytest

import topica


def test_english_stopwords_and_tokenize_accepts_iterable():
    # #3: a bundled stopword frozenset, and tokenize accepts any iterable.
    assert isinstance(topica.ENGLISH_STOPWORDS, frozenset)
    assert "the" in topica.ENGLISH_STOPWORDS and "cat" not in topica.ENGLISH_STOPWORDS
    toks = topica.tokenize("The cat and the dog ran", stopwords=topica.ENGLISH_STOPWORDS)
    assert toks == ["cat", "dog", "ran"]
    # a plain set and a list work too
    assert topica.tokenize("a big cat", stopwords={"a"}) == ["big", "cat"]
    assert topica.tokenize("a big cat", stopwords=["a", "big"]) == ["cat"]


def test_corpus_documents_round_trip():
    # #11: Corpus can recover its token lists.
    docs = [["cat", "dog"], ["star", "moon", "star"]]
    c = topica.Corpus.from_documents(docs)
    assert c.documents() == docs


def test_prepare_pyldavis_accepts_corpus():
    # #11: prepare_pyldavis takes a Corpus (no manual re-tokenizing).
    docs = [["cat", "dog", "pet"]] * 10 + [["star", "moon", "sky"]] * 10
    c = topica.Corpus.from_documents(docs)
    m = topica.LDA(2, seed=1)
    m.fit(c, iterations=100)
    out = topica.prepare_pyldavis(m, c)  # must not raise on a Corpus
    assert out is not None


def test_keyatm_warns_on_oov_keywords():
    # #9: out-of-vocabulary keywords warn instead of silently doing nothing.
    docs = [["health", "care", "doctor"]] * 10 + [["tax", "econ", "budget"]] * 10
    c = topica.Corpus.from_documents(docs)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        topica.KeyATM({"Health": ["health", "zzznotaword"]}, num_topics=2, seed=1).fit(c, iters=20)
    msgs = " ".join(str(x.message) for x in w)
    assert "zzznotaword" in msgs and "vocabulary" in msgs


def test_empty_clustering_warns_and_diagnostics_guard():
    # #6: degenerate embeddings -> 0 clusters: a warning, and diagnostics raise a
    # clear error instead of leaking numpy's "Mean of empty slice".
    rng = np.random.default_rng(0)
    docs = [["a", "b", "c"]] * 30
    emb = rng.normal(0, 1e-3, (30, 5))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m = topica.BERTopic(min_cluster_size=15, seed=1)
        m.fit(docs, emb)
    if m.num_topics == 0:
        assert any("no clusters" in str(x.message) for x in w)
        with pytest.raises(ValueError, match="no topics"):
            topica.label_topics(m.topic_word, m.vocabulary)


def test_citation_handle():
    # #14: programmatic citation.
    assert "Caren" in topica.__citation__ and "topica" in topica.__citation__


@pytest.mark.parametrize(
    "make",
    [
        lambda: topica.LDA(-3),
        lambda: topica.DMR(-1),
        lambda: topica.CTM(-2),
        lambda: topica.STM(-2),
        lambda: topica.GSDMM(-1),
        lambda: topica.PA(-1, 3),
        lambda: topica.PA(3, -1),
        lambda: topica.PT(-2),
        lambda: topica.PT(2, num_pseudo=-1),
        lambda: topica.HLDA(depth=-1),
        lambda: topica.KeyATM({"a": ["x"]}, num_topics=-5),
    ],
)
def test_negative_count_is_value_error(make):
    # #13: a negative count raises a clean ValueError, not a raw OverflowError.
    with pytest.raises(ValueError):
        make()


def test_zero_num_topics_still_guarded():
    # #13: the existing zero guard keeps working.
    with pytest.raises(ValueError, match="num_topics must be >= 1"):
        topica.LDA(0)


def test_search_k_labels_its_coherence_metric():
    # #14: search_k reports UMass; label it so its scale isn't confused with c_v.
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12
    rows = topica.search_k(docs, [2, 3], iterations=60, num_samples=1)
    assert all(r["coherence_metric"] == "u_mass" for r in rows)


def test_report_is_callable():
    # #12: report(model) works as a one-call overview (alias for summary).
    assert callable(topica.report)
    docs = [["cat", "dog"], ["star", "moon"]] * 8
    m = topica.LDA(2, seed=1)
    m.fit(docs, iterations=50)
    assert topica.report(m) == topica.summary(m)
    assert "num_topics" in topica.report(m)
