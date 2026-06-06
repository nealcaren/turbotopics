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


def test_flexible_first_arg_accepts_model_or_matrix():
    # #10: frex/relevance/label_topics/topic_correlation/find_thoughts accept a
    # fitted model (the failing convention) as well as the raw matrix.
    docs = [["cat", "dog", "pet", "vet"]] * 12 + [["star", "moon", "sky", "sun"]] * 12
    m = topica.LDA(2, seed=1)
    m.fit(docs, iterations=150)
    texts = [" ".join(d) for d in docs]

    # model-first (previously raised "float() argument ... not 'topica.LDA'")
    assert topica.topic_correlation(m).cor.shape == (2, 2)
    assert len(topica.find_thoughts(m, texts, topic=0)) == 3
    assert len(topica.frex(m)) == 2
    assert set(topica.label_topics(m)[0]) == {"prob", "frex", "lift", "score"}
    assert len(topica.relevance(m, topic=0)) == m.topic_word.shape[1]  # capped at vocab

    # matrix-first still works (backward compatible)
    assert topica.topic_correlation(m.doc_topic).cor.shape == (2, 2)
    assert len(topica.frex(m.topic_word, m.vocabulary)) == 2

    # a bare matrix with no vocabulary gives a clear message, not a cryptic one
    with pytest.raises(ValueError, match="vocabulary is required"):
        topica.frex(m.topic_word)


def test_search_k_labels_its_coherence_metric():
    # #14: search_k reports UMass; label it so its scale isn't confused with c_v.
    docs = [["cat", "dog", "pet"]] * 12 + [["star", "moon", "sky"]] * 12
    rows = topica.search_k(docs, [2, 3], iterations=60, num_samples=1)
    assert all(r["coherence_metric"] == "u_mass" for r in rows)


def _planted_embeddings(seed=0):
    rng = np.random.default_rng(seed)
    vocab = [f"a{i}" for i in range(8)] + [f"b{i}" for i in range(8)]
    word_emb = np.vstack([rng.normal([3, 0], 0.2, (8, 2)),
                          rng.normal([-3, 0], 0.2, (8, 2))])
    idx = {w: i for i, w in enumerate(vocab)}
    docs = [[f"a{i}" for i in rng.integers(0, 8, 6)] for _ in range(30)] + \
           [[f"b{i}" for i in rng.integers(0, 8, 6)] for _ in range(30)]
    doc_emb = np.array([word_emb[[idx[w] for w in d]].mean(0) for d in docs])
    doc_emb = doc_emb + rng.normal(0, 0.05, doc_emb.shape)
    return docs, vocab, word_emb, doc_emb


def test_top2vec_centroid_default_and_kwarg():
    # #8: topic_neighbors(0, n=8) must not raise (topic is the first positional);
    # and top_words defaults to the centroid view when word_embeddings are present,
    # giving Top2Vec a headline distinct from BERTopic's c-TF-IDF.
    docs, vocab, word_emb, doc_emb = _planted_embeddings()
    tv = topica.Top2Vec(min_cluster_size=8, seed=1)
    tv.fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)

    neigh = [w for w, _ in tv.topic_neighbors(0, n=4)]  # previously raised
    assert len(neigh) == 4

    centroid = [w for w, _ in tv.top_words(4, topic=0)]              # default
    assert centroid == neigh                                        # centroid view
    ctfidf = [w for w, _ in tv.top_words(4, topic=0, representation="c-tf-idf")]
    assert isinstance(ctfidf, list)
    assert tv.topic_word.shape[0] == tv.num_topics                  # matrix stays c-TF-IDF


def test_top2vec_centroid_requires_word_vectors():
    # #8: without word_embeddings, top_words falls back to c-TF-IDF and an explicit
    # centroid request gives a clear error.
    docs, _, _, doc_emb = _planted_embeddings()
    tv = topica.Top2Vec(min_cluster_size=8, seed=1)
    tv.fit(docs, doc_emb)
    assert isinstance(tv.top_words(3, topic=0), list)  # c-TF-IDF default, no raise
    with pytest.raises(ValueError, match="word_embeddings"):
        tv.top_words(3, topic=0, representation="centroid")


def _three_blobs(seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[4, 0], [-4, 0], [0, 4]], float)
    doc_emb, docs = [], []
    for c in range(3):
        for _ in range(25):
            doc_emb.append(centers[c] + rng.normal(0, 0.3, 2))
            docs.append([f"w{c}_{i}" for i in rng.integers(0, 5, 6)])
    return docs, np.array(doc_emb)


@pytest.mark.parametrize("model_cls", ["BERTopic", "Top2Vec"])
@pytest.mark.parametrize("clusterer", ["kmeans", "agglomerative"])
def test_swappable_clusterer_assigns_every_doc(model_cls, clusterer):
    # #7: KMeans / agglomerative assign every document (no -1 noise bucket) to a
    # fixed number of clusters, unlike HDBSCAN.
    docs, doc_emb = _three_blobs()
    cls = getattr(topica, model_cls)
    m = cls(min_cluster_size=8, clusterer=clusterer, num_clusters=3, seed=1)
    m.fit(docs, doc_emb)
    assert m.num_topics == 3
    assert -1 not in set(m.labels)  # no noise bucket


def test_clusterer_validation():
    # #7: clear errors for the new knobs.
    with pytest.raises(ValueError, match="needs num_clusters"):
        topica.BERTopic(clusterer="kmeans")
    with pytest.raises(ValueError, match="unknown clusterer"):
        topica.Top2Vec(clusterer="dbscan")
    with pytest.raises(ValueError, match="num_clusters must be >= 1"):
        topica.Top2Vec(clusterer="kmeans", num_clusters=-2)


def test_report_is_callable():
    # #12: report(model) works as a one-call overview (alias for summary).
    assert callable(topica.report)
    docs = [["cat", "dog"], ["star", "moon"]] * 8
    m = topica.LDA(2, seed=1)
    m.fit(docs, iterations=50)
    assert topica.report(m) == topica.summary(m)
    assert "num_topics" in topica.report(m)
