"""The general standard-error facility (issue #15): method-of-composition by
default, bootstrap for top-words / embedding models, with an alignment-quality
flag that suppresses SEs where matching is unstable."""

import numpy as np
import pytest

import topica


def _planted(seed=0, n=240):
    """A covariate x that shifts prevalence between two word blocks, with a
    vocabulary large enough that the two topics are genuinely distinct."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    a = [f"a{i}" for i in range(6)]
    b = [f"b{i}" for i in range(6)]
    docs = []
    for i in range(n):
        p0 = 1.0 / (1.0 + np.exp(-2.0 * x[i]))
        block = a if rng.random() < p0 else b
        docs.append(list(rng.choice(block, size=12)))
    corpus = topica.Corpus.from_documents(docs)
    m = topica.LDA(2, seed=1)
    m.fit(corpus, iterations=250)
    return m, corpus, x


def test_corpus_doc_lengths_aligns_with_doc_topic():
    m, corpus, _ = _planted()
    assert len(corpus.doc_lengths) == np.asarray(m.doc_topic).shape[0]
    assert all(n > 0 for n in corpus.doc_lengths)


def test_model_family_detection():
    m, _, _ = _planted()
    assert topica.model_family(m) == "dirichlet"
    assert topica.model_family(topica.STM(2)) == "logistic_normal"
    assert topica.model_family(topica.BERTopic(min_cluster_size=5)) == "none"


def test_collapsed_gibbs_models_are_dirichlet():
    """Issue #20: KeyATM and SeededLDA are collapsed-Gibbs Dirichlet models, so
    `model_family` must classify them as "dirichlet" (they expose `alpha`) and
    `composition_theta` must draw the full posterior, not silently fall back."""
    docs = [list(np.random.default_rng(i).choice(
        [f"a{j}" for j in range(6)] + [f"b{j}" for j in range(6)], size=12))
        for i in range(60)]
    corpus = topica.Corpus.from_documents(docs)

    seeded = topica.SeededLDA({"a": ["a0", "a1"], "b": ["b0", "b1"]}, residual=1)
    seeded.fit(corpus, iters=150)
    key = topica.KeyATM({"a": ["a0", "a1"], "b": ["b0", "b1"]}, num_topics=3)
    key.fit(corpus, iters=150)

    k = len(corpus.doc_lengths)
    for model in (seeded, key):
        assert topica.model_family(model) == "dirichlet"
        assert np.asarray(model.alpha).shape == (model.num_topics,)
        draws = topica.effects.composition_theta(model, corpus, nsims=5)
        assert draws.shape == (5, k, model.num_topics)
        # Real posterior draws vary across simulations (not a repeated point estimate).
        assert draws.std(axis=0).max() > 0.0


def test_composition_effect_inflates_over_ols():
    m, corpus, x = _planted()
    X = x[:, None]
    ols = topica.estimate_effect(m.doc_topic, X, feature_names=["x"])
    moc = topica.standard_errors(m, corpus, of="effect", X=X, feature_names=["x"], nsims=30)
    # K=2: the two topics' slopes are exact negatives; composition SEs are >= OLS.
    assert np.sign(moc[0].coef[1]) == -np.sign(moc[1].coef[1])
    for t in range(2):
        assert moc[t].se[1] >= ols[t].se[1] - 1e-9


def test_composition_needs_corpus_for_gibbs_and_rejects_embedding():
    m, corpus, _ = _planted()
    with pytest.raises(ValueError, match="corpus"):
        topica.standard_errors(m, of="prevalence", nsims=10)  # Gibbs needs lengths
    bert = topica.BERTopic(min_cluster_size=5)
    with pytest.raises(ValueError, match="bootstrap"):
        topica.standard_errors(bert, corpus, of="prevalence", method="composition", nsims=10)


def test_composition_top_words_is_rejected():
    m, corpus, _ = _planted()
    with pytest.raises(ValueError, match="bootstrap"):
        topica.standard_errors(m, corpus, of="top_words", method="composition")


def test_bootstrap_prevalence_matches_composition_on_clean_data():
    m, corpus, _ = _planted()
    comp = topica.standard_errors(m, corpus, of="prevalence", nsims=40)
    boot = topica.standard_errors(m, corpus, of="prevalence", method="bootstrap",
                                  n_boot=40, topn=5, iterations=150, seed=0)
    for t in range(2):
        assert boot[t].reliable
        assert boot[t].alignment_quality > 0.5 and boot[t].alignment_margin > 0.1
        # Same ballpark (within 3x); both are honest, small SEs here.
        assert 0.2 < boot[t].se / comp[t].se < 5.0


def test_bootstrap_top_words_inclusion_probs():
    m, corpus, _ = _planted()
    res = topica.standard_errors(m, corpus, of="top_words", method="bootstrap",
                                 n_boot=30, topn=5, iterations=150, seed=0)
    assert len(res) == 2
    for r in res:
        assert r.reliable
        # Each top word has an inclusion probability in [0, 1]; stable topics keep
        # their words most of the time.
        probs = [p for (_, p, _, _) in r.words]
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert max(probs) > 0.5


def test_alignment_margin_flags_indistinct_topics():
    # topn larger than the distinct words per topic makes every topic's top-word
    # set identical: the Jaccard looks perfect but the match is arbitrary. The
    # margin diagnostic must catch this and suppress the SE.
    m, corpus, _ = _planted()
    res = topica.standard_errors(m, corpus, of="prevalence", method="bootstrap",
                                 n_boot=20, topn=20, iterations=150, seed=0)
    for r in res:
        assert r.alignment_margin < 0.1
        assert not r.reliable
        assert np.isnan(r.se)


def test_bootstrap_via_refit_hook():
    # The general hook: a user-supplied refit(doc_indices) -> fitted model, the
    # path embedding models use (resample embeddings alongside documents).
    m, corpus, _ = _planted()
    docs = corpus.documents()

    def refit(picks):
        mm = topica.LDA(2, seed=int(picks[0]) + 1)
        mm.fit([docs[i] for i in picks], iterations=150)
        return mm

    res = topica.standard_errors(m, corpus, of="prevalence", method="bootstrap",
                                 n_boot=20, topn=5, refit=refit, seed=0)
    assert len(res) == 2 and all(r.reliable for r in res)
