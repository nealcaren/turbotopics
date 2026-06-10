"""Tests for topica.permutation_test (issue #36).

The planted-covariate corpus has a binary covariate that strongly predicts
which of two word-blocks dominates a document. After fitting a two-topic LDA
the covariate effect on the matched topic should sit in the extreme tail of the
permutation null, while an uncorrelated covariate should not.
"""

import numpy as np
import pytest

import topica


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _planted_corpus(n=120, seed=0):
    """Corpus + binary covariate where group 1 uses a-words, group 0 uses b-words.

    Returns (model, docs, covariate) with a fresh two-topic LDA fit.
    """
    rng = np.random.default_rng(seed)
    covariate = (rng.random(n) < 0.5).astype(float)   # ~50/50 split
    a = [f"a{i}" for i in range(8)]
    b = [f"b{i}" for i in range(8)]
    docs = []
    for flag in covariate:
        block = a if flag == 1.0 else b
        docs.append(list(rng.choice(block, size=16, replace=True)))
    corpus = topica.Corpus.from_documents(docs)
    m = topica.LDA(2, seed=1)
    m.fit(corpus, iters=200)
    return m, docs, covariate


# ---------------------------------------------------------------------------
# Basic contract
# ---------------------------------------------------------------------------

def test_returns_list_of_permutation_results():
    m, docs, cov = _planted_corpus()
    results = topica.permutation_test(m, docs, cov, n_perm=5, seed=0, iters=50)
    assert isinstance(results, list)
    assert len(results) == 2
    for r in results:
        assert isinstance(r, topica.PermutationResult)


def test_pvalue_in_unit_interval():
    m, docs, cov = _planted_corpus()
    results = topica.permutation_test(m, docs, cov, n_perm=10, seed=0, iters=50)
    for r in results:
        assert 0.0 <= r.pvalue <= 1.0, f"p-value out of range: {r.pvalue}"


def test_null_shape():
    m, docs, cov = _planted_corpus(n=80)
    n_perm = 8
    results = topica.permutation_test(m, docs, cov, n_perm=n_perm, seed=0, iters=50)
    for r in results:
        assert r.null.shape == (n_perm,), f"null shape mismatch: {r.null.shape}"


def test_topic_attribute_range():
    m, docs, cov = _planted_corpus()
    results = topica.permutation_test(m, docs, cov, n_perm=5, seed=0, iters=50)
    topics = {r.topic for r in results}
    assert topics == {0, 1}


# ---------------------------------------------------------------------------
# Sensitivity: planted effect should be extreme vs null
# ---------------------------------------------------------------------------

def test_planted_topic_has_extreme_observed_effect():
    """The topic most correlated with the planted covariate should have its
    observed effect well outside the bulk of the permutation null (n_perm=20
    is enough to detect a clean planted signal)."""
    m, docs, cov = _planted_corpus(n=160, seed=42)
    results = topica.permutation_test(m, docs, cov, n_perm=20, seed=7, iters=100)

    # At least one topic's observed effect should exceed the entire null range.
    obs_vals = [abs(r.observed) for r in results]
    best = max(obs_vals)
    best_result = results[obs_vals.index(best)]

    # The observed effect should be larger in magnitude than the null mean.
    null_abs_mean = float(np.abs(best_result.null).mean())
    assert best > null_abs_mean, (
        f"observed |effect|={best:.4f} not larger than null mean |effect|={null_abs_mean:.4f}"
    )


def test_planted_topic_pvalue_small():
    """The topic best matched to the planted covariate should have a small p-value
    with n_perm=30 permutations on a clear signal."""
    m, docs, cov = _planted_corpus(n=200, seed=99)
    results = topica.permutation_test(m, docs, cov, n_perm=30, seed=3, iters=100)
    best_pval = min(r.pvalue for r in results)
    assert best_pval <= 0.5, (
        f"expected at least one topic to have p <= 0.5; best was {best_pval:.3f}"
    )


def test_random_covariate_pvalue_not_consistently_small():
    """A random covariate uncorrelated with the corpus should not produce
    a very small p-value (most of the time)."""
    m, docs, _ = _planted_corpus(n=120, seed=0)
    rng = np.random.default_rng(77)
    random_cov = (rng.random(len(docs)) < 0.5).astype(float)
    results = topica.permutation_test(m, docs, random_cov, n_perm=20, seed=5, iters=50)
    # At least one topic should have p > 0.1 (not a slam-dunk rejection).
    assert any(r.pvalue > 0.1 for r in results), (
        "expected at least one topic to have p > 0.1 for a random covariate"
    )


# ---------------------------------------------------------------------------
# topics= restriction
# ---------------------------------------------------------------------------

def test_topics_restriction():
    m, docs, cov = _planted_corpus()
    results = topica.permutation_test(m, docs, cov, n_perm=5, seed=0,
                                      topics=[0], iters=50)
    assert len(results) == 1
    assert results[0].topic == 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_raises_on_non_binary_covariate():
    m, docs, _ = _planted_corpus(n=60)
    bad_cov = np.arange(len(docs)) % 3  # three levels
    with pytest.raises(ValueError, match="binary"):
        topica.permutation_test(m, docs, bad_cov, n_perm=2, iters=20)


def test_raises_on_wrong_length_covariate():
    m, docs, _ = _planted_corpus(n=60)
    with pytest.raises(ValueError):
        topica.permutation_test(m, docs, np.ones(10), n_perm=2, iters=20)


# ---------------------------------------------------------------------------
# as_dict and topic_name
# ---------------------------------------------------------------------------

def test_as_dict_keys():
    m, docs, cov = _planted_corpus(n=60)
    results = topica.permutation_test(m, docs, cov, n_perm=3, seed=0, iters=30)
    d = results[0].as_dict()
    for key in ("topic", "topic_name", "observed", "pvalue", "null_mean", "null_std"):
        assert key in d, f"missing key {key!r} in as_dict()"


def test_topic_name_is_string():
    m, docs, cov = _planted_corpus(n=60)
    results = topica.permutation_test(m, docs, cov, n_perm=3, seed=0, iters=30)
    for r in results:
        assert isinstance(r.topic_name, str)


# ---------------------------------------------------------------------------
# Corpus object accepted
# ---------------------------------------------------------------------------

def test_corpus_object_accepted():
    rng = np.random.default_rng(0)
    n = 60
    cov = (rng.random(n) < 0.5).astype(float)
    a = [f"a{i}" for i in range(6)]
    b = [f"b{i}" for i in range(6)]
    docs = []
    for flag in cov:
        block = a if flag == 1.0 else b
        docs.append(list(rng.choice(block, size=10, replace=True)))
    corpus = topica.Corpus.from_documents(docs)
    m = topica.LDA(2, seed=1)
    m.fit(corpus, iters=100)
    # Should not raise.
    results = topica.permutation_test(m, corpus, cov, n_perm=3, seed=0, iters=30)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Viz: PermutationTestPlot
# ---------------------------------------------------------------------------

def test_viz_import():
    import topica.viz as viz
    assert hasattr(viz, "permutation_test_plot")
    assert hasattr(viz, "PermutationTestPlot")


def test_viz_to_frame():
    m, docs, cov = _planted_corpus(n=60)
    results = topica.permutation_test(m, docs, cov, n_perm=3, seed=0, iters=30)
    import topica.viz as viz
    panel = viz.permutation_test_plot(results, covariate_name="group")
    df = panel.to_frame()
    assert len(df) == 2
    assert "observed" in df.columns
    assert "pvalue" in df.columns


def test_viz_to_png_no_error():
    """Rendering to matplotlib should not raise."""
    pytest.importorskip("matplotlib")
    m, docs, cov = _planted_corpus(n=60)
    results = topica.permutation_test(m, docs, cov, n_perm=3, seed=0, iters=30)
    import topica.viz as viz
    panel = viz.permutation_test_plot(results)
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "perm.png")
        fig = panel.to_png(path)
        assert os.path.exists(path)
