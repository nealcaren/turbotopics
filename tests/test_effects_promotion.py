"""The model-neutral prevalence tools: top-level/effects aliases resolve to the
same functions as their old model-namespaced homes, and estimate_effect runs the
method of composition on Gibbs models via dirichlet_theta_samples."""

import numpy as np
import pytest

import topica


def test_aliases_point_to_one_implementation():
    assert topica.estimate_effect is topica.effects.estimate_effect
    assert topica.estimate_effect is topica.stm.estimate_effect
    assert topica.by_strata is topica.effects.by_strata is topica.keyatm.by_strata
    assert topica.top_topics is topica.keyatm.top_topics
    assert topica.posterior_theta_samples is topica.stm.posterior_theta_samples


def _corpus(n=80):
    a = ["econ", "tax", "jobs", "budget"]
    b = ["war", "troop", "iraq", "border"]
    docs = [list(np.tile(a if i % 2 == 0 else b, 3)) for i in range(n)]
    x = np.array([[float(i % 2)] for i in range(n)])
    return docs, x


def test_dirichlet_theta_samples_shape_and_determinism():
    docs, _ = _corpus()
    m = topica.LDA(num_topics=2, seed=1)
    m.fit(docs, iterations=150)
    lengths = np.array([len(d) for d in docs])
    a = topica.dirichlet_theta_samples(m.doc_topic, lengths, nsims=15, seed=0)
    b = topica.dirichlet_theta_samples(m.doc_topic, lengths, nsims=15, seed=0)
    assert a.shape == (15, len(docs), 2)
    assert np.allclose(a.sum(axis=2), 1.0)
    assert np.array_equal(a, b)  # deterministic for a fixed seed
    # Longer documents give tighter (lower-variance) draws.
    short = topica.dirichlet_theta_samples(m.doc_topic, np.full(len(docs), 5.0), nsims=40, seed=0)
    long = topica.dirichlet_theta_samples(m.doc_topic, np.full(len(docs), 500.0), nsims=40, seed=0)
    assert long.var(axis=0).mean() < short.var(axis=0).mean()


def test_estimate_effect_on_gibbs_draws_and_point():
    docs, x = _corpus()
    m = topica.LDA(num_topics=2, seed=1)
    m.fit(docs, iterations=200)
    lengths = np.array([len(d) for d in docs])
    draws = topica.dirichlet_theta_samples(m.doc_topic, lengths, nsims=20, seed=0)

    moc = topica.estimate_effect(draws, x, feature_names=["grp"])
    ols = topica.estimate_effect(m.doc_topic, x, feature_names=["grp"])
    assert len(moc) == 2 and len(ols) == 2
    # Method of composition propagates topic uncertainty, so its standard errors
    # are at least as large as the point-estimate OLS ones.
    moc_se = moc[0].as_dict()["grp"]["se"]
    ols_se = ols[0].as_dict()["grp"]["se"]
    assert moc_se >= ols_se - 1e-9


def test_dirichlet_validation():
    with pytest.raises(ValueError):
        topica.dirichlet_theta_samples(np.zeros((4, 3)), np.zeros(5))  # mismatched lengths
    with pytest.raises(ValueError):
        topica.dirichlet_theta_samples(np.zeros(4), np.zeros(4))  # theta not 2-D
