"""keyATM covariate model: the document-topic prior is a DMR on covariates,
alpha_{d,k} = exp(x_d . lambda_k), matching the keyATM R package. These check
that a planted covariate effect on topic prevalence is recovered, that the base
model is unaffected, and the output API."""

import numpy as np
import pytest

import topica

ECON = ["tax", "market", "trade", "fiscal", "budget", "deficit"]
SOC = ["abortion", "gay", "marriage", "church", "family", "prayer"]
SEEDS = {"economic": ECON[:4], "social": SOC[:4]}


def _corpus(seed=0):
    rng = np.random.default_rng(seed)
    docs, party = [], []
    for i in range(300):
        is_d = i % 2 == 0
        heavy, light = (SOC, ECON) if is_d else (ECON, SOC)
        docs.append(rng.choice(heavy, 9).tolist() + rng.choice(light, 3).tolist())
        party.append(1.0 if is_d else 0.0)
    return docs, np.array(party)


@pytest.fixture(scope="module")
def cov_model():
    docs, party = _corpus()
    m = topica.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, covariates=party.reshape(-1, 1), feature_names=["is_D"], iters=600)
    return m, docs, party


def test_feature_effects_shape_and_names(cov_model):
    m, _, _ = cov_model
    assert m.feature_names == ["intercept", "is_D"]
    assert m.feature_effects.shape == (2, 2)  # (K, intercept + 1 covariate)


def test_recovers_covariate_effect_on_prevalence(cov_model):
    m, _, _ = cov_model
    si, ei = m.topic_names.index("social"), m.topic_names.index("economic")
    # Being a Democrat raises the social topic relative to the economic topic.
    assert m.feature_effects[si, 1] - m.feature_effects[ei, 1] > 1.0


def test_theta_tracks_covariate(cov_model):
    m, _, party = cov_model
    th = m.doc_topic
    si = m.topic_names.index("social")
    assert th[party == 1, si].mean() > th[party == 0, si].mean() + 0.2


def test_deterministic(cov_model):
    m, docs, party = cov_model
    m2 = topica.KeyATM(SEEDS, num_topics=2, seed=1)
    m2.fit(docs, covariates=party.reshape(-1, 1), feature_names=["is_D"], iters=600)
    assert np.allclose(m.feature_effects, m2.feature_effects)


def test_base_model_has_no_feature_effects():
    docs, _ = _corpus()
    m = topica.KeyATM(SEEDS, num_topics=2, seed=1)
    m.fit(docs, iters=200)
    assert m.feature_names == []
    with pytest.raises(RuntimeError):
        _ = m.feature_effects


def test_covariate_row_count_validated():
    docs, party = _corpus()
    m = topica.KeyATM(SEEDS, num_topics=2, seed=1)
    with pytest.raises(ValueError):
        m.fit(docs, covariates=party[:-1].reshape(-1, 1), iters=10)
