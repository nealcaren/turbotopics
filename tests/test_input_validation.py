"""Input-validation edge cases.

Adversarial inputs must fail loudly, not silently corrupt the fit:
  1. NaN/Inf float hyperparameters are rejected at construction.
  2. A corpus with no words (all documents empty) is rejected at fit.
  3. Misusing the diagnostics (raw matrix, non-integer topn) gives a clear error.
  4. coherence on an empty reference corpus errors rather than returning NaN.
  5. frex rejects out-of-range frequency weights.
"""

import numpy as np
import pytest

import topica
from topica import LDA

NAN = float("nan")
INF = float("inf")
DOCS = [["a", "b", "c"], ["b", "c", "d"], ["a", "d", "e"]] * 5


# --- 1. Non-finite hyperparameters -----------------------------------------

@pytest.mark.parametrize("bad", [NAN, INF, -INF, 0.0, -1.0])
def test_lda_rejects_nonfinite_beta(bad):
    with pytest.raises(ValueError):
        LDA(num_topics=2, beta=bad)


@pytest.mark.parametrize("bad", [NAN, INF, 0.0, -1.0])
def test_lda_rejects_nonfinite_alpha_sum(bad):
    with pytest.raises(ValueError):
        LDA(num_topics=2, alpha_sum=bad)


def test_other_constructors_reject_nan():
    # A representative float hyperparameter on each model family.
    cases = [
        (topica.DMR, dict(num_topics=2, beta=NAN)),
        (topica.DMR, dict(num_topics=2, prior_variance=NAN)),
        (topica.LabeledLDA, dict(alpha=NAN)),
        (topica.HDP, dict(eta=NAN)),
        (topica.HDP, dict(alpha=NAN)),
        (topica.DTM, dict(num_topics=2, chain_variance=NAN)),
        (topica.SupervisedLDA, dict(num_topics=2, alpha=NAN)),
        (topica.SAGE, dict(num_topics=2, prior_variance=NAN)),
    ]
    for cls, kw in cases:
        with pytest.raises(ValueError):
            cls(**kw)


def test_valid_hyperparameters_still_construct_and_fit():
    m = LDA(num_topics=2, beta=0.01, alpha_sum=1.0, seed=42)
    m.fit(DOCS, iterations=20)
    assert not np.any(np.isnan(m.topic_word))
    assert m.topic_word.shape == (2, len(m.vocabulary))


# --- 2. Degenerate (empty) corpus ------------------------------------------

def test_empty_documents_rejected():
    m = LDA(num_topics=2, seed=42)
    with pytest.raises(ValueError):
        m.fit([[], [], []], iterations=20)


def test_all_words_filtered_rejected():
    m = LDA(num_topics=2, seed=42)
    # Every word occurs once; min_doc_freq via a Corpus that prunes everything.
    from topica import Corpus
    with pytest.raises(ValueError):
        Corpus.from_documents([["a"], ["b"], ["c"]], min_doc_freq=5)


# --- 3 & 4. Diagnostic misuse ----------------------------------------------

@pytest.fixture(scope="module")
def model():
    m = LDA(num_topics=2, seed=42)
    m.fit(DOCS, iterations=30)
    return m


def test_topic_diversity_rejects_non_integer_topn(model):
    with pytest.raises(ValueError):
        topica.topic_diversity(model.topic_word, model.vocabulary)  # vocab as topn


def test_coherence_rejects_raw_matrix(model):
    with pytest.raises(ValueError):
        topica.coherence(model.topic_word, DOCS)


def test_coherence_rejects_empty_texts(model):
    with pytest.raises(ValueError):
        topica.coherence(model, [])


def test_diagnostics_on_model_still_work(model):
    assert 0.0 < topica.topic_diversity(model) <= 1.0
    assert topica.coherence(model, DOCS).shape == (2,)


# --- 5. frex weight range --------------------------------------------------

@pytest.mark.parametrize("w", [-1.0, 2.0, -0.01, 1.01, NAN])
def test_frex_rejects_bad_weight(model, w):
    with pytest.raises(ValueError):
        topica.frex(model, w=w)


def test_frex_valid_weight_works(model):
    out = topica.frex(model, w=0.5)
    assert len(out) == 2 and isinstance(out[0][0][0], str)
