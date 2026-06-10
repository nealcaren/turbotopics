"""Tests for select_model / plot_models (issue #37: best-of-N runs at fixed K)."""

import numpy as np
import pytest

import topica
from topica.validation import SelectModelResult

# ---------------------------------------------------------------------------
# Shared tiny corpus
# ---------------------------------------------------------------------------

DOCS = [["cat", "dog", "fish"]] * 20 + [["sun", "moon", "star"]] * 20


# ---------------------------------------------------------------------------
# Basic contract: LDA
# ---------------------------------------------------------------------------

class TestSelectModelLDA:
    """select_model with model='lda' (default)."""

    @pytest.fixture(scope="class")
    def result(self):
        return topica.select_model(DOCS, K=2, runs=3, iters=60, seed=0)

    def test_returns_select_model_result(self, result):
        assert isinstance(result, SelectModelResult)

    def test_n_models(self, result):
        assert len(result.models) == 3

    def test_coherence_array_length(self, result):
        assert len(result.coherence) == 3

    def test_exclusivity_array_length(self, result):
        assert len(result.exclusivity) == 3

    def test_run_seeds_array_length(self, result):
        assert len(result.run_seeds) == 3

    def test_coherence_values_are_finite(self, result):
        assert all(np.isfinite(v) for v in result.coherence)

    def test_exclusivity_values_are_positive(self, result):
        assert all(v > 0 for v in result.exclusivity)

    def test_models_are_fitted_lda(self, result):
        for m in result.models:
            assert hasattr(m, "topic_word")
            assert m.topic_word.shape[0] == 2

    def test_seeds_differ(self, result):
        seeds = list(result.run_seeds)
        assert len(set(seeds)) == len(seeds)


# ---------------------------------------------------------------------------
# Basic contract: STM
# ---------------------------------------------------------------------------

class TestSelectModelSTM:
    """select_model with model='stm'."""

    @pytest.fixture(scope="class")
    def result(self):
        prev = np.ones((len(DOCS), 1))
        return topica.select_model(DOCS, K=2, runs=3, model="stm",
                                   prevalence=prev, iters=30, seed=0)

    def test_n_models(self, result):
        assert len(result.models) == 3

    def test_coherence_length(self, result):
        assert len(result.coherence) == 3

    def test_exclusivity_length(self, result):
        assert len(result.exclusivity) == 3

    def test_models_have_bound(self, result):
        for m in result.models:
            assert hasattr(m, "bound")
            assert np.isfinite(m.bound)


# ---------------------------------------------------------------------------
# Early discard
# ---------------------------------------------------------------------------

class TestEarlyDiscard:
    """fraction= keeps only the top ceil(fraction * runs) survivors."""

    def test_fraction_reduces_survivors(self):
        # 3 runs, keep top 34% → ceil(0.34 * 3) = 2 survivors
        result = topica.select_model(DOCS, K=2, runs=3, iters=60, seed=0,
                                     fraction=0.34)
        assert len(result.models) == 2
        assert len(result.coherence) == 2
        assert len(result.exclusivity) == 2

    def test_fraction_one_keeps_all(self):
        result = topica.select_model(DOCS, K=2, runs=3, iters=60, seed=0,
                                     fraction=1.0)
        assert len(result.models) == 3

    def test_fraction_small_keeps_at_least_one(self):
        result = topica.select_model(DOCS, K=2, runs=3, iters=60, seed=0,
                                     fraction=0.01)
        assert len(result.models) >= 1

    def test_burn_in_iters_respected(self):
        # Smoke-test: does not raise and returns fitted models.
        result = topica.select_model(DOCS, K=2, runs=3, iters=60, seed=0,
                                     fraction=0.5, burn_in_iters=10)
        assert len(result.models) >= 1

    def test_stm_early_discard(self):
        prev = np.ones((len(DOCS), 1))
        result = topica.select_model(DOCS, K=2, runs=3, model="stm",
                                     prevalence=prev, iters=30, seed=0,
                                     fraction=0.5)
        assert 1 <= len(result.models) <= 3


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_invalid_model_raises():
    with pytest.raises(ValueError, match="model must be"):
        topica.select_model(DOCS, K=2, runs=2, model="hmm")


def test_invalid_runs_raises():
    with pytest.raises(ValueError, match="runs must be"):
        topica.select_model(DOCS, K=2, runs=0)


def test_invalid_fraction_raises():
    with pytest.raises(ValueError, match="fraction must be"):
        topica.select_model(DOCS, K=2, runs=2, fraction=1.5)


# ---------------------------------------------------------------------------
# plot_models
# ---------------------------------------------------------------------------

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")


@pytest.fixture(scope="module")
def lda_result():
    return topica.select_model(DOCS, K=2, runs=3, iters=60, seed=0)


def test_plot_models_returns_axes(lda_result):
    ax = topica.plot_models(lda_result)
    import matplotlib.axes
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_models_axes_labels(lda_result):
    ax = topica.plot_models(lda_result)
    assert "coherence" in ax.get_xlabel().lower()
    assert "exclusivity" in ax.get_ylabel().lower()


def test_plot_models_accepts_existing_axes(lda_result):
    import matplotlib.pyplot as plt
    _, ax_in = plt.subplots()
    ax_out = topica.plot_models(lda_result, ax=ax_in)
    assert ax_out is ax_in
    plt.close("all")


def test_plot_models_no_labels(lda_result):
    ax = topica.plot_models(lda_result, label_runs=False)
    assert ax is not None


def test_plot_models_one_run():
    """Single-run result should not raise."""
    result = topica.select_model(DOCS, K=2, runs=1, iters=60, seed=0)
    ax = topica.plot_models(result)
    assert ax is not None


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------

def test_select_model_exported():
    assert hasattr(topica, "select_model")
    assert hasattr(topica, "plot_models")
    assert hasattr(topica, "SelectModelResult")
