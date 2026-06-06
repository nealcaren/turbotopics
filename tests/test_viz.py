"""The visualization toolkit: capability descriptor, panel contract, honest gating."""

import warnings

import numpy as np
import pytest

import topica

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import topica.viz as viz  # noqa: E402


@pytest.fixture(scope="module")
def covariate_lda():
    rng = np.random.default_rng(0)
    n = 240
    x = rng.normal(size=n)
    docs = []
    for i in range(n):
        p0 = 1.0 / (1.0 + np.exp(-2.0 * x[i]))
        block = ["a0", "a1", "a2", "a3", "a4"] if rng.random() < p0 else \
                ["b0", "b1", "b2", "b3", "b4"]
        docs.append(list(rng.choice(block, size=12)))
    corpus = topica.Corpus.from_documents(docs)
    m = topica.LDA(2, seed=1)
    m.fit(corpus, iterations=300)
    return m, corpus, x, [" ".join(d) for d in docs]


@pytest.fixture(scope="module")
def bertopic():
    rng = np.random.default_rng(1)
    docs = [["x0", "x1", "x2"]] * 40 + [["y0", "y1", "y2"]] * 40
    emb = np.vstack([rng.normal([0, 0], 0.4, (40, 2)), rng.normal([6, 0], 0.4, (40, 2))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = topica.BERTopic(min_cluster_size=15, seed=1)
        m.fit(docs, emb)
    return m


# --- capability descriptor -------------------------------------------------

def test_capabilities_generative(covariate_lda):
    m = covariate_lda[0]
    cap = viz.capabilities(m)
    assert cap.prob_simplex_words and cap.soft_theta
    assert cap.theta_posterior == "dirichlet"
    assert cap.word_weight_label == "P(w | topic)"
    assert "frex" in cap.word_modes


def test_capabilities_cluster(bertopic):
    cap = viz.capabilities(bertopic)
    assert not cap.prob_simplex_words and not cap.soft_theta
    assert cap.theta_posterior == "none"
    assert cap.word_weight_label == "c-TF-IDF weight"
    assert cap.word_modes == ["prob"]


# --- the .to_frame() / .to_png() contract ----------------------------------

def test_every_panel_frames_and_renders(covariate_lda):
    m, corpus, x, texts = covariate_lda
    panels = [
        viz.coherence_frontier(m, texts),
        viz.search_k(topica.search_k(
            [t.split() for t in texts], [2, 3], iterations=80, num_samples=1)),
        viz.effect_plot(m, corpus, X=x[:, None], feature_names=["x"], nsims=15),
        viz.term_barchart(m, topic=0, mode="frex", n=6),
        viz.topic_similarity(m),
    ]
    for p in panels:
        df = p.to_frame()
        assert len(df) >= 1
        fig = p.to_png()  # returns a Figure, doesn't raise
        assert fig is not None


# --- honest gating ---------------------------------------------------------

def test_frex_mode_refused_on_ctfidf(bertopic):
    with pytest.raises(ValueError, match="not valid for BERTopic"):
        viz.term_barchart(bertopic, topic=0, mode="frex")
    # but prob mode works and is labeled c-TF-IDF
    tb = viz.term_barchart(bertopic, topic=0, mode="prob", n=4)
    assert tb.cap.word_weight_label == "c-TF-IDF weight"


def test_effect_plot_refuses_ci_without_posterior(bertopic):
    rng = np.random.default_rng(0)
    ep = viz.effect_plot(bertopic, X=rng.normal(size=(80, 1)), feature_names=["z"])
    assert ep.has_ci is False
    assert "no theta posterior" in ep.note
    ep.to_png()  # still draws point estimates


def test_topic_similarity_metric_switches(covariate_lda, bertopic):
    assert "Jensen-Shannon" in viz.topic_similarity(covariate_lda[0]).metric
    assert viz.topic_similarity(bertopic).metric == "cosine distance"


# --- dashboard -------------------------------------------------------------

def test_dashboard_picks_panels_and_exports(covariate_lda):
    m, corpus, x, texts = covariate_lda
    d = viz.dashboard(m, texts, corpus=corpus, X=x[:, None])
    assert set(d.panels) == {"similarity", "terms", "frontier", "effect"}
    frames = d.to_frame()
    assert set(frames) == set(d.panels)
    assert d.to_png() is not None


# --- interactive build (skips if altair absent) ----------------------------

def test_interactive_browser():
    alt = pytest.importorskip("altair")  # noqa: F841
    rng = np.random.default_rng(0)
    docs = [["a", "b", "c"]] * 15 + [["x", "y", "z"]] * 15
    m = topica.LDA(2, seed=1)
    m.fit(docs, iterations=100)
    chart = viz.term_topic_browser(m, n=5)
    html = chart.to_html()
    assert "vega" in html.lower()
