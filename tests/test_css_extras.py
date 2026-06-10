"""CSS-workflow additions: Fighting Words, the metadata-preserving document
splitter, the highlighted close-reading export, the coherence-exclusivity
frontier, bootstrap topic stability, and clustered / GLM-link estimate_effect.
"""

import numpy as np
import pytest

import topica
from topica import stm


# ---------------------------------------------------------------------------
# Fighting Words (Monroe-Colaresi-Quinn)
# ---------------------------------------------------------------------------

class TestFightingWords:
    def _corpora(self):
        a = [["tax", "cut", "growth", "jobs", "market"]] * 30
        b = [["climate", "carbon", "green", "planet", "energy"]] * 30
        return a, b

    def test_distinguishes_groups(self):
        a, b = self._corpora()
        scored = topica.fighting_words(a, b, prior=0.05)
        words = [w for w, _ in scored]
        assert words[0] in {"tax", "cut", "growth", "jobs", "market"}     # top -> A
        assert words[-1] in {"climate", "carbon", "green", "planet", "energy"}  # bottom -> B
        # z-scores are sorted descending.
        z = [s for _, s in scored]
        assert z == sorted(z, reverse=True)

    def test_shared_words_are_neutral(self):
        a = [["tax", "the", "of"]] * 40
        b = [["green", "the", "of"]] * 40
        d = dict(topica.fighting_words(a, b, prior=0.05))
        assert abs(d["the"]) < abs(d["tax"])     # shared word near zero

    def test_top_helper_and_informative(self):
        a, b = self._corpora()
        top = topica.top_fighting_words(a, b, n=3)
        assert set(top) == {"a", "b"} and len(top["a"]) == 3
        # Informative prior runs and returns the full vocabulary.
        scored = topica.fighting_words(a, b, informative=True)
        assert len(scored) == 10

    def test_min_count_filter(self):
        a = [["common", "common", "rare_a"]]
        b = [["common", "common", "rare_b"]]
        words = [w for w, _ in topica.fighting_words(a, b, min_count=2)]
        assert words == ["common"]


# ---------------------------------------------------------------------------
# Document splitter
# ---------------------------------------------------------------------------

class TestSplitDocuments:
    def test_propagates_metadata(self):
        long = "word " * 500
        chunks, meta = topica.split_documents([long.strip()], [{"year": 1920, "id": "a"}],
                                          max_words=100, min_words=20)
        assert len(chunks) == len(meta) > 1
        for j, row in enumerate(meta):
            assert row["year"] == 1920 and row["id"] == "a"
            assert row["parent"] == 0 and row["chunk"] == j

    def test_token_input_returns_tokens(self):
        doc = ["w"] * 250
        chunks, meta = topica.split_documents([doc], max_words=100, min_words=20)
        assert all(isinstance(c, list) for c in chunks)
        assert sum(len(c) for c in chunks) == 250          # no text lost

    def test_short_doc_one_chunk(self):
        chunks, meta = topica.split_documents(["just a short sentence."], max_words=100)
        assert len(chunks) == 1 and meta[0]["chunk"] == 0

    def test_runt_tail_merged(self):
        # 230 words, max 100 -> would be 100/100/30; the 30 merges into the prior.
        chunks, _ = topica.split_documents([("w " * 230).strip()], max_words=100,
                                       min_words=50, sentence_aware=False)
        assert all(len(c.split()) >= 50 for c in chunks)

    def test_metadata_length_mismatch(self):
        with pytest.raises(ValueError):
            topica.split_documents(["a", "b"], [{"x": 1}])


# ---------------------------------------------------------------------------
# Model-based helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def two_topic():
    docs = [["mob", "lynch", "south", "murder"]] * 40 + [["school", "child", "teach", "college"]] * 40
    m = topica.LDA(num_topics=2, seed=1)
    m.fit(docs, iters=400)
    return m, docs


class TestFindThoughtsHtml:
    def test_html_highlights_keywords(self, two_topic):
        m, docs = two_topic
        texts = [" ".join(d) for d in docs]
        html = topica.find_thoughts_html(m, texts, n_docs=2, n_words=4)
        assert "<mark>" in html and "Topic 0" in html and "Topic 1" in html

    def test_markdown_mode(self, two_topic):
        m, docs = two_topic
        texts = [" ".join(d) for d in docs]
        md = topica.find_thoughts_html(m, texts, n_docs=1, markdown=True)
        assert "**" in md and "### Topic" in md

    def test_alignment_checked(self, two_topic):
        m, docs = two_topic
        with pytest.raises(ValueError):
            topica.find_thoughts_html(m, ["only one text"])


class TestQualityFrontier:
    def test_returns_per_topic_arrays(self, two_topic):
        m, _ = two_topic
        qf = topica.quality_frontier(m, n=5)
        for key in ("topic", "coherence", "exclusivity", "prevalence"):
            assert qf[key].shape == (2,)
        np.testing.assert_allclose(qf["prevalence"].sum(), 1.0, atol=1e-6)


class TestBootstrapStability:
    def test_stable_topics_score_high(self, two_topic):
        _, docs = two_topic
        res = topica.bootstrap_stability(docs, k=2, n_boot=4, iters=200, topn=4)
        assert res["stability"].shape == (2,)
        assert 0.0 <= res["mean"] <= 1.0
        # Two clean, well-separated topics should be highly reproducible.
        assert res["mean"] > 0.5

    def test_accepts_corpus_object(self, two_topic):
        # Issue #27: the docstring promises a Corpus is accepted (like its
        # siblings perplexity / prepare_pyldavis), so it must not raise.
        _, docs = two_topic
        corpus = topica.Corpus.from_documents(docs)
        res = topica.bootstrap_stability(corpus, k=2, n_boot=2, iters=80, topn=4)
        assert res["stability"].shape == (2,)
        assert 0.0 <= res["mean"] <= 1.0

    def test_stable_across_changing_vocabulary(self):
        # Each document carries its block's shared words plus a unique filler
        # token, so every bootstrap resample produces a *different* vocabulary.
        # Matching topics by word-index (the original bug) collapses to ~0 here;
        # matching by word string keeps clearly-separated blocks stable.
        rng = np.random.default_rng(0)
        blocks = [["alpha", "bravo", "charlie"], ["xray", "yankee", "zulu"]]
        docs = []
        for i in range(120):
            blk = blocks[i % 2]
            docs.append(blk + [blk[int(rng.integers(3))], f"uniq_{i}"])
        res = topica.bootstrap_stability(docs, k=2, n_boot=6, iters=200, topn=3)
        assert res["mean"] > 0.4          # two clean blocks must stay reproducible


# ---------------------------------------------------------------------------
# estimate_effect: clustered SEs and GLM links
# ---------------------------------------------------------------------------

class TestEstimateEffectExtras:
    def _data(self):
        rng = np.random.default_rng(0)
        D = 200
        groups = np.repeat(np.arange(20), 10)
        x = rng.normal(size=D)
        t0 = np.clip(0.3 + 0.05 * x + 0.1 * rng.normal(size=D), 0.01, 0.99)
        theta = np.column_stack([t0, 1 - t0])
        return theta, x, groups

    def test_cluster_keeps_coef_changes_se(self):
        theta, x, groups = self._data()
        base = stm.estimate_effect(theta, x, feature_names=["x"])[0].as_dict()
        clus = stm.estimate_effect(theta, x, feature_names=["x"], cluster=groups)[0].as_dict()
        assert np.isclose(base["x"]["coef"], clus["x"]["coef"])   # same point estimate
        assert clus["x"]["se"] != base["x"]["se"]                 # different uncertainty

    def test_identity_no_cluster_is_legacy_ols(self):
        theta, x, _ = self._data()
        eff = stm.estimate_effect(theta, x, feature_names=["x"])[0]
        X = np.column_stack([np.ones(len(x)), x])
        beta = np.linalg.lstsq(X, theta[:, 0], rcond=None)[0]
        np.testing.assert_allclose(eff.coef, beta, atol=1e-10)

    def test_logit_link_runs(self):
        theta, x, _ = self._data()
        eff = stm.estimate_effect(theta, x, feature_names=["x"], link="logit")[0].as_dict()
        assert np.isfinite(eff["x"]["coef"]) and eff["x"]["se"] > 0

    def test_cluster_composes_with_method_of_composition(self):
        theta, x, groups = self._data()
        draws = np.stack([theta + 0.001 * np.random.default_rng(s).normal(size=theta.shape)
                          for s in range(5)])
        eff = stm.estimate_effect(draws, x, feature_names=["x"], cluster=groups)
        assert len(eff) == 2 and np.isfinite(eff[0].as_dict()["x"]["se"])

    def test_bad_link_and_cluster(self):
        theta, x, groups = self._data()
        with pytest.raises(ValueError):
            stm.estimate_effect(theta, x, link="probit")
        with pytest.raises(ValueError):
            stm.estimate_effect(theta, x, cluster=groups[:5])
