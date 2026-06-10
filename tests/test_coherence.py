"""Topic coherence (u_mass / c_uci / c_npmi / c_v) and topic diversity.

The measures are validated by their defining properties: a topic of words that
always co-occur should score higher than a topic of words that never do, the
normalized measures must respect their ranges, and the gensim-style API should
accept both explicit word lists and fitted models.
"""

import numpy as np
import pytest

import topica
from topica import LDA

ALL_TYPES = ["u_mass", "c_uci", "c_npmi", "c_v"]


@pytest.fixture(scope="module")
def reference():
    """A corpus where {a,b,c} always co-occur and the r* words are scattered."""
    rng = np.random.default_rng(0)
    texts = []
    for _ in range(500):
        d = []
        if rng.random() < 0.5:
            d += ["a", "b", "c"] * 3
        d += [f"r{int(rng.integers(200))}" for _ in range(6)]
        rng.shuffle(d)
        texts.append(d)
    return texts


COHERENT = ["a", "b", "c"]
INCOHERENT = ["r1", "r2", "r3"]


class TestCoherenceRanksTopics:
    @pytest.mark.parametrize("ct", ALL_TYPES)
    def test_coherent_beats_incoherent(self, reference, ct):
        s = topica.coherence([COHERENT, INCOHERENT], reference, coherence_type=ct, topn=3)
        assert s.shape == (2,)
        assert s[0] > s[1], f"{ct}: coherent {s[0]} !> incoherent {s[1]}"

    def test_per_topic_shape(self, reference):
        s = topica.coherence([COHERENT, INCOHERENT, ["a", "b", "c"]], reference, coherence_type="c_v", topn=3)
        assert s.shape == (3,)


class TestRanges:
    def test_npmi_in_unit_range(self, reference):
        s = topica.coherence([COHERENT, INCOHERENT], reference, coherence_type="c_npmi", topn=3)
        assert np.all(s >= -1.0001) and np.all(s <= 1.0001)

    def test_cv_nonnegative_ish(self, reference):
        # C_v is a cosine of non-negative-ish context vectors; in [0, 1].
        s = topica.coherence([COHERENT, INCOHERENT], reference, coherence_type="c_v", topn=3)
        assert np.all(s >= -0.01) and np.all(s <= 1.0001)

    def test_umass_nonpositive_for_rare(self, reference):
        s = topica.coherence([INCOHERENT], reference, coherence_type="u_mass", topn=3)
        assert s[0] <= 0.0


class TestApi:
    def test_invalid_type_raises(self, reference):
        with pytest.raises(ValueError):
            topica.coherence([COHERENT], reference, coherence_type="c_bogus")

    def test_accepts_fitted_model(self, reference):
        docs = [["cat", "dog", "pet"]] * 20 + [["star", "moon", "sky"]] * 20
        m = LDA(num_topics=2, seed=1)
        m.fit(docs, iters=300)
        s = topica.coherence(m, docs, coherence_type="c_npmi", topn=3)
        assert s.shape == (2,)
        assert np.all(np.isfinite(s))

    def test_accepts_word_prob_pairs(self, reference):
        topics = [[("a", 0.5), ("b", 0.3), ("c", 0.2)]]
        s = topica.coherence(topics, reference, coherence_type="c_v", topn=3)
        assert s.shape == (1,)

    def test_window_size_override(self, reference):
        s = topica.coherence([COHERENT], reference, coherence_type="c_npmi", topn=3, window_size=5)
        assert s.shape == (1,) and np.isfinite(s[0])

    def test_default_is_cv(self, reference):
        a = topica.coherence([COHERENT], reference, topn=3)
        b = topica.coherence([COHERENT], reference, coherence_type="c_v", topn=3)
        assert np.allclose(a, b)


class TestDiversity:
    def test_disjoint_is_one(self):
        assert topica.topic_diversity([["a", "b", "c"], ["d", "e", "f"]], topn=3) == 1.0

    def test_identical_is_half(self):
        # two identical 3-word topics: 3 unique / 6 total.
        assert topica.topic_diversity([["a", "b", "c"], ["a", "b", "c"]], topn=3) == 0.5

    def test_accepts_model(self):
        docs = [["cat", "dog", "pet"]] * 20 + [["star", "moon", "sky"]] * 20
        m = LDA(num_topics=2, seed=1)
        m.fit(docs, iters=300)
        d = topica.topic_diversity(m, topn=3)
        assert 0.0 < d <= 1.0


class TestAnalysisContract:
    """Any object exposing the analysis contract -- ``topic_word`` /
    ``doc_topic`` / ``vocabulary`` -- works with the model-agnostic diagnostics,
    even without a ``top_words`` method. This pins the extensibility guarantee:
    a foreign model that presents the two matrices inherits the stack for free.
    """

    def test_duck_typed_model_works_without_top_words(self):
        docs = [["cat", "dog", "pet"]] * 30 + [["star", "moon", "sky"]] * 30
        m = LDA(num_topics=2, seed=1)
        m.fit(docs, iters=300)

        class Contract:  # the four members, NO top_words
            topic_word = m.topic_word
            doc_topic = m.doc_topic
            vocabulary = m.vocabulary

        c = Contract()
        # coherence / topic_diversity derive top words from topic_word + vocabulary
        np.testing.assert_allclose(
            topica.coherence(c, docs), topica.coherence(m, docs)
        )
        assert topica.topic_diversity(c) == topica.topic_diversity(m)
        np.testing.assert_allclose(topica.exclusivity(c), topica.exclusivity(m))
