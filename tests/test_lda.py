"""Tests for the LDA class."""

import math

import numpy as np
import numpy.testing as npt
import pytest

from topica import LDA, Corpus


# ---------------------------------------------------------------------------
# Small helper: fit a model quickly
# ---------------------------------------------------------------------------

def _quick_model(docs, seed=42, num_topics=2, **kwargs):
    """Return a fitted LDA with fast settings."""
    model = LDA(num_topics, seed=seed, **kwargs)
    model.fit(docs, iterations=200, num_samples=3, sample_interval=5)
    return model


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestLDAConstructor:
    def test_num_topics_available_before_fit(self):
        model = LDA(5)
        assert model.num_topics == 5

    def test_num_topics_less_than_one_raises(self):
        with pytest.raises(ValueError):
            LDA(0)

    def test_beta_zero_raises(self):
        with pytest.raises(ValueError):
            LDA(2, beta=0.0)

    def test_beta_negative_raises(self):
        with pytest.raises(ValueError):
            LDA(2, beta=-0.5)


# ---------------------------------------------------------------------------
# Unfitted-access raises RuntimeError
# ---------------------------------------------------------------------------

class TestUnfittedAccess:
    PROPERTIES = ["topic_word", "doc_topic", "vocabulary", "doc_names", "alpha", "beta"]

    @pytest.mark.parametrize("prop", PROPERTIES)
    def test_property_raises_before_fit(self, prop):
        model = LDA(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            getattr(model, prop)

    def test_log_likelihood_raises_before_fit(self):
        model = LDA(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.log_likelihood()

    def test_top_words_raises_before_fit(self):
        model = LDA(2)
        with pytest.raises(RuntimeError, match="not fitted"):
            model.top_words()


# ---------------------------------------------------------------------------
# Output shapes and basic properties after fit on list[list[str]]
# ---------------------------------------------------------------------------

class TestFitOnTokenLists:
    def test_topic_word_shape(self, toy_docs):
        model = _quick_model(toy_docs)
        # 2 topics, 7 unique words (cat dog fish planet star moon rocket)
        assert model.topic_word.shape == (2, 7)

    def test_doc_topic_shape(self, toy_docs):
        model = _quick_model(toy_docs)
        assert model.doc_topic.shape == (30, 2)

    def test_doc_topic_rows_sum_to_one(self, toy_docs):
        model = _quick_model(toy_docs)
        row_sums = model.doc_topic.sum(axis=1)
        npt.assert_allclose(row_sums, np.ones(30), atol=1e-6)

    def test_alpha_shape(self, toy_docs):
        model = _quick_model(toy_docs)
        assert model.alpha.shape == (2,)

    def test_beta_is_float(self, toy_docs):
        model = _quick_model(toy_docs)
        assert isinstance(model.beta, float)

    def test_num_topics_after_fit(self, toy_docs):
        model = _quick_model(toy_docs)
        assert model.num_topics == 2

    def test_vocabulary_length(self, toy_docs):
        model = _quick_model(toy_docs)
        assert len(model.vocabulary) == model.topic_word.shape[1]

    def test_doc_names_length(self, toy_docs):
        model = _quick_model(toy_docs)
        assert len(model.doc_names) == model.doc_topic.shape[0]

    def test_log_likelihood_finite(self, toy_docs):
        model = _quick_model(toy_docs)
        ll = model.log_likelihood()
        assert isinstance(ll, float)
        assert math.isfinite(ll)

    def test_repr_contains_num_topics(self, toy_docs):
        model = _quick_model(toy_docs)
        assert "2" in repr(model)


# ---------------------------------------------------------------------------
# Fit on Corpus object
# ---------------------------------------------------------------------------

class TestFitOnCorpus:
    def test_topic_word_shape(self, toy_corpus):
        model = _quick_model(toy_corpus)
        assert model.topic_word.shape == (2, toy_corpus.num_words)

    def test_doc_topic_shape(self, toy_corpus):
        model = _quick_model(toy_corpus)
        assert model.doc_topic.shape == (toy_corpus.num_docs, 2)

    def test_doc_topic_rows_sum_to_one(self, toy_corpus):
        model = _quick_model(toy_corpus)
        npt.assert_allclose(model.doc_topic.sum(axis=1), np.ones(toy_corpus.num_docs), atol=1e-6)

    def test_vocabulary_matches_corpus(self, toy_corpus):
        model = _quick_model(toy_corpus)
        assert sorted(model.vocabulary) == sorted(toy_corpus.vocabulary)

    def test_doc_names_match_corpus(self, toy_corpus):
        model = _quick_model(toy_corpus)
        assert model.doc_names == toy_corpus.doc_names


# ---------------------------------------------------------------------------
# Topic recovery
# ---------------------------------------------------------------------------

class TestTopicRecovery:
    def test_animal_and_space_land_in_different_topics(self, toy_docs):
        """The two clusters should map to different dominant topics."""
        model = _quick_model(toy_docs)
        vocab = model.vocabulary
        tw = model.topic_word
        animal_topic = int(tw[:, vocab.index("cat")].argmax())
        space_topic = int(tw[:, vocab.index("planet")].argmax())
        assert animal_topic != space_topic, (
            "Expected animal and space clusters in different topics, "
            f"but both mapped to topic {animal_topic}"
        )


# ---------------------------------------------------------------------------
# top_words
# ---------------------------------------------------------------------------

class TestTopWords:
    def test_topic_none_returns_list_of_lists(self, toy_docs):
        model = _quick_model(toy_docs)
        result = model.top_words(5)
        assert isinstance(result, list)
        assert len(result) == 2
        for topic_list in result:
            assert isinstance(topic_list, list)
            for item in topic_list:
                assert len(item) == 2
                word, prob = item
                assert isinstance(word, str)
                assert isinstance(prob, float)

    def test_topic_int_returns_single_list(self, toy_docs):
        model = _quick_model(toy_docs)
        result = model.top_words(5, topic=0)
        assert isinstance(result, list)
        assert len(result) == 5
        for word, prob in result:
            assert isinstance(word, str)
            assert isinstance(prob, float)

    def test_probabilities_descending_all_topics(self, toy_docs):
        model = _quick_model(toy_docs)
        for topic_list in model.top_words(7):
            probs = [p for _, p in topic_list]
            assert probs == sorted(probs, reverse=True)

    def test_probabilities_descending_single_topic(self, toy_docs):
        model = _quick_model(toy_docs)
        probs = [p for _, p in model.top_words(7, topic=1)]
        assert probs == sorted(probs, reverse=True)

    def test_topic_out_of_range_raises(self, toy_docs):
        model = _quick_model(toy_docs)
        with pytest.raises(ValueError):
            model.top_words(5, topic=10)

    def test_topic_negative_out_of_range_raises(self, toy_docs):
        model = _quick_model(toy_docs)
        with pytest.raises((ValueError, OverflowError)):
            model.top_words(5, topic=-1)


# ---------------------------------------------------------------------------
# save_topic_word / save_doc_topic
# ---------------------------------------------------------------------------

class TestSaveFiles:
    def test_save_topic_word_nonempty_tsv(self, toy_docs, tmp_path):
        model = _quick_model(toy_docs)
        path = tmp_path / "topic_word.tsv"
        model.save_topic_word(str(path))
        assert path.exists()
        content = path.read_text()
        assert len(content) > 0
        # header line
        first_line = content.splitlines()[0]
        assert "topic" in first_line and "word" in first_line and "probability" in first_line

    def test_save_doc_topic_nonempty_tsv(self, toy_docs, tmp_path):
        model = _quick_model(toy_docs)
        path = tmp_path / "doc_topic.tsv"
        model.save_doc_topic(str(path))
        assert path.exists()
        content = path.read_text()
        assert len(content) > 0
        # at least one data line beyond the header
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) >= 2

    def test_save_topic_word_tab_separated(self, toy_docs, tmp_path):
        model = _quick_model(toy_docs)
        path = tmp_path / "tw.tsv"
        model.save_topic_word(str(path))
        for line in path.read_text().splitlines()[1:5]:  # skip header
            parts = line.split("\t")
            assert len(parts) == 3

    def test_save_doc_topic_tab_separated(self, toy_docs, tmp_path):
        model = _quick_model(toy_docs)
        path = tmp_path / "dt.tsv"
        model.save_doc_topic(str(path))
        # data lines (skip header)
        for line in path.read_text().splitlines()[1:5]:
            parts = line.split("\t")
            assert len(parts) >= 2
