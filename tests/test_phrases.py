"""Tests for topica.phrases — pure-Python collocation extraction.

Imported directly by file path so the compiled Rust extension is not required.
"""

import importlib.util
import math
import pathlib

import pytest

# ---------------------------------------------------------------------------
# Direct import — no compiled extension needed
# ---------------------------------------------------------------------------

import sys

spec = importlib.util.spec_from_file_location(
    "phrases",
    pathlib.Path(__file__).resolve().parents[1] / "python" / "topica" / "phrases.py",
)
phrases_mod = importlib.util.module_from_spec(spec)
# Register before exec so @dataclass can resolve the module's __dict__.
sys.modules["phrases"] = phrases_mod
spec.loader.exec_module(phrases_mod)

learn_phrases  = phrases_mod.learn_phrases
apply_phrases  = phrases_mod.apply_phrases
export_phrases = phrases_mod.export_phrases
Phrases        = phrases_mod.Phrases


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_planted_corpus(seed=42):
    """A corpus where 'new york' always co-occurs.

    Generates 200 documents.  Every document contains the pair ['new', 'york']
    exactly once (planted collocation) plus 10 random words drawn from a
    50-word vocabulary.  'new' and 'york' never appear separately, so
    count('new') == count('york') == count('new', 'york') == 200.

    With 10 noise words per document the average document length is 12 tokens,
    giving a default (len-vocab) score for the planted pair of ~0.25 (above threshold=0.1).
    """
    import random
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(50)]
    docs = []
    for _ in range(200):
        noise = [rng.choice(vocab) for _ in range(10)]
        # Insert planted bigram at a random position
        pos = rng.randint(0, len(noise))
        doc = noise[:pos] + ["new", "york"] + noise[pos:]
        docs.append(doc)
    return docs


def _make_corpus_with_triplet(seed=0):
    """Corpus where 'new york city' always co-occurs together.

    200 documents each containing the triplet plus 10 noise words.
    With 10 noise words the average doc length is 13 tokens, giving
    a default (len-vocab) score for each pair in the triplet of ~0.26 (above threshold=0.1).
    """
    import random
    rng = random.Random(seed)
    vocab = [f"w{i}" for i in range(50)]
    docs = []
    for _ in range(200):
        noise = [rng.choice(vocab) for _ in range(10)]
        pos = rng.randint(0, len(noise))
        doc = noise[:pos] + ["new", "york", "city"] + noise[pos:]
        docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# Test 1: planted collocation is detected and merged
# ---------------------------------------------------------------------------

class TestDetectAndMerge:
    def test_planted_bigram_detected(self):
        """'new york' should be in the learned phrase set."""
        docs = _make_planted_corpus()
        model = learn_phrases(docs, min_count=5, threshold=0.1)
        assert ("new", "york") in model.bigrams, (
            "('new', 'york') should be a detected collocation"
        )

    def test_incidental_pairs_not_detected(self):
        """Random adjacent noise-word pairs should not pass the threshold."""
        docs = _make_planted_corpus()
        model = learn_phrases(docs, min_count=5, threshold=0.1)
        # The noise vocabulary has 50 words; any adjacent pair of two distinct
        # noise words should appear far less often than 'new york'.
        for (a, b) in model.bigrams:
            assert not (a.startswith("w") and b.startswith("w")), (
                f"Noise pair ({a!r}, {b!r}) should not survive the threshold"
            )

    def test_transform_merges_planted_bigram(self):
        """apply_phrases should replace ['new', 'york'] with ['new_york']."""
        docs = _make_planted_corpus()
        model = learn_phrases(docs, min_count=5, threshold=0.1)
        transformed = apply_phrases(docs, model)

        for orig, new_doc in zip(docs, transformed):
            # Every original document had exactly one 'new york' pair.
            assert "new_york" in new_doc, "Merged token should be present"
            assert "new" not in new_doc, "'new' should have been consumed"
            assert "york" not in new_doc, "'york' should have been consumed"

    def test_transform_preserves_non_collocated_tokens(self):
        """Noise tokens that are not part of a collocation pass through unchanged."""
        docs = _make_planted_corpus()
        model = learn_phrases(docs, min_count=5, threshold=0.1)
        transformed = apply_phrases(docs, model)

        for orig, new_doc in zip(docs, transformed):
            # Remove the two tokens that form the collocation.
            orig_noise = [t for t in orig if t not in ("new", "york")]
            new_noise   = [t for t in new_doc if t != "new_york"]
            assert orig_noise == new_noise, (
                "Non-collocated tokens should be unchanged and in the same order"
            )

    def test_method_transform_identical_to_apply_phrases(self):
        """Phrases.transform and apply_phrases must return identical results."""
        docs = _make_planted_corpus()
        model = learn_phrases(docs, min_count=5, threshold=0.1)
        assert model.transform(docs) == apply_phrases(docs, model)


# ---------------------------------------------------------------------------
# Test 2: min_count filter
# ---------------------------------------------------------------------------

class TestMinCount:
    def test_rare_bigram_excluded_by_min_count(self):
        """A bigram that appears fewer times than min_count must be excluded."""
        # Single document containing a rare bigram exactly once.
        docs = [["rare", "bigram"]] + [["foo", "bar", "baz"] for _ in range(50)]
        model = learn_phrases(docs, min_count=5, threshold=0.0)
        assert ("rare", "bigram") not in model.bigrams, (
            "A bigram with count < min_count must not be accepted"
        )

    def test_frequent_bigram_accepted_above_min_count(self):
        """A bigram appearing exactly min_count times must be accepted (score>=threshold)."""
        # Create exactly min_count occurrences of ('alpha', 'beta').
        # Use threshold=0 so only min_count gates acceptance.
        min_c = 5
        docs = [["alpha", "beta"]] * min_c + [["other", f"w{i}"] for i in range(50)]
        model = learn_phrases(docs, min_count=min_c, threshold=0.0)
        assert ("alpha", "beta") in model.bigrams, (
            "A bigram with count == min_count should be accepted (score >= threshold=0)"
        )

    def test_below_min_count_excluded(self):
        """One fewer occurrence than min_count must be excluded."""
        min_c = 5
        docs = [["alpha", "beta"]] * (min_c - 1) + [["other", f"w{i}"] for i in range(50)]
        model = learn_phrases(docs, min_count=min_c, threshold=0.0)
        assert ("alpha", "beta") not in model.bigrams


# ---------------------------------------------------------------------------
# Test 3: NPMI scoring
# ---------------------------------------------------------------------------

class TestNPMI:
    def test_npmi_values_in_range(self):
        """All NPMI scores must lie in [-1, 1]."""
        docs = _make_planted_corpus()
        model = learn_phrases(docs, min_count=5, threshold=-1.0, scoring="npmi")
        for (a, b), sc in model.bigrams.items():
            assert -1.0 <= sc <= 1.0, (
                f"NPMI score for ({a!r}, {b!r}) = {sc} is outside [-1, 1]"
            )

    def test_npmi_planted_collocation_scores_high(self):
        """The planted ('new', 'york') pair should have NPMI near 1."""
        docs = _make_planted_corpus()
        # Use very low threshold so all bigrams above min_count are kept.
        model = learn_phrases(docs, min_count=5, threshold=-1.0, scoring="npmi")
        assert ("new", "york") in model.bigrams, "Planted pair not found under NPMI"
        sc = model.bigrams[("new", "york")]
        assert sc > 0.9, (
            f"Planted collocation should have NPMI near 1, got {sc:.4f}"
        )

    def test_npmi_collocation_ranks_highest(self):
        """The planted collocation should have the highest NPMI among all bigrams."""
        docs = _make_planted_corpus()
        model = learn_phrases(docs, min_count=5, threshold=-1.0, scoring="npmi")
        if not model.bigrams:
            pytest.skip("No bigrams found — corpus may need adjustment")
        best_pair = max(model.bigrams, key=lambda k: model.bigrams[k])
        assert best_pair == ("new", "york"), (
            f"Expected ('new', 'york') to rank highest, got {best_pair!r}"
        )

    def test_npmi_threshold_filters(self):
        """Setting threshold=0.5 for NPMI should keep only highly associated pairs."""
        docs = _make_planted_corpus()
        model_loose = learn_phrases(docs, min_count=5, threshold=-1.0, scoring="npmi")
        model_tight = learn_phrases(docs, min_count=5, threshold=0.5, scoring="npmi")
        # Tight model should have fewer or equal bigrams.
        assert len(model_tight.bigrams) <= len(model_loose.bigrams)
        # Planted collocation should still be present under tight threshold.
        assert ("new", "york") in model_tight.bigrams


# ---------------------------------------------------------------------------
# Test 4: trigram composition
# ---------------------------------------------------------------------------

class TestTrigrams:
    def test_trigram_via_composition(self):
        """Applying two phrase passes to a triplet corpus produces 'new_york_city'."""
        docs = _make_corpus_with_triplet()

        # Pass 1: learn and apply bigrams.
        p1   = learn_phrases(docs, min_count=5, threshold=0.1)
        docs1 = apply_phrases(docs, p1)

        # After pass 1 every document should have 'new_york' (or similar prefix).
        # (new, york) pair appears in every doc so it will be merged first,
        # leaving 'new_york' adjacent to 'city'.
        bigram_token_present = any("new_york" in doc for doc in docs1)
        assert bigram_token_present, "Pass-1 bigram 'new_york' should be present"

        # Pass 2: learn trigrams from bigram-merged corpus.
        p2    = learn_phrases(docs1, min_count=5, threshold=0.1)
        docs2 = apply_phrases(docs1, p2)

        # Every document should now contain the trigram token.
        for doc in docs2:
            assert "new_york_city" in doc, (
                f"Expected 'new_york_city' in document, got: {doc}"
            )

    def test_trigram_does_not_appear_in_single_pass(self):
        """A single phrase pass must NOT produce 'new_york_city' directly."""
        docs = _make_corpus_with_triplet()
        p1    = learn_phrases(docs, min_count=5, threshold=0.1)
        docs1 = apply_phrases(docs, p1)
        for doc in docs1:
            assert "new_york_city" not in doc, (
                "Trigram should not appear after a single bigram pass"
            )


# ---------------------------------------------------------------------------
# Test 5: apply_phrases leaves non-collocated text unchanged
# ---------------------------------------------------------------------------

class TestNoCollocations:
    def test_docs_unchanged_when_no_phrases_detected(self):
        """Documents should be returned unmodified when the phrase set is empty."""
        docs = [["hello", "world"], ["foo", "bar", "baz"]]
        # Tiny corpus — nothing will pass min_count=5.
        model = learn_phrases(docs, min_count=5, threshold=0.1)
        assert len(model.bigrams) == 0
        result = apply_phrases(docs, model)
        assert result == docs

    def test_empty_phrase_model_on_large_corpus(self):
        """Explicitly constructed Phrases with no bigrams leaves all docs intact."""
        docs = [["a", "b", "c"], ["d", "e", "f"]]
        empty_model = Phrases(bigrams={}, delimiter="_")
        result = apply_phrases(docs, empty_model)
        assert result == docs

    def test_non_collocated_tokens_in_mixed_corpus(self):
        """Tokens not forming a collocation pass through unchanged."""
        docs = _make_planted_corpus()
        model = learn_phrases(docs, min_count=5, threshold=0.1)
        # Construct a document with no 'new'/'york' pair.
        solo_doc = [["hello", "world", "foo"]]
        result = apply_phrases(solo_doc, model)
        assert result == solo_doc, (
            "Document with no collocations should be returned unchanged"
        )


# ---------------------------------------------------------------------------
# Test 6: export_phrases
# ---------------------------------------------------------------------------

class TestExportPhrases:
    def test_export_returns_sorted_list(self):
        """export_phrases should return (phrase_str, score) sorted desc by score."""
        docs = _make_planted_corpus()
        model = learn_phrases(docs, min_count=5, threshold=0.1)
        exported = export_phrases(model)
        assert isinstance(exported, list)
        scores = [sc for _, sc in exported]
        assert scores == sorted(scores, reverse=True), "Scores should be in descending order"

    def test_export_contains_planted_phrase(self):
        """The exported list should contain 'new_york' as a phrase string."""
        docs = _make_planted_corpus()
        model = learn_phrases(docs, min_count=5, threshold=0.1)
        exported = export_phrases(model)
        phrase_strings = [ph for ph, _ in exported]
        assert "new_york" in phrase_strings

    def test_export_uses_custom_delimiter(self):
        """export_phrases should respect a non-default delimiter."""
        docs = _make_planted_corpus()
        model = learn_phrases(docs, min_count=5, threshold=0.1, delimiter="-")
        exported = export_phrases(model)
        phrase_strings = [ph for ph, _ in exported]
        assert "new-york" in phrase_strings

    def test_export_empty_model(self):
        """An empty Phrases model should produce an empty export list."""
        model = Phrases(bigrams={})
        assert export_phrases(model) == []


# ---------------------------------------------------------------------------
# Test 7: edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_docs(self):
        """learn_phrases on an empty list should return an empty Phrases."""
        model = learn_phrases([])
        assert model.bigrams == {}

    def test_single_token_docs(self):
        """Documents with one token produce no bigrams."""
        docs = [["hello"]] * 20
        model = learn_phrases(docs, min_count=5, threshold=0.0)
        assert model.bigrams == {}

    def test_custom_delimiter_in_transform(self):
        """The custom delimiter must appear in merged tokens."""
        docs = _make_planted_corpus()
        model = learn_phrases(docs, min_count=5, threshold=0.1, delimiter="-")
        transformed = apply_phrases(docs, model)
        for doc in transformed:
            if "new-york" in doc:
                break
        else:
            pytest.fail("Custom delimiter '-' not found in any merged token")

    def test_invalid_scoring_raises(self):
        """An unknown scoring name should raise ValueError."""
        with pytest.raises(ValueError, match="scoring must be"):
            learn_phrases([["a", "b"]], scoring="unknown")

    def test_greedy_non_overlapping(self):
        """Greedy left-to-right merge: overlapping bigrams are handled correctly."""
        # If both (a,b) and (b,c) are collocations in 'a b c',
        # greedy left-to-right picks (a,b) → ['a_b', 'c'].
        docs = [["a", "b", "c"]] * 20
        # Make both pairs appear; use threshold=0 so both are learned.
        model = learn_phrases(docs, min_count=5, threshold=0.0)
        # Manually add both pairs to bigrams to force the overlap scenario.
        model.bigrams[("a", "b")] = 99.0
        model.bigrams[("b", "c")] = 99.0
        result = apply_phrases([["a", "b", "c"]], model)
        assert result == [["a_b", "c"]], (
            f"Greedy left-to-right should yield ['a_b', 'c'], got {result}"
        )
