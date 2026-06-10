"""Statistical parity against the original Java MALLET.

Independent implementations with different RNGs are never byte-identical, so we
assert topic *agreement*: on a planted-topic corpus, topica's LDA and Java
MALLET should recover the same topics (high aligned top-word overlap). Skips
cleanly when the `mallet` CLI is not installed.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parity"))
import mallet_parity  # noqa: E402

pytestmark = pytest.mark.parity


@pytest.mark.skipif(not mallet_parity.mallet_available(), reason="mallet CLI not installed")
def test_lda_matches_java_mallet():
    r = mallet_parity.lda_parity(iters=600)
    # Planted disjoint-vocabulary topics: both implementations recover them, so
    # aligned topics should overlap almost entirely.
    assert r["mean_jaccard"] > 0.8, r
    assert r["mean_cosine"] > 0.9, r


@pytest.mark.skipif(not mallet_parity.java_drivers_available(), reason="mallet jars / javac not available")
def test_labeled_matches_java_mallet():
    r = mallet_parity.labeled_parity(iters=600)
    # Topics correspond to labels and align by name; the per-label topic-word
    # distributions should be nearly identical to Java MALLET's LabeledLDA.
    assert r["mean_cosine"] > 0.95, r


@pytest.mark.skipif(not mallet_parity.java_drivers_available(), reason="mallet jars / javac not available")
def test_dmr_matches_java_mallet():
    r = mallet_parity.dmr_parity(iters=600)
    # DMR fits feature weights by L-BFGS (implementation-specific), so this is a
    # statistical check: topics align, and the covariate effect agrees in sign
    # and is substantial in both implementations.
    assert r["topic_cosine"] > 0.9, r
    assert r["mallet_effect"] > 1.0 and r["topica_effect"] > 1.0, r
