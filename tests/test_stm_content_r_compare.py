"""Cross-implementation validation of the STM content model against R `stm`.

Guards the content path (where the topic-collapse bug lived) against the
reference implementation: both engines must SEPARATE the two topics, and their
per-group word distributions must agree. Skips when Rscript / `stm` is absent.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parity"))
import stm_content_r_compare  # noqa: E402

pytestmark = pytest.mark.parity


@pytest.mark.skipif(
    not stm_content_r_compare.r_stm_available(),
    reason="Rscript with the 'stm' package not available",
)
def test_content_model_matches_r_stm():
    r = stm_content_r_compare.run(verbose=False)
    for g in r["cosine"]:
        # Both engines must separate the topics (collapse would be ~1.0).
        assert r["r_topic_sep"][g] < 0.5, r
        assert r["tt_topic_sep"][g] < 0.5, r
        # Per-group word distributions agree with R.
        assert r["cosine"][g] > 0.7, r
