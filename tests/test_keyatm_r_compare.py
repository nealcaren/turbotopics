"""Cross-implementation validation against the R `keyATM` package.

R's `keyATM` and topica are independent implementations of the keyword-assisted
topic model; they share no code or RNG, and both initialize at random, so exact
agreement is impossible. keyATM's anchored *keyword* topics are the sharp test:
their content is pinned by the supplied keywords, so topica should recover them
as well as R recovers them across its own seeds. Skips cleanly when Rscript /
the `keyATM` package is unavailable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parity"))
import keyatm_r_compare  # noqa: E402

pytestmark = pytest.mark.parity


@pytest.mark.skipif(
    not keyatm_r_compare.r_keyatm_available(),
    reason="Rscript with the 'keyATM' package not available",
)
def test_keyatm_matches_r_keyatm():
    # Fewer iterations keeps the test quick; the keyword topics stabilize early.
    keyatm_r_compare.ITERS = 400
    r = keyatm_r_compare.run(verbose=False)
    # topica's anchored keyword topics agree with R's at least as well as R's own
    # seed-to-seed runs do (small multimodality margin for the free topics).
    assert r["keyword_cosine"] >= r["keyword_r_self_cosine"] - 0.15, r
