"""Cross-implementation validation against the R `stm` package.

R's `stm` and turbotopics are independent implementations of the structural
topic model; they share no code or RNG. On the multimodal gadarian K=3 problem
exact agreement is impossible, so the assertion is that turbotopics' Spectral
solution lands as close to R's Spectral solution as R's own init variants land
to each other. Skips cleanly when Rscript / the `stm` package is unavailable.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parity"))
import stm_r_compare  # noqa: E402

pytestmark = pytest.mark.parity


@pytest.mark.skipif(
    not stm_r_compare.r_stm_available(), reason="Rscript with the 'stm' package not available"
)
def test_stm_matches_r_stm():
    r = stm_r_compare.run(verbose=False)
    # turbotopics' Spectral fit is in the same neighborhood as R's Spectral fit:
    # no further from it than R's own Spectral-vs-Random basins differ (plus a
    # small multimodality margin).
    assert r["spectral_cosine"] >= r["r_spec_vs_rand"] - 0.15, r
