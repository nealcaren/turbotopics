# Parity Testing Report

You asked me to test the library against the original reference implementations
and report back, without changing model code. I added several focused parity and
contract tests, rebuilt the local Python extension so the installed package
matched the current source, ran the expanded parity suite, and documented the
remaining caveats.

## What I Used

- topica dev environment: `.venv-dev`
- topica extension rebuild:

```bash
maturin develop --release --features python
```

- R reference packages through `Rscript`: `stm`, `keyATM`
- Java MALLET through the `mallet` CLI plus existing Java driver tests
- pytest parity marker:

```bash
.venv-dev/bin/pytest -q -m parity -rs
```

## Current Result

The expanded parity-marked suite passes locally except for the embedding-model
reference dependency we are currently ignoring:

```text
20 passed, 1 skipped, 887 deselected
```

Skipped:

- `tests/test_top2vec_parity.py::test_top2vec_matches_bertopic`

Reason: `BERTopic` with `umap` / `hdbscan` is not installed. This is not a
failure of the non-embedding topic-model parity work.

## Tests Added

### Default contracts

Added `tests/test_reference_default_contracts.py`.

This checks topica's public defaults against the original packages where the
defaults are directly introspectable:

- R `stm`: `init`, `em_iters`, `em_tol`
- R `keyATM`: beta, keyword beta, gamma priors, alpha-estimation default,
  weighting default, threading default
- Java MALLET: shared LDA defaults such as beta, burn-in, thread count,
  symmetric-alpha flag, and iterations

It also pins known default differences as current topica behavior rather than
silently assuming parity.

Known default differences:

- Java MALLET default alpha sum is `5.0`; topica uses `alpha_sum=None`, which
  resolves to `num_topics`.
- Java MALLET default `optimize_interval` is `0`; topica defaults to `50`.
- Java MALLET default `random_seed=0` means clock/random seed; topica defaults
  to reproducible `seed=42`.
- R `keyATM` base alpha is effectively `1 / K`; topica exposes `alpha=0.1`.

Recommendation: leave the code unchanged for now. If we later want stricter
drop-in compatibility, revisit these as deliberate API changes.

### SeededLDA and GSDMM contracts

Added `tests/test_seeded_gsdmm_contracts.py`.

These are not live external-package parity tests. Instead, they pin exact
algorithmic contracts where a byte-identical maintained reference runner is not
currently available:

- `SeededLDA`: seed words get `weight * 100` extra topic-word prior mass.
- `SeededLDA`: seeded tokens initialize into their named topics.
- `SeededLDA`: `weight=0.0` removes the extra seed-word pseudocount while
  preserving seeded initialization.
- `GSDMM`: public `topic_word` and `doc_topic` follow the Movie Group Process
  smoothing and soft-assignment formulas.
- `GSDMM`: trace log-likelihood and effective cluster count match the formula
  after a recorded sweep.

Focused result:

```text
4 passed
```

Combined with the existing guided/GSDMM tests:

```text
24 passed
```

### STM deterministic helper parity

Added `tests/test_stm_helper_r_parity.py`.

These tests compare deterministic post-hoc helpers against R `stm` on fixed
matrices, avoiding the non-convex fitted-model problem:

- R `stm::calcfrex` vs `topica.stm.frex`
- R `stm::labelTopics` probability/FREX/score labels vs
  `topica.stm.label_topics`
- R `stm::topicCorr(method="simple")` vs `topica.stm.topic_correlation`
- R `stm::estimateEffect(..., uncertainty="None")` coefficients and standard
  errors vs `topica.stm.estimate_effect`

Result in the parity suite: passed.

Note: FREX uses ECDF ranks and ties are common. The tests treat selected tied
word sets as the contract, not arbitrary tie ordering.

### MALLET state contracts

Added `tests/test_mallet_state_contracts.py`.

These tests train a tiny corpus with Java MALLET, read MALLET's `--output-state`,
and verify topica's `LDA.load_state` preserves the mathematical state:

- token assignments
- alpha and beta
- vocabulary order
- state-derived smoothed topic-word probabilities
- overlapping MALLET diagnostics XML fields that share the same scale

Result in the parity suite: passed.

Important caveat: `topica.LDA.diagnostics()` is not a byte-for-byte clone of
MALLET diagnostics XML. Some fields overlap and now pass (`tokens`,
`document_entropy`), but others differ in scale or definition. Examples:

- MALLET `rank_1_docs` is a ratio; topica reports a count.
- Coherence values differ, indicating a formula/scale mismatch or different
  diagnostic convention.

This is now documented as a real compatibility boundary, not hidden.

## Existing Parity Tests That Passed

The existing parity checks still pass after adding the new tests:

- Python binding vs topica `train` CLI byte-for-byte output
- topica LDA vs Java MALLET statistical topic agreement
- topica LabeledLDA vs Java MALLET label-aligned topic-word agreement
- topica DMR vs Java MALLET topic agreement and covariate-effect direction
- topica STM vs R `stm` statistical agreement
- topica STM content model vs R `stm`
- topica keyATM vs R `keyATM`

## Where We Are

I would now describe the library as reference-validated for:

- `LDA`
- `LabeledLDA`
- `DMR`
- `STM`
- STM content/SAGE path
- base `KeyATM`

With stronger deterministic support added for:

- STM helper functions
- MALLET state import
- SeededLDA seed-prior behavior
- GSDMM Movie Group Process formulas

I would not yet claim the whole non-embedding library has equally strong
external parity. The next work should focus on:

- R `keyATM` covariate and dynamic parity
- CTM-as-STM or CTM-vs-reference parity
- optional R `seededlda` parity if the package is available
- stronger tests for HDP, DTM, SupervisedLDA, PT, PA, HLDA, and LightLDA

The detailed checklist is in `notes/model-parity-validation-todo.md`.
