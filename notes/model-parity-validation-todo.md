# Model Parity Validation TODO

Goal: get every non-embedding model to the strongest defensible validation
level without changing model code. Use exact parity where deterministic
contracts exist, external statistical parity where independent samplers or
optimizers differ, paper-formula contract tests where no maintained reference
runner is practical, and replication-style tests for complex non-convex models.

## Tier 1: Already Strong, Tighten Edges

- [ ] `LDA`: add exact MALLET state and diagnostics parity.
  - [x] Compare MALLET `--output-state` loaded by `LDA.load_state`.
  - [x] Compare exact state-derived topic-word formulas.
  - [x] Compare overlapping MALLET diagnostics XML fields to `LDA.diagnostics()`.
  - [x] Keep default-contract differences documented.
  - [ ] Decide whether to expose MALLET-compatible diagnostic scales for fields
    that currently differ (`rank_1_docs`, coherence, and related XML-only fields).
- [ ] `LabeledLDA`: add Java MALLET small-corpus count/state checks if practical.
  - [ ] Pin label order.
  - [ ] Pin topic-label alignment.
  - [ ] Pin smoothing formula behavior.
- [ ] `DMR`: strengthen Java MALLET comparison.
  - [ ] Check feature-name/order contracts.
  - [ ] Check coefficient rank/sign consistency.
  - [ ] Compare predictions on a fixed covariate grid.

## Tier 2: R Package Families

- [x] `STM`: add deterministic helper parity.
  - [x] R `stm::labelTopics` vs `topica.stm.label_topics`.
  - [x] R FREX calculations vs `topica.stm.frex`.
  - [x] R `topicCorr` vs `topica.stm.topic_correlation`.
  - [x] R `estimateEffect`-style regressions vs `topica.stm.estimate_effect` on fixed matrices.
- [x] `CTM`: validate as STM without covariates against R `stm`. (`parity/ctm_r_compare.py`.)
  - [x] Statistical topic alignment. (Spectral-matched cosine 0.967 vs R's own 0.778 seed spread.)
  - [x] Bound monotonicity. (topica bound increases at 100% of EM steps.)
  - [ ] Held-out transform check (optional follow-up).
- [ ] `SAGE` / STM content path: expand current STM content parity.
  - [ ] Multi-group content effects.
  - [ ] Group ordering.
  - [ ] Content distribution normalization.
- [x] `KeyATM`: expand R `keyATM` parity beyond the base model.
  - [x] Base model. (`parity/keyatm_r_compare.py`: keyword cosine vs R within R's seed spread.)
  - [x] Covariate model. (`parity/keyatm_models_r_compare.py`: keyword phi 0.896 vs R-self 0.930.)
  - [x] Dynamic model. (same file: keyword phi 0.903; chronological time index.)
  - [x] Weighted LDA. (`KeyATM.weighted_lda`; token weighting validated against keyATM.)
  - [x] Output analogs: `model_fit`, `pi`, `alpha`, topic order, time labels, covariate
    coefficient signs (rating effect sign agreement 1.00; time-trend sign agreement 1.00).

## Tier 3: Guided And Short-Text Models

- [ ] `SeededLDA`: add live R `seededlda` parity if installed.
  - [ ] Compare seeded-topic top words.
  - [ ] Compare seed prior/default contracts.
  - [x] Pin exact seeded-prior formula and seeded initialization contracts.
- [ ] `GSDMM`: use paper-formula plus optional Python/R reference package if available.
  - [x] Pin Movie Group Process output formulas.
  - [x] Pin trace likelihood and effective cluster count formulas.
  - [ ] Add document-removal sampling probability checks on a hand-built state if exposed.
  - [ ] Add planted short-text statistical recovery benchmark.

## Tier 4: Long-Tail Probabilistic Models

- [ ] `HDP`: compare to `tomotopy.HDPModel` if available.
  - [ ] Discovered K.
  - [ ] Top-word recovery.
  - [ ] Concentration trace sanity.
  - [ ] Chinese restaurant franchise formula checks if exposed.
- [ ] `DTM`: compare to `tomotopy.DTModel` or `gensim` LdaSeq where available.
  - [ ] Trend direction over time.
  - [ ] Time-slice topic-word normalization.
  - [ ] Smoothness under low chain variance.
- [ ] `SupervisedLDA`: compare to R `lda` package if available.
  - [ ] Regression coefficient sign/magnitude on synthetic labeled data.
  - [ ] Prediction correlation with reference package.
- [ ] `PT`: paper-formula and planted-corpus validation.
  - [ ] Pseudo-document assignment invariants.
  - [ ] Better-than-baseline behavior on short mixed-topic synthetic corpus.
- [ ] `PA`: paper-formula tests plus planted hierarchy recovery.
  - [ ] Super-topic/sub-topic matrix contracts.
  - [ ] Known DAG/co-occurrence structure recovery.
- [ ] `HLDA`: paper-formula plus hierarchy recovery.
  - [ ] Nested CRP path constraints.
  - [ ] Tree shape.
  - [ ] Level-specific vocabulary on synthetic hierarchical corpus.
- [ ] `LightLDA`: algorithm-contract tests.
  - [ ] Alias/MH proposal acceptance math on hand states.
  - [ ] Statistical comparison to sparse LDA on planted corpora.

## Execution Order

- [x] 1. Add STM helper parity.
- [x] 2. Add MALLET state/diagnostics exact tests.
- [x] 3. Add R `keyATM` covariate/dynamic parity. (`parity/keyatm_models_r_compare.py`)
- [x] 4. Add CTM-as-STM parity. (`parity/ctm_r_compare.py`)
- [ ] 5. Add SeededLDA optional R parity.
- [ ] 6. Work through HDP, DTM, sLDA, PT, PA, HLDA, and LightLDA with formula plus optional reference tests.

After each tier, update `docs/default-parity-testing-note.md` with the test name,
reference used, pass/skip/fail status, and the exact claim the test supports.
