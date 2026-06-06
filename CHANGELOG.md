# Changelog

All notable changes to topica are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once released.

## [Unreleased]

### Added

- `reducer="umap"` now ships in the wheel for `BERTopic` / `Top2Vec` (opt-in at
  runtime, no special build). PCA stays the default. The UMAP discovery fit is not
  reproducible (the `umap-rs` optimizer's negative sampling is unseeded) and emits
  a warning saying so; following BERTopic's fit-vs-predict split, the prediction
  phase is deterministic regardless — `transform` never re-runs the reducer — so a
  fitted model still maps documents reproducibly. Use `reducer="pca"` for a fully
  reproducible fit, or `clusterer="kmeans"` to empty the `-1` bucket deterministically.
- `topica.diagnostics(model, texts)` — a one-call per-topic table (coherence,
  exclusivity, FREX, size, prevalence, top words, and optional bootstrap
  stability) as a pandas DataFrame, consolidating the scattered quality
  functions. It reads a model's analysis surface, so it works for every model
  and sidesteps the model-vs-matrix first-argument friction.
- `topica.perplexity(model, held_out)` — model-agnostic document-completion
  held-out perplexity (infer each held-out document's mixture from half its
  tokens, score the other half), a K-comparable signal for justifying a topic
  count across the generative models. (`LDA` keeps its rigorous left-to-right
  estimator as `LDA.perplexity` / `LDA.evaluate`.)
- `bootstrap_stability(..., reference=model)` measures stability of an
  already-fitted model's topics (matching resamples back to it) rather than a
  fresh full-corpus fit.

### Changed

- The post-hoc analysis module moved from `topica.diagnostics` to
  `topica.validation`, freeing the verb-like `diagnostics` name for the new
  one-call function. Its helpers stay importable (`from topica import
  validation`) and every function remains available top-level (`topica.frex`,
  `topica.coherence`, …).

## [0.8.0] - 2026-06-06

### Added

- `topica.standard_errors(model, corpus, of=..., method=...)` — one entry point
  for uncertainty on the quantities people publish (#15). `method="composition"`
  (default) auto-detects the model family, draws the right θ posterior
  (logistic-normal for STM/CTM, Dirichlet for the Gibbs models), and pools by
  Rubin's rules for `of="effect"`/`"prevalence"`. `method="bootstrap"` refits on
  resampled documents for `of="top_words"` and the embedding models, matching
  topics across refits and reporting `alignment_quality`/`alignment_margin` so it
  can flag and suppress SEs where the matching is unstable (split/merge or
  indistinct topics).
- `Corpus.doc_lengths` — per-document token counts in the pruned vocabulary,
  parallel to a model's `doc_topic` rows (needed by `dirichlet_theta_samples`).
- `estimate_effect` and `by_strata` now accept the fitted model directly and draw
  θ internally (with `corpus=`/`nsims=`), so the sampler no longer has to be
  wired by hand. `topica.model_family(model)` exposes the detection.

## [0.7.1] - 2026-06-06

### Added

- `BERTopic` and `Top2Vec` accept `clusterer="kmeans"` / `"agglomerative"` with
  `num_clusters=K`, a swappable alternative to the default HDBSCAN that assigns
  every document to a cluster (no `-1` noise bucket) (#7).
- `topica.report(model)` is now a callable one-line overview (an alias for
  `summary`), so the natural `report(model)` call works instead of raising
  `'module' object is not callable` (#12).
- A bundled `text -> llm_embed -> BERTopic` example and an `llm_embed`
  cross-reference in every embedding model's docstring (#5).

### Changed

- `Top2Vec.top_words()` now returns the centroid representation (vocabulary
  nearest the cluster centroid) by default when fit with `word_embeddings`, so
  its headline output is distinct from `BERTopic`'s shared c-TF-IDF; pass
  `representation="c-tf-idf"` for the shared view. `topic_neighbors` is now
  `(topic, *, n=10)`, so `topic_neighbors(0, n=8)` reads naturally (#8).
- `frex`, `label_topics`, `relevance`, `topic_correlation`, and `find_thoughts`
  now accept a fitted model or the raw matrix as the first argument (vocabulary
  derived from the model when omitted), matching `exclusivity` and the intrusion
  tests; a bare matrix with no vocabulary raises a clear message (#10).
- The model-neutral analysis surface moved from the `topica.report` module to
  `topica.analysis` (its functions remain available top-level, e.g.
  `topica.topic_info`, `topica.plot_report`), freeing the verb-like `report`
  name for the new callable (#12).

### Fixed

- A negative count (`num_topics`, `num_pseudo`, `num_super`, `num_sub`, `depth`)
  now raises a clean `ValueError` instead of leaking PyO3's
  `OverflowError: can't convert negative int to unsigned` (#13).

## [0.7.0] - 2026-06-06

### Added

- Embedding-based models: `BERTopic` and `Top2Vec` (embedding-clustering pipeline,
  class-based TF-IDF, `merge_topics` / `reduce_outliers`), `ETM` (per-document
  variational EM and an amortized VAE inference path via `inference="vae"`), and
  `FASTopic` (optimal transport, a hand-coded reverse-mode Sinkhorn).
- Model-neutral analysis surface (`topica.report`, `topica.effects`), including
  `plot_report` — a one-figure model overview — and `topic_info` /
  `topics_over_time` / `topics_per_class`.
- LLM topic labeling and embeddings as plumbing: `llm_topic_labels`,
  `topic_label_prompts`, `llm_backend`, and `llm_embed` (with caching via
  `save_embeddings` / `load_embeddings`). The core takes any callable; an optional
  `topica[llm]` extra adds the `llm` library and the ollama plugin.
- Polars support: `from_dataframe`, `align`, and `design_matrix` accept Polars
  frames alongside pandas.
- A Citing page collecting per-model references, a `LICENSE` file, `CITATION.cff`,
  `CONTRIBUTING.md`, and this changelog.

### Validated

- R-parity checks for the keyATM covariate and dynamic models and for `CTM`
  (as `stm` with no covariates), alongside the existing base keyATM and STM checks.
