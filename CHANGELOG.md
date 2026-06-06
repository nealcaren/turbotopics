# Changelog

All notable changes to topica are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once released.

## [Unreleased]

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
