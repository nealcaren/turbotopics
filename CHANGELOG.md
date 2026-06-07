# Changelog

All notable changes to topica are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once released.

## [Unreleased]

### Added

- `topica.viz` — four more panels, continuing the toolkit's deferred roadmap:
  - `topic_health` — flags **dead** topics (expected mass share below
    `min_mass_frac`) and **near-duplicate** topics (φ-cosine above `dup_threshold`),
    off the same `topic_sizes` / topic-word surfaces the rest of the toolkit uses.
    Essential for honest reporting and for HDP, which returns many near-zero-mass
    topics by construction.
  - `prevalence_heatmap` — a groups × topics heatmap of mean topic prevalence
    (`by_strata`), with method-of-composition intervals in `.to_frame()` when a
    corpus and `nsims` are given.
  - `topics_over_time` — per-topic prevalence trajectories as small multiples (the
    readable replacement for a streamgraph), with optional method-of-composition CI
    ribbons.
  - `topic_correlation` — the honest, closure-corrected correlation layer
    (`clr` / `partial` / η-space `eta` / labeled-biased `raw`), drawn as a
    zero-centered diverging heatmap; refused for hard/degenerate-θ cluster models.
  - `dashboard()` now assembles these by introspection: topic-health always, the
    group heatmap with `groups=`, the time small-multiples with `timestamps=`, and
    the correlation layer for soft-θ models.
- `topica.project(data, n_components=2, method=...)` — a numpy-native projection
  primitive backed by topica's own Rust core: `"pca"` (default, deterministic,
  distance-faithful), `"umap"` (`umap-rs`), or `"tsne"` (new **`bhtsne`** Barnes-Hut
  reducer, pure Rust). UMAP and t-SNE warn that they are non-metric and not
  reproducible. No Python UMAP/sklearn dependency.
- `topica.viz.document_map` — the deferred 4th panel: a 2-D projection of the
  *document* cloud (a supplement figure). Coordinates come from the document
  embeddings you pass, or, for a count/soft-θ model, the clr-transformed θ simplex;
  a hard-θ cluster model with no embeddings is refused. PCA reports variance
  explained; UMAP/t-SNE carry the non-metric caveat and the seed. Density via alpha
  clouds / hexbin (never convex hulls), Okabe–Ito palette for small K else
  gray-all + `highlight_topic=`, a separate `-1` outlier layer, and stratified
  subsampling with a "showing N of D" badge. `dashboard(..., doc_embeddings=)` adds
  it.
- `topica.viz.document_inspector` — read one document the way the model read it: its
  θ mixture, its words shaded by attributed topic (`argmax_t p(t | w, d)` from θ and
  φ, so it needs no per-token assignments), and the `find_thoughts` neighbors of its
  dominant topic. Refused for hard/degenerate-θ cluster models.
- `topica.viz.content_covariate` — for an STM/SAGE content model, one topic's wording
  across covariate groups as a words × groups `p(w | topic, group)` heatmap (the
  union of each group's top words), surfacing the per-group distribution instead of a
  reference snapshot. `.contrast(...)` wraps the model's `word_contrast`. Refused for
  a model fit without a content covariate.
- `dashboard()` adds the content-wording panel for content models, and the inspector
  when `inspect_doc=` is given. The generic panels now collapse a content model's
  per-group (K, G, V) topic-word to its marginal, and the dashboard assembles every
  panel best-effort (a model that cannot support one is skipped, not fatal).

### Changed

- The interactive (`.to_html()`) backend is now **Plotly only**; the Altair
  dependency is dropped. `term_topic_browser` (a seriated heatmap plus a topic
  dropdown) and the dashboard report render with Plotly (WebGL), the same stack as
  the document map.
- Packaging simplified: the static `viz` and interactive `viz-interactive` extras
  are **merged into one `viz` extra** (matplotlib, pandas, scipy, plotly), and a new
  **`all`** extra installs everything in one shot. The base install stays
  `numpy`-only.
- `viz` design polish (from two independent expert reviews): the topic-similarity
  heatmap anchors its color scale at 0 (no contrast-stretch) and labels the colorbar
  `1 − <metric>`; the covariate effect plot drops sign-coded red/blue for a single
  neutral color (position already encodes sign); heatmaps share `SEQ`/`DIV` colormap
  constants; the coherence frontier gains a prevalence size legend; `topics_over_time`
  shares its y-axis by default; `search_k` is faceted (one metric per panel) instead
  of a triple twin-axis.

### Fixed

- CTM/STM expose `topic_covariance` (the fitted logistic-normal prior Σ over η,
  shape (K−1, K−1)), and `viz.topic_correlation(model, method="eta")` now uses it —
  the model's own covariance rather than an empirical re-correlation of η posterior
  means, which it had been mislabeling as "the model's covariance."
- `viz.term_barchart` FREX / relevance / score modes no longer crash on a SAGE
  content model (they now route through the group-averaged marginal, as `prob`/`lift`
  already did); the descriptor advertised these modes but they raised.
- `viz.dashboard` records skipped panels in `.skipped` and warns, instead of
  silently swallowing every failure (so a real error is visible, not indistinguishable
  from "not applicable").
- `find_thoughts` uses `argpartition` for the top-n (O(D)) rather than a full sort.
- The document map no longer prints a `seed=` for UMAP/t-SNE (neither fit is
  reproducible), and the docs no longer claim the interactive browser links a
  heatmap click to the barchart (it is a dropdown).

## [0.10.0] - 2026-06-06

### Added

- `AGENTS.md` — a working guide for LLM agents (Claude Code, Cursor, …) helping a
  social scientist run topica. It maps the API onto the text-analysis workflow
  (question → corpus → choose K → fit → validate → measure effects → report) with
  explicit handoffs, and draws the line on what the researcher owns (the question,
  K, topic labels, covariate choice, whether a result matters) versus what topica
  and the agent supply (mechanics, honest diagnostics, refusal to fabricate
  uncertainty).
- `topica.viz` — a manuscript-first visualization toolkit (the honest successor to
  pyLDAvis). Each view is a panel with `.to_frame()` (the numbers, always),
  `.to_png()` (matplotlib, for papers), and `.to_html()` (Altair, for the
  interactive subset). Panels read a per-model capability descriptor and switch
  their statistics/labels on it: c-TF-IDF `topic_word` disables the FREX/lift
  modes and is labeled as such, effect-plot CIs are refused where there is no θ
  posterior (and ghosted where the bootstrap flags a topic unreliable), and
  uncertainty is labeled for what it is. Panels: `coherence_frontier`, `search_k`,
  `effect_plot`, `term_barchart`, `topic_similarity` (a seriated K×K heatmap, the
  pyLDAvis replacement), `term_topic_browser` (linked interactive), and a
  `dashboard()` composite. New extras: `topica[viz]` (matplotlib/pandas/scipy) and
  `topica[viz-interactive]` (altair).

## [0.9.0] - 2026-06-06

### Added

- `topica.mmr(model, word_embeddings, diversity=...)` — maximal-marginal-relevance
  top words: rerank a topic's candidate words to cut redundant near-synonyms,
  balancing `topic_word` relevance against word-embedding similarity (BERTopic's
  `MaximalMarginalRelevance`). Accepts a model or a `(K, V)` matrix.
- `save` / `load` for the embedding-cluster models (`BERTopic`, `Top2Vec`), so a
  discovered fit can be frozen and reloaded — the way to keep a good (stochastic)
  UMAP discovery fit, since the prediction phase is deterministic. The loaded
  model's `transform` reproduces the original.
- `topica.add_ngrams(docs, ngram_range=(1, 2), min_df=...)` — expand pre-tokenized
  documents with contiguous n-grams (the mechanical analog of scikit-learn's
  `CountVectorizer(ngram_range=, min_df=)`), so an embedding model's c-TF-IDF topic
  words can include bigrams. Keeps every document, so it stays aligned with
  per-document embeddings. The exhaustive complement to `learn_phrases`.
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
