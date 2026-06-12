# Changelog

All notable changes to topica are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once released.

## [Unreleased]

### Added

- `LDA(..., init="spectral")` seeds the initial token-topic assignment from a
  deterministic anchor-word topic-word matrix (the same spectral recovery STM
  and CTM use) instead of a uniform random draw. It does not speed convergence,
  but it improves topic coherence at larger K (a robust +2 to +3 mean-coherence
  points across seeds at K=50 and K=100 on the poliblog corpus; a wash at small
  K), the fine-grained regime where the sparse sampler already pays off. It
  falls back to the random draw when the corpus is too small for anchor
  recovery. The default stays `init="random"`, so MALLET byte-parity and
  same-seed determinism are unchanged.

### Changed

- The collapsed-Gibbs samplers (LDA, DMR, LabeledLDA, SeededLDA, KeyATM, PA, PT,
  HDP, GSDMM, SAGE) now draw from a fast non-cryptographic PRNG (PCG) instead of
  ChaCha8. Gibbs sampling needs uniform draws, not cryptographic entropy, and PCG
  is faster: single-threaded, HDP is ~2x faster, LDA ~10% and keyATM ~9% (#67).
  Fits remain reproducible from a fixed seed, but **the random stream changed**,
  so a given seed now yields different (still-deterministic) topics than in
  0.15.0; pin a topica version if you need to reproduce an earlier fit exactly.
  The variational models (CTM, STM, STS, DTM, supervised LDA, ETM, ProdLDA,
  FASTopic) and the embedding-cluster models are unchanged.

### Fixed

- `HDP` no longer runs away to hundreds of topics on real corpora (#68). The
  concentration resampler was a positive-feedback loop: the Escobar-West update
  draws `gamma` from `Gamma(a + K, ...)`, whose mean grows with the topic count
  `K`, so more topics raised `gamma`, which created more topics, irreversibly
  (K reached 774 with gamma at 102 over 800 sweeps on a 3,500-document corpus).
  `resample_conc` now defaults to `False` (fixed concentrations give a stable,
  reproducible topic count; `gamma` sets the granularity directly), and the
  opt-in resampling path caps the concentrations so it stays bounded. Default
  concentrations remain `alpha=0.1`, `gamma=0.1` (the reference convention).

## [0.15.0] - 2026-06-10

This release completes the structural-topic-model and keyATM drop-in parity work
and rounds out the model-agnostic effect-estimation surface. It also moves the
heavy CI (wheels, sdist) to release tags and builds the test job optimized, so a
normal push runs only the fast test suite.

### Added

- `permutation_test(model, covariate, ...)` for a binary prevalence covariate: a
  distribution-free check on whether a topic's prevalence differs across the two
  groups, returning a `PermutationResult` per topic (#36).
- `select_model` / `plot_models`: fit N models at a fixed K under different seeds
  and pick the best by a held-out or coherence criterion, mirroring R `stm`'s
  `selectModel`; returns a `SelectModelResult` (#37).
- `prep_documents` / `plot_removed`: R `stm`-style preprocessing diagnostics that
  report how many documents, words, and tokens each vocabulary threshold removes,
  with metadata re-alignment via the `Corpus`'s kept indices (#41).
- A uniform convergence interface on every iterative model: `model.fit_history`
  (per-iteration `(iter, objective)`) and `model.converged`. The collapsed-Gibbs
  models gained an opt-in early stop (`convergence_tol` / `check_every`, default
  off so the full `iters` run is bit-for-bit unchanged); the variational models
  trace and early-stop on the ELBO (#46).
- `prevalence_ci(model, groups, ...)`: model-neutral per-group topic-prevalence
  credible bands read directly from a model's posterior theta draws (the
  draws-based companion to `by_strata`). `time_prevalence_ci(model, timestamps)`
  is the dynamic-keyATM wrapper that pins the period order to `time_labels`, so
  the dynamic time trend now carries the HMM posterior's own uncertainty rather
  than a generic ribbon (#42).
- Covariate-aware `stm.transform(model, docs, prevalence=/formula=/X=)`: held-out
  topic inference that builds each new document's prior from its covariates and
  the fitted `gamma` (`mu_d = X_d gamma`), matching R `stm`'s `fitNewDocuments`.
  A model-neutral `align_corpus(new_docs, model)` maps new tokens onto the fitted
  vocabulary (dropping out-of-vocabulary tokens) before transform (#39).
- `STM.fit(gamma_prior="pooled"|"l1", gamma_enet=...)`: an L1/elastic-net prior on
  the prevalence coefficients, fit by coordinate descent with an AIC-selected
  penalty, for high-dimensional prevalence designs (a factor with many levels).
  `"pooled"` (ridge, the default) is unchanged; `gamma_enet` is the elastic-net
  mix (R `stm`'s `gamma.enet`) (#40).

### Fixed

- `search_k(held_out=...)` now composes with a `make_heldout` split: it dispatches
  on the `Heldout` type and reports the held-out log-likelihood, instead of
  raising a `TypeError` from the legacy perplexity path (#55).

### Changed

- CI: `build-wheels` and `sdist` run only on release tags (`v*`) and manual
  dispatch, not on every push/PR; the wheels are consumed only by the release
  job. The 3-platform test job still runs on every push/PR and now builds with
  `--release` plus a cached Rust toolchain, cutting the test legs from roughly
  twenty minutes to a few. Committed tests no longer assume a macOS-only
  `/private/tmp`.

## [0.14.0] - 2026-06-10

This release makes the estimator interface uniform across the whole library and
adds two publication-grade quantities of interest. Every estimator now meets a
documented contract, checked in CI.

### Added

- `predicted_prevalence(model, ...)`: predicted topic prevalence at chosen
  covariate values, with difference contrasts and continuous prediction curves,
  and simulation-based confidence intervals. Model-agnostic (STM, CTM, the
  covariate keyATM, LDA, ...), built on the method-of-composition draws, so it is
  the same call regardless of model family. A `viz.predicted_prevalence_plot`
  renders the forest and curve figures (#35, #43).
- `make_heldout` / `eval_heldout`: R `stm`-style document-completion held-out
  log-likelihood, model-agnostic via each model's `transform`; `search_k` now
  reports a held-out metric for STM/CTM, not only LDA (#38).
- Estimator conformance facility: `topica.check_conformance(model)`, a
  registry-driven `tests/test_conformance.py`, and a contributor contract at
  `docs/contributing/estimator-contract.md`. New estimators that drop part of
  the contract fail CI.
- `theta_draws` and `doc_lengths` on the remaining Dirichlet models (DMR, SAGE,
  PA, PT, HDP, LabeledLDA, SupervisedLDA), so `composition_theta`,
  `standard_errors`, and `predicted_prevalence` work for them with no `corpus=`
  re-thread. SupervisedLDA draws from its variational Dirichlet posterior.
- Held-out `transform` on KeyATM, SeededLDA, SAGE, PA, and PT, so held-out
  perplexity, `eval_heldout`, and out-of-sample inference now work for the
  keyword, seeded, and anchored models.
- Settable `topic_names` on every estimator (default `["topic_0", ...]`).
- `coherence`, `save`/`load`, and `doc_names` on the neural and cluster models
  (ETM, FASTopic, ProdLDA, BERTopic, Top2Vec) where they were missing.

### Changed

- **Breaking:** the fit iteration count is the canonical keyword `iters` on every
  estimator (previously `iterations` for the collapsed-Gibbs models and
  `em_iters` for the variational ones); `search_k` likewise takes `iters`. No
  deprecation aliases.
- **Breaking:** ETM, ProdLDA, and FASTopic take the training length as
  `fit(iters=...)` rather than a constructor `epochs` / `em_iters` argument.

## [0.13.0] - 2026-06-10

### Added

- The Gibbs/Dirichlet models (`LDA`, `KeyATM` base/covariate/dynamic,
  `SeededLDA`) retain thinned post-burn-in MCMC document-topic draws as
  `model.theta_draws` (shape `(num_draws, num_docs, num_topics)`, f32). On by
  default (`keep_theta_draws=True`, `num_theta_draws=25`); pass
  `keep_theta_draws=False` to skip the store. `composition_theta` (and
  `standard_errors` / `estimate_effect` with `method="composition"`) prefers
  these real cross-sweep posterior samples over the within-document Dirichlet
  approximation, and needs no `corpus=` when they are present. Retention rides
  on sweeps that already run, so it adds negligible fit time (#31).
- The same models expose `model.doc_lengths` (per-document token counts, in
  `doc_topic` row order), so the Dirichlet-approximation fallback is also
  self-sufficient: `composition_theta(model)` works without re-threading the
  `Corpus`, even with `keep_theta_draws=False`. Passing `corpus=` still takes
  precedence (#32).

### Changed

- Standard errors for the Gibbs models now reflect genuine topic-estimation
  uncertainty (the cross-sweep posterior variance of theta), which grows when
  topics overlap and shrinks when the model is confident. Values therefore
  differ from 0.12.1, where the Dirichlet approximation added length-only
  `1/N_d` sampling noise regardless of identifiability; the new intervals can be
  wider or narrower depending on the corpus. Fit with `keep_theta_draws=False`
  to recover the prior approximation behavior.

## [0.12.1] - 2026-06-08

### Fixed

- The `viz` topic-correlation panel (`TopicCorrelation`, and the correlation
  sub-panel of `plot_report`) masks its always-1.0 diagonal instead of drawing
  it. The self-correlation carried no information yet saturated the diverging
  color scale and visually dominated the panel; the diagonal now renders as a
  neutral background so the off-diagonal structure reads on a scale set by the
  strongest real correlation. `to_frame()` is unchanged and still reports the
  true diagonal.

### Docs

- Paper: added `ProdLDA` to the model-family table (count-based, with its
  bibliography entry), and a worked example that aligns two model families and
  compares their covariate effects with method-of-composition uncertainty.

## [0.12.0] - 2026-06-08

### Added

- `alpha` getter on the collapsed-Gibbs Dirichlet models that lacked it —
  `KeyATM`, `SeededLDA`, `LabeledLDA`, `SupervisedLDA`, `DMR`, `PA`, `PT`, and
  `SAGE` — returning the per-topic document-topic Dirichlet prior aligned with
  `doc_topic`'s columns (the estimated/asymmetric prior where one is fitted, the
  symmetric prior otherwise, and `exp(lambda_intercept)` for `DMR`'s
  per-document prior). This is what `effects.model_family` keys "dirichlet" off,
  so it is the mechanism behind the `composition_theta` fix below (#20, #21).

### Fixed

- `effects.model_family` misclassified every collapsed-Gibbs model except `LDA`
  and `HDP` as `"none"`, so `composition_theta` raised for them and `viz`
  effect/uncertainty panels silently fell back to point estimates. With `alpha`
  now exposed, `KeyATM`, `SeededLDA`, `LabeledLDA`, `SupervisedLDA`, `DMR`, `PA`,
  `PT`, and `SAGE` are correctly `"dirichlet"`; `GSDMM` stays `"none"` by design
  (a Dirichlet mixture, not an admixture) (#20, #21).
- `dirichlet_theta_samples` double-counted the symmetric prior on the `prior > 0`
  path, biasing draws toward uniform; the default `prior = 0` path is unchanged
  (#26).
- `find_thoughts` and `document_intrusion` (and `representative_docs` /
  `topic_info` through them) now raise on a `texts` / `doc_topic` length
  mismatch, the guard their siblings already had, so a document dropped by
  vocabulary pruning can no longer be returned in place of a real one;
  `plot_report`'s per-class panel gets the same alignment check (#24).
- Stopped swallowing exceptions that quietly degraded results: bootstrap refits
  and held-out `transform` now choose their call arity by inspecting the
  signature instead of treating any `TypeError` as an arity mismatch (which had
  re-run every resample at the default seed); `quality_frontier` warns when a
  windowed `coherence_type` is requested without `texts`; `plot_report` warns and
  names any panel it drops; the top-words fallback warns before discarding custom
  (e.g. FREX) weighting (#25).
- API-surface drift: the `DMR` type stub (copied from `STM`) now matches the real
  `fit(data, features, ...)` signature and exposes `feature_effects` (not the
  nonexistent `prevalence_effects`); `coherence` and the analysis surface work
  for `SAGE` via its group marginal and reject `DTM`'s time-sliced `topic_word`
  with a clear message; the `viz` capability descriptor marks `HLDA` and `DTM`
  (no usable `doc_topic`) as not soft-theta; `bootstrap_stability` accepts a
  `Corpus`, as its docstring promised (#27).

## [0.11.0] - 2026-06-07

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

### Fixed

- Input validation hardened against adversarial edge cases:
  - Non-finite float hyperparameters (`NaN`/`Inf` for `beta`, `alpha`,
    `prior_variance`, `chain_variance`, `eta`, `alpha_sum`, and the rest) are now
    rejected at construction instead of silently producing a `NaN` fit.
  - A corpus with no words — all documents empty, or everything pruned by frequency
    filtering — is rejected at fit instead of yielding a degenerate `(K, 0)` model.
  - `coherence` / `topic_diversity` raise a clear error on a non-integer `topn` or a
    raw `topic_word` matrix, and `coherence` errors on an empty reference corpus
    instead of returning `NaN`.
  - `frex` rejects frequency weights outside `[0, 1]`.
- `coherence` / `topic_diversity` now accept any object satisfying the analysis
  contract (`topic_word` + `vocabulary`): top words are derived from the matrix when
  the model exposes no `top_words` method.

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
