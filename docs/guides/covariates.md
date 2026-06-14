# Covariates & STM

The Structural Topic Model lets topics depend on document metadata in two ways:
**prevalence** (how much a document discusses each topic) and **content** (how a
topic is worded). For the publication-grade version of this workflow, with proper
uncertainty and clustered errors, see [Measure effects properly](../publishing/effects.md).

!!! tip "Importing the covariate helpers"
    All of the design-matrix and effect helpers are top-level: `topica.one_hot`,
    `topica.design_matrix`, `topica.spline`, `topica.interaction`,
    `topica.estimate_effect`, and `topica.posterior_theta_samples`. That is the
    canonical path used throughout these docs. The same names are also reachable
    under `topica.stm.*` (they are the identical objects, kept as a compatibility
    alias), but prefer the top-level form.

## End to end: from a DataFrame to effects

The whole covariate workflow in one block: build an aligned corpus from a
DataFrame, turn the metadata into a design matrix with an R-style formula, fit
the STM, and read the effects as a tidy table. Every step uses the canonical
top-level helpers.

```python
import pandas as pd
import topica

# df has columns: text, party, year
corpus = topica.from_dataframe(df, text_col="text")     # metadata kept aligned

# Design matrix from a formula (needs the optional topica[formula] extra).
# corpus.metadata is the surviving rows, already aligned to the documents.
X, names = topica.design_matrix("~ party + spline(year, df=3)", corpus.metadata)

# Pick K with a safe, direction-aware selector, then fit at that K.
scan = topica.search_k(corpus, [10, 20, 30], model="stm", prevalence=X, iters=200)
model = topica.STM(num_topics=scan.best_k(), seed=1)
model.fit(corpus, prevalence=X, prevalence_names=names)

# Effects with method-of-composition uncertainty, as a tidy long table.
draws = topica.posterior_theta_samples(model, nsims=50, seed=0)
effects = topica.estimate_effect(draws, X, feature_names=names)
table = pd.concat([e.to_frame() for e in effects], ignore_index=True)
```

If you would rather not add the `formulaic` dependency, replace `design_matrix`
with hand-built blocks: `X, names = topica.one_hot(df["party"])` combined with
`topica.spline` / `topica.interaction` via `numpy.hstack`.

## Prevalence covariates

```python
import topica

X, names = topica.one_hot(party)                      # design matrix + column names
model = topica.STM(num_topics=20, seed=1)
model.fit(docs, prevalence=X, prevalence_names=names)

model.prevalence_effects        # learned γ
topica.topic_correlation(model.doc_topic)
```

## Content covariates

A content model makes the topic-word distribution vary by group (the SAGE
mechanism), so the same topic is phrased differently across, say, conservative
and liberal sources:

```python
model = topica.STM(num_topics=20, seed=1)
model.fit(docs, prevalence=X, content=source, content_names=groups)

model.topic_word_by_group        # per-group β
# words that most distinguish how a topic is worded across two groups:
# model.word_contrast(topic, "liberal", "conservative")
```

## Estimating effects

Regress topic proportions on covariates with honest uncertainty, using the method
of composition, optionally with clustered standard errors and GLM links:

```python
import pandas as pd
import topica

draws = topica.posterior_theta_samples(model, nsims=50, seed=0)
effects = topica.estimate_effect(
    draws, X, feature_names=names,
    cluster=source_id,     # cluster-robust SEs for nested data
    # link="logit",        # keep predictions in [0, 1]
)

# One tidy row per (topic, feature): coef, se, z, ci_low, ci_high, r_squared
table = pd.concat([e.to_frame() for e in effects], ignore_index=True)
```

Build non-linear and interaction terms with `topica.spline` and
`topica.interaction`. Full detail and the journal-grade treatment are in the
[Publishing](../publishing/effects.md) track.

## Predicted prevalence

`topica.predicted_prevalence` computes predicted topic prevalence at chosen
covariate values with simulation-based credible intervals — the direct
counterpart of R `stm`'s `plot.estimateEffect`. Three modes mirror `stm`'s
`method` argument:

```python
import topica

# Point grid: predicted prevalence when party is "D" vs "R"
pp = topica.predicted_prevalence(
    model,
    formula="~ party + year",
    data=meta,
    at={"party": ["D", "R"]},
)
for result in pp:
    print(result.topic_name, result.estimate, result.ci_low, result.ci_high)

# Continuous sweep: prevalence as year varies, other covariates at their means
pp = topica.predicted_prevalence(
    model, formula="~ party + year", data=meta,
    continuous="year",
)

# Contrast: difference in prevalence between two covariate settings
pp = topica.predicted_prevalence(
    model, formula="~ party", data=meta,
    contrast={"party": ["D", "R"]},
)
```

## Permutation test for binary covariates

`topica.permutation_test` assesses whether a binary covariate genuinely shifts
topic prevalence, using permutation resampling rather than parametric
assumptions:

```python
results = topica.permutation_test(
    model, corpus=docs, covariate=treated,   # treated: 0/1 array
    n_perm=100, seed=0,
)
for r in results:
    print(f"Topic {r.topic_name}: observed={r.observed:.3f}  p={r.pvalue:.3f}")
```

Each `PermutationResult` carries the observed covariate effect, the full null
distribution (`r.null`), and a two-sided p-value.

## L1/elastic-net prior for high-dimensional designs

When the prevalence design matrix has many columns (many dummies, interaction
terms, or a wide feature matrix), add `gamma_prior="l1"` to `STM.fit` to
penalize the prevalence coefficients:

```python
model = topica.STM(num_topics=20, seed=1)
model.fit(
    docs, prevalence=X, prevalence_names=names,
    gamma_prior="l1",     # elastic-net with full L1 (lasso)
    gamma_enet=1.0,       # alpha=1.0 is pure L1; 0 < alpha < 1 mixes L2
)
```

The default `gamma_prior="pooled"` uses the OLS pooled regression from the
original STM paper. Use `"l1"` when p (number of prevalence covariates)
approaches or exceeds the number of documents.

## Choosing K for STM

Use `search_k`, the coherence×exclusivity frontier, and an `HDP` sanity check.
See [Choose and justify K](../publishing/choosing-k.md).
