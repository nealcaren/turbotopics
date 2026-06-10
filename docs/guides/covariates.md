# Covariates & STM

The Structural Topic Model lets topics depend on document metadata in two ways:
**prevalence** (how much a document discusses each topic) and **content** (how a
topic is worded). For the publication-grade version of this workflow, with proper
uncertainty and clustered errors, see [Measure effects properly](../publishing/effects.md).

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
from topica import stm

draws = stm.posterior_theta_samples(model, nsims=50, seed=0)
effects = stm.estimate_effect(
    draws, X, feature_names=names,
    cluster=source_id,     # cluster-robust SEs for nested data
    # link="logit",        # keep predictions in [0, 1]
)
for e in effects:
    print(e.as_dict())
```

Build non-linear and interaction terms with `stm.spline` and `stm.interaction`.
Full detail and the journal-grade treatment are in the
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
