# The models

Every model shares the same shape: construct with hyperparameters and a `seed`,
`fit(documents, ...)`, then read `topic_word` (φ), `doc_topic` (θ),
`top_words(n)`, `coherence(n)`, and `save` / `load`. Full signatures are in the
[API reference](../api/models.md).

## Choosing a model

| If you want to… | Use |
|-----------------|-----|
| Discover themes, fast and standard | [`LDA`](#lda) |
| Relate topic prevalence to metadata | [`STM`](../guides/covariates.md), [`DMR`](#dmr) |
| Let topics correlate | [`CTM`](#ctm), `STM` |
| Have topics worded differently by group | [`SAGE`](#sage), `STM` (content) |
| Let the data choose the number of topics | [`HDP`](#hdp) |
| Track topics that drift over time | [`DTM`](#dtm) |
| Tie topics to known labels | [`LabeledLDA`](#labeledlda) |
| Shape topics to predict an outcome | [`SupervisedLDA`](#supervisedlda) |
| Steer topics with known keywords | [`SeededLDA`, `KeyATM`](guided.md) |
| Model short texts (tweets, answers) | [`PT`, `GSDMM`](short-text.md) |
| Build a topic hierarchy | `PA`, `HLDA` |

## LDA

Classic Latent Dirichlet Allocation via MALLET's fast SparseLDA collapsed-Gibbs
sampler. Fits are bit-for-bit reproducible, with optional approximate
multi-threaded training.

```python
import turbotopics as tt
model = tt.LDA(num_topics=20, seed=42)
model.fit(docs, iterations=1000)
model.top_words(10)
```

### Sampler choice: SparseLDA vs LightLDA

`LDA` ships two interchangeable samplers for the *same model*, selected with
`sampler=`:

- **`"sparse"`** (default) — MALLET's SparseLDA collapsed-Gibbs sampler,
  `O(K_d + K_w)` per token. Near-optimal for the topic counts typical of social
  science; the faster choice at every scale we tested (up to `K = 600` on a
  2,000-document corpus).
- **`"lightlda"`** — the alias-table Metropolis-Hastings sampler of
  [Yuan et al. (2015)](https://arxiv.org/abs/1412.1576). It draws from cheap
  word- and document-proposal alias tables and corrects with an MH accept/reject
  step, for `O(1)` amortized work per token. This pays off only in the
  web-scale regime it was designed for — very large `K`, long documents, and
  large vocabularies — where SparseLDA's buckets stop being sparse.

```python
model = tt.LDA(num_topics=500, seed=1, sampler="lightlda", mh_steps=2)
model.fit(docs, iterations=1000)
```

Both samplers target the same posterior and recover the same topics (matched
coherence on shared corpora); LightLDA is also useful as an independent
cross-check of a SparseLDA fit. Use the default unless you have a specific
large-`K` reason not to.

## DMR

Dirichlet-Multinomial Regression: each document's topic prior depends on its
metadata, `α_d = exp(Xγ)`. The learned `feature_effects` show how covariates
shift topic propensity.

```python
import numpy as np
X, names = tt.one_hot(party)
model = tt.DMR(num_topics=20, seed=1)
model.fit(docs, X, feature_names=names)
```

## LabeledLDA

Supervised: each label is a topic, and a document's tokens are restricted to its
labels. Empty labels fall back to unconstrained LDA.

## SAGE

Content-covariate topics via an additive log-linear model: the *same* topic is
worded differently across groups. `word_contrast(topic, a, b)` shows the words
that most distinguish two groups' phrasing.

## CTM

The Correlated Topic Model (logistic-normal): topics can co-occur, unlike LDA's
Dirichlet. This is the engine STM builds on; `topic_correlation` reports the
learned structure. Fit by parallel variational EM.

## STM

The full Structural Topic Model: CTM core plus **prevalence** and **content**
covariates. This is the workhorse for social science; it has its own
[guide](covariates.md).

## HDP

A nonparametric model that **infers** the number of topics rather than taking
`K` as input. Useful as a sanity check on the `K` you chose elsewhere.

```python
hdp = tt.HDP(eta=0.3, seed=1)
hdp.fit(docs, iters=300)
print(hdp.num_topics, "topics inferred")
```

## DTM

The Dynamic Topic Model: a fixed number of topics whose word distributions
**drift** across ordered time slices. `word_evolution(topic, word)` traces one
word's probability through time, and `word_drift(topic)` reports *which* words
rose and fell most within a topic — what makes its vocabulary evolve.

```python
dtm = tt.DTM(num_topics=10, chain_variance=0.05, seed=1)
dtm.fit(docs, times, em_iters=20)   # `times` = per-doc slice index

drift = dtm.word_drift(topic=3)     # first vs last slice by default
print("rising: ", [w for w, _ in drift["rising"][:5]])
print("falling:", [w for w, _ in drift["falling"][:5]])
```

## SupervisedLDA

Topics shaped to predict a per-document real-valued response (Blei & McAuliffe).
`coefficients` give each topic's pull on the outcome, and `predict` scores new
documents.

## Guided topics

`SeededLDA` and `KeyATM` steer named topics with a few seed words each, for when you know the themes you expect. See the [guided-topics guide](guided.md).

## Short-text and hierarchy models

`PT` and `GSDMM` are built for short documents; see the
[short-text guide](short-text.md). `PA` (Pachinko Allocation) and `HLDA`
(hierarchical, nested-CRP) recover super-/sub-topic structure.
