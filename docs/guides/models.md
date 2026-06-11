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
| Measure topic sentiment/discourse from covariates | [`STS`](#sts) |
| Let the data choose the number of topics | [`HDP`](#hdp) |
| Track topics that drift over time | [`DTM`](#dtm) |
| Tie topics to known labels | [`LabeledLDA`](#labeledlda) |
| Shape topics to predict an outcome | [`SupervisedLDA`](#supervisedlda) |
| Steer topics with known keywords | [`keyATM`, `seededlda`](guided.md) |
| Sharper, more coherent topics at scale | [`ProdLDA`](#prodlda) |
| Model short texts (tweets, answers) | [`PT`, `GSDMM`](short-text.md) |
| Build a topic hierarchy | `PA`, `HLDA` |

## LDA

Classic Latent Dirichlet Allocation via MALLET's fast SparseLDA collapsed-Gibbs
sampler. Fits are bit-for-bit reproducible, with optional approximate
multi-threaded training.

```python
import topica
model = topica.LDA(num_topics=20, seed=42)
model.fit(docs, iters=1000)
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
model = topica.LDA(num_topics=500, seed=1, sampler="lightlda", mh_steps=2)
model.fit(docs, iters=1000)
```

Both samplers target the same posterior and recover the same topics (matched
coherence on shared corpora); LightLDA is also useful as an independent
cross-check of a SparseLDA fit. Use the default unless you have a specific
large-`K` reason not to.

## STM

The full Structural Topic Model: CTM core plus **prevalence** and **content**
covariates. This is the workhorse for social science; it has its own
[guide](covariates.md).

## STS

The Structural Topic and Sentiment-Discourse model (Chen & Mankad 2024) extends
STM with a per-document, per-topic **continuous sentiment-discourse** latent that
shifts the wording within a topic, with both topic prevalence and sentiment driven
by document covariates. Use it when you want to measure not just *which* topics a
covariate predicts, but *how* — the tone and slant with which each topic is
discussed.

```python
m = topica.STS(num_topics=10, seed=1)
m.fit(docs, sentiment_seed=rating, prevalence=X, prevalence_names=names)

m.doc_topic          # topic prevalence θ
m.sentiment          # per-document topic sentiment-discourse α^(s)
m.prevalence_effects # covariate → prevalence
m.sentiment_effects  # covariate → sentiment-discourse
m.topic_word_at(2.0) # how the topic is worded at high sentiment
```

`sentiment_seed` (one value per document — e.g. a star rating) seeds the sentiment
and defines the aggregation groups for the topic-word estimation. `kappa_estimation`
selects the topic-word estimator: `"ridge"` (default, fast) or `"lasso"` (matches
the reference R `sts` exactly, at higher cost); the two agree closely on
well-conditioned corpora. Validated against the authors' R `sts` implementation in
`parity/sts_r_compare.py` — on the political-blog corpus topica's STS aligns with
the published fit in the mid-0.90s (topic-word cosine), the same neighborhood as
topica's STM matches R's STM.

## CTM

The Correlated Topic Model (logistic-normal): topics can co-occur, unlike LDA's
Dirichlet. This is the engine STM builds on; `topic_correlation` reports the
learned structure. Fit by parallel variational EM.

## DMR

Dirichlet-Multinomial Regression: each document's topic prior depends on its
metadata, `α_d = exp(Xγ)`. The learned `feature_effects` show how covariates
shift topic propensity.

```python
import numpy as np
X, names = topica.one_hot(party)
model = topica.DMR(num_topics=20, seed=1)
model.fit(docs, X, feature_names=names)
```

## DTM

The Dynamic Topic Model: a fixed number of topics whose word distributions
**drift** across ordered time slices. `word_evolution(topic, word)` traces one
word's probability through time, and `word_drift(topic)` reports *which* words
rose and fell most within a topic — what makes its vocabulary evolve.

```python
dtm = topica.DTM(num_topics=10, chain_variance=0.05, seed=1)
dtm.fit(docs, times, iters=20)   # `times` = per-doc slice index

drift = dtm.word_drift(topic=3)     # first vs last slice by default
print("rising: ", [w for w, _ in drift["rising"][:5]])
print("falling:", [w for w, _ in drift["falling"][:5]])
```

## HDP

A nonparametric model that **infers** the number of topics rather than taking
`K` as input. Useful as a sanity check on the `K` you chose elsewhere.

```python
hdp = topica.HDP(gamma=0.5, eta=0.3, seed=1)
hdp.fit(docs, iters=300)
print(hdp.num_topics, "topics inferred")
```

`gamma` is the main lever on the inferred count: larger values discover more
topics (the conservative default `0.1` lands near a handful, like the reference
implementations). By default the concentrations are held fixed, which gives a
stable, reproducible topic count; `resample_conc=True` lets the model adapt them
to the data instead, useful for exploration but more liberal about adding topics.

## Guided topics

`keyATM` and `seededlda` steer named topics with a few seed words each, for when you know the themes you expect. See the [guided-topics guide](guided.md).

## ProdLDA

ProdLDA ([Srivastava & Sutton 2017](https://arxiv.org/abs/1703.01488)) keeps
LDA's document model but replaces the word-level *mixture* of topics with a
*product of experts*: the word distribution is `softmax(βθ)` with an unnormalized
β, rather than `softmax(β)·θ`. This sharper word model reliably yields more
coherent topics than collapsed-Gibbs LDA. Inference is an amortized variational
autoencoder (the AVITM framework): an encoder network maps a document's bag of
words to a logistic-normal posterior over θ, trained by minibatch Adam on the
ELBO. There is no PyTorch dependency; the network is hand-coded in the Rust core.

```python
model = topica.ProdLDA(num_topics=20, seed=1)
theta = model.fit_transform(docs)      # one encoder pass per document
model.top_words(10)
```

Two details follow the paper's recipe for avoiding *component collapse* (topics
decaying onto the prior early in training): batch normalization on the encoder
heads and decoder, and high-momentum Adam (`β₁ = 0.99`). Because inference is
amortized, `transform` maps new documents with a single forward pass rather than
re-running an optimizer. ProdLDA is bag-of-words (no embeddings); for the
embedding-factored generative model see [`ETM`](embedding.md).

## Short-text models

`PT` and `GSDMM` are built for short documents; see the
[short-text guide](short-text.md).

## SupervisedLDA

Topics shaped to predict a per-document real-valued response (Blei & McAuliffe).
`coefficients` give each topic's pull on the outcome, and `predict` scores new
documents.

## LabeledLDA

Supervised: each label is a topic, and a document's tokens are restricted to its
labels. Empty labels fall back to unconstrained LDA.

## SAGE

Content-covariate topics via an additive log-linear model: the *same* topic is
worded differently across groups. `word_contrast(topic, a, b)` shows the words
that most distinguish two groups' phrasing.

## Hierarchy models

`PA` (Pachinko Allocation) and `HLDA` (hierarchical, nested-CRP) recover
super-/sub-topic structure.
