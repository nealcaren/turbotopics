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

## Choosing K for STM

Use `search_k`, the coherence×exclusivity frontier, and an `HDP` sanity check.
See [Choose and justify K](../publishing/choosing-k.md).
