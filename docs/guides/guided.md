# Guided topics (seed words)

Plain LDA is unsupervised: you label the topics after the fact and have no
control over whether the themes you care about appear. **Guided** models let you
inject prior knowledge as a few **seed words per topic**, so a topic forms around
words you already believe belong together. You seed *topics with keywords*, not
*documents with labels*, so there is no hand-coding.

This is squarely a social-science tool: it improves [measurement validity and
reproducibility](../publishing/validation.md), the things reviewers push on.
topica has two, matching the two standard R packages.

## SeededLDA

Seed-word priors steer some topics; `residual` unseeded topics are learned
freely. Faithful to the `seededlda` package (Watanabe): a seed word gets a
`weight × 100` prior pseudocount in its topic, plus seeded initialization.

```python
import topica

model = topica.SeededLDA(
    {"economy": ["jobs", "wages", "tax"],
     "immigration": ["border", "visa", "deport"]},
    residual=3,           # 3 extra unseeded topics
    seed=1,
)
model.fit(docs, iters=2000)

model.topic_names         # ['economy', 'immigration', 'residual_1', ...]
for t in range(model.num_topics):
    print(model.topic_names[t], [w for w, _ in model.top_words(8, topic=t)])
```

## KeyATM

The Keyword-Assisted Topic Model (Eshima, Imai & Sasaki 2024) is the modern,
well-validated version. A token in a keyword topic comes either from a
distribution over only that topic's keywords or from the topic's full
distribution; the learned mix is the **keyword rate**.

```python
model = topica.KeyATM(
    {"economy": ["jobs", "wages", "tax"],
     "immigration": ["border", "visa", "deport"]},
    num_topics=10,        # 2 keyword topics + 8 regular topics
    seed=1,
)
model.fit(docs, iters=1500)

model.keyword_rate        # per-topic share drawn from the keyword distribution
```

### Covariate keyATM

Pass `covariates` to let document metadata shape topic prevalence, the keyATM
covariate model. The document-topic prior becomes a Dirichlet-multinomial
regression, `α_{d,k} = exp(x_d · λ_k)` (Mimno & McCallum 2008, the same engine as
[`DMR`](models.md#dmr)), so you can ask whether a covariate moves a named topic.
An intercept is prepended; the learned coefficients are in `feature_effects`.

```python
import numpy as np
is_dem = np.array([...]).reshape(-1, 1)          # one row per document
model = topica.KeyATM(seeds, num_topics=2, seed=1)
model.fit(docs, covariates=is_dem, feature_names=["is_dem"], iters=1000)

model.feature_names       # ['intercept', 'is_dem']
model.feature_effects     # (num_topics, 2): coefficient of each covariate per topic
```

A larger `feature_effects[k, j]` means covariate `j` raises topic `k`'s
prevalence. For uncertainty, pair the fitted `doc_topic` with
[`estimate_effect`](covariates.md).

### Dynamic keyATM

Pass `timestamps` (one per document) to let topic prevalence shift over time.
This is the keyATM dynamic model, a Chib (1998) change-point hidden Markov model:
the timeline is split into `num_states` latent regimes, each with its own
document-topic prior, and the model estimates where prevalence changes. Following
the keyATM Supreme Court application (Eshima, Imai & Sasaki 2024, Section 3.3),
documents carry a year and the model recovers when each topic rises or falls.

```python
model = topica.KeyATM(seeds, num_topics=14, seed=1)
model.fit(docs, timestamps=years, num_states=5, iters=3000)

model.time_labels        # ['1946', '1947', ..., '2012']  (T distinct timestamps)
model.time_state         # [0, 0, 1, 1, ..., 4]  regime of each segment
model.time_prevalence    # (T, num_topics): smoothed prevalence path, rows sum to 1
model.transition_matrix  # (num_states, num_states), left-to-right
```

Documents may be passed in any order; they are sorted by timestamp internally and
`doc_topic` is returned in the original order. Plot a column of `time_prevalence`
against `time_labels` to see a topic's trajectory.

## Which to use

- **`KeyATM`** is the better-validated choice and the one with the political-
  science following; prefer it for new work.
- **`SeededLDA`** is simpler and maps directly onto the `seededlda` workflow.

Both feed the same [diagnostics](diagnostics.md), [effects](../publishing/effects.md),
and [validation](../publishing/validation.md) as every other model.

!!! note "Faithful to the references"
    On a shared corpus with identical seeds, topica recovers the same
    seeded-topic vocabulary as R's `seededlda` and the same keyword-topic words
    as R's `keyATM` (verified word-for-word against both packages).
