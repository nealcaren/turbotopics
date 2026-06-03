# Guided topics (seed words)

Plain LDA is unsupervised: you label the topics after the fact and have no
control over whether the themes you care about appear. **Guided** models let you
inject prior knowledge as a few **seed words per topic**, so a topic forms around
words you already believe belong together. You seed *topics with keywords*, not
*documents with labels*, so there is no hand-coding.

This is squarely a social-science tool: it improves [measurement validity and
reproducibility](../publishing/validation.md), the things reviewers push on.
turbotopics has two, matching the two standard R packages.

## SeededLDA

Seed-word priors steer some topics; `residual` unseeded topics are learned
freely. Faithful to the `seededlda` package (Watanabe): a seed word gets a
`weight × 100` prior pseudocount in its topic, plus seeded initialization.

```python
import turbotopics as tt

model = tt.SeededLDA(
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
model = tt.KeyATM(
    {"economy": ["jobs", "wages", "tax"],
     "immigration": ["border", "visa", "deport"]},
    num_topics=10,        # 2 keyword topics + 8 regular topics
    seed=1,
)
model.fit(docs, iters=1500)

model.keyword_rate        # per-topic share drawn from the keyword distribution
```

## Which to use

- **`KeyATM`** is the better-validated choice and the one with the political-
  science following; prefer it for new work.
- **`SeededLDA`** is simpler and maps directly onto the `seededlda` workflow.

Both feed the same [diagnostics](diagnostics.md), [effects](../publishing/effects.md),
and [validation](../publishing/validation.md) as every other model.

!!! note "Faithful to the references"
    On a shared corpus with identical seeds, turbotopics recovers the same
    seeded-topic vocabulary as R's `seededlda` and the same keyword-topic words
    as R's `keyATM` (verified word-for-word against both packages).
