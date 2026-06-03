# 2. Choose and justify K

!!! quote "The principle"
    **K is a research decision, not a tuning parameter.** Multiple values of `K`
    are often defensible; your job is to pick one for a reason and show your
    conclusions don't hinge on it.

`K` sets the *granularity* of your themes: roughly, `K=10` for broad themes,
`K=30` for specific topics, `K=100` for fine distinctions. There is no single
"correct" `K`, and you should resist any procedure that pretends otherwise.

## Three converging justifications

Good practice combines all three:

**1. Theory-driven.** How many themes would you expect in this corpus? What level
of granularity answers *your* research question? Start from theory and adjust.

**2. Diagnostic-guided.** Scan a range and look at quality metrics:

| Metric | What it measures | Reading |
|--------|------------------|---------|
| Coherence (c_v / UMass) | Do a topic's top words co-occur? | Higher is better |
| Exclusivity | Are words distinctive to a topic? | Higher is better |
| Held-out perplexity | Fit on unseen documents | Lower is better |

!!! warning "Do not just maximize coherence"
    A model with `K=5` may have higher mean coherence yet miss distinctions that
    matter to your argument. Coherence trades off against exclusivity and against
    substantive richness. Use the metrics to *inform* a judgment, not to replace
    it.

**3. Interpretability-focused.** For each candidate `K`: can you label every
topic? Do the topics make substantive sense? How many are "junk" (stopwords,
artifacts)? Do topics split and merge sensibly as `K` grows?

## A concrete procedure

```python
import turbotopics as tt
import numpy as np

# 1) Scan a theoretically plausible range.
held_out = test_docs                     # a held-out split for perplexity
results = tt.search_k(
    train_docs, ks=[10, 15, 20, 25, 30],
    held_out=held_out, iterations=800,
)
for r in results:
    print(f"K={r['k']:>3}  coherence={r['coherence']:.3f}  "
          f"exclusivity={r['exclusivity']:.3f}  perplexity={r.get('perplexity'):.0f}")
```

Then, for the two or three best candidates, fit the model and **read the
topics**. Count how many you can label, and look at the coherence×exclusivity
spread per topic:

```python
model = tt.STM(num_topics=20, seed=1)
model.fit(docs, prevalence=X)

frontier = tt.quality_frontier(model, n=10)   # per-topic coherence & exclusivity
# scatter frontier["coherence"] vs frontier["exclusivity"];
# weak topics cluster in the lower-left.
```

A nonparametric model is a useful sanity check on your choice: it *infers* a
topic count rather than taking one.

```python
hdp = tt.HDP(eta=0.3, seed=1)
hdp.fit(docs, iters=300)
print("HDP suggests ~", hdp.num_topics, "topics")
```

## Report sensitivity

Pick the `K` that balances metrics, interpretability, and theory, then **show
your finding survives nearby `K`**. Re-run the headline result at `K-5` and
`K+5`; if a covariate effect or a key topic only appears at one exact `K`, say
so. Reviewers read "we used K=20" charitably only when followed by "results were
robust to K ∈ {15, 25}."

→ Next: [Validate the topics](validation.md).
