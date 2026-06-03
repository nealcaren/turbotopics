# Diagnostics & validation

All of these are **model-agnostic**: they take any fitted model's `topic_word` /
`doc_topic`, so they work the same across LDA, STM, HDP, and the rest. They're
exported at the top level (`topica.<name>`) and in `topica.diagnostics`. For how
to *use* them to make an analysis publishable, see
[Validate the topics](../publishing/validation.md).

## Quality metrics

```python
import topica

model.coherence(10)                                   # per-topic UMass (built in)
topica.coherence(model, texts, coherence_type="c_v")      # windowed, human-aligned
topica.exclusivity(model, n=10)                           # per topic
topica.topic_diversity(model, topn=25)                    # fraction of unique top words

qf = topica.quality_frontier(model, n=10)                 # coherence, exclusivity, prevalence
# qf["coherence"], qf["exclusivity"] -> the canonical STM quality scatter
```

!!! tip "Coherence is fast, even at large K"
    `topica.coherence` runs its co-occurrence counting in the Rust core, scoring only
    the word pairs that actually occur within a topic's top-N rather than a full
    vocabulary×vocabulary matrix. `c_v` on a 500-topic model that took minutes in
    a pure-Python loop now takes a fraction of a second. Two habits still help on
    very large corpora: compute coherence **once** on the final model (never
    inside a fit loop), and pass a **document sample** as `texts` — coherence is
    an estimate, and a few thousand documents give the same ranking. `u_mass`
    (document-level, no sliding window) remains the cheapest option for quick
    `K`-selection sweeps.

## Labeling and interpretation

```python
topica.label_topics(model.topic_word, model.vocabulary, n=10)   # prob / frex / lift / score
topica.frex(model.topic_word, model.vocabulary, n=10)           # frequent + exclusive
topica.relevance(model.topic_word, model.vocabulary, lam=0.6)   # LDAvis relevance
topica.find_thoughts(model.doc_topic, texts, topic=0, n=3)      # representative docs
topica.find_thoughts_html(model, texts, n_docs=3)               # highlighted close-reading
```

## Human validation: intrusion tests

```python
topica.word_intrusion(model, n_words=5, seed=0)           # top words + an intruder
topica.document_intrusion(model, texts=texts, n_docs=3)   # top docs + an intruder
```

## Stability and model selection

```python
topica.search_k(docs, ks=[10, 20, 30], held_out=test)     # coherence/exclusivity/perplexity per K
topica.bootstrap_stability(docs, k=20, n_boot=50)         # per-topic stability under resampling
topica.align_topics(model_a, model_b)                     # one-to-one match across fits
topica.topic_stability([model_a, model_b], topn=10)       # cross-fit term overlap
topica.check_residuals(model, docs)                       # Taddy dispersion: is K too small?
```

## Visualization

```python
viz = topica.prepare_pyldavis(model, docs)                # pyLDAvis PreparedData if installed
qf, fig = topica.quality_frontier(model, plot=True)       # matplotlib scatter if installed
```
