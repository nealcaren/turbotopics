# Diagnostics & validation

All of these are **model-agnostic**: they take any fitted model's `topic_word` /
`doc_topic`, so they work the same across LDA, STM, HDP, and the rest. They're
exported at the top level (`tt.<name>`) and in `turbotopics.diagnostics`. For how
to *use* them to make an analysis publishable, see
[Validate the topics](../publishing/validation.md).

## Quality metrics

```python
import turbotopics as tt

model.coherence(10)                                   # per-topic UMass (built in)
tt.coherence(model, texts, coherence_type="c_v")      # windowed, human-aligned
tt.exclusivity(model, n=10)                           # per topic
tt.topic_diversity(model, topn=25)                    # fraction of unique top words

qf = tt.quality_frontier(model, n=10)                 # coherence, exclusivity, prevalence
# qf["coherence"], qf["exclusivity"] -> the canonical STM quality scatter
```

## Labeling and interpretation

```python
tt.label_topics(model.topic_word, model.vocabulary, n=10)   # prob / frex / lift / score
tt.frex(model.topic_word, model.vocabulary, n=10)           # frequent + exclusive
tt.relevance(model.topic_word, model.vocabulary, lam=0.6)   # LDAvis relevance
tt.find_thoughts(model.doc_topic, texts, topic=0, n=3)      # representative docs
tt.find_thoughts_html(model, texts, n_docs=3)               # highlighted close-reading
```

## Human validation: intrusion tests

```python
tt.word_intrusion(model, n_words=5, seed=0)           # top words + an intruder
tt.document_intrusion(model, texts=texts, n_docs=3)   # top docs + an intruder
```

## Stability and model selection

```python
tt.search_k(docs, ks=[10, 20, 30], held_out=test)     # coherence/exclusivity/perplexity per K
tt.bootstrap_stability(docs, k=20, n_boot=50)         # per-topic stability under resampling
tt.align_topics(model_a, model_b)                     # one-to-one match across fits
tt.topic_stability([model_a, model_b], topn=10)       # cross-fit term overlap
tt.check_residuals(model, docs)                       # Taddy dispersion: is K too small?
```

## Visualization

```python
viz = tt.prepare_pyldavis(model, docs)                # pyLDAvis PreparedData if installed
qf, fig = tt.quality_frontier(model, plot=True)       # matplotlib scatter if installed
```
