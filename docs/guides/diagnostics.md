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

For readable labels, `llm_topic_labels` asks an LLM to name each topic from its
top words and representative documents. topica is the plumbing: it assembles the
prompt and you bring the model. Pass any callable (your own client, a local
`ollama` endpoint) as `call`, or name a model through the optional
[`llm`](https://llm.datasette.io/) adapter, which reaches every provider and
local models via plugins.

```python
# Bring your own callable (no extra dependency):
labels = topica.llm_topic_labels(model, texts, call=my_model_fn, set_labels=True)

# Or name a model via the `llm` adapter (pip install "topica[llm]"):
backend = topica.llm_backend("gpt-4o-mini", temperature=0)   # pin for stability
labels = topica.llm_topic_labels(model, texts, call=backend, set_labels=True)

topica.topic_label_prompts(model, texts)[0]   # inspect exactly what the model sees
```

`set_labels=True` flows the labels into `topic_info` and `plot_report`. LLM labels
are a convenience, not a reproducible measurement: pin the model and temperature,
and keep `label_topics` (FREX / probability / lift) as the defensible descriptors.

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

## Convergence

Every iterative model exposes a uniform convergence interface. `model.fit_history`
is a list of `(iteration, objective)` pairs — the ELBO/bound for variational
models (STM, CTM, ProdLDA, ETM, FASTopic) and the per-token log-likelihood for
collapsed-Gibbs models (LDA, keyATM, SeededLDA, …). `model.converged` is `True`
if a tolerance criterion was met during `fit`, `False` if the model ran to the
iteration cap, and `None` for models with no iterative objective (BERTopic,
Top2Vec).

```python
model = topica.LDA(num_topics=20, seed=1)
model.fit(docs, iters=500)

model.converged        # True / False / None
model.fit_history      # [(10, -7.43), (20, -7.31), ...]
```

On collapsed-Gibbs models you can enable early stopping by passing
`convergence_tol` and `check_every` to `fit`:

```python
model.fit(docs, iters=1000, convergence_tol=1e-4, check_every=10)
# stops as soon as the relative change in log-likelihood over one check
# interval drops below 1e-4, rather than running all 1000 sweeps.
```

The cluster models (BERTopic, Top2Vec) and structurally non-iterative models
(DTM, HLDA) return an empty `fit_history` and `converged` of `False` or `None`;
they satisfy the contract without early-stop support.

## Visualization

```python
viz = topica.prepare_pyldavis(model, docs)                # pyLDAvis PreparedData if installed
qf, fig = topica.quality_frontier(model, plot=True)       # matplotlib scatter if installed
```
