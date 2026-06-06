# Visualization

`topica.viz` is a manuscript-first visualization toolkit: an honest successor to
pyLDAvis that works across model families, exports the numbers behind every
figure, and refuses to draw what it cannot justify.

```python
import topica.viz as viz

viz.coherence_frontier(model, texts).to_png("quality.png")
viz.effect_plot(model, corpus, formula="~ year", data=meta).to_png("effects.pdf")
viz.term_barchart(model, topic=3, mode="frex").to_frame()
```

Every view is a **panel** with three renderers:

- `.to_frame()` — a pandas DataFrame of the numbers behind the picture. Always
  available; this is your reproducibility and reviewer-armor.
- `.to_png(path)` — a matplotlib figure (PNG / PDF / SVG by extension). The
  publication renderer.
- `.to_html(path)` — an interactive (Altair) build, for the few views interaction
  genuinely helps. Needs the `topica[viz-interactive]` extra.

Install the static stack with `pip install topica[viz]` (matplotlib, pandas,
scipy) and the interactive subset with `pip install topica[viz-interactive]`.

## Honest by capability

The panels read a per-model **capability descriptor** (`viz.capabilities(model)`)
and switch their statistics and labels on it, so they never overclaim:

- A **c-TF-IDF** `topic_word` (BERTopic / Top2Vec) is not a probability, so the
  FREX / lift / relevance modes are disabled and the bars are labeled "c-TF-IDF
  weight," not "P(w | topic)."
- An **effect-plot confidence interval** is drawn only where a θ posterior exists.
  For an embedding/cluster model the panel shows point estimates and says so; pass
  `method="bootstrap"` for intervals. A topic the bootstrap flags as unreliable is
  drawn as a ghosted point, not a band.
- The uncertainty is **labeled for what it is** — a Gibbs model's
  Dirichlet-conditional (within-document) uncertainty is not a logistic-normal
  posterior.

## The covariate effect plot

The results figure for an STM paper: each topic's prevalence response to a
covariate, with method-of-composition intervals, diverging color centered at zero.

```python
ep = viz.effect_plot(model, corpus, formula="~ party", data=meta, nsims=50)
ep.to_png("effects.pdf")   # publication figure
ep.to_frame()              # coef / se / ci / reliable per topic
```

![Covariate effect plot](../images/viz_effect.png)

## Choosing K

Reuse `search_k` / `quality_frontier`, with the data export and a clean figure:

```python
rows = topica.search_k(docs, ks=[10, 20, 30, 40], held_out=test_docs)
viz.search_k(rows).to_png("choose_k.png")          # coherence / exclusivity / perplexity vs K
viz.coherence_frontier(model, texts).to_png("frontier.png")   # per-topic, defend dropping topics
```

## The pyLDAvis replacement: terms + a seriated similarity heatmap

Instead of a spurious 2-D "intertopic map," topica shows the K×K topic-similarity
matrix at full fidelity, ordered by hierarchical clustering (√Jensen-Shannon for
probability topics, cosine for c-TF-IDF), paired with a term barchart:

```python
viz.topic_similarity(model).to_png("similarity.png")
viz.term_barchart(model, topic=3, mode="frex", error_bars=True).to_png("terms.png")

# interactive: click a topic in the heatmap, the barchart follows
viz.term_topic_browser(model).to_html("explore.html")
```

`error_bars=True` adds top-word inclusion-probability bars (a bootstrap, so it is
off by default).

## One-call report

`dashboard()` introspects the descriptor and your arguments to assemble the
applicable panels:

```python
report = viz.dashboard(model, texts, corpus=corpus, formula="~ year", data=meta)
report.to_html("report.html")    # interactive browser + static panels, self-contained
report.to_png("report.png")      # the static panels stacked
report.to_frame()                # {panel_name: DataFrame}
```
