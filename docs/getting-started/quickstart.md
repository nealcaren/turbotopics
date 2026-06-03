# Quickstart

Every model takes pre-tokenized documents, a `list[list[str]]` or a
[`Corpus`](../guides/preprocessing.md), and returns NumPy arrays.

## Fit a model

```python
import topica

animal_docs = [["cat", "dog", "fish", "cat", "dog"]] * 15
space_docs  = [["planet", "star", "moon", "rocket", "planet"]] * 15
documents   = animal_docs + space_docs

model = topica.LDA(num_topics=2, seed=42)
model.fit(documents, iterations=1000)
```

## Read the results

```python
print(model.topic_word.shape)   # (2, 7)  — φ, topics × words
print(model.doc_topic.shape)    # (30, 2) — θ, docs × topics (rows sum to 1)

for i, words in enumerate(model.top_words(5)):
    print(f"Topic {i}:", "  ".join(f"{w}({p:.2f})" for w, p in words))
```

```
Topic 0: cat(0.40)  dog(0.40)  fish(0.20)  planet(0.00)  star(0.00)
Topic 1: planet(0.40)  star(0.20)  moon(0.20)  rocket(0.20)  cat(0.00)
```

Fits are deterministic for a fixed `seed`.

## Score and validate

```python
# Per-topic coherence and exclusivity — the standard quality pair.
coherence   = model.coherence(n=10)             # UMass, per topic
exclusivity = topica.exclusivity(model, n=10)       # per topic
diversity   = topica.topic_diversity(model, topn=25)

# Windowed, human-aligned coherence (gensim-style):
cv = topica.coherence(model, documents, coherence_type="c_v", topn=10)
```

## Infer topics for new documents

```python
new_docs = [["cat", "dog"], ["rocket", "moon"]]
theta = model.transform(new_docs, seed=0)       # (2, num_topics), rows sum to 1
print(theta.argmax(axis=1))                     # dominant topic per document
```

## Where to go next

- [The models](../guides/models.md): pick the right one for your question.
- [Covariates & STM](../guides/covariates.md): relate topics to metadata.
- [Diagnostics & validation](../guides/diagnostics.md): choose `K`, prove stability.
- [Worked example: Du Bois in *The Crisis*](../examples/dubois.md): the whole workflow end to end.
