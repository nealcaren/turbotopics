# Embedding topics (BERTopic & Top2Vec)

The models elsewhere in topica learn topics from word counts. **BERTopic** and
**Top2Vec** instead start from *embeddings*: they reduce a set of document
vectors, cluster them, and read one topic off each cluster. They are the most
widely used topic models today, and topica ports both onto one Rust pipeline,
`reduce → cluster → represent`, with no PyTorch, no UMAP/numba, and no
sentence-transformers in the shipped wheel.

You bring the embeddings. topica does not call an embedding model; you pass a
document-vector matrix (and, for Top2Vec, a matching word-vector matrix) from
wherever you like, a sentence-transformer, an API, or a local model such as
ollama. Everything downstream is in the wheel.

```python
import numpy as np, topica

docs = [["the", "economy", "and", "jobs"], ["pitching", "and", "home", "runs"], ...]
doc_emb = embed([" ".join(d) for d in docs])   # (num_docs, E), your embedder
```

## BERTopic

BERTopic defines a topic by **class-based TF-IDF** over its documents' words, so
it needs only the document embeddings. The topic count is discovered by the
clustering, not set in advance.

```python
model = topica.BERTopic(min_cluster_size=15, seed=1)
model.fit(docs, doc_emb)

model.num_topics                       # discovered
model.top_words(8, topic=0)            # [(word, c-TF-IDF weight), ...]
model.topic_word                       # (num_topics, vocab), row-normalized c-TF-IDF
model.doc_topic                        # (num_docs, num_topics) soft membership
model.labels                           # hard cluster per doc; -1 is noise
```

Two BERTopic features carry over. `nr_topics` merges the most similar topics down
to a target count:

```python
model = topica.BERTopic(min_cluster_size=15, nr_topics=10, seed=1)
model.fit(docs, doc_emb)
```

and `approximate_distribution` gives a soft topic distribution by sliding a
window over a document's words and comparing each window's c-TF-IDF to every
topic. It is the default `doc_topic`, and you can also run it on new documents:

```python
dist = model.approximate_distribution(new_docs, window=4, stride=1)  # (n, num_topics)
```

## Top2Vec

Top2Vec places each topic *in the embedding space*: the topic vector is the mean
of its documents' embeddings, and its words are the vocabulary terms nearest that
vector. Pass `word_embeddings` with the aligned `vocabulary` (same space as the
document embeddings) to get those nearest-word topics.

```python
vocab = sorted({w for d in docs for w in d})
word_emb = embed(vocab)                          # (len(vocab), E)

model = topica.Top2Vec(min_cluster_size=15, seed=1)
model.fit(docs, doc_emb, word_embeddings=word_emb, vocabulary=vocab)

model.topic_neighbors(8, topic=0)      # [(word, cosine), ...] nearest word vectors
model.top_words(8, topic=0)            # [(word, weight), ...] class-TF-IDF
model.topic_vectors                    # (num_topics, E) topic positions
```

Without `word_embeddings` Top2Vec still fits and exposes `top_words` (c-TF-IDF);
`topic_neighbors` is what the word vectors light up.

## The shared surface

Both models expose topica's standard fitted surface, so they slot in alongside
every other model: `topic_word` (`num_topics × vocab`), `doc_topic`
(`num_docs × num_topics`), `top_words`, `num_topics`, `topic_names`,
`vocabulary`, and `labels`. The embedding-native additions are `topic_vectors`
and `topic_neighbors` (Top2Vec) and `approximate_distribution` (BERTopic).

## Tuning and notes

- `min_cluster_size` is the main dial: larger gives fewer, broader topics; smaller
  gives more, finer ones. `min_samples` (default `min_cluster_size`) sets how
  aggressively sparse documents are called noise (label `-1`).
- `n_components` is the dimensionality the embeddings are reduced to before
  clustering. topica's default reducer is a randomized PCA, which is fast and
  deterministic but separates less sharply than UMAP; on closely spaced themes it
  can merge clusters a UMAP-based run would split.
- Results are reproducible for a fixed `seed`.

!!! note "Faithful to the references"
    On a shared task with shared document embeddings, topica's `Top2Vec` and
    `BERTopic` recover the same clusters as the Python `BERTopic` package (same
    topics, matching assignments). The difference is the dependency footprint:
    topica runs the pipeline in Rust with none of `torch`, `umap-learn`, or
    `hdbscan` installed.
