# Embedding topics

The models elsewhere in topica learn topics from word counts. The models here
start from *embeddings*, in three flavors. **BERTopic** and **Top2Vec** *cluster*
document embeddings and read one topic off each cluster; **ETM** is *generative*,
LDA with the topic-word distribution factored through embeddings; **FASTopic**
reads topics off two *optimal-transport* plans between embedding sets. topica fits
all four with no PyTorch, no UMAP/numba, and no sentence-transformers in the
shipped wheel.

You bring the embeddings. topica does not call an embedding model; you pass a
document-vector matrix (and, for Top2Vec, a matching word-vector matrix) from
wherever you like, a sentence-transformer, an API, or a local model such as
ollama. Everything downstream is in the wheel.

If you would rather not wire up an embedder yourself, `topica.llm_embed` produces
the matrix through Simon Willison's [`llm`](https://llm.datasette.io/) library
(the optional `topica[llm]` extra), which reaches OpenAI embeddings and local
sentence-transformers via plugins:

```python
doc_emb = topica.llm_embed(texts, model="text-embedding-3-small")          # API
doc_emb = topica.llm_embed(texts, model="sentence-transformers/all-MiniLM-L6-v2")  # local
```

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

## ETM

ETM (the Embedded Topic Model) is not a clustering pipeline; it is LDA with the
topic-word distribution factored through embeddings,
`β_{k,v} = softmax(ρ_v · α_k)`, and a logistic-normal document prior. Each topic
is a *point* `α_k` in the embedding space, and semantically related words share
topic mass even when a topic never saw them. You bring the word embeddings `ρ`;
topica fits the topic embeddings `α` and the prior by the same variational EM as
[`CTM`](models.md), no PyTorch.

```python
import topica

vocab = sorted({w for d in docs for w in d})
word_emb = embed(vocab)                          # (len(vocab), E)

model = topica.ETM(num_topics=20, seed=1)
model.fit(docs, word_emb, vocab)

model.topic_word                       # (num_topics, vocab) β
model.doc_topic                        # (num_docs, num_topics) θ
model.topic_embeddings                 # (num_topics, E) the α points
model.top_words(8, topic=0)
model.bound, model.converged           # the variational evidence bound
```

Because ETM is generative and mixed-membership, you get a proper `θ` and the full
[effects](../publishing/effects.md) and diagnostics stack, not a hard partition.
It fits in a fraction of a second on a few thousand documents.

### Inference: EM or VAE

ETM has two inference engines, selected with `inference=`. The default `"em"` is
the per-document variational EM above: accurate per document, but it runs an
optimizer for every document, so it does not minibatch. `"vae"` is the reference's
amortized autoencoder, an encoder network that maps a document's word counts
straight to its topic proportions. It trains by minibatch Adam, scales to large
corpora, and maps a new document with a single encoder pass rather than a
per-document optimization.

```python
model = topica.ETM(num_topics=20, inference="vae",
                   hidden_size=800, epochs=150, batch_size=1000, lr=0.005, seed=1)
model.fit(docs, word_emb, vocab)
model.transform(new_docs)              # fast: one encoder forward pass
```

The reference fits the VAE with PyTorch autograd; topica hand-codes the encoder's
forward and backward (every gradient checked against finite differences) and steps
with Adam, so the VAE path is the same model with no PyTorch. Both engines return
the same surface (`topic_word`, `doc_topic`, `topic_embeddings`); `bound` is the
variational bound for EM and the ELBO for VAE. The trade is the usual one: EM is
more accurate per document, the VAE scales.

## FASTopic

FASTopic also drops the encoder, but it is not a clustering pipeline and not a
generative LDA. It places topics, words, and documents in one embedding space and
reads the topic proportions `theta` and topic-word matrix `beta` straight off two
*optimal-transport* plans: documents are transported to topics, topics to words.
You bring the document embeddings; topica learns the topic embeddings, the word
embeddings (in the same space), and the transport marginals, minimizing a
bag-of-words reconstruction plus the two transport costs.

```python
import topica

model = topica.FASTopic(num_topics=20, seed=1)
theta = model.fit_transform(docs, doc_emb)   # (num_docs, num_topics)

model.topic_word                       # (num_topics, vocab) beta
model.doc_topic                        # (num_docs, num_topics) theta
model.topic_embeddings                 # (num_topics, E) topic points
model.word_embeddings                  # (vocab, E) learned word points
model.top_words(8, topic=0)
model.loss_history                     # the objective at each epoch
```

Unlike Top2Vec and BERTopic, FASTopic is mixed-membership: each document gets a
full `theta` over topics, so it carries the [effects](../publishing/effects.md)
and diagnostics stack. New documents are mapped to topics by a distance-softmax
over the fitted topic embeddings, so `transform` needs only their embeddings, no
tokens:

```python
theta_new = model.transform(new_doc_emb)   # (n, num_topics)
```

The reference trains by autodiff through the unrolled Sinkhorn iterations; topica
has no autodiff, so it differentiates the fixed point of a hand-coded reverse-mode
Sinkhorn (every gradient checked against finite differences) and steps with Adam.
`dt_alpha`/`tw_alpha` are the inverse entropic regularizations for the two
transport problems (reference defaults 3.0 and 2.0); larger is sharper.

## Inspecting and adjusting clustering models

Top2Vec and BERTopic produce hard `labels` (`-1` is a noise/outlier document), so
they support two post-hoc edits. `reduce_outliers()` reassigns every `-1`
document to the topic whose words best explain it and rebuilds the topic-word
matrix, returning how many it moved. `merge_topics([[3, 7], [1, 2]])` collapses
groups of topics you decide to combine, rebuilding the representation and
renumbering topics. Both also gain `transform`/`fit_transform` for held-out
documents, and the c-TF-IDF knobs `bm25=` and `reduce_frequent=` on BERTopic.

For a quick read of any fitted model (not just these), `topica.topic_info(model,
texts)` returns per-topic size, prevalence, top words, and representative
documents, with an outlier row when present; `topica.topics_over_time(model,
timestamps)` and `topica.topics_per_class(model, groups)` summarize prevalence by
time or group; and `topica.set_topic_labels(model, {...})` stores your own labels.

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
  clustering. The default reducer is a randomized PCA: fast, deterministic, and
  dependency-free, but it separates less sharply than UMAP and on closely spaced
  themes can merge clusters a UMAP run would split.
- `reducer="umap"` switches to a faithful UMAP reducer (with `n_neighbors`), which
  separates real document embeddings much better. On a four-newsgroup slice it
  recovers four clean topics where PCA merges them into two. UMAP is opt-in at
  build time: install topica built with the `umap` feature
  (`maturin develop --features python,umap`); asking for it without that build
  raises a clear error. The default build stays lean and PCA-only.
- Results are reproducible for a fixed `seed`.

!!! note "Faithful to the references"
    On a shared task with shared document embeddings, topica's `Top2Vec` and
    `BERTopic` recover the same clusters as the Python `BERTopic` package (same
    topics, matching assignments). The difference is the dependency footprint:
    topica runs the pipeline in Rust with none of `torch`, `umap-learn`, or
    `hdbscan` installed.
