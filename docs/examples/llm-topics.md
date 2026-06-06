# LLM embeddings and labels: Du Bois's *Crisis* essays

This worked example runs the **embedding-native** path end to end with the
[`llm`](https://llm.datasette.io/) library doing the two jobs that touch a model:
generating the document embeddings and naming the topics. Everything between is
pure topica. The corpus is the 704 essays W. E. B. Du Bois wrote for *The Crisis*
between 1910 and 1934 (the same corpus as the [Du Bois example](dubois.md), here
modeled from sentence embeddings rather than word counts).

!!! info "Focus of this example"
    `llm_embed` (cached) · an embedding model (`FASTopic`) · `llm_topic_labels` ·
    `plot_report`. For the count-based workflow on this corpus see
    [Du Bois](dubois.md).

    Needs the optional extras and two `llm` plugins for the fully local path:
    ```bash
    pip install "topica[llm,viz]" llm-sentence-transformers llm-ollama
    ```
    Reproducible with [`examples/llm_topics.py`](https://github.com/nealcaren/topica/blob/main/examples/llm_topics.py).

## 1. Corpus

```python
import csv, topica

rows = list(csv.DictReader(open("examples/dubois_crisis.csv")))
texts = [r["text"] for r in rows]                       # raw essay text for embedding
decade = [f"{r['decade']}s" for r in rows]              # 1910s / 1920s / 1930s
stop = open("examples/english-stoplist.txt").read().split()
docs = [topica.tokenize(t, stopwords=stop, min_length=4) for t in texts]   # tokens for the model
```

## 2. Embed — once, cached

`llm_embed` produces the document-vector matrix the embedding models need. Naming
a local `sentence-transformers` model keeps it offline and free; `cache=` writes
the matrix to disk so re-running the script reloads it instead of re-embedding
(embeddings are the costly step).

```python
doc_emb = topica.llm_embed(
    texts, model="sentence-transformers/all-MiniLM-L6-v2", cache="crisis_emb.npz"
)
doc_emb.shape          # (704, 384)
```

## 3. Model

`FASTopic` reads topics off optimal-transport plans between the document, topic,
and word embeddings — mixed-membership, so every essay gets a topic distribution.

```python
model = topica.FASTopic(num_topics=10, epochs=200, seed=1)
model.fit(docs, doc_emb)

for t in range(model.num_topics):
    print(t, " ".join(w for w, _ in model.top_words(6, topic=t)))
```

```
0 lazy sisters manners latin antipathy hawaii
1 peoples africa revolution india religion british
2 murderer imprisonment unconstitutional murders punished lynchers
3 drama yonder almighty thou king shadows
4 coöperative consumers survey agricultural pays capitalize
5 imprisoned toussaint petition legion bentley methodist
6 officers miss night street louis town
7 taft darrow wilson's woodrow appointment politician
8 bruce hayti porters sympathize pullman ireland
9 segregation voters votes republican voting vote
```

## 4. Name the topics with an LLM

topica is the plumbing: it builds a prompt from each topic's top words and
representative essays, and you bring the model. Here a local `ollama` model labels
them; `temperature=0` keeps the labels stable. `set_labels=True` stores them so
they flow into `topic_info` and the report.

```python
backend = topica.llm_backend("llama3.2", temperature=0)        # or "gpt-4o-mini"
labels = topica.llm_topic_labels(model, texts, call=backend, set_labels=True)

topica.topic_label_prompts(model, texts)[1]   # inspect exactly what the model saw
```

This step needs a labeling model available through `llm` (a local one via
`llm-ollama`, or a hosted one with an API key); the output labels are whatever
your model returns. The deterministic descriptors (`label_topics`: FREX /
probability / lift) remain the defensible naming for publication; the LLM labels
are the readable shorthand. With `set_labels=True` they replace the default labels
everywhere, including the report below.

## 5. Report

```python
fig = topica.plot_report(model, texts=docs, timestamps=decade, n=6,
                         title="FASTopic on W.E.B. Du Bois's Crisis essays (K=10)")
fig.savefig("crisis_report.png", dpi=200)
```

![A plot_report figure for the Du Bois Crisis essays: topic prevalence with top
words, the coherence-vs-exclusivity quality plot, the topic correlation heatmap,
and topic shares across the 1910s, 1920s, and 1930s.](../images/llm_workflow_report.png)

The figure above is the actual output of steps 1–3 and 5 (MiniLM embeddings, the
FASTopic fit, real top words). It shows topica's **default** topic labels — each
topic's top words — because the rendered figure was built without a labeling
model; run step 4 with your own model to swap in readable names. The time panel
reads as intellectual history: the Africa / Pan-African topic (1) and the
economic-cooperation topic (4) climb steadily from the 1910s to the 1930s,
tracking the arc of Du Bois's later writing.
