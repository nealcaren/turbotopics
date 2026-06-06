"""End-to-end embedding topics with the `llm` library: embed, model, label, report.

Uses topica.llm_embed to produce document embeddings (cached so the corpus is
embedded once), fits FASTopic, labels the topics with an LLM, and writes a
plot_report figure. Reproducible with a local embedder and a local labeling model,
so no API key is required:

    pip install "topica[llm,viz]" llm-sentence-transformers llm-ollama
    python examples/llm_topics.py

Swap the model names for OpenAI ("text-embedding-3-small", "gpt-4o-mini") if you
prefer the hosted versions.
"""

import csv
import os

import topica

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CRISIS = os.path.join(ROOT, "examples", "dubois_crisis.csv")
STOP = os.path.join(ROOT, "examples", "english-stoplist.txt")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # local, offline
LABEL_MODEL = "llama3.2"                                 # local via llm-ollama


def main():
    rows = list(csv.DictReader(open(CRISIS)))
    stop = list(open(STOP).read().split())
    texts = [r["text"] for r in rows]
    decade = [f"{r['decade']}s" for r in rows]
    docs = [topica.tokenize(t, stopwords=stop, min_length=4) for t in texts]

    # 1. Embed once, cached. Re-runs reload from disk instead of re-embedding.
    doc_emb = topica.llm_embed(texts, model=EMBED_MODEL, cache=os.path.join(HERE, "crisis_emb.npz"))

    # 2. Fit an embedding-native, mixed-membership model.
    model = topica.FASTopic(num_topics=10, epochs=200, seed=1)
    model.fit(docs, doc_emb)

    # 3. Name the topics with an LLM (pin temperature for stable labels).
    backend = topica.llm_backend(LABEL_MODEL, temperature=0)
    labels = topica.llm_topic_labels(model, texts, call=backend, set_labels=True)
    for t, label in enumerate(labels):
        print(f"{t:2d}  {label}")

    # 4. A one-figure report, with the LLM labels flowing into it.
    fig = topica.plot_report(model, texts=docs, timestamps=decade, n=6,
                             title="Du Bois's Crisis essays")
    fig.savefig(os.path.join(HERE, "crisis_report.png"), dpi=110, bbox_inches="tight")
    print("wrote crisis_report.png")


if __name__ == "__main__":
    main()
