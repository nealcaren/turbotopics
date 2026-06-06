"""BERTopic's "Best Practices" tutorial, translated to topica.

The original (Maarten Grootendorst's BERTopic tutorial) runs the
embed -> reduce -> cluster -> represent pipeline on ArXiv ML abstracts. topica
fits the same pipeline with no PyTorch / UMAP / numba / sentence-transformers in
its wheel: you bring the embeddings, topica does the rest. This script mirrors the
tutorial step for step and notes where topica does things differently.

Run it (e.g. in Colab):

    pip install topica datasets sentence-transformers
    python bertopic_best_practices.py

What maps cleanly:
  - UMAP + HDBSCAN + class-based TF-IDF      -> topica.BERTopic(reducer="umap", ...)
  - get_topic_info()                          -> topica.diagnostics(model, texts)
  - get_topic(i)                              -> model.top_words(n, topic=i)
  - set_topic_labels(...)                     -> topica.set_topic_labels(model, ...)
  - approximate_distribution(window, stride)  -> model.approximate_distribution(...)
  - reduce_outliers(...)                      -> model.reduce_outliers()
  - transform(new_docs)                       -> model.transform(new_docs)

Where topica differs, on purpose:
  - **Tokenization.** BERTopic tokenizes raw strings internally (CountVectorizer);
    topica takes pre-tokenized documents, so we call topica.tokenize first. (Bigrams
    / ngram_range are not modeled: topica's c-TF-IDF is over unigrams.)
  - **Reproducibility.** UMAP has no random_state equivalent here; the discovery fit
    is stochastic (topica warns). The *prediction* phase (transform) is
    deterministic. Use reducer="pca" for a fully reproducible fit.
  - **Representations.** Instead of KeyBERT / MMR / PartOfSpeech, topica reports FREX
    (frequency-exclusivity) words — the defensible label for publication — and, with
    the optional `topica[llm]` extra, LLM labels via topica.llm_topic_labels.
  - **Visualization.** topica ships a static composite figure, topica.plot_report,
    rather than the interactive plotly views.
  - **Serialization.** save/load for the embedding models is not yet available.
"""

import numpy as np

import topica


def main():
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    # --- Data -------------------------------------------------------------
    dataset = load_dataset("CShorten/ML-ArXiv-Papers")["train"]
    abstracts = dataset["abstract"]
    titles = dataset["title"]  # noqa: F841  (used for hover labels in the original)

    # --- Pre-calculate embeddings (the tutorial's all-MiniLM-L6-v2) -------
    # topica is bring-your-own-vectors. (topica.llm_embed(abstracts,
    # model="sentence-transformers/all-MiniLM-L6-v2") is the in-package route.)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

    # --- Tokenize (topica is pre-tokenized; drop English stopwords) -------
    docs = [topica.tokenize(a, stopwords=topica.ENGLISH_STOPWORDS) for a in abstracts]

    # --- Reduce (UMAP) + cluster (HDBSCAN) + represent (c-TF-IDF) ---------
    topic_model = topica.BERTopic(
        reducer="umap", n_neighbors=15, n_components=5,  # UMAP(n_neighbors=15, n_components=5)
        min_cluster_size=150,                            # HDBSCAN(min_cluster_size=150)
        seed=42,
    )
    theta = topic_model.fit_transform(docs, embeddings)  # like topics, probs = fit_transform(...)
    print(f"discovered {topic_model.num_topics} topics; doc-topic matrix {theta.shape}")

    # --- get_topic_info() -> the one-call diagnostics table --------------
    table = topica.diagnostics(topic_model, abstracts)
    print(table.head(20).to_string())

    # --- get_topic(1) -> top words of a topic ----------------------------
    print("topic 1:", [w for w, _ in topic_model.top_words(10, topic=1)])

    # --- Representations: FREX (always) + optional LLM labels ------------
    # FREX words are already a column in `table`. For LLM labels (the tutorial's
    # OpenAI representation), with the topica[llm] extra:
    #   topica.llm_topic_labels(topic_model, abstracts,
    #                           llm_model="gpt-4o-mini", set_labels=True)

    # --- Custom labels ----------------------------------------------------
    topica.set_topic_labels(topic_model, {1: "Space Travel", 7: "Religion"})

    # --- Topic-document distribution (window / stride) -------------------
    topic_distr = topic_model.approximate_distribution(docs, window=8, stride=4)
    abstract_id = 10
    print(f"\nabstract {abstract_id} topic distribution:",
          np.round(topic_distr[abstract_id], 3))

    # --- Outlier reduction ------------------------------------------------
    moved = topic_model.reduce_outliers()
    print(f"reduce_outliers reassigned {moved} documents")

    # --- Visualize: a static composite report ----------------------------
    fig = topica.plot_report(topic_model, texts=abstracts)
    fig.savefig("topica_report.png", dpi=150, bbox_inches="tight")
    print("wrote topica_report.png")

    # --- Inference on new documents --------------------------------------
    new_theta = topic_model.transform(docs[:100])
    print("inference doc-topic matrix:", new_theta.shape)


if __name__ == "__main__":
    main()
