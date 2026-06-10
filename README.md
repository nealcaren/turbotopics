# Topica: fast, all-purpose topic modeling for Python

`topica` is a fast topic-modeling library for Python with more than a dozen models, built for social scientists who want to move from text data to publishable results in a single workflow. It brings together models and tools usually split across JVM software like MALLET and R packages like `stm`, and runs them on a parallel Rust core competitive with the standard implementations, with every fit reproducible from a fixed seed. Each model comes with the validation, covariate-effect, and reporting tools to meet the standards reviewers expect.

```bash
pip install topica
```

The core needs only NumPy. Optional extras add features without weighing the core
down: `topica[viz]` (matplotlib plots), `topica[formula]` (R-style formulas),
`topica[polars]` (Polars frames), and `topica[llm]` (LLM labels and embeddings,
OpenAI or local via ollama). PyTorch is never required.

```python
from topica import LDA

model = LDA(num_topics=2, seed=42)
model.fit([["cat", "dog", "fish"]] * 15 + [["planet", "star", "moon"]] * 15, iters=1000)

for i, words in enumerate(model.top_words(3)):
    print(f"Topic {i}:", " ".join(w for w, _ in words))
```

See the [getting-started guide](https://nealcaren.github.io/topica/getting-started/quickstart/) and the [worked examples](https://nealcaren.github.io/topica/examples/dubois/) for end-to-end analyses.

## Models

**Count-based models** learn topics from word counts (collapsed Gibbs, variational EM, or an amortized VAE):

| Model | What it's for |
|-------|---------------|
| **`LDA`** | Classic topics via fast collapsed-Gibbs (SparseLDA); optional multi-threaded and LightLDA alias samplers |
| **`ProdLDA`** | Sharper, more coherent topics via a product-of-experts word model, fit as an amortized VAE (no PyTorch) |
| **`DMR`** | Topics conditioned on document metadata (Dirichlet-multinomial regression) |
| **`LabeledLDA`** | Supervised topics tied to document labels |
| **`CTM`** | Correlated topics (logistic-normal) |
| **`STM`** | The Structural Topic Model: correlated topics with prevalence **and** content covariates |
| **`SAGE`** | Content-covariate topics: the same topic worded differently across groups |
| **`HDP`** | Nonparametric LDA that *infers* the number of topics |
| **`DTM`** | Dynamic topics that evolve across time slices |
| **`SupervisedLDA`** | Topics shaped to predict a per-document response |
| **`PT` / `GSDMM`** | Short-text models for tweets, survey answers, headlines |
| **`SeededLDA` / `KeyATM`** | Guided topics steered by seed words |
| **`PA` / `HLDA`** | Topic hierarchies (Pachinko, nested-CRP) |

**Embedding-based models** start from document embeddings you supply (no PyTorch, no UMAP/numba in the wheel):

| Model | What it's for |
|-------|---------------|
| **`BERTopic`** | Cluster document embeddings, label topics by class-TF-IDF; topic reduction and a soft per-document distribution |
| **`Top2Vec`** | Topics as points in the embedding space; topic words are the nearest word vectors |
| **`ETM`** | Generative LDA with the topic-word distribution factored through embeddings (`β = softmax(ρ·α)`); per-document EM or an amortized VAE (`inference="vae"`) |
| **`FASTopic`** | Topics read off two optimal-transport plans between document, topic, and word embeddings |

Every model exposes the same shape: `fit(docs, …)`, then `topic_word` (φ), `doc_topic` (θ), `top_words(n)`, and `save`/`load`. The count-based variational models (`CTM`/`STM`/`SupervisedLDA`/`DTM`) parallelize across cores while staying bit-for-bit deterministic. The embedding models split into two kinds: `BERTopic` and `Top2Vec` run the `reduce → cluster → represent` pipeline, while `ETM` and `FASTopic` are generative and mixed-membership; all of them take vectors from any embedder (sentence-transformers, an API, a local model such as ollama). Full guides: [the models](https://nealcaren.github.io/topica/guides/models/) and [embedding topics](https://nealcaren.github.io/topica/guides/embedding/).

## Diagnostics & analysis

Model-agnostic: they work on any fitted model's `topic_word`/`doc_topic`:

- **Quality:** `coherence` (`u_mass`, `c_v`, `c_uci`, `c_npmi`; computed in the Rust core), `exclusivity`, `topic_diversity`, `quality_frontier`
- **Labeling:** `label_topics` (prob / FREX / lift / score), `frex`, `relevance`, `find_thoughts`, `topic_table`, `summary`
- **Validation:** `word_intrusion`, `document_intrusion`, `bootstrap_stability`, `search_k`
- **Comparison:** `fighting_words` (weighted log-odds) for contrasting corpora
- **Covariate effects:** `estimate_effect` (method of composition, **cluster-robust SEs**, GLM links), `topic_correlation`, and the design helpers `spline` / `interaction` / `one_hot` (an `stm`-style API); `posterior_theta_samples` draws θ for the logistic-normal models (STM/CTM)
- **Preprocessing:** `tokenize`, `learn_phrases` / `apply_phrases`, `split_documents`, the `Corpus` class

See [diagnostics](https://nealcaren.github.io/topica/guides/diagnostics/) and [covariate effects](https://nealcaren.github.io/topica/guides/covariates/).

## Performance

topica runs on a parallel Rust core. It is several times faster than R `stm` — the single-threaded field standard — for the structural and other variational models, and it matches the hand-tuned compiled samplers core for core: parity with Java MALLET on plain LDA and with the C++ `keyATM` on keyword models. On the political-blog corpus (2,000 documents, fit time only, same iterations on both sides):

| Model | Reference | topica speedup |
|-------|-----------|----------------|
| STM | R `stm` | **3–6× single-threaded, ~10–22× multithreaded** |
| LDA | Java MALLET | parity single-threaded, **~2×** multithreaded |
| keyATM | R `keyATM` | parity single-threaded, **~2×** multithreaded |

Every fit is reproducible from a fixed seed and validated against its reference. See [Benchmarks](https://nealcaren.github.io/topica/benchmarks/) for the full methodology, and reproduce the table with `python benchmarks/speed_vs_r.py`.

## Install from source

```bash
pip install maturin
git clone https://github.com/nealcaren/topica && cd topica
python -m venv .venv && source .venv/bin/activate
maturin develop --release --features python
```

Requires `numpy >= 1.21`. Use `--release` (the debug build is much slower).

## Acknowledgements

Topica stands on a generation of open topic-modeling research and code. Each entry below lists the reference, its authors and year, and the topica class(es) it underlies; the other models are Rust ports or reimplementations, validated against these reference implementations.

- [**MALLET**](https://github.com/mimno/Mallet) (McCallum, 2002) — `LDA`, `DMR`, `LabeledLDA`: the SparseLDA sampler, Dirichlet-multinomial regression, and hyperparameter optimization. `LDA` binds David Mimno's [**RustMallet**](https://github.com/mimno/RustMallet) (Apache-2.0), reproducing its `train` CLI byte-for-byte; against Java MALLET (a different RNG) it recovers the same topics (cosine 1.000)
- [**stm**](https://github.com/bstewart/stm) (Roberts, Stewart & Tingley, 2019) — `STM`, `CTM`, `SAGE`: variational EM, `estimateEffect`, `searchK`, FREX, spectral initialization, and the method of composition
- [**lda-c / ctm-c / dtm**](https://github.com/blei-lab) and [**hdp**](https://github.com/blei-lab/hdp) (Blei lab, 2006–2007) — `CTM`, `DTM`, `HDP`: the CTM, Dynamic Topic Model, and HDP samplers
- [**gensim**](https://github.com/piskvorky/gensim) (Řehůřek & Sojka, 2010) — `DTM`: coherence measures and the `LdaSeqModel` DTM reference
- [**tomotopy**](https://github.com/bab2min/tomotopy) (bab2min, 2020) — API conventions (`summary`, the short-text models)
- [**keyATM**](https://github.com/keyATM/keyATM) (Eshima, Imai & Sasaki, 2024) — `KeyATM`: the base, covariate, and dynamic models, the information-theory token weighting, and the Chib (1998) change-point HMM, validated against the package
- [**seededlda**](https://github.com/koheiw/seededlda) (Watanabe, 2023) — `SeededLDA`: the seeded-prior scheme
- [**LightLDA**](https://github.com/microsoft/LightLDA) (Yuan et al., 2015) — `LDA`: the alias-table Metropolis-Hastings sampler
- **GSDMM** (Yin & Wang, 2014) — `GSDMM`: the movie-group-process mixture for short text
- [**ProdLDA / AVITM**](https://arxiv.org/abs/1703.01488) (Srivastava & Sutton, 2017) — `ProdLDA`: autoencoding variational inference and the product-of-experts word model
- [**BERTopic**](https://github.com/MaartenGr/BERTopic) (Grootendorst, 2022) and [**Top2Vec**](https://github.com/ddangelov/Top2Vec) (Angelov, 2020) — `BERTopic`, `Top2Vec`: the embedding-clustering pipeline, class-based TF-IDF, and the `reduce → cluster → represent` design
- [**ETM**](https://github.com/adjidieng/ETM) (Dieng, Ruiz & Blei, 2020) — `ETM`: the Embedded Topic Model (per-document variational EM and an amortized VAE)
- [**FASTopic**](https://github.com/BobXWu/FASTopic) (Wu et al., 2024) — `FASTopic`: the optimal-transport topic model

The embedding-native models build on two pure-Rust crates: [**petal-clustering**](https://github.com/petabi/petal-clustering) for HDBSCAN and [**umap-rs**](https://github.com/wilsonzlin/umap-rs) for the optional UMAP reducer, both BLAS-free.

Full citations for every model and reference implementation, and how to cite topica, are on the [Citing](https://nealcaren.github.io/topica/citing/) page.

## License

Apache-2.0 — see [LICENSE](LICENSE).
