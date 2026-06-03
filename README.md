# Topica: fast, all-purpose topic modeling for Python

📖 **[Documentation](https://nealcaren.github.io/topica/)**: guides, a full API reference, worked examples, and a [*Publishing in a social science journal*](https://nealcaren.github.io/topica/publishing/) methodology track.

`topica` is a fast topic-modeling library for Python with more than a dozen models, built for social scientists who want to move from text data to publishable results in a single workflow. It brings together models and tools usually split across JVM software like MALLET and R packages like `stm`, and runs them on a parallel Rust core competitive with the standard implementations, with every fit reproducible from a fixed seed. Each model comes with the validation, covariate-effect, and reporting tools to meet the standards reviewers expect.

```bash
pip install topica            # once published; pre-built abi3 wheels, no Rust toolchain needed
```

```python
from topica import LDA

model = LDA(num_topics=2, seed=42)
model.fit([["cat", "dog", "fish"]] * 15 + [["planet", "star", "moon"]] * 15, iterations=1000)

for i, words in enumerate(model.top_words(5)):
    print(f"Topic {i}:", " ".join(w for w, _ in words))
```

See the [getting-started guide](https://nealcaren.github.io/topica/getting-started/) and the [worked examples](https://nealcaren.github.io/topica/examples/dubois/) for end-to-end analyses.

## Models

| Model | What it's for |
|-------|---------------|
| **`LDA`** | Classic topics via fast collapsed-Gibbs (SparseLDA); optional multi-threaded and LightLDA alias samplers |
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

Every model exposes the same shape: `fit(docs, …)`, then `topic_word` (φ), `doc_topic` (θ), `top_words(n)`, `transform(new_docs)`, and `save`/`load`. The variational models (`CTM`/`STM`/`SupervisedLDA`/`DTM`) parallelize across cores while staying bit-for-bit deterministic. Full guide: [the models](https://nealcaren.github.io/topica/guides/models/).

## Diagnostics & analysis

Model-agnostic: they work on any fitted model's `topic_word`/`doc_topic`:

- **Quality:** `coherence` (`u_mass`, `c_v`, `c_uci`, `c_npmi`; computed in the Rust core), `exclusivity`, `topic_diversity`, `quality_frontier`
- **Labeling:** `label_topics` (prob / FREX / lift / score), `frex`, `relevance`, `find_thoughts`, `topic_table`, `summary`
- **Validation:** `word_intrusion`, `document_intrusion`, `bootstrap_stability`, `search_k`
- **Comparison:** `fighting_words` (weighted log-odds) for contrasting corpora
- **`stm` toolkit:** `estimate_effect` (method of composition, **cluster-robust SEs**, GLM links), `posterior_theta_samples`, `spline`, `interaction`, `one_hot`, `topic_correlation`
- **Preprocessing:** `tokenize`, `learn_phrases` / `apply_phrases`, `split_documents`, the `Corpus` class

See [diagnostics](https://nealcaren.github.io/topica/guides/diagnostics/) and [covariate effects](https://nealcaren.github.io/topica/guides/covariates/).

## Install from source

```bash
pip install maturin
git clone https://github.com/nealcaren/topica && cd topica
python -m venv .venv && source .venv/bin/activate
maturin develop --release --features python
```

Requires `numpy >= 1.21`. Use `--release` (the debug build is much slower).

## Acknowledgements

Topica stands on a generation of open topic-modeling research and code. The `LDA` core binds David Mimno's [**RustMallet**](https://github.com/mimno/RustMallet) and reproduces [**MALLET**](https://github.com/mimno/Mallet)'s `train` output bit-for-bit; the other models are Rust ports or reimplementations, validated against their reference implementations:

- [**MALLET**](https://github.com/mimno/Mallet) (McCallum): SparseLDA, DMR, hyperparameter optimization
- [**stm**](https://github.com/bstewart/stm) (Roberts, Stewart & Tingley): the Structural Topic Model, `estimateEffect`, `searchK`, FREX, spectral initialization, method of composition
- [**lda-c / ctm-c / dtm**](https://github.com/blei-lab) and [**hdp**](https://github.com/blei-lab/hdp) (Blei lab): the CTM, Dynamic Topic Model, and HDP samplers
- [**gensim**](https://github.com/piskvorky/gensim): coherence measures and the `LdaSeqModel` DTM reference
- [**tomotopy**](https://github.com/bab2min/tomotopy) (bab2min): API conventions (`summary`, short-text models)
- [**keyATM**](https://github.com/keyATM/keyATM) (Eshima, Imai & Sasaki): the full keyword-assisted family. `KeyATM` ports their base, covariate, and dynamic models, including the information-theory token weighting and the Chib (1998) change-point HMM for topic prevalence over time, validated against the package
- [**seededlda**](https://github.com/koheiw/seededlda) (Watanabe): seeded LDA
- [**LightLDA**](https://github.com/microsoft/LightLDA) (Yuan et al.): the alias-table Metropolis-Hastings sampler
- **GSDMM** (Yin & Wang 2014): the movie-group-process mixture for short text

Underlying methods are credited to their authors in the [documentation](https://nealcaren.github.io/topica/) and the source. The SparseLDA scheme is Yao, Mimno & McCallum (KDD 2009). If you use `KeyATM`, please cite the original work:

> Eshima, S., Imai, K., & Sasaki, T. (2024). Keyword-Assisted Topic Models. *American Journal of Political Science*, 68(2), 730–750. [doi:10.1111/ajps.12779](https://doi.org/10.1111/ajps.12779)

## License

Apache-2.0. Builds on [RustMallet](https://github.com/mimno/RustMallet) (Apache-2.0).
