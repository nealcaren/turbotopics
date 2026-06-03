# Topica

`topica` is a topic-modeling library with a Rust core and a numpy-native Python
API. It covers a family of models (from classic LDA to the Structural Topic
Model), fits them in native code (no JVM, no pure-Python inner loops), and keeps
every fit **deterministic for a given seed**. It is built for working social
scientists: every model ships with the validation, covariate-effect, and
reporting tools that contemporary social science journals expect, so the output
is an analysis that holds up in peer review, not a quick demonstration. Pass
pre-tokenized `list[list[str]]` (or a `Corpus`); get back plain numpy arrays.

```bash
pip install topica
```

```python
import topica

docs = [["cat", "dog", "fish"]] * 15 + [["planet", "star", "moon"]] * 15
model = topica.LDA(num_topics=2, seed=42)
model.fit(docs, iterations=1000)

for i, words in enumerate(model.top_words(5)):
    print(f"Topic {i}:", " ".join(w for w, _ in words))
```

## Why topica

- **One package, many models.** LDA, DMR, Labeled LDA, SAGE, CTM, the full STM
  (prevalence **and** content covariates), HDP, dynamic topics, supervised LDA,
  and short-text models. See [the models](guides/models.md).
- **Built for social science.** Covariate effects with the method of
  composition, **clustered standard errors**, GLM links, Fighting Words,
  intrusion tests, bootstrap stability, and `searchK`: the things reviewers ask
  for. See [covariates](guides/covariates.md) and [diagnostics](guides/diagnostics.md).
- **Fast and deterministic.** A Rust core with bit-for-bit reproducible fits. The
  variational models parallelize across cores automatically.
- **No heavy dependencies.** NumPy only. Optional integrations (pyLDAvis,
  matplotlib) light up if installed.

## The model families

| Model | What it's for |
|-------|---------------|
| [`LDA`](api/models.md#topica.LDA) | Classic topics via fast collapsed-Gibbs (SparseLDA) |
| [`DMR`](api/models.md#topica.DMR) | Topics conditioned on document metadata |
| [`LabeledLDA`](api/models.md#topica.LabeledLDA) | Supervised topics tied to document labels |
| [`CTM`](api/models.md#topica.CTM) | Correlated topics (logistic-normal) |
| [`STM`](api/models.md#topica.STM) | Structural Topic Model: prevalence **and** content covariates |
| [`SAGE`](api/models.md#topica.SAGE) | The same topic worded differently across groups |
| [`HDP`](api/models.md#topica.HDP) | Nonparametric LDA that *infers* the number of topics |
| [`DTM`](api/models.md#topica.DTM) | Dynamic topics that evolve across time slices |
| [`SupervisedLDA`](api/models.md#topica.SupervisedLDA) | Topics shaped to predict a per-document response |
| [`PT`](guides/short-text.md) / [`GSDMM`](guides/short-text.md) | Short-text models for tweets, survey answers |
| [`SeededLDA`](guides/guided.md) / [`KeyATM`](guides/guided.md) | Guided topics steered by seed words |
| `PA` / `HLDA` | Topic hierarchies (Pachinko, nested-CRP) |

## Worked examples

Three end-to-end analyses on real, redistributable corpora:

- [**W.E.B. Du Bois in *The Crisis***](examples/dubois.md): 704 articles,
  1910–1934, the full workflow from preprocessing to dynamic topics.
- [**Gadarian immigration experiment**](examples/gadarian.md): the canonical STM
  vignette, reproduced.
- [**Political blogs**](examples/poliblog.md): STM with ideology and time
  covariates.

---

topica is open source on [GitHub](https://github.com/nealcaren/topica).
