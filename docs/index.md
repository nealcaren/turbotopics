# turbotopics

**Fast, all-purpose topic modeling for Python.** A Rust core gives you MALLET's
algorithm without the JVM, the Structural Topic Model without R, and a unified
toolkit of diagnostics, validation, and covariate analysis — all returning plain
NumPy arrays.

```bash
pip install turbotopics
```

```python
import turbotopics as tt

docs = [["cat", "dog", "fish"]] * 15 + [["planet", "star", "moon"]] * 15
model = tt.LDA(num_topics=2, seed=42)
model.fit(docs, iterations=1000)

for i, words in enumerate(model.top_words(5)):
    print(f"Topic {i}:", " ".join(w for w, _ in words))
```

## Why turbotopics

- **One package, many models.** LDA, DMR, Labeled LDA, SAGE, CTM, the full STM
  (prevalence **and** content covariates), HDP, dynamic topics, supervised LDA,
  and short-text models — see [the models](guides/models.md).
- **Built for social science.** Covariate effects with the method of
  composition, **clustered standard errors**, GLM links, Fighting Words,
  intrusion tests, bootstrap stability, and `searchK` — the things reviewers ask
  for. See [covariates](guides/covariates.md) and [diagnostics](guides/diagnostics.md).
- **Fast and deterministic.** A Rust core with bit-for-bit reproducible fits; the
  variational models parallelize across cores automatically.
- **No heavy dependencies.** NumPy only. Optional integrations (pyLDAvis,
  matplotlib) light up if installed.

## The model families

| Model | What it's for |
|-------|---------------|
| [`LDA`](api/models.md#turbotopics.LDA) | Classic topics via fast collapsed-Gibbs (SparseLDA) |
| [`DMR`](api/models.md#turbotopics.DMR) | Topics conditioned on document metadata |
| [`LabeledLDA`](api/models.md#turbotopics.LabeledLDA) | Supervised topics tied to document labels |
| [`CTM`](api/models.md#turbotopics.CTM) | Correlated topics (logistic-normal) |
| [`STM`](api/models.md#turbotopics.STM) | Structural Topic Model: prevalence **and** content covariates |
| [`SAGE`](api/models.md#turbotopics.SAGE) | The same topic worded differently across groups |
| [`HDP`](api/models.md#turbotopics.HDP) | Nonparametric LDA that *infers* the number of topics |
| [`DTM`](api/models.md#turbotopics.DTM) | Dynamic topics that evolve across time slices |
| [`SupervisedLDA`](api/models.md#turbotopics.SupervisedLDA) | Topics shaped to predict a per-document response |
| [`PT`](guides/short-text.md) / [`GSDMM`](guides/short-text.md) | Short-text models for tweets, survey answers |
| `PA` / `HLDA` | Topic hierarchies (Pachinko, nested-CRP) |

## Worked examples

Three end-to-end analyses on real, redistributable corpora:

- [**W.E.B. Du Bois in *The Crisis***](examples/dubois.md) — 704 articles,
  1910–1934: the full workflow from preprocessing to dynamic topics.
- [**Gadarian immigration experiment**](examples/gadarian.md) — the canonical STM
  vignette, reproduced.
- [**Political blogs**](examples/poliblog.md) — STM with ideology and time
  covariates.

---

turbotopics is open source on [GitHub](https://github.com/nealcaren/turbotopics).
