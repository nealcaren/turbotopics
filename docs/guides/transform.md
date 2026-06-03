# Held-out inference

After fitting, `transform` infers topic proportions (θ) for **new, unseen**
documents while holding the fitted topics (φ) fixed: a held-out test set, or
freshly collected texts.

```python
import topica

model = topica.LDA(num_topics=20, seed=42)
model.fit(train_docs, iterations=1000)

theta = model.transform(new_docs, seed=0)     # (len(new_docs), num_topics)
theta.argmax(axis=1)                          # dominant topic per new document
```

`transform` accepts a `list[list[str]]` or a `Corpus`. Out-of-vocabulary tokens
are dropped; a document with no in-vocabulary tokens gets the prior. Rows sum to
1, and results are deterministic for a fixed `seed`.

## Available across the model families

Each model uses the same inference it uses at fit time:

| Model | Inference |
|-------|-----------|
| `LDA`, `LabeledLDA`, `SupervisedLDA` | collapsed Gibbs against fixed φ |
| `HDP` | collapsed Gibbs over the discovered topics |
| `DMR` | collapsed Gibbs with `α_d = exp(Xγ)`; pass held-out `features` |
| `CTM`, `STM` | Laplace **variational** E-step against the logistic-normal prior |

For `CTM` / `STM`, the variational `transform` reproduces the model's own
training θ to ~`1e-3`. It is the same inference R's `stm` runs in
`fitNewDocuments`, not an approximation.

```python
# DMR with held-out covariates:
theta = dmr.transform(new_docs, features=new_X)

# STM variational inference (no Gibbs parameters needed):
theta = stm_model.transform(new_docs)
```
