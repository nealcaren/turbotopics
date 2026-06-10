# Estimator contract

This page specifies the uniform interface every topica estimator must expose.
Follow it when adding a new model or porting an existing one. The conformance
test (`tests/test_conformance.py`) checks every registered estimator against
this contract on every CI run, so gaps in a new model fail the suite rather
than silently degrading the toolkit.

## Constructor

```python
Model(num_topics, *, seed=42, **hyperparameters)
```

Every estimator takes `seed` in its constructor (not in `fit`). If the model
requires non-standard arguments (for example a seed dict for `KeyATM`, a depth
for `HLDA`, or super/sub counts for `PA`), those come first; `seed` is always a
keyword-only argument.

## The three tiers

### Tier 0 — every estimator (the floor)

All 20 estimators in the registry must pass every Tier 0 check.

**fit signature**

```python
def fit(self, data, ..., *, iters: int = <default>, ...) -> Self:
    ...
```

`iters` is the canonical iteration-count keyword. It controls the number of
Gibbs sweeps, EM steps, or epochs, depending on the model family. Use `iters`
even where the underlying algorithm calls them something else. The `validation`
module's `_accepts_kwarg` helper checks for this at import time; do not hide
`iters` behind `**kwargs` unless you explicitly check `'iters' in kwargs`.

**Properties and methods**

| Name | Shape / return type | Notes |
|------|---------------------|-------|
| `topic_word` | `(K, V)` float array | Each row sums to 1 for generative models; c-TF-IDF weights for cluster models |
| `doc_topic` | `(D, K)` float array | Each row sums to 1 for mixed-membership models |
| `vocabulary` | sequence of str, length V | Vocabulary aligned with `topic_word` columns |
| `num_topics` | int | Number of topics K |
| `topic_names` | list of str, length K | User-set labels; default `["topic_0", "topic_1", ...]` |
| `doc_names` | list of str, length D | Row labels for `doc_topic`; default document indices as strings |
| `top_words(n)` | list of K lists of `(word, weight)` | Top n words per topic |
| `coherence(n)` | array of K floats | Per-topic UMass coherence over top n words |
| `save(path)` | — | Serialize to disk |
| `load(path)` | — | Class method; restore from disk |

### Tier 1 — generative models

Models whose `model_family` is `"dirichlet"` or `"logistic_normal"` must also
expose:

```python
def transform(self, docs) -> np.ndarray:  # shape (n_docs, K)
    ...
```

`transform` infers the topic mixture for new documents (without updating the
model). It is the basis for held-out perplexity, `eval_heldout`, and `search_k`.
Models with structural reasons for not supporting held-out inference (keyword-
constrained models such as `KeyATM` and `SeededLDA`, or group-anchored models
such as `SAGE` and `PA`) are listed in `EXEMPT` in `topica.conformance` and are
not required to implement `transform`.

### Tier 2 — family-specific attributes

**Dirichlet family** (`model_family == "dirichlet"`, i.e. collapsed-Gibbs models):

| Name | Shape | Notes |
|------|-------|-------|
| `alpha` | `(K,)` float array | Per-topic document-topic Dirichlet prior |
| `theta_draws` | `(S, D, K)` float array | Retained MCMC draws of the doc-topic matrix; shape `(0,)` when not collected (pass `keep_theta_draws=True` to `fit`) |
| `doc_lengths` | `(D,)` int array | Tokens per document; used by the Dirichlet theta sampler in `composition_theta` |

**Logistic-normal family** (`model_family == "logistic_normal"`, i.e. STM/CTM):

| Name | Shape | Notes |
|------|-------|-------|
| `eta_mean` | `(D, K-1)` float array | Variational mean of the logistic-normal document representation |
| `eta_cov` | `(D, K-1, K-1)` float array | Variational covariance |

## Principled exemptions

Some requirements will never apply to certain models because their statistics
differ structurally. These exemptions are recorded in
`topica.conformance.EXEMPT`. Do not add a workaround to make a model fake-pass
a requirement it cannot honestly satisfy; add it to `EXEMPT` instead with a
brief reason.

Current permanent exemptions:

| Model | Requirement | Reason |
|-------|-------------|--------|
| `HLDA` | `doc_topic` | Topic tree; documents have paths, not a (D, K) theta matrix |
| `HLDA` | `num_topics` | Tree model with `num_nodes`; no fixed scalar K |
| `HLDA` | `doc_names` | No static per-document row index |
| `HLDA` | `coherence` | Node-indexed tree structure is incompatible with flat (K, V) coherence |
| `HLDA` | `transform` | No flat K-topic generative distribution |
| `DTM` | `doc_topic` | Time-sliced; no single static theta matrix |
| `DTM` | `doc_names` | No static per-document row index |
| `DTM` | `coherence` | Time-varying phi is incompatible with flat (K, V) coherence |
| `DTM` | `transform` | No time-slice-free held-out inference |
| `BERTopic` | `coherence` | c-TF-IDF topic_word is not a probability; use `topica.coherence()` externally |
| `BERTopic` | `doc_names` | Exposes cluster `labels`, not a doc_names property |
| `Top2Vec` | `coherence` | Same as BERTopic |
| `Top2Vec` | `doc_names` | Same as BERTopic |
| `GSDMM` | `topic_names` | Mixture model (one topic per document); topic_names absent |
| `SAGE` | `transform` | Keyword-anchored model; no held-out transform |
| `PA` | `transform` | Super/sub-topic model; no held-out transform |
| `PT` | `transform` | Pseudo-topic model; no held-out transform |
| `KeyATM` | `transform` | Keyword-constrained; no held-out transform |
| `SeededLDA` | `transform` | Seeded variant of KeyATM; no held-out transform |

## Checking your model

Run the conformance test in isolation:

```bash
VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python -m pytest tests/test_conformance.py -v
```

Or call the helper directly from Python:

```python
import topica

violations = topica.check_conformance(MyNewModel(num_topics=5))
for v in violations:
    print(v)
```

An empty list means the model satisfies every applicable requirement. A non-
empty list tells you exactly which attributes or methods are missing before you
open a pull request.

## Adding a new estimator

1. Implement the model in `src/<name>.rs` with PyO3 bindings in `src/python.rs`
   (or in pure Python in `python/topica/`).
2. Export it from `python/topica/__init__.py`.
3. Add it to `REGISTRY` in `python/topica/conformance.py` with a zero-arg
   factory lambda and the correct `model_family` string.
4. Add the class to `_topica.pyi` to keep the stub in sync.
5. Run `topica.check_conformance(MyModel())` to find any missing attributes.
6. For each missing attribute:
   - If it is a structural impossibility for this model, add it to `EXEMPT`
     with a clear reason.
   - Otherwise, implement it so the check passes.
7. Run the full conformance test. All cells for the new model must either pass,
   be exempted, or (temporarily) be listed in `KNOWN_GAPS`.

Do not add entries to `KNOWN_GAPS` for a brand-new model unless you intend to
fix them in an immediately following commit. The purpose of `KNOWN_GAPS` is to
track pre-existing drift, not to grant new models a blanket deferred
implementation.
