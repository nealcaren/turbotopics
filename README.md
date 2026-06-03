# turbotopics — fast, all-purpose topic modeling for Python

📖 **Documentation: [nealcaren.github.io/turbotopics](https://nealcaren.github.io/turbotopics/)** — guides, a full API reference, and a [*Publishing in a social science journal*](https://nealcaren.github.io/turbotopics/publishing/) methodology track.

`turbotopics` is a topic-modeling library with a Rust core and a numpy-native Python API. It covers a family of models, from classic LDA to the Structural Topic Model, fits them in native code (no JVM, no pure-Python inner loops), and keeps every fit **deterministic for a given seed**.

**Models:**

| Model | What it's for |
|-------|---------------|
| **`LDA`** | Classic topics via fast collapsed-Gibbs (SparseLDA), with optional multi-threaded training |
| **`DMR`** | Topics conditioned on document metadata (Dirichlet-multinomial regression) |
| **`LabeledLDA`** | Supervised topics tied to document labels |
| **`CTM`** | Correlated topics (logistic-normal) |
| **`STM`** | The Structural Topic Model: correlated topics with prevalence **and** content covariates |
| **`SAGE`** | Content-covariate topics — the same topic worded differently across groups |
| **`HDP`** | Nonparametric LDA that *infers* the number of topics |
| **`DTM`** | Dynamic topics that evolve across time slices |
| **`SupervisedLDA`** | Topics shaped to predict a per-document response |
| **`PT`** | Pseudo-document topics for short texts (Zuo et al. 2016) |
| **`GSDMM`** | One-topic-per-document mixture for short texts — tweets, survey answers (Yin & Wang 2014) |
| **`SeededLDA`** | Guided topics steered by seed words (seededlda; Watanabe) |
| **`KeyATM`** | Keyword-assisted topics (Eshima, Imai & Sasaki 2024) |
| **`PA`** | Pachinko Allocation: super-/sub-topic hierarchy |
| **`HLDA`** | Hierarchical LDA over a nested-CRP topic tree |

Every model takes pre-tokenized `list[list[str]]` (or a `Corpus`) and returns numpy arrays for downstream analysis, plus model-agnostic diagnostics (coherence, exclusivity, FREX/labeling, intrusion tests, topic alignment, Fighting Words), an `stm`-style toolkit (covariate effects with clustered SEs and GLM links, topic correlation, `searchK`), and fit diagnostics (held-out perplexity). The variational models (`CTM`/`STM`/`SupervisedLDA`/`DTM`) parallelize across cores automatically while staying bit-for-bit deterministic.

The implementations are validated, not approximations. The `LDA` core binds David Mimno's [RustMallet](https://github.com/mimno/RustMallet) and reproduces MALLET's `train` output bit-for-bit; the other models are Rust ports checked against their reference implementations: Java MALLET, the R `stm` package, and Blei's dynamic-topic-model code.

---

## Install

### From PyPI (once published)

```bash
pip install turbotopics
```

Pre-built abi3 wheels are provided for CPython >= 3.9 on Linux, macOS, and Windows. No Rust toolchain needed.

### From source (development)

```bash
pip install maturin
git clone https://github.com/nealcaren/turbotopics
cd turbotopics
python -m venv .venv && source .venv/bin/activate
maturin develop --release --features python
```

A `numpy` dependency is required (`numpy >= 1.21`). Use `--release` for an optimized build (the debug build is much slower).

---

## Quickstart

```python
from turbotopics import LDA

# Pre-tokenized documents — any list[list[str]]
animal_docs = [["cat", "dog", "fish", "cat", "dog"]] * 15
space_docs  = [["planet", "star", "moon", "rocket", "planet"]] * 15
documents   = animal_docs + space_docs

# Fit
model = LDA(num_topics=2, seed=42)
model.fit(documents, iterations=1000)

# Shapes
print(model.topic_word.shape)   # (2, 7)  — φ matrix (topics × words)
print(model.doc_topic.shape)    # (30, 2) — θ matrix (docs × topics), rows sum to 1

# Top words per topic
for i, words in enumerate(model.top_words(5)):
    print(f"Topic {i}:", "  ".join(f"{w}({p:.3f})" for w, p in words))

# Vocabulary in column order of topic_word
print(model.vocabulary)
```

Expected output (with `seed=42`):

```
Topic 0: cat(0.396)  dog(0.396)  fish(0.199)  planet(0.002)  star(0.002)
Topic 1: planet(0.396)  star(0.199)  moon(0.199)  rocket(0.199)  cat(0.002)
```

### A full worked example

[**`examples/dubois_tutorial.ipynb`**](examples/dubois_tutorial.ipynb) is an end-to-end notebook that runs the whole library (preprocessing, phrase detection, LDA, STM with topic prevalence over time via the method of composition, DTM, HDP, and held-out `transform`) over a corpus of 706 W.E.B. Du Bois articles from *The Crisis* (1910–1951), bundled as [`examples/dubois_crisis.csv`](examples/dubois_crisis.csv). It renders with outputs on GitHub; a script version is in [`examples/dubois_tutorial.py`](examples/dubois_tutorial.py).

---

## STM: Structural Topic Model (covariate-aware correlated topics)

The Structural Topic Model (Roberts, Stewart & Tingley) extends the Correlated Topic Model with **prevalence covariates**: the prior mean over latent topic weights is a regression on document-level variables, `μ_d = X_d γ`. Covariates (publication year, author group, treatment condition, etc.) shift *which* topics a document discusses. The model fits the CTM's variational E-step per document, then refits γ by ridge regression each M-step.

**When to use STM:** when you want correlated topics *and* you have metadata that explains variation in topic use. Use CTM when you only want the correlation structure; use DMR when you want covariate-driven topic prevalence under a Gibbs sampler.

**Two kinds of covariates:** STM supports `prevalence` covariates (shift *which* topics a document discusses) and `content` covariates (shift *how* a topic is worded across groups). You can use either or both; at least one is required (otherwise use CTM).

### Example

```python
import numpy as np
from turbotopics import STM, stm

# ── 1. Build a corpus with a strong binary covariate ────────────────────────
rng = np.random.default_rng(0)
policy_words = ["policy", "reform", "senate", "vote", "bill"]
culture_words = ["film", "music", "gallery", "festival", "art"]

docs, x = [], []
for _ in range(125):   # group x=1: mostly policy
    docs.append(rng.choice(policy_words, size=8).tolist()
                + rng.choice(culture_words, size=2).tolist())
    x.append(1.0)
for _ in range(125):   # group x=0: mostly culture
    docs.append(rng.choice(culture_words, size=8).tolist()
                + rng.choice(policy_words, size=2).tolist())
    x.append(0.0)

# ── 2. Fit STM with a (N, 1) prevalence matrix ──────────────────────────────
X = np.array(x).reshape(-1, 1)   # shape (250, 1); intercept prepended internally

model = STM(num_topics=2, seed=1)
model.fit(docs, X, prevalence_names=["is_policy"], em_iters=60)

# ── 3. Standard output arrays ────────────────────────────────────────────────
print(model.topic_word.shape)       # (2, 10)
print(model.doc_topic.shape)        # (250, 2), rows sum to 1
print(model.topic_correlation)      # (2, 2) Pearson correlation of θ
print(model.prevalence_effects)     # (2, 1): [intercept, is_policy] × [topic-1 ref]
print(model.feature_names)          # ['intercept', 'is_policy']

# ── 4. Estimate how the covariate shifts topic prevalence ───────────────────
effects = stm.estimate_effect(model.doc_topic, X, feature_names=["is_policy"])
for eff in effects:
    d = eff.as_dict()
    print(f"Topic {d['topic']}  R²={d['r_squared']:.3f}  "
          f"is_policy: coef={d['is_policy']['coef']:+.3f}  z={d['is_policy']['z']:+.1f}")
```

Expected output (topic order may vary by seed):

```
Topic 0  R²=0.99  is_policy: coef=+0.71  z=+8418.6
Topic 1  R²=0.99  is_policy: coef=-0.71  z=-8418.6
```

The `stm.estimate_effect` function (see the [stm analysis toolkit section](#stm-style-analysis-toolkit)) performs OLS of each topic's θ on the covariates. A large positive z for a topic means that covariate predicts higher prevalence of that topic. Use `feature_names=["is_policy"]` (length F, not including intercept) to label the columns.

### Key properties

| Property | Shape | Description |
|----------|-------|-------------|
| `topic_word` | `(K, V)` | Topic-word distributions β. |
| `doc_topic` | `(D, K)` | Document-topic proportions θ; rows sum to 1. |
| `topic_correlation` | `(K, K)` | Pearson correlation of θ across documents. Symmetric, unit diagonal. Same as CTM. |
| `prevalence_effects` | `(F+1, K-1)` | Learned γ; row 0 is the intercept (prepended internally). For interpretation use `stm.estimate_effect`. |
| `feature_names` | `list[str]` length F+1 | `"intercept"` first, then your `prevalence_names`. |
| `eta_mean` | `(D, K-1)` | Per-document variational posterior means λ of η. |
| `eta_cov` | `(D, K-1, K-1)` | Per-document variational posterior covariances ν. With `eta_mean`, the logistic-normal posterior used for method-of-composition uncertainty. |

### Covariate effects with proper uncertainty (method of composition)

`estimate_effect` on the point `doc_topic` gives OLS standard errors, but those treat θ as if it were observed exactly, understating uncertainty. R `stm` instead uses the **method of composition**: draw θ from the model's posterior, run the regression on each draw, and pool by Rubin's rules. turbotopics exposes the STM/CTM variational posterior (`eta_mean`, `eta_cov`), so you can do the same:

```python
from turbotopics import STM, stm

model = STM(num_topics=20, seed=1)
model.fit(docs, X, prevalence_names=["treatment", "pid"], em_iters=80)

# Draw theta from the variational posterior and pool the regressions.
draws   = stm.posterior_theta_samples(model, nsims=25, seed=1)   # (25, D, K)
effects = stm.estimate_effect(draws, X, feature_names=["treatment", "pid"])

for e in effects:
    ti = e.feature_names.index("treatment")
    print(f"topic {e.topic}: treatment coef={e.coef[ti]:+.3f}  z={e.z[ti]:+.2f}")
```

The pooled standard errors include the topic-estimation uncertainty, so z-statistics are honest (point OLS can report absurdly large z when θ is near-deterministic). For **nonlinear** prevalence (`~ s(day)`) and **interactions** (`~ treatment * party`), build the design matrix with `stm.spline` and `stm.interaction`:

```python
import numpy as np

day_basis, day_names = stm.spline(day, df=5)               # nonlinear time trend
tp, tp_names = stm.interaction(treatment, party)            # treatment x party
X = np.column_stack([treatment, party, day_basis, tp])
names = ["treatment", "party"] + day_names + tp_names
effects = stm.estimate_effect(draws, X, feature_names=names)
```

### sigma_shrink

Identical to CTM: shrinks Σ toward its diagonal at each M-step. Default `0.0` learns the full covariance; non-zero values regularize on small corpora.

### STM content covariates: topic wording varies by group

In addition to (or instead of) prevalence covariates, STM supports a `content` covariate: one group label per document. When `content` is supplied, the model learns per-group word distributions for each topic, so the same topic uses different vocabulary in different groups (a SAGE model embedded inside the STM variational inference engine). Use it when you want to ask *how* a topic is expressed differently across groups (political parties, languages, time periods, outlets) rather than just *how prevalent* each topic is.

Pass a list of group labels (strings or ints) as `content=`. Use `content_names` to fix a specific group order; otherwise groups are sorted.

```python
import numpy as np
from turbotopics import STM

# ── Bilingual corpus: same two topics (weather / food) in English and German ─
en_weather = ["rain", "sun", "cloud", "wind", "storm"]
de_weather = ["regen", "sonne", "wolke", "sturm", "nebel"]
en_food    = ["bread", "cheese", "wine", "apple", "meat"]
de_food    = ["brot",  "kaese",  "wein",  "apfel", "fleisch"]

rng = np.random.default_rng(42)
docs, groups = [], []

for _ in range(50):   # EN weather-heavy
    docs.append(rng.choice(en_weather, 10).tolist() + rng.choice(en_food, 2).tolist())
    groups.append("en")
for _ in range(50):   # EN food-heavy
    docs.append(rng.choice(en_food, 10).tolist() + rng.choice(en_weather, 2).tolist())
    groups.append("en")
for _ in range(50):   # DE weather-heavy
    docs.append(rng.choice(de_weather, 10).tolist() + rng.choice(de_food, 2).tolist())
    groups.append("de")
for _ in range(50):   # DE food-heavy
    docs.append(rng.choice(de_food, 10).tolist() + rng.choice(de_weather, 2).tolist())
    groups.append("de")

# ── Fit content-only STM ────────────────────────────────────────────────────
model = STM(num_topics=2, seed=1)
model.fit(docs, content=groups, em_iters=60)

# ── Per-group word distributions ─────────────────────────────────────────────
print(model.groups)                      # ['de', 'en']  (sorted)
print(model.topic_word_by_group.shape)   # (2, 2, 20)  — (topics, groups, words)

gi_en = model.groups.index("en")
gi_de = model.groups.index("de")

for t in range(model.num_topics):
    en_idx  = model.topic_word_by_group[t, gi_en, :].argsort()[-5:][::-1]
    de_idx  = model.topic_word_by_group[t, gi_de, :].argsort()[-5:][::-1]
    print(f"Topic {t}  EN top-5: {[model.vocabulary[i] for i in en_idx]}")
    print(f"         DE top-5: {[model.vocabulary[i] for i in de_idx]}")

# ── word_contrast: which words most distinguish EN vs DE wording ─────────────
for t in range(model.num_topics):
    contrast = model.word_contrast(t, "en", "de", n=5)
    print(f"Topic {t} words favouring EN over DE:")
    for word, log_ratio in contrast:
        print(f"  {word:12s}  log-ratio={log_ratio:+.2f}")
```

Expected output (topic index may vary by seed):

```
['de', 'en']
(2, 2, 20)
Topic 0  EN top-5: ['cloud', 'apple', 'rain', 'wine', 'storm']
         DE top-5: ['kaese', 'sturm', 'fleisch', 'brot', 'sonne']
Topic 1  EN top-5: ['cloud', 'apple', 'rain', 'wine', 'storm']
         DE top-5: ['kaese', 'sturm', 'fleisch', 'brot', 'sonne']
Topic 0 words favouring EN over DE:
  cloud         log-ratio=+4.50
  apple         log-ratio=+4.49
  rain          log-ratio=+4.45
  ...
```

#### Combining prevalence and content

Prevalence and content covariates can be used together. The model simultaneously learns per-document topic shifts (from `prevalence`) and per-group vocabulary shifts (from `content`):

```python
X = np.ones((200, 1))   # some prevalence covariate, shape (N, F)

model = STM(num_topics=2, seed=1)
model.fit(
    docs,
    X,
    prevalence_names=["my_covariate"],
    content=groups,
    em_iters=60,
)

print(model.prevalence_effects.shape)    # (2, 1)  — (F+1, K-1)
print(model.feature_names)              # ['intercept', 'my_covariate']
print(model.topic_word_by_group.shape)  # (2, 2, V)
print(model.groups)                     # ['de', 'en']
```

#### Content-only STM: key properties

| Property | Shape | Description |
|----------|-------|-------------|
| `topic_word` | `(K, V)` | Group-marginal topic-word distributions. |
| `topic_word_by_group` | `(K, G, V)` | Per-group word distributions; each `(K, g, :)` row sums to 1. `RuntimeError` if fit without content. |
| `groups` | `list[str]` length G | Group names in index order. `RuntimeError` if fit without content. |
| `doc_topic` | `(D, K)` | Document-topic proportions θ; rows sum to 1. |

#### `word_contrast(topic, group_a, group_b, n=10)`

Returns the `n` words that most distinguish how `topic` is worded in `group_a` versus `group_b`, as `list[(word, log_ratio)]`. Positive log-ratio means the word favours `group_a`; negative means it favours `group_b`. Groups may be named by their string label or by their integer index into `model.groups`. Raises `ValueError` if `topic` is out of range or either group is unknown, `RuntimeError` if the model was fit without content.

---

## CTM: correlated topics (the STM core)

The Correlated Topic Model (Blei & Lafferty 2007) places a **logistic-normal prior** with a full covariance matrix Σ over the per-document topic proportions. Because documents draw from a multivariate normal before the softmax, topics can be positively or negatively correlated, which LDA's Dirichlet prior cannot represent. `CTM` is the only variational (non-Gibbs) model in `turbotopics`.

The model is fit by **variational EM**: the E-step approximates each document's posterior over latent topic weights via Newton's method (a port of STM's `lhoodcpp`/`gradcpp`/`hpbcpp` Laplace approximation); the M-step updates the global topic-word distributions β and the covariance Σ. This is the inference engine that STM's prevalence and content covariate models build on.

The main output is `topic_correlation`: a `(num_topics, num_topics)` matrix giving the Pearson correlation of θ across documents. Topics that tend to co-occur in the same documents get a higher (less negative) correlation; topics that rarely co-occur get a lower (more negative) correlation. All correlations may be negative because topic proportions live on a simplex, so what matters is the *relative* ordering.

### Example

```python
import numpy as np
from turbotopics import CTM

# --- Build a corpus with a known co-occurrence structure ---
# Topics A and B co-occur often; topic C is isolated.
rng = np.random.default_rng(0)
vocab_a = ["policy", "reform", "senate", "vote", "bill"]
vocab_b = ["election", "campaign", "ballot", "primary", "candidate"]
vocab_c = ["galaxy", "nebula", "orbit", "comet", "telescope"]

docs = []
# 40%: A+B together
for _ in range(120):
    docs.append(rng.choice(vocab_a, 8).tolist() + rng.choice(vocab_b, 8).tolist())
# 20%: A alone
for _ in range(60):
    docs.append(rng.choice(vocab_a, 8).tolist())
# 20%: B alone
for _ in range(60):
    docs.append(rng.choice(vocab_b, 8).tolist())
# 20%: C alone
for _ in range(60):
    docs.append(rng.choice(vocab_c, 8).tolist())

# --- Fit CTM ---
model = CTM(num_topics=3, seed=1)
model.fit(docs, em_iters=50)

# --- Top words per topic ---
for i, words in enumerate(model.top_words(5)):
    print(f"Topic {i}:", "  ".join(f"{w}({p:.3f})" for w, p in words))

# --- Inspect the correlation matrix ---
print("\ntopic_correlation:")
print(np.round(model.topic_correlation, 3))
# Topics corresponding to A and B should be *more correlated* (less negative)
# than either is with C, because A and B frequently appear in the same documents.
```

Expected output (topic index order may vary by seed):

```
Topic 0: galaxy(0.239)  telescope(0.218)  ...   ← C
Topic 1: policy(0.239)  reform(0.219)  ...       ← A
Topic 2: election(0.215)  campaign(0.209)  ...   ← B

topic_correlation:
[[ 1.    -0.576 -0.602]
 [-0.576  1.    -0.304]
 [-0.602 -0.304  1.   ]]
```

Here A (topic 1) and B (topic 2) have correlation −0.30, while A–C and B–C are −0.58 and −0.60. The A/B pair is far less anti-correlated than either pair involving C, the co-occurrence structure we built in.

### When to use CTM vs LDA

Use CTM when the research question involves **which topics co-occur** (or are mutually exclusive) across documents. LDA cannot model inter-topic dependence because its Dirichlet prior has independent marginals. CTM is the right model when you want a correlation network over topics, or when you plan to use STM's covariate machinery and need the logistic-normal document model underneath.

### `sigma_shrink`: regularising the covariance

`sigma_shrink ∈ [0, 1]` shrinks Σ toward its diagonal at each M-step. Setting it to `0.0` (the default) learns a fully off-diagonal Σ; values closer to `1.0` pull the estimate toward a diagonal (i.e., toward decorrelated topics). Use a non-zero value with small corpora where the full Σ is under-identified.

```python
model = CTM(num_topics=10, sigma_shrink=0.3, seed=42)
model.fit(docs, em_iters=100)
```

---

## Working with a Corpus

Passing `list[list[str]]` to `fit()` builds an internal corpus with no frequency filtering. For filtering, document metadata, or loading preprocessed data, use the `Corpus` class.

### Building from token lists

```python
from turbotopics import Corpus

corpus = Corpus.from_documents(
    documents,
    doc_names=["doc{:03d}".format(i) for i in range(len(documents))],
    doc_labels=["animals"] * 15 + ["space"] * 15,  # optional metadata
    min_doc_freq=2,          # drop words appearing in fewer than 2 documents
    max_doc_fraction=0.95,   # drop words appearing in more than 95% of docs
    stopwords=["the", "a"],  # optional inline stoplist
)

print(corpus.num_docs, corpus.num_words, corpus.total_tokens)
model.fit(corpus, iterations=1000)
```

Documents emptied by filtering are dropped automatically.

### Loading from a text file

```python
# Plain text — one document per line
corpus = Corpus.from_text_file("docs.txt")

# MALLET-style TSV — columns: id, label, text
corpus = Corpus.from_text_file(
    "docs.tsv",
    format="tsv",           # "plain" (default) or "tsv"
    id_column=0,
    label_column=1,
    text_column=2,
    stopwords=open("stopwords.txt").read().splitlines(),
    min_doc_freq=5,
    max_doc_fraction=0.90,
)
```

`token_regex=None` (the default) uses `DEFAULT_TOKEN_REGEX`, a Unicode-letter pattern matching the upstream `preprocess` CLI. Tokenization lowercases all tokens.

### Tokenizing raw text: `turbotopics.tokenize()`

For convenience when building `list[list[str]]` input outside of `Corpus.from_text_file`, use the module-level `tokenize` function. It applies the same regex-based tokenization as the corpus loader.

```python
from turbotopics import tokenize

tokens = tokenize("Well-known methods for the U.S.A. study.")
# ['well-known', 'methods', 'for', 'the', 'u.s.a', 'study']

tokens = tokenize(
    "The quick brown fox jumps.",
    stopwords=["the", "a"],    # drop stopwords
    min_length=4,              # drop tokens shorter than 4 characters
)
# ['quick', 'brown', 'jumps']

# Custom regex (e.g. ASCII letters only)
tokens = tokenize("Hello, world! 123 test.", token_regex=r"[a-zA-Z]+")
# ['hello', 'world', 'test']

# Build list[list[str]] for a set of raw strings
documents = [tokenize(text, stopwords=stoplist) for text in raw_texts]
model.fit(documents, iterations=1000)
```

`lowercase=True` (default) lowercases all tokens before matching. An invalid `token_regex` raises `ValueError`.

### Save and load

```python
corpus.save("corpus.corp")           # binary format, compatible with CLI tools
corpus2 = Corpus.load("corpus.corp") # reload for reuse
```

The `.corp` format is shared with the `preprocess`/`train`/`show`/`analyze` CLI tools from the upstream RustMallet project.

---

## Saving and loading models

Every fitted model serializes to a compact binary file with `save(path)` and reloads with the class's `load(path)`, so you train once and reuse the model later without refitting:

```python
from turbotopics import LDA

model = LDA(num_topics=50, seed=1)
model.fit(documents, iterations=1000)
model.save("my_model.tt")

# ...later, or in another process:
model = LDA.load("my_model.tt")
print(model.topic_word.shape)
theta = model.transform(new_documents)   # inference still works after loading
```

This works for **every model** (`LDA`, `DMR`, `LabeledLDA`, `SAGE`, `CTM`, `STM`, `HDP`, `DTM`, `SupervisedLDA`). The full fitted state is preserved, including each model's distinguishing outputs: STM's variational posterior `eta_mean`/`eta_cov` and prevalence effects, SupervisedLDA's regression coefficients, the LDA sampler state needed for `transform`, and so on. Saving an unfitted model raises; loading a corrupt or non-model file raises `ValueError`.

---

## Labeled LDA: supervised topics from document labels

Labeled LDA (Ramage et al. 2009) is a supervised extension of LDA where each document carries a set of string labels. Every distinct label becomes a topic, and a document's tokens are constrained to be assigned only to its labels' topics. The model learns a dedicated word distribution for each label, which keeps topic interpretation straightforward, and the `doc_topic` matrix gives the per-label proportions for every document.

**When to use Labeled LDA:** when you have pre-existing label annotations on documents (e.g. tags, categories, codes) and want to learn the vocabulary associated with each label, or to measure each document's mixture across its assigned labels. The number of topics is set automatically from the labels; you do not specify it.

### Example

```python
import numpy as np
from turbotopics import LabeledLDA

# Pre-tokenized documents, one label list per document
docs = [
    ["football", "soccer", "goalkeeper", "penalty"],   # sports
    ["election", "senate", "ballot", "campaign"],       # politics
    ["algorithm", "database", "network", "software"],  # tech
    ["basketball", "playoff", "defense", "rebound"],   # sports
    ["congress", "legislation", "senate", "vote"],     # politics + sports: two labels
    ["basketball", "election", "campaign"],            # sports + politics
]
labels = [
    ["sports"],
    ["politics"],
    ["tech"],
    ["sports"],
    ["politics", "sports"],
    ["sports", "politics"],
]

# Fit — num_topics is inferred from the union of all labels
model = LabeledLDA(alpha=0.1, seed=42)
model.fit(docs, labels, iterations=500)

# Topic order: sorted union of all labels
print(model.labels)       # ['politics', 'sports', 'tech']
print(model.num_topics)   # 3

# Top words per label — use model.labels to map column index to label name
for t, label_name in enumerate(model.labels):
    words = model.top_words(5, topic=t)
    print(f"{label_name}:", "  ".join(f"{w}({p:.3f})" for w, p in words))

# doc_topic: label proportions for each document (rows sum to 1)
# Only a document's own labels are non-zero
print(model.doc_topic.shape)      # (6, 3)
print(model.doc_topic.sum(axis=1))  # all 1.0

# For a document with labels ["politics", "sports"], the "tech" column is 0
# For a single-label document, that label's column is ~1.0
```

### Fixing topic order with `label_names`

By default topics are ordered by sorted label names. Pass `label_names` to fix a custom order:

```python
model.fit(docs, labels, label_names=["tech", "sports", "politics"])
# model.labels == ["tech", "sports", "politics"]
# topic 0 is "tech", topic 1 is "sports", topic 2 is "politics"
```

### Unconstrained documents

A document with an empty label list `[]` is treated as unconstrained: all topics are allowed. This is useful when some documents lack annotations:

```python
labels_with_missing = [["sports"], ["politics"], []]   # third doc unconstrained
model.fit(docs[:3], labels_with_missing, label_names=["politics", "sports", "tech"])
# The third doc's doc_topic row sums to 1; any topic may be non-zero
```

### Coherence

```python
import numpy as np
c = model.coherence(n=10)   # shape (num_topics,); UMass coherence per label-topic
print("Mean coherence:", np.mean(c))
```

---

## SAGE: content covariates (topics worded differently by group)

SAGE (Sparse Additive Generative model) is a content-covariate topic model. Like LDA, it learns shared latent topics, but unlike LDA the word distribution for each topic varies by a document-level **group** covariate. The generative model is additive in log-space:

```
log β_{k,g,v} = m_v + κT_{k,v} + κC_{g,v} + κI_{k,g,v}
```

where `m_v` is a corpus background, `κT_{k,v}` is the topic deviation, `κC_{g,v}` is the group deviation, and `κI_{k,g,v}` is their interaction. The κ deviations are MAP-estimated by L-BFGS with a Gaussian (L2) prior.

**When to use SAGE:** when you want to ask *how* a topic is expressed differently across groups, not just *how prevalent* a topic is. Typical uses: comparing how different political parties discuss the same issue, how the same themes appear in texts from different time periods or languages, or how news outlets frame shared topics with different vocabulary.

**SAGE vs DMR:** DMR models variation in topic *prevalence* (the document-topic prior shifts by covariates). SAGE models variation in topic *content* (the same topic uses different words in different groups). Use DMR when you want to know whether a covariate predicts topic proportions; use SAGE when you want to know whether a covariate predicts topic vocabulary.

### Example

```python
import numpy as np
from turbotopics import SAGE

# Two groups ("en" / "de") with disjoint vocabulary for the same two topics
en_weather = ["rain", "sun", "cloud", "wind", "storm"]
de_weather = ["regen", "sonne", "wolke", "sturm", "nebel"]
en_food    = ["bread", "cheese", "wine", "apple", "meat"]
de_food    = ["brot",  "kaese",  "wein",  "apfel", "fleisch"]

rng = np.random.default_rng(0)
docs, groups = [], []

# English docs: weather-heavy and food-heavy
for _ in range(50):
    docs.append(rng.choice(en_weather, size=10).tolist() + rng.choice(en_food, size=2).tolist())
    groups.append("en")
for _ in range(50):
    docs.append(rng.choice(en_food, size=10).tolist() + rng.choice(en_weather, size=2).tolist())
    groups.append("en")

# German docs: same two topic proportions, different vocabulary
for _ in range(50):
    docs.append(rng.choice(de_weather, size=10).tolist() + rng.choice(de_food, size=2).tolist())
    groups.append("de")
for _ in range(50):
    docs.append(rng.choice(de_food, size=10).tolist() + rng.choice(de_weather, size=2).tolist())
    groups.append("de")

# Fit SAGE
model = SAGE(num_topics=2, seed=1)
model.fit(docs, groups, iterations=500)

# topic_word is 3D: (num_topics, num_groups, num_words)
print(model.topic_word.shape)    # e.g. (2, 2, 20)
print(model.groups)              # ['de', 'en'] (sorted)

# Per-group top words: same topic, different vocabulary
for t in range(model.num_topics):
    print(f"\nTopic {t}:")
    en_words = [w for w, _ in model.top_words(t, group="en", n=5)]
    de_words = [w for w, _ in model.top_words(t, group="de", n=5)]
    print(f"  en: {en_words}")
    print(f"  de: {de_words}")

# Group-marginal top words (averaged across groups)
for t in range(model.num_topics):
    marginal = [w for w, _ in model.top_words(t, n=5)]
    print(f"Topic {t} marginal: {marginal}")

# word_contrast: which words most distinguish how a topic is worded in 'en' vs 'de'
contrast = model.word_contrast(0, group_a="en", group_b="de", n=5)
print("\nTopic 0 words favouring 'en' over 'de':")
for word, log_ratio in contrast:
    print(f"  {word:12s}  log-ratio={log_ratio:+.2f}")
```

Expected output (with `seed=1`, topic order may vary):

```
(2, 2, 20)
['de', 'en']

Topic 0:
  en: ['apple', 'wine', 'meat', 'bread', 'cheese']
  de: ['kaese', 'brot', 'fleisch', 'apfel', 'wein']

Topic 1:
  en: ['rain', 'cloud', 'storm', 'wind', 'sun']
  de: ['sonne', 'regen', 'sturm', 'nebel', 'wolke']

Topic 0 words favouring 'en' over 'de':
  apple         log-ratio=+4.87
  wine          log-ratio=+4.83
  bread         log-ratio=+4.82
  meat          log-ratio=+4.79
  cheese        log-ratio=+4.76
```

### group_names: fixing group order

By default groups are ordered by their sorted labels. Pass `group_names` to fix a specific order:

```python
model.fit(docs, groups, group_names=["en", "de"])
# model.groups == ["en", "de"]
# topic_word[:, 0, :] is the 'en' distribution; topic_word[:, 1, :] is 'de'
```

### Accessing top_words by index

`top_words(topic, group=...)` accepts a group label (str) or a group index (int):

```python
model.top_words(0, group="en")   # by name
model.top_words(0, group=1)      # by index into model.groups
# both return the same list
```

---

## DMR: topics conditioned on document metadata

Dirichlet-Multinomial Regression (DMR; Mimno & McCallum, 2008) is an extension of LDA where each document gets its own topic prior, shaped by observed covariates. Instead of a shared alpha vector, the per-document prior is:

```
α_{d,t} = exp(λ_t · x_d)
```

where `x_d` is a vector of document features (year, author group, treatment condition, outlet, etc.) and `λ_t` is a learned weight vector for topic `t`. The weights `λ` are estimated by L-BFGS during Gibbs sampling, with a Gaussian prior on the weights for regularization. The `feature_effects` property holds the learned `λ` matrix; its values tell you which covariates raise or lower each topic's prevalence.

**When to use DMR:** when you want covariate-aware topics, for example to ask whether news articles from different outlets have different topic distributions, whether topic prevalence changes over time, or whether a treatment group produces different topic patterns than a control group.

**Fidelity note:** DMR involves L-BFGS optimization of the feature weights, so (unlike plain LDA) it is **not** bit-identical to Java MALLET. Fidelity is validated statistically, by covariate recovery and comparable perplexity, not byte-for-byte.

### Example

```python
import numpy as np
import pandas as pd
from turbotopics import DMR, one_hot

# ── 1. Build documents and features ─────────────────────────────────────────

# Suppose each document has a publication year and an outlet type
years   = [2010, 2012, 2015, 2010, 2018, 2015, 2020, 2012]
outlets = ["broadsheet", "tabloid", "broadsheet", "tabloid",
           "broadsheet", "tabloid",  "broadsheet", "broadsheet"]
# (docs is your list[list[str]] or Corpus)
docs = [
    ["economy", "trade", "market"],
    ["celebrity", "gossip", "scandal"],
    ["economy", "policy", "election"],
    ["celebrity", "crime", "scandal"],
    ["climate", "policy", "election"],
    ["gossip", "celebrity", "crime"],
    ["climate", "trade", "policy"],
    ["economy", "election", "policy"],
]

# Continuous feature: center the year
year_feature = np.array(years, dtype=float)
year_feature -= year_feature.mean()

# Categorical feature: one-hot encode outlet (drop_first omits the reference level)
outlet_matrix, outlet_names = one_hot(outlets, drop_first=True, prefix="outlet_")
# outlet_names = ["outlet_tabloid"]  (broadsheet is the dropped reference)

# Stack all features into a single (num_docs, F) array
features = np.column_stack([year_feature, outlet_matrix])
feature_names = ["year_centered"] + outlet_names

# ── 2. Fit DMR ───────────────────────────────────────────────────────────────

model = DMR(num_topics=3, seed=42, prior_variance=1.0)
model.fit(
    docs,
    features,
    feature_names=feature_names,
    iterations=1000,
)

# ── 3. Interpret feature effects ─────────────────────────────────────────────

# feature_effects shape: (num_topics, num_features)
# Column 0 is the intercept (prepended automatically); then your feature columns.
df = pd.DataFrame(
    model.feature_effects,
    columns=model.feature_names,       # ["intercept", "year_centered", "outlet_tabloid"]
    index=[f"Topic {t}" for t in range(model.num_topics)],
)
print(df)
# Positive values in "year_centered" mean that topic grew more prevalent over time.
# Positive values in "outlet_tabloid" mean tabloids use that topic more than broadsheets.

# Top words per topic to label them
for i, words in enumerate(model.top_words(5)):
    print(f"Topic {i}:", "  ".join(f"{w}({p:.3f})" for w, p in words))

# Document-topic matrix (num_docs, num_topics), rows sum to 1
print(model.doc_topic.shape)
```

The `one_hot` helper encodes any categorical covariate into a feature matrix ready for `DMR.fit`:

```python
matrix, names = one_hot(values, drop_first=True, prefix="outlet_")
# drop_first=True omits the first sorted category as the reference level (standard practice)
# names = column labels to pass as feature_names (without "intercept", which is prepended)
```

---

## HDP: inferring the number of topics

The Hierarchical Dirichlet Process topic model (Teh, Jordan, Beal & Blei 2006) is **nonparametric LDA**: instead of fixing `num_topics`, it *infers* how many topics the corpus supports. A corpus-level Dirichlet process (concentration `gamma`) defines a shared, unbounded menu of topics; each document is its own Dirichlet process (concentration `alpha`) drawing a mixture from that menu. New topics appear as the data demands and unused ones die off, so `K` is learned rather than chosen.

It is fit by the **direct-assignment Gibbs sampler** (the Chinese Restaurant Franchise), with a sequential-CRF initialization that grows a compact, well-separated topic set, and by default resamples both concentration parameters from the data (a port of the `blei-lab/hdp` updates). You usually don't tune `alpha`/`gamma`; pass a corpus and read `num_topics` off the fitted model.

**When to use HDP:** when you don't know `K` and want the model to find it, or to sanity-check a `K` you picked for LDA/CTM. HDP tends to slightly over-segment, which is normal; treat the inferred `K` as a grounded estimate, not an exact count.

### Example

```python
from turbotopics import HDP

# `documents` is a list[list[str]] (or a Corpus).
model = HDP(seed=1)            # alpha/gamma resampled from the data by default
model.fit(documents, iters=150)

print(model.num_topics)        # the INFERRED number of topics, K
print(model.alpha, model.gamma)  # the fitted DP concentrations
print(model.topic_word.shape)  # (K, num_words) — rows sum to 1
print(model.doc_topic.shape)   # (num_docs, K) — rows sum to 1

for t, words in enumerate(model.top_words(8)):
    print(f"topic {t}:", " ".join(w for w, _ in words))
```

### Constructor and key properties

`HDP(*, alpha=1.0, gamma=1.0, eta=0.01, seed=42, resample_conc=True)` — `alpha` (document-level) and `gamma` (corpus-level) are the initial DP concentrations, resampled each sweep when `resample_conc=True`; `eta` is the topic-word Dirichlet. All three must be `> 0`.

| Property | Shape | Meaning |
|---|---|---|
| `num_topics` | `int` | The **inferred** number of topics K. |
| `topic_word` | `(K, V)` | φ — topic-word distributions; rows sum to 1. |
| `doc_topic` | `(D, K)` | θ — document-topic mixtures; rows sum to 1. |
| `alpha`, `gamma` | `float` | The fitted DP concentrations. |

Methods: `fit(data, *, iters=150)`, `top_words(n=10, *, topic=None)`, `coherence(n=10)`; plus `vocabulary` and `doc_names`. Fitting is deterministic for a fixed `seed`.

---

## DTM: topics that evolve over time

The Dynamic Topic Model (Blei & Lafferty 2006) lets a topic's vocabulary **drift across time**. You split the corpus into ordered time slices (years, say); each topic's word distribution at slice *t* is tied to its distribution at slice *t−1* through a Gaussian random walk, so the topic evolves smoothly rather than being re-estimated independently per slice. You can then trace *how* a topic's language changes over time: which words rise and fall within a persistent theme.

Inference is variational with Kalman smoothing over the slices, a port of Blei's C `dtm` and its gensim `LdaSeqModel` reincarnation: a per-document LDA variational E-step against each slice's topics, and a per-topic state-space M-step (optimized here with L-BFGS on the same `f_obs`/`df_obs` objective gensim hands to its conjugate-gradient solver).

**When to use DTM:** when documents are timestamped and you care about *change*, how a topic's terminology shifts year to year, not just which topics exist. The key knob is `chain_variance`: small (the 0.005 default) keeps topics nearly stationary; larger values let them move more freely between slices.

### Example

```python
from turbotopics import DTM

# documents in any order; `times[d]` is document d's slice index (0-based, contiguous).
model = DTM(num_topics=5, seed=1)
model.fit(documents, times, em_iters=20)

print(model.num_times)              # number of time slices (inferred as max(times)+1)

# A topic's top words at each slice — watch them change:
for t in range(model.num_times):
    words = model.top_words(topic=0, time=t, n=6)
    print(f"slice {t}:", " ".join(w for w, _ in words))

# Trace one word's probability within a topic over time (great for plotting):
traj = model.word_evolution(topic=0, word="climate")   # shape (num_times,)

# Or see which words drive a topic's drift (first vs last slice):
drift = model.word_drift(topic=0, n=10)
print("rising:", [w for w, _ in drift["rising"]], "falling:", [w for w, _ in drift["falling"]])

# Full topic-word matrix at a given slice (rows sum to 1):
phi_t0 = model.topic_word(time=0)   # shape (num_topics, num_words)
```

### Constructor and key methods

`DTM(num_topics, *, alpha=0.01, chain_variance=0.005, obs_variance=0.5, seed=42)` — `num_topics >= 2`; `chain_variance` controls how much a topic may drift between adjacent slices; `obs_variance` is the observation noise. All three must be `> 0`.

| Member | Returns | Meaning |
|---|---|---|
| `fit(data, times, *, em_iters=20)` | `None` | `times` is each document's integer slice index, 0-based and contiguous. |
| `topic_word(time)` | `(K, V)` array | Topic-word distributions at one slice; rows sum to 1. |
| `word_evolution(topic, word)` | `(num_times,)` array | A word's probability trajectory in a topic across slices (`word` is a string or id). |
| `word_drift(topic, *, n=10, from_time=0, to_time=None)` | `dict` | Words that rose/fell most within a topic between two slices: `{"rising": [(word, Δ)], "falling": [...]}`. |
| `top_words(topic, time, n=10)` | `list[(word, prob)]` | Top words for a topic at one slice. |
| `num_topics`, `num_times`, `bound`, `vocabulary` | — | K, slice count, final ELBO, and the vocab. |

Fitting is deterministic for a fixed `seed`.

---

## Supervised LDA: topics that predict a response

Supervised LDA (Blei & McAuliffe 2007) attaches a **real-valued response** to each document and fits the topics so they *predict* it. The response is modeled as `y_d ~ N(ηᵀ z̄_d, σ²)`, a linear regression on the document's empirical topic frequencies `z̄_d`. Because the response feeds back into inference, topics are pulled toward directions that explain `y`, and the fitted coefficients `η` tell you how each topic moves the response. You can then `predict` the response for new, unlabeled documents.

Use it when documents come with an outcome you care about (a rating, a price, a sentiment score, a vote share) and you want topics that are both interpretable *and* predictive, plus a per-topic read on which themes push the outcome up or down.

Inference is the variational EM of Blei & McAuliffe: a per-document coordinate ascent on the variational parameters (with the response-coupling term that ties a document's words together through `η`/`σ²`), then an M-step that re-estimates the topics `β`, the coefficients `η` (by the regression normal equations), and the noise variance `σ²`.

### Example

```python
import numpy as np
from turbotopics import SupervisedLDA

# `documents` is list[list[str]]; `y` is one real number per document.
model = SupervisedLDA(num_topics=10, seed=1)
model.fit(documents, y, em_iters=25)

print(model.coefficients)   # eta: shape (num_topics,) — how each topic moves y
print(model.sigma2)         # fitted response variance

# Which topics push the response up / down?
order = np.argsort(model.coefficients)
print("lowers y most:", model.top_words(6, topic=int(order[0])))
print("raises  y most:", model.top_words(6, topic=int(order[-1])))

# Predict the response for new, unlabeled documents (OOV words are ignored):
y_hat = model.predict(new_documents)   # shape (len(new_documents),)
```

### Constructor and key members

`SupervisedLDA(num_topics, *, alpha=0.1, seed=42)` — `num_topics >= 2`, `alpha > 0`.

| Member | Returns | Meaning |
|---|---|---|
| `fit(data, y, *, em_iters=25, var_iters=15)` | `None` | `y` is the per-document response (length = #docs). |
| `predict(data, *, var_iters=20)` | `(n_docs,)` array | ŷ for new documents; out-of-vocabulary words ignored. |
| `coefficients` | `(K,)` array | Regression weights η — each topic's effect on the response. |
| `sigma2` | `float` | Fitted response variance. |
| `topic_word`, `doc_topic` | arrays | φ `(K, V)` and θ `(D, K)` (rows sum to 1). |
| `top_words(n=10, *, topic=None)`, `coherence(n=10)` | — | As in the other models. |

Fitting is deterministic for a fixed `seed`.

---

## Inferring topics for new documents

After fitting a model, use `transform()` to infer the document-topic distribution (θ) for documents that were not part of training, such as a held-out test set or freshly collected texts. The fitted topic-word distributions (φ) are held fixed; only the new documents' topic assignments are inferred.

```python
from turbotopics import LDA, tokenize

# Train on existing documents
train_docs = [tokenize(text) for text in train_texts]
model = LDA(num_topics=20, seed=42)
model.fit(train_docs, iterations=1000)

# Infer topic distributions for new documents
new_docs = [tokenize(text) for text in new_texts]
theta = model.transform(new_docs, seed=0)  # shape (len(new_docs), num_topics)

# Each row sums to 1.0
print(theta.shape)           # (n_new_docs, 20)
print(theta[0].sum())        # 1.0

# Dominant topic for each new document
dominant_topics = theta.argmax(axis=1)
print(dominant_topics)       # e.g. [3, 7, 3, 12, ...]
```

`transform()` accepts the same `data` types as `fit()`: a `list[list[str]]` or a `Corpus`. Out-of-vocabulary tokens are dropped silently; a document whose tokens are all OOV receives the prior θ (which, for models with a learned asymmetric prior such as DMR, need not be uniform). Results are deterministic for a fixed `seed`.

**Held-out inference is available across the model families**, each using the same inference procedure it uses at fit time:

| Model | Inference | Notes |
|-------|-----------|-------|
| `LDA`, `LabeledLDA`, `SupervisedLDA` | collapsed Gibbs against fixed φ | `LabeledLDA`/`SupervisedLDA` infer over all topics (labels/response unused) |
| `HDP` | collapsed Gibbs over the discovered topics | symmetric prior with the learned concentration |
| `DMR` | collapsed Gibbs with `α_d = exp(Xγ)` | pass held-out `features` to set the prior, else the intercept-only baseline is used |
| `CTM`, `STM` | Laplace **variational** E-step against fixed β and the logistic-normal prior | reproduces the model's own training θ to ~1e-3; `STM` uses the covariate-free baseline μ |

For `CTM`/`STM` the variational `transform` is the *same* inference R's `stm` runs in `fitNewDocuments`, so held-out θ is consistent with the fitted document-topic matrix.

---

## Exploring topics and documents

### Top documents per topic

`top_documents(topic, n=10)` returns the `n` training documents most strongly associated with a topic, sorted by descending θ weight.

```python
# Which documents best represent topic 3?
for doc_name, weight in model.top_documents(3, n=5):
    print(f"{doc_name}  θ={weight:.3f}")
```

Returns a `list` of `(doc_name, weight)` tuples. `doc_name` is a value from `model.doc_names`. Raises `ValueError` if `topic` is out of range.

### Similar documents

`similar_documents(doc, n=10)` finds the `n` training documents most similar to a given document (by index), ranked by ascending Jensen-Shannon divergence of their θ vectors.

```python
# Documents most similar to document 0
for doc_name, divergence in model.similar_documents(0, n=5):
    print(f"{doc_name}  JS={divergence:.4f}")
```

The query document itself is excluded from the results. Raises `ValueError` if `doc` is out of range.

### Topic divergence matrix

`topic_divergence` is a **property** (no parentheses) that returns a `(num_topics, num_topics)` numpy array of pairwise Jensen-Shannon divergences between topic-word distributions (base 2, values in [0, 1]). The diagonal is 0; high off-diagonal values indicate distinct topics, low values indicate redundancy.

```python
import numpy as np

D = model.topic_divergence   # property — no ()
print(D.shape)               # (num_topics, num_topics)
print(D[0, 1])               # divergence between topics 0 and 1; close to 1 = very different

# Find the most similar pair of topics (excluding diagonal)
np.fill_diagonal(D, np.inf)
i, j = np.unravel_index(D.argmin(), D.shape)
print(f"Most similar topics: {i} and {j}  (JS={D[i,j]:.3f})")
```

---

## Progress Callback

`fit()` accepts an optional `progress` callable invoked every `progress_interval` iterations. The GIL is released during sampling and re-acquired for the callback, so it is safe to print or update a progress bar.

```python
def report(iteration: int, ll_per_token: float) -> None:
    print(f"iter {iteration:5d}  LL/token = {ll_per_token:.4f}")

model = LDA(num_topics=20, seed=42)
model.fit(
    corpus,
    iterations=1000,
    progress=report,
    progress_interval=50,   # call every 50 iterations
)
```

---

## Determinism

The sampler is fully deterministic for a given `seed`. Passing the same seed to `LDA(...)` and the same corpus and `fit()` arguments always produces identical `topic_word` and `doc_topic` arrays. Different seeds produce different results.

```python
m1 = LDA(num_topics=10, seed=42)
m2 = LDA(num_topics=10, seed=42)
m1.fit(corpus, iterations=500)
m2.fit(corpus, iterations=500)
assert (m1.topic_word == m2.topic_word).all()  # True
```

---

## Hyperparameters

| Parameter | Default | Behavior |
|-----------|---------|----------|
| `num_topics` | — | Number of latent topics K. |
| `alpha_sum` | `num_topics` | Sum of the symmetric Dirichlet prior over topics. Defaults to K (≈ 1.0 per topic). Lower values push documents toward fewer topics. |
| `beta` | `0.01` | Per-word Dirichlet prior for topic-word distributions. Lower values concentrate each topic on fewer words. |
| `optimize_interval` | `50` | Run Minka fixed-point updates for α and β every N iterations after burn-in. Set to `0` to disable optimization and use fixed priors. |
| `burn_in` | `200` | Gibbs iterations to run before hyperparameter optimization begins. |
| `num_threads` | `1` | Number of parallel sampler threads. `1` (default) is the exact, CLI-bit-identical sequential path. `>1` enables MALLET-style approximate parallel Gibbs sampling. Clamped to `≥1` (`0` behaves like `1`). See [Parallel training](#parallel-training). |
| `iterations` | `1000` | Total Gibbs sampling sweeps in `fit()`. |
| `num_samples` | `5` | Number of equally-spaced snapshots averaged to produce the final φ/θ matrices. Averaging reduces sampling noise relative to reading from a single state. |
| `sample_interval` | `25` | Gibbs sweeps between snapshots during the averaging phase. |

---

## Parallel training

`LDA` supports opt-in multi-threaded training via the `num_threads` constructor parameter.

```python
from turbotopics import LDA

model = LDA(num_topics=50, num_threads=8)
model.fit(documents, iterations=1000)
```

**`num_threads=1` (default):** the exact, single-threaded SparseLDA sampler. Results are bit-identical to the upstream `train` CLI for the same corpus, seed, and parameters.

**`num_threads>1`:** MALLET-style approximate parallel Gibbs sampling. During each sweep, documents are partitioned across worker threads; each thread samples against a private copy of the topic-word counts, then changes are reconciled. This is an *approximation*: results will differ from the single-threaded path. But:

- **Deterministic for a fixed `(num_threads, seed)` pair.** Two fits with the same `num_threads` and `seed` produce identical `topic_word` matrices.
- **Valid token bookkeeping.** `doc_topic` rows still sum to 1; all downstream methods (`transform`, `coherence`, `diagnostics`, `perplexity`, etc.) work normally.
- **Comparable topic quality.** Held-out perplexity and topic coherence are on par with the sequential path; the approximation does not degrade topic recovery.

**Speedup** is corpus- and hardware-dependent. On a ~5 000-document / 627 000-token corpus, roughly 2.5–3.5× has been observed at 8 threads, with better scaling when there are more topics and more tokens per thread. Gains taper past ~8 threads when the vocabulary is large, because each worker must copy the topic-word count table per sweep.

**Tradeoff summary:** not bit-identical to the single-threaded / CLI path, but deterministic for a fixed `num_threads`+`seed`, and produces coherent topics suitable for downstream analysis.

### Variational models parallelize automatically (and exactly)

The variational-EM models (`CTM`, `STM`, `SupervisedLDA`, and `DTM`) parallelize their per-document E-step across all available cores with no configuration. Because each document's variational update is independent, the work fans out across a thread pool and the resulting sufficient statistics are then summed back in document order. Unlike the approximate LDA parallel sampler above, **this is exact: the fit is bit-for-bit identical regardless of how many threads run it.** Determinism for a fixed `seed` holds across machines and core counts.

Control the thread count with the standard `RAYON_NUM_THREADS` environment variable (e.g. `RAYON_NUM_THREADS=1` to force single-threaded). The E-step is the dominant per-iteration cost, so on a multicore machine this is roughly a several-fold speedup; the exact factor is hardware- and corpus-dependent.

For large vocabularies, the STM/CTM spectral (anchor-word) initialization also switches from the exact dense V×V co-occurrence to a random-projected approximation (Johnson-Lindenstrauss, the same approach R `stm` uses), which keeps the one-time init cost from growing quadratically in the vocabulary size. Small and moderate vocabularies use the exact path unchanged. Spectral init remains deterministic and seed-independent in both regimes.

---

## Model Evaluation & Diagnostics

### Choosing the number of topics with held-out perplexity

Held-out perplexity measures how well a fitted model predicts tokens it has not seen during training: lower is better. A practical strategy is to fit models over a range of candidate topic counts and choose the K whose held-out perplexity is lowest.

```python
from turbotopics import LDA

# Pre-tokenized documents split into train / held-out sets
train_docs = [...]   # list[list[str]]
held_docs  = [...]   # separate held-out split

results = {}
for k in [5, 10, 20, 50]:
    model = LDA(k, seed=42)
    model.fit(train_docs, iterations=1000)
    results[k] = model.perplexity(held_docs, num_particles=10, seed=0)
    print(f"k={k:3d}  perplexity={results[k]:.2f}")

best_k = min(results, key=results.get)
print(f"\nBest k: {best_k}")
```

`perplexity()` uses the Wallach (2009) left-to-right particle-filter estimator. The `num_particles` parameter controls the accuracy/speed trade-off; 10 is a reasonable default. Results are deterministic for a fixed `seed`.

Out-of-vocabulary (OOV) tokens, words absent from the training vocabulary, are dropped silently and counted in `evaluate()`'s `num_oov` field. If all tokens in the held-out set are OOV, `perplexity` is `nan` and `num_tokens` is 0.

```python
# Full diagnostics from evaluate()
result = model.evaluate(held_docs, num_particles=10, seed=0)
print(result)
# {'log_likelihood': -1234.5, 'perplexity': 18.3, 'num_tokens': 5000, 'num_oov': 12}
```

### Topic coherence

Every model has a built-in `coherence(n=10)` method that returns per-topic **UMass** coherence (Mimno et al. 2011), an intrinsic measure computed directly from training-corpus co-occurrence; higher (closer to 0) is more coherent.

```python
import numpy as np

c = model.coherence(n=10)   # top-n words per topic; returns shape (num_topics,)
print(c)                     # e.g. [-2.1, -1.1]  — all values <= 0
print("Mean coherence:", np.mean(c))
```

For the windowed, PMI-based measures that correlate better with human judgement (and that anyone coming from gensim's `CoherenceModel` will expect), use the module-level `turbotopics.coherence(...)` with a `coherence_type=` switch:

```python
import turbotopics as tt

# `documents` is the reference corpus (your training docs, or an external
# corpus like a Wikipedia dump for a more human-aligned signal).
cv    = tt.coherence(model, documents, coherence_type="c_v",    topn=10)
cnpmi = tt.coherence(model, documents, coherence_type="c_npmi", topn=10)
cuci  = tt.coherence(model, documents, coherence_type="c_uci",  topn=10)
umass = tt.coherence(model, documents, coherence_type="u_mass", topn=10)

print("C_v per topic:", np.round(cv, 3), "  mean:", cv.mean())
```

| `coherence_type` | What it is | Window | Range |
|---|---|---|---|
| `"c_v"` (default) | Indirect cosine over NPMI context vectors (Röder et al. 2015) — best human correlation | 110 | ~`[0, 1]` |
| `"c_npmi"` | Mean pairwise normalized PMI | 10 | `[-1, 1]` |
| `"c_uci"` | Mean pairwise PMI (Newman et al. 2010) | 10 | unbounded |
| `"u_mass"` | Document co-occurrence (Mimno 2011) | — | `(-inf, 0]` |

The first argument is a fitted model (its top words are read automatically) or an explicit list of word lists. Override the sliding window with `window_size=`. Pair coherence with **topic diversity** (Dieng et al. 2020), the fraction of unique words across all topics' top-N (1.0 = no repetition):

```python
print("Topic diversity:", tt.topic_diversity(model, topn=25))
```

### Exclusivity and the coherence–exclusivity plot

`tt.exclusivity(model, n=10)` returns per-topic **exclusivity**: how concentrated each topic's top words are in that topic rather than shared across topics. It is the companion to per-topic coherence in the standard STM topic-quality workflow: plot one against the other and good topics sit toward the upper-right (coherent *and* distinctive).

```python
import numpy as np

coh  = model.coherence(10)        # per-topic UMass coherence
excl = tt.exclusivity(model, 10)  # per-topic exclusivity, same shape
for t, (c, e) in enumerate(zip(coh, excl)):
    print(f"topic {t}: coherence={c:+.2f}  exclusivity={e:.3f}")
# scatter coh (x) vs excl (y) to spot weak topics in the lower-left
```

### Human validation: intrusion tests

The intrusion tests of Chang et al. (2009, *Reading Tea Leaves*) are the standard way social scientists validate that topics are humanly interpretable, and they work on any fitted model.

`tt.word_intrusion(model, n_words=5, seed=0)` builds, per topic, its top words plus one **intruder** (a word salient in another topic but rare here). Show the shuffled `words` to a human; if they can reliably pick the `intruder`, the topic is coherent.

```python
for t in tt.word_intrusion(model, n_words=5, seed=0):
    print(f"Topic {t['topic']}: {t['words']}")
    print(f"  answer -> intruder '{t['intruder']}' at position {t['intruder_index']}")
```

`tt.document_intrusion(model, texts=None, n_docs=3, seed=0)` does the same with documents: each topic's most representative documents plus one where the topic is nearly absent. Pass `texts` to get `texts` previews alongside the `doc_indices`; the answer key is `intruder_index`. Both are deterministic for a fixed `seed`.

### Diagnostics table

`diagnostics(n)` returns one dict per topic with the following fields:

| Key | Type | Description |
|-----|------|-------------|
| `topic` | `int` | Topic index. |
| `tokens` | `int` | Total token count assigned to this topic across the training corpus. |
| `coherence` | `float` | UMass coherence for the top-n words (≤ 0). |
| `exclusivity` | `float` | Fraction of top-word probability mass exclusive to this topic (0–1); higher means the topic's words appear primarily in this topic. |
| `effective_words` | `float` | exp(entropy of φ_t) — effective vocabulary size of the topic; lower means more concentrated. |
| `rank1_docs` | `int` | Number of documents for which this is the dominant topic. Sum across topics ≤ num_docs. |
| `alpha` | `float` | Per-topic α value (possibly asymmetric after hyperparameter optimization). |
| `top_words` | `list[str]` | Top-n words by probability (length ≤ n). |

Load into a DataFrame for easy inspection:

```python
import pandas as pd

df = pd.DataFrame(model.diagnostics())
print(df[["topic", "coherence", "exclusivity", "effective_words", "rank1_docs", "top_words"]])
```

---

## stm-style analysis toolkit

`turbotopics.stm` is a pure-Python (numpy) post-hoc analysis toolkit that mirrors the user-facing functions of the R `stm` (Structural Topic Model) package. It operates on the `topic_word` (φ) and `doc_topic` (θ) arrays produced by any fitted turbotopics model (`LDA`, `DMR`, or `LabeledLDA`), so no extra fitting step is needed.

**Uncertainty:** `estimate_effect` does ordinary OLS when given a point θ. Given posterior draws of θ from an `STM`/`CTM` fit, it instead uses the **method of composition** that R `stm` uses: each draw is regressed and the results are pooled by Rubin's rules, so the standard errors propagate topic-estimation uncertainty, not just OLS sampling error. See [Covariate effects with proper uncertainty](#covariate-effects-with-proper-uncertainty-method-of-composition) below. Confidence intervals use a normal approximation (no scipy dependency).

### Quick example

```python
from turbotopics import LDA, stm
import numpy as np

# ── 1. Fit a model (or load pre-fitted arrays) ───────────────────────────────
animal_docs = [["cat", "dog", "fish"]] * 40
space_docs  = [["planet", "star", "moon"]] * 40
docs = animal_docs + space_docs

# Binary covariate: 1 = space document, 0 = animal document
x = np.array([0] * 40 + [1] * 40, dtype=float)

model = LDA(num_topics=2, seed=42)
model.fit(docs, iterations=500)

# ── 2. estimate_effect: regress topic proportions on covariates ───────────────
effects = stm.estimate_effect(
    model.doc_topic,
    x,
    feature_names=["is_space"],
)

# Print a small summary table
for eff in effects:
    d = eff.as_dict()
    print(f"Topic {d['topic']}  R²={d['r_squared']:.3f}")
    print(f"  is_space: coef={d['is_space']['coef']:+.3f}  "
          f"z={d['is_space']['z']:+.1f}  "
          f"95% CI {d['is_space']['ci']}")

# Or, with pandas:
# import pandas as pd
# rows = []
# for eff in effects:
#     d = eff.as_dict()
#     for feat in eff.feature_names:
#         rows.append({"topic": d["topic"], "feature": feat,
#                      "coef": d[feat]["coef"], "z": d[feat]["z"]})
# pd.DataFrame(rows)

# ── 3. label_topics: prob / FREX / lift / score word lists ────────────────────
labels = stm.label_topics(model.topic_word, model.vocabulary, n=5)
for t, d in enumerate(labels):
    print(f"Topic {t}")
    print("  prob: ", [w for w, _ in d["prob"]])
    print("  frex:", [w for w, _ in d["frex"]])

# frex() alone for just the FREX words:
frex_words = stm.frex(model.topic_word, model.vocabulary, n=5)
for t, words in enumerate(frex_words):
    print(f"Topic {t} FREX:", [w for w, _ in words])

# ── 4. topic_correlation: correlation network ─────────────────────────────────
tc = stm.topic_correlation(model.doc_topic, threshold=0.05)
print("Correlation matrix:\n", tc.cor)
print("Edges (i, j, corr):", tc.edges)

# ── 5. find_thoughts: representative documents per topic ──────────────────────
texts = [" ".join(d) for d in docs]
thoughts = stm.find_thoughts(model.doc_topic, texts=texts, topic=0, n=3)
for idx, prop, text in thoughts:
    print(f"  doc {idx}  θ={prop:.3f}  '{text}'")

# ── 6. search_k: fit across topic counts, compare quality metrics ─────────────
train = docs[:60]
held  = docs[60:]

results = stm.search_k(
    train,
    ks=[2, 3, 5],
    held_out=held,          # omit to skip perplexity
    iterations=300,
)
for row in results:
    print(f"k={row['k']}  coherence={row['coherence']:.3f}  "
          f"exclusivity={row['exclusivity']:.3f}  "
          f"perplexity={row.get('perplexity', 'n/a')}")
```

### API summary

turbotopics is a **general** topic-modeling tool, so the model-agnostic post-hoc analyses (labeling, interpretation, comparison, visualization) are exported at the **top level** (and also live in `turbotopics.diagnostics`). They take any fitted model's `topic_word`/`doc_topic`, not just an STM. The handful of genuinely *structural* operations (regressing topics on covariates) stay in the `turbotopics.stm` submodule.

**General diagnostics** (`tt.<name>`, also `tt.diagnostics.<name>`; the `stm.<name>` aliases still work):

| Function | Returns | Notes |
|----------|---------|-------|
| `frex(topic_word, vocabulary, *, w=0.5, n=10)` | `list[list[(word, score)]]` | FREX (frequency–exclusivity) top words per topic. |
| `label_topics(topic_word, vocabulary, *, n=10)` | `list[dict]` | Per-topic word lists: keys `prob`, `frex`, `lift`, `score`. |
| `topic_correlation(doc_topic, *, threshold=0.05)` | `TopicCorrelation` | Correlation network: `.cor`, `.adjacency`, `.edges`. |
| `find_thoughts(doc_topic, texts=None, *, topic, n=3)` | `list[(idx, prop, text)]` | Top-n docs for a topic, sorted by descending proportion. |
| `search_k(docs, ks, *, held_out=None, iterations=500, ...)` | `list[dict]` | Fit LDA per K; report coherence, exclusivity, perplexity. |
| `relevance(topic_word, vocabulary, *, topic=None, lam=0.6, n=10, term_frequency=None)` | `list[(word, score)]` | LDAvis relevance (Sievert & Shirley 2014); the FREX cousin the LDAvis slider tunes. |
| `prepare_pyldavis(model, docs, **kwargs)` | `PreparedData` or `PyLDAvisInputs` | Build the LDAvis intertopic-distance view; returns `pyLDAvis`'s object if installed, else the input arrays. |
| `check_residuals(model, docs, *, tol=0.01)` | `ResidualCheck` | Taddy (2012) residual-dispersion test for whether K is too small (faithful to stm's `checkResiduals`). `.dispersion`, `.pvalue`, `.df`. |
| `align_topics(a, b, *, metric="cosine")` | `list[(topic_a, topic_b, dist)]` | One-to-one topic matching across two fits (Hungarian); `metric` ∈ `cosine`/`js`. |
| `topic_stability(runs, *, topn=10, metric="cosine")` | `float` | Term-centric stability across fits (Greene et al. 2014): mean top-N Jaccard of matched topics. |
| `exclusivity(model, *, n=10)` | `ndarray (K,)` | Per-topic exclusivity (see [Exclusivity](#exclusivity-and-the-coherenceexclusivity-plot)). |
| `word_intrusion` / `document_intrusion` | `list[dict]` | Human-validation intrusion tests (see [Human validation](#human-validation-intrusion-tests)). |
| `quality_frontier(model, *, n=10, texts=None, plot=False)` | `dict` (or `(dict, fig)`) | Per-topic coherence, exclusivity, prevalence — the data behind the coherence-vs-exclusivity quality plot. |
| `bootstrap_stability(docs, *, k, n_boot=20, ...)` | `dict` | Refit on bootstrap resamples; per-topic mean Jaccard flags topics that dissolve. Answers the "fishing expedition" critique. |
| `find_thoughts_html(model, texts, *, n_docs=3, n_words=8, markdown=False)` | `str` | Representative documents per topic with the topic's words highlighted, for close reading in a notebook. |
| `fighting_words(corpus_a, corpus_b, *, prior=0.01, informative=False)` | `list[(word, z)]` | Monroe-Colaresi-Quinn weighted log-odds: words that distinguish two corpora, with significance. `top_fighting_words(...)` returns the top-n per side. |
| `split_documents(texts, metadata=None, *, max_words=200, min_words=50)` | `(chunks, chunk_meta)` | Segment long documents into comparable chunks, copying each source's metadata onto every chunk. |

**Structural topic model** (`turbotopics.stm` — covariates on topic prevalence):

| Function | Returns | Notes |
|----------|---------|-------|
| `stm.estimate_effect(doc_topic, X, *, feature_names=None, topics=None, add_intercept=True, ci=0.95, cluster=None, link="identity")` | `list[TopicEffect]` | Regress each topic's θ on covariates. `doc_topic` 2-D → OLS; 3-D (posterior draws) → method-of-composition (Rubin pooling). `cluster=` (per-doc group labels) → cluster-robust CR1 SEs for nested data; `link=` `"logit"`/`"log"` → fractional-logit / quasi-Poisson GLM. |
| `stm.posterior_theta_samples(model, nsims=25, seed=0)` | `ndarray (nsims, D, K)` | Draw θ from an `STM`/`CTM` variational posterior (`eta_mean`/`eta_cov`); feed to `estimate_effect` for proper uncertainty. |
| `stm.spline(x, df=4, knots=None)` | `(ndarray (n, df), names)` | Restricted cubic-spline basis for nonlinear terms (R `stm`'s `s(x)`). |
| `stm.interaction(a, b, name="interaction")` | `(ndarray, names)` | Pairwise-product interaction columns (R `stm`'s `a*b`). |

`TopicEffect` fields: `.topic`, `.feature_names`, `.coef`, `.se`, `.z`, `.ci_low`, `.ci_high`, `.r_squared`, `.as_dict()`. The OLS intercept is prepended by default (`add_intercept=True`); pass `add_intercept=False` to suppress it.

---

## API Reference

### `LDA`

#### Constructor

```python
LDA(num_topics: int, *, alpha_sum: float | None = None, beta: float = 0.01,
    optimize_interval: int = 50, burn_in: int = 200, seed: int = 42,
    num_threads: int = 1)
```

Raises `ValueError` if `num_topics < 1` or `beta <= 0`. `num_threads` is clamped to `≥1`; `num_threads=0` behaves like `1`.

#### Methods

| Signature | Returns | Notes |
|-----------|---------|-------|
| `fit(data, *, iterations=1000, num_samples=5, sample_interval=25, progress=None, progress_interval=50)` | `None` | `data` is a `Corpus` or `list[list[str]]`. Token-list input builds an internal corpus with no frequency filtering. |
| `top_words(n=10, *, topic=None)` | `list[list[tuple[str, float]]]` or `list[tuple[str, float]]` | All topics if `topic=None`; single topic list otherwise. Tuples are `(word, probability)`. Raises `ValueError` if `topic` is out of range. |
| `log_likelihood()` | `float` | MALLET-formula model log-likelihood of the final sampler state. |
| `evaluate(data, *, num_particles=10, seed=None)` | `dict` | Held-out evaluation via the Wallach (2009) left-to-right estimator. Returns `{"log_likelihood": float, "perplexity": float, "num_tokens": int, "num_oov": int}`. OOV tokens are dropped and counted. All-OOV input gives `num_tokens=0` and `perplexity=nan`. Raises `ValueError` if `num_particles < 1`. |
| `perplexity(data, *, num_particles=10, seed=None)` | `float` | Convenience wrapper over `evaluate()`; returns `perplexity` directly. Lower is better. |
| `coherence(n=10)` | `numpy.ndarray` shape `(num_topics,)` | UMass coherence per topic, computed intrinsically from the training corpus. Values are ≤ 0; higher (closer to 0) is better. Use `numpy.mean()` for a summary score. |
| `diagnostics(n=10)` | `list[dict]` | One dict per topic. Keys: `topic`, `tokens`, `coherence`, `exclusivity`, `effective_words`, `rank1_docs`, `alpha`, `top_words`. Feed directly to `pandas.DataFrame(model.diagnostics())`. |
| `transform(data, *, iterations=100, burn_in=10, num_samples=10, sample_interval=5, seed=None)` | `numpy.ndarray` shape `(num_new_docs, num_topics)` | Infer document-topic θ for new documents under the fixed fitted model (sklearn-style). `data` is a `Corpus` or `list[list[str]]`; OOV tokens dropped; all-OOV doc gets the prior θ. Rows sum to 1. Deterministic per `seed`. |
| `top_documents(topic, n=10)` | `list[tuple[str, float]]` | Training documents most associated with `topic`, sorted by descending θ weight. Tuples are `(doc_name, weight)`. Raises `ValueError` if `topic` is out of range. |
| `similar_documents(doc, n=10)` | `list[tuple[str, float]]` | Training documents most similar to document index `doc`, sorted by ascending Jensen-Shannon divergence of θ vectors. Query doc excluded. Raises `ValueError` if `doc` is out of range. |
| `save_topic_word(path: str)` | `None` | TSV: `topic\tword\tprobability`. |
| `save_doc_topic(path: str)` | `None` | TSV: `doc[\tlabel]\ttopic_0\t...`. |

#### Properties

All properties below raise `RuntimeError("model is not fitted yet; call fit() first")` before `fit()` is called. `num_topics` is available before `fit()`.

| Property | Type | Description |
|----------|------|-------------|
| `topic_word` | `numpy.ndarray` shape `(num_topics, num_words)` | φ matrix — topic-word distributions. |
| `doc_topic` | `numpy.ndarray` shape `(num_docs, num_topics)` | θ matrix — document-topic distributions. Rows sum to 1. |
| `vocabulary` | `list[str]` | Words in column order of `topic_word`. |
| `doc_names` | `list[str]` | Document identifiers in row order of `doc_topic`. |
| `alpha` | `numpy.ndarray` shape `(num_topics,)` | Per-topic α values (possibly asymmetric after optimization). |
| `beta` | `float` | Final β per word. |
| `num_topics` | `int` | Number of topics (available before `fit()`). |
| `num_threads` | `int` | Number of sampler threads (as clamped; `1` = sequential exact path). |
| `topic_divergence` | `numpy.ndarray` shape `(num_topics, num_topics)` | Pairwise Jensen-Shannon divergence between topic-word distributions (base 2, values in [0, 1]). Zero diagonal; symmetric. Low off-diagonal values indicate redundant topics. Access as a property (no parentheses). |

---

### `Corpus`

#### Static constructors

| Signature | Returns | Notes |
|-----------|---------|-------|
| `Corpus.from_documents(documents, *, doc_names=None, doc_labels=None, stopwords=None, min_doc_freq=1, max_doc_fraction=1.0)` | `Corpus` | `doc_names`/`doc_labels` must match `len(documents)` if provided. Documents emptied by filtering are dropped. |
| `Corpus.from_text_file(path, *, format="plain", id_field=False, id_column=0, label_column=1, text_column=2, token_regex=None, stopwords=None, min_doc_freq=1, max_doc_fraction=1.0)` | `Corpus` | `format` is `"plain"` or `"tsv"`. `token_regex=None` uses `DEFAULT_TOKEN_REGEX`. Raises `ValueError` on unknown `format`. |
| `Corpus.load(path)` | `Corpus` | Loads a binary `.corp` file produced by `save()` or the `preprocess` CLI. |

#### Methods and properties

| Name | Type / Signature | Description |
|------|-----------------|-------------|
| `save(path: str)` | `None` | Write binary corpus; reusable by CLI tools. |
| `num_docs` | `int` | Number of documents. |
| `num_words` | `int` | Vocabulary size. |
| `total_tokens` | `int` | Total token count across all documents. |
| `vocabulary` | `list[str]` | All word types. |
| `doc_names` | `list[str]` | Document identifiers. |
| `doc_labels` | `list[str]` | Document labels (empty strings if none). |

---

### `DMR`

Dirichlet-Multinomial Regression topic model. Per-document topic prior `α_{d,t} = exp(λ_t · x_d)` learned by L-BFGS from document features. See the [DMR section](#dmr-topics-conditioned-on-document-metadata) for a usage guide.

**Not bit-identical to Java MALLET.** L-BFGS optimization makes byte-for-byte reproducibility impossible; fidelity is validated statistically (covariate recovery, comparable perplexity).

#### Constructor

```python
DMR(num_topics: int, *, beta: float = 0.01, optimize_interval: int = 50,
    burn_in: int = 200, seed: int = 42, prior_variance: float = 1.0,
    lbfgs_iters: int = 20)
```

- `prior_variance` — Gaussian prior variance σ² on λ (smaller = stronger shrinkage toward zero).
- `lbfgs_iters` — caps L-BFGS steps per optimization round.
- Raises `ValueError` if `num_topics < 1`, `beta <= 0`, or `prior_variance <= 0`.

#### Methods

| Signature | Returns | Notes |
|-----------|---------|-------|
| `fit(data, features, *, feature_names=None, iterations=1000, num_samples=5, sample_interval=25, progress=None, progress_interval=50)` | `None` | `data` is a `Corpus` or `list[list[str]]`. `features` is a `(num_docs, F)` numpy array or list of float lists; an intercept column is prepended internally. `feature_names` (length F) names the F user-supplied columns; `"intercept"` is always prepended. Row count of `features` must equal `num_docs`; all rows must have the same width. |
| `top_words(n=10, *, topic=None)` | `list[list[tuple[str, float]]]` or `list[tuple[str, float]]` | Same shape contract as `LDA.top_words`. |
| `coherence(n=10)` | `numpy.ndarray` shape `(num_topics,)` | UMass coherence per topic (intrinsic; values ≤ 0). |

#### Properties

All properties below raise `RuntimeError("model is not fitted yet; call fit() first")` before `fit()` is called. `num_topics` is available before `fit()`.

| Property | Type | Description |
|----------|------|-------------|
| `topic_word` | `numpy.ndarray` shape `(num_topics, num_words)` | φ matrix — topic-word distributions. |
| `doc_topic` | `numpy.ndarray` shape `(num_docs, num_topics)` | θ matrix — document-topic distributions. Rows sum to 1. |
| `feature_effects` | `numpy.ndarray` shape `(num_topics, num_features)` | Learned λ matrix. Column 0 is the intercept; positive entries raise that topic's prevalence. The headline DMR output. |
| `feature_names` | `list[str]` | Column labels for `feature_effects`; `"intercept"` is always first. |
| `vocabulary` | `list[str]` | Words in column order of `topic_word`. |
| `doc_names` | `list[str]` | Document identifiers in row order of `doc_topic`. |
| `num_topics` | `int` | Number of topics (available before `fit()`). |

---

### `LabeledLDA`

Supervised topic model (Ramage et al. 2009). Each label becomes a topic; documents are constrained to their labels' topics. The number of topics equals the number of distinct labels (inferred automatically from the data). See the [Labeled LDA section](#labeled-lda-supervised-topics-from-document-labels) for a usage guide.

#### Constructor

```python
LabeledLDA(*, alpha: float = 0.1, beta: float = 0.01, seed: int = 42)
```

- `alpha` — symmetric per-topic Dirichlet prior over topics (scalar, applied to each topic).
- `beta` — per-word Dirichlet prior for topic-word distributions.
- Raises `ValueError` if `alpha <= 0` or `beta <= 0`.

#### Methods

| Signature | Returns | Notes |
|-----------|---------|-------|
| `fit(data, labels, *, label_names=None, iterations=1000, num_samples=5, sample_interval=25, progress=None, progress_interval=50)` | `None` | `data` is a `Corpus` or `list[list[str]]`. `labels` is `list[list[str]]` — one label list per document; length must equal `num_docs`. Topic set = sorted union of all labels, or `label_names` (which also fixes topic order). A document with an empty label list is treated as unconstrained (all topics allowed). Raises `ValueError` if no labels found. |
| `top_words(n=10, *, topic=None)` | `list[list[tuple[str, float]]]` or `list[tuple[str, float]]` | Same shape contract as `LDA.top_words`. Raises `ValueError` if `topic` is out of range. |
| `coherence(n=10)` | `numpy.ndarray` shape `(num_topics,)` | UMass coherence per topic (intrinsic; values ≤ 0). |

#### Properties

All properties below raise `RuntimeError("model is not fitted yet; call fit() first")` before `fit()` is called.

| Property | Type | Description |
|----------|------|-------------|
| `topic_word` | `numpy.ndarray` shape `(num_topics, num_words)` | φ matrix — topic-word distributions. |
| `doc_topic` | `numpy.ndarray` shape `(num_docs, num_topics)` | θ matrix — per-label proportions. Only columns corresponding to a document's labels are non-zero; rows sum to 1. |
| `labels` | `list[str]` | Label name per topic, in column order of `doc_topic` and `topic_word`. |
| `vocabulary` | `list[str]` | Words in column order of `topic_word`. |
| `doc_names` | `list[str]` | Document identifiers in row order of `doc_topic`. |
| `num_topics` | `int` | Number of topics = number of distinct labels (raises `RuntimeError` before fit). |

---

### `CTM`

Correlated Topic Model (Blei & Lafferty 2007). Topics drawn from a logistic-normal prior with full covariance, so they can correlate, unlike LDA. The only variational (non-Gibbs) model in `turbotopics`. See the [CTM section](#ctm-correlated-topics-the-stm-core) for a usage guide.

#### Constructor

```python
CTM(num_topics: int, *, sigma_shrink: float = 0.0, seed: int = 42)
```

- `num_topics >= 2` (else `ValueError`).
- `sigma_shrink ∈ [0, 1]` shrinks Σ toward its diagonal at each M-step; `0.0` learns the full covariance.
- Raises `ValueError` if `num_topics < 2` or `sigma_shrink` outside `[0, 1]`.

#### Methods

| Signature | Returns | Notes |
|-----------|---------|-------|
| `fit(data, *, em_iters=50)` | `None` | `data` is a `Corpus` or `list[list[str]]`. Runs variational EM for `em_iters` iterations. |
| `top_words(n=10, *, topic=None)` | `list[list[tuple[str, float]]]` or `list[tuple[str, float]]` | All topics if `topic=None`; single topic list otherwise. Tuples are `(word, probability)`. Raises `ValueError` if `topic` is out of range. |
| `coherence(n=10)` | `numpy.ndarray` shape `(num_topics,)` | UMass coherence per topic (intrinsic; values ≤ 0). |

#### Properties

All properties below raise `RuntimeError("model is not fitted yet; call fit() first")` before `fit()` is called. `num_topics` is available before `fit()`.

| Property | Type | Description |
|----------|------|-------------|
| `topic_word` | `numpy.ndarray` shape `(num_topics, num_words)` | β matrix — topic-word distributions. |
| `doc_topic` | `numpy.ndarray` shape `(num_docs, num_topics)` | θ matrix — document-topic distributions (softmax of latent η). Rows sum to 1. |
| `topic_correlation` | `numpy.ndarray` shape `(num_topics, num_topics)` | Pearson correlation of θ across documents. Symmetric, unit diagonal. **The distinguishing CTM output.** |
| `vocabulary` | `list[str]` | Words in column order of `topic_word`. |
| `doc_names` | `list[str]` | Document identifiers in row order of `doc_topic`. |
| `num_topics` | `int` | Number of topics (available before `fit()`). |

---

### `STM`

Structural Topic Model (Roberts, Stewart & Tingley). CTM core with prevalence covariates: the prior topic mean is `μ_d = X_d γ`, so document metadata shifts which topics a document discusses. See the [STM section](#stm-structural-topic-model-covariate-aware-correlated-topics) for a usage guide.

#### Constructor

```python
STM(num_topics: int, *, sigma_shrink: float = 0.0, seed: int = 42)
```

- `num_topics >= 2` (else `ValueError`).
- `sigma_shrink ∈ [0, 1]` shrinks Σ toward its diagonal at each M-step (else `ValueError`).

#### Methods

| Signature | Returns | Notes |
|-----------|---------|-------|
| `fit(data, prevalence, *, prevalence_names=None, em_iters=50)` | `None` | `data` is a `Corpus` or `list[list[str]]`. `prevalence` is a `(num_docs, F)` numpy array or list of float lists; an intercept column is prepended. `prevalence_names` (length F) names the covariates. Row count must equal `num_docs`; all rows must have the same width. |
| `top_words(n=10, *, topic=None)` | `list[list[tuple[str, float]]]` or `list[tuple[str, float]]` | Same shape contract as `LDA.top_words`. Raises `ValueError` if `topic` is out of range. |
| `coherence(n=10)` | `numpy.ndarray` shape `(num_topics,)` | UMass coherence per topic (intrinsic; values ≤ 0). |

#### Properties

All properties below raise `RuntimeError("model is not fitted yet; call fit() first")` before `fit()` is called. `num_topics` is available before `fit()`.

| Property | Type | Description |
|----------|------|-------------|
| `topic_word` | `numpy.ndarray` shape `(num_topics, num_words)` | β matrix — topic-word distributions. |
| `doc_topic` | `numpy.ndarray` shape `(num_docs, num_topics)` | θ matrix — document-topic distributions. Rows sum to 1. |
| `topic_correlation` | `numpy.ndarray` shape `(num_topics, num_topics)` | Pearson correlation of θ across documents. Symmetric, unit diagonal. **Same as CTM.** |
| `prevalence_effects` | `numpy.ndarray` shape `(num_features, num_topics-1)` | Learned γ. Row 0 is the intercept (prepended internally); `num_features = F + 1`. **The headline STM output.** For inference use `stm.estimate_effect(model.doc_topic, X)`. |
| `feature_names` | `list[str]` length `F+1` | `"intercept"` first, then the user-supplied `prevalence_names` (or `"feature_0"`, `"feature_1"`, ... if omitted). |
| `vocabulary` | `list[str]` | Words in column order of `topic_word`. |
| `doc_names` | `list[str]` | Document identifiers in row order of `doc_topic`. |
| `num_topics` | `int` | Number of topics (available before `fit()`). |

---

### `SAGE`

Content-covariate topic model. Topics are shared across documents, but each topic's word distribution varies by a document-level group covariate. See the [SAGE section](#sage-content-covariates-topics-worded-differently-by-group) for a usage guide.

#### Constructor

```python
SAGE(num_topics: int, *, alpha: float = 0.1, prior_variance: float = 1.0,
     optimize_interval: int = 50, burn_in: int = 100, seed: int = 42,
     lbfgs_iters: int = 20)
```

- `prior_variance` — Gaussian prior variance σ² on the κ deviations (smaller = stronger shrinkage toward the background).
- `lbfgs_iters` — caps L-BFGS steps per optimization round.
- Raises `ValueError` if `num_topics < 1` or `prior_variance <= 0`.

#### Methods

| Signature | Returns | Notes |
|-----------|---------|-------|
| `fit(data, groups, *, group_names=None, iterations=1000, num_samples=5, sample_interval=25, progress=None, progress_interval=50)` | `None` | `data` is a `Corpus` or `list[list[str]]`. `groups` is one group label per document (list of str or int); length must equal `num_docs`. `group_names` fixes group order (default: sorted union). Raises `ValueError` if `groups` length mismatches, or a label is not in `group_names`. |
| `top_words(topic, *, group=None, n=10)` | `list[tuple[str, float]]` | Top n words for `topic` in the specified group (by name or index), or group-averaged if `group=None`. Raises `ValueError` on bad topic or group. |
| `word_contrast(topic, group_a, group_b, n=10)` | `list[tuple[str, float]]` | Top n words most distinguishing how `topic` is worded in `group_a` vs `group_b` (positive log-ratio favours `group_a`). Raises `ValueError` on bad topic. |
| `coherence(n=10)` | `numpy.ndarray` shape `(num_topics,)` | Group-averaged UMass coherence per topic (intrinsic; values ≤ 0). |

#### Properties

All properties below raise `RuntimeError("model is not fitted yet; call fit() first")` before `fit()` is called. `num_topics` is available before `fit()`.

| Property | Type | Description |
|----------|------|-------------|
| `topic_word` | `numpy.ndarray` shape `(num_topics, num_groups, num_words)` | Per-group word distributions. `topic_word[k, g, :]` is the distribution for topic k in group g. |
| `topic_word_marginal` | `numpy.ndarray` shape `(num_topics, num_words)` | Group-averaged word distributions; equivalent to `topic_word.mean(axis=1)`. |
| `doc_topic` | `numpy.ndarray` shape `(num_docs, num_topics)` | Document-topic proportions. Rows sum to 1. |
| `groups` | `list[str]` | Group labels in index order (column order of the `num_groups` axis). |
| `vocabulary` | `list[str]` | Words in column order of `topic_word`. |
| `doc_names` | `list[str]` | Document identifiers in row order of `doc_topic`. |
| `num_topics` | `int` | Number of topics (available before `fit()`). |
| `num_groups` | `int` | Number of distinct groups. |

---

### `HDP`

Hierarchical Dirichlet Process topic model — nonparametric LDA that infers the number of topics. See the [HDP section](#hdp-inferring-the-number-of-topics) for a usage guide.

#### Constructor

```python
HDP(*, alpha: float = 1.0, gamma: float = 1.0, eta: float = 0.01,
    seed: int = 42, resample_conc: bool = True)
```

- `alpha` (document-level) and `gamma` (corpus-level) are the initial DP concentrations; resampled from the data each sweep when `resample_conc=True`.
- `eta` — topic-word Dirichlet (base measure).
- Raises `ValueError` unless `alpha`, `gamma`, `eta` are all `> 0`.

#### Methods

| Signature | Returns | Notes |
|-----------|---------|-------|
| `fit(data, *, iters=150)` | `None` | `data` is a `Corpus` or `list[list[str]]`. Runs `iters` Gibbs sweeps; the inferred K is then `num_topics`. Deterministic for a fixed `seed`. |
| `top_words(n=10, *, topic=None)` | `list[tuple[str, float]]` or list thereof | Top n words for one topic or all topics. |
| `coherence(n=10)` | `numpy.ndarray` shape `(num_topics,)` | UMass coherence per topic. |

#### Properties

All raise `RuntimeError` before `fit()` except `alpha`/`gamma` (which return the current concentration value).

| Property | Type | Description |
|----------|------|-------------|
| `num_topics` | `int` | The **inferred** number of topics K. |
| `topic_word` | `numpy.ndarray` shape `(K, num_words)` | φ — topic-word distributions; rows sum to 1. |
| `doc_topic` | `numpy.ndarray` shape `(num_docs, K)` | θ — document-topic mixtures; rows sum to 1. |
| `alpha`, `gamma` | `float` | The fitted DP concentrations. |
| `vocabulary` | `list[str]` | Words in column order of `topic_word`. |
| `doc_names` | `list[str]` | Document identifiers in row order of `doc_topic`. |

---

### `DTM`

Dynamic Topic Model — topics whose word distributions evolve over time slices. See the [DTM section](#dtm-topics-that-evolve-over-time) for a usage guide.

#### Constructor

```python
DTM(num_topics: int, *, alpha: float = 0.01, chain_variance: float = 0.005,
    obs_variance: float = 0.5, seed: int = 42)
```

- `chain_variance` — how much a topic may drift between adjacent slices (larger = freer).
- Raises `ValueError` if `num_topics < 2` or any of `alpha`, `chain_variance`, `obs_variance` is `<= 0`.

#### Methods

| Signature | Returns | Notes |
|-----------|---------|-------|
| `fit(data, times, *, em_iters=20)` | `None` | `data` is a `Corpus` or `list[list[str]]`. `times` is each document's integer time-slice index (0-based, contiguous; slice count is `max(times)+1`). Raises `ValueError` if `len(times)` mismatches the document count, an index is negative, or a slice is empty. Deterministic for a fixed `seed`. |
| `topic_word(time)` | `numpy.ndarray` shape `(num_topics, num_words)` | Topic-word distributions at one slice; rows sum to 1. |
| `word_evolution(topic, word)` | `numpy.ndarray` shape `(num_times,)` | A word's probability trajectory in a topic across slices. `word` is a vocabulary string or integer id. |
| `top_words(topic, time, n=10)` | `list[tuple[str, float]]` | Top n words for `topic` at slice `time`. |

#### Properties

`num_topics` is available before `fit()`; the rest raise `RuntimeError` before fitting.

| Property | Type | Description |
|----------|------|-------------|
| `num_topics` | `int` | Number of topics. |
| `num_times` | `int` | Number of time slices. |
| `bound` | `float` | The final variational bound (ELBO) reached during fitting. |
| `vocabulary` | `list[str]` | Words in column order of `topic_word`. |

---

### `SupervisedLDA`

Supervised LDA — topics fit to predict a per-document real-valued response. See the [Supervised LDA section](#supervised-lda-topics-that-predict-a-response) for a usage guide.

#### Constructor

```python
SupervisedLDA(num_topics: int, *, alpha: float = 0.1, seed: int = 42)
```

- Raises `ValueError` if `num_topics < 2` or `alpha <= 0`.

#### Methods

| Signature | Returns | Notes |
|-----------|---------|-------|
| `fit(data, y, *, em_iters=25, var_iters=15)` | `None` | `data` is a `Corpus` or `list[list[str]]`. `y` is the per-document response; `len(y)` must equal the document count (else `ValueError`). Deterministic for a fixed `seed`. |
| `predict(data, *, var_iters=20)` | `numpy.ndarray` shape `(n_docs,)` | Predict ŷ for new documents (a `Corpus` or `list[list[str]]`). Out-of-vocabulary words are ignored. |
| `top_words(n=10, *, topic=None)` | `list[tuple[str, float]]` or list thereof | Top words for one topic or all topics. |
| `coherence(n=10)` | `numpy.ndarray` shape `(num_topics,)` | UMass coherence per topic. |

#### Properties

`num_topics` is available before `fit()`; the rest raise `RuntimeError` before fitting.

| Property | Type | Description |
|----------|------|-------------|
| `topic_word` | `numpy.ndarray` shape `(num_topics, num_words)` | φ — topic-word distributions; rows sum to 1. |
| `doc_topic` | `numpy.ndarray` shape `(num_docs, num_topics)` | θ — document-topic mixtures; rows sum to 1. |
| `coefficients` | `numpy.ndarray` shape `(num_topics,)` | Regression coefficients η — each topic's effect on the response. |
| `sigma2` | `float` | The fitted response variance σ². |
| `num_topics` | `int` | Number of topics. |
| `vocabulary` | `list[str]` | Words in column order of `topic_word`. |
| `doc_names` | `list[str]` | Document identifiers in row order of `doc_topic`. |

---

### Module-level functions

| Signature | Returns | Notes |
|-----------|---------|-------|
| `tokenize(text, *, lowercase=True, stopwords=None, token_regex=None, min_length=1)` | `list[str]` | Regex-tokenize a string using the same pattern as the corpus loader. Lowercases by default; drops short tokens and stopwords. Useful for building `list[list[str]]` input outside `Corpus.from_text_file`. `token_regex=None` uses `DEFAULT_TOKEN_REGEX`. Raises `ValueError` on an invalid regex. |
| `one_hot(values, *, drop_first=True, prefix="")` | `tuple[numpy.ndarray, list[str]]` | One-hot encode a categorical covariate. `values` is a list of category labels. `drop_first=True` omits the first sorted category as a reference level (standard for regression). `prefix` is prepended to each category name. Returns `(matrix, names)` where `matrix` has shape `(len(values), num_categories)` and `names` are the column labels. Pass `names` as `feature_names` and `matrix` as `features` in `DMR.fit`. |

### Module constants

| Name | Value | Description |
|------|-------|-------------|
| `DEFAULT_TOKEN_REGEX` | Unicode-letter pattern | Minimum-length-2 pattern used by `from_text_file` and `tokenize` when `token_regex=None`. Matches the upstream `preprocess` CLI. |
| `__version__` | `"0.1.0"` | Package version. |

---

## Performance

turbotopics fits in native Rust, and its variational models parallelize the per-document E-step across cores (deterministically: the result is bit-for-bit identical regardless of thread count). This makes the Structural Topic Model substantially faster than the reference R `stm` package on the same data.

The table times **STM fit only** (excluding startup), matched on `K`, EM iterations (30), and Spectral initialization, on fixed-seed synthetic corpora. R `stm` is single-threaded by design, so turbotopics is shown both pinned to one core (apples-to-apples) and on all cores (its default). Measured on a 14-core machine:

| docs | vocab | K | R `stm` | turbotopics (1 core) | turbotopics (all cores) |
|------:|------:|--:|--------:|---------------------:|------------------------:|
| 1,000 |   500 | 10 |  3.1s | 0.49s — **6.3×** | 0.12s — **25×** |
| 2,000 | 2,000 | 10 |  6.7s | 1.41s — **4.6×** | 0.43s — **16×** |
| 5,000 | 5,000 | 20 | 26.6s | 8.95s — **3.0×** | 2.67s — **10×** |

Even on a single core, turbotopics runs roughly **3–6× faster per iteration** than R `stm`, from the native Rust inner loop with no per-iteration interpreter overhead. Spreading the E-step across cores (the default) brings it to **~10–25×** in these configurations.

**Caveats, stated plainly:** this compares *per-iteration* cost (both run a fixed 30 EM iterations rather than to convergence), on synthetic data, on one machine. Treat the numbers as indicative, not a guarantee: speed depends on corpus, vocabulary, `K`, and hardware. The benchmark is in the repo, so you can run your own regime:

```bash
python benchmarks/bench_stm.py                      # turbotopics on all cores
RAYON_NUM_THREADS=1 python benchmarks/bench_stm.py  # single-threaded
```

(The comparison column needs `Rscript` with the `stm` package installed; without it the script prints turbotopics timings only.)

---

## Comparison to Alternatives

| | **turbotopics** | **gensim LdaModel** | **MALLET Java CLI** |
|---|---|---|---|
| Language / runtime | Python + Rust | Python (C extensions) | Python or shell + JVM |
| Algorithm | SparseLDA (three-bucket Gibbs) | Collapsed variational Bayes or online LDA | SparseLDA (three-bucket Gibbs) |
| Hyperparameter optimization | Yes (Minka fixed-point, built-in) | Optional (auto) | Yes (Minka fixed-point, built-in) |
| Determinism | Yes (seed controls all randomness) | Partial (multicore non-deterministic by default) | Yes |
| Input | Pre-tokenized `list[list[str]]` or `Corpus` | `Dictionary` + `corpus` (bag-of-words) | Text file via `mallet import-file` |
| Output | numpy arrays, ready for analysis | Gensim model objects; numpy via conversion | TSV files on disk |
| JVM required | No | No | Yes |
| MALLET-compatible corpus format | Yes (`.corp` round-trip with CLI tools) | No | Yes |

`turbotopics` uses native Rust with no JVM startup overhead. See [Performance](#performance) above for measured STM timings against R `stm`; comparisons against other Python LDA implementations depend heavily on corpus size and hardware.

---

## Relationship to Upstream RustMallet

This package wraps the library crate of [RustMallet](https://github.com/mimno/RustMallet), David Mimno's Rust reimplementation of MALLET's SparseLDA sampler. The upstream project also ships four standalone CLI tools (`preprocess`, `analyze`, `train`, and `show`) that form a pipeline for working with text files directly. Those tools remain available; the Python bindings here provide the same algorithm as a library you can call from code.

---

## License and Credits

Licensed under the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

Algorithm and Rust implementation by David Mimno and contributors. The SparseLDA scheme is described in:

> Yao, L., Mimno, D., & McCallum, A. (2009). Efficient methods for topic model inference on streaming document collections. *Proceedings of KDD 2009*.

The original MALLET toolkit is from Andrew McCallum's group at UMass Amherst.
