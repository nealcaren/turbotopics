# Political blogs: validation and clustered errors

This example carries the **validation** and **effects** load of the
[publishing workflow](../publishing/index.md): choosing `K` with a scan,
validating topics, and the headline task of estimating covariate effects with
**clustered standard errors** for nested data. The corpus is `poliblog5k` from
the R `stm` package: 2,000 political-blog posts (2008 U.S. campaign), each tagged
with the blog's `rating` (Conservative / Liberal), the `day`, and the `blog` it
came from.

!!! info "Focus of this example"
    K selection · topic validation · **clustered SEs** · GLM links. For corpus
    cleaning and dynamic topics see [Du Bois](dubois.md); for the experimental
    effect estimation see [Gadarian](gadarian.md).

    Data: [`examples/poliblog.csv`](https://github.com/nealcaren/turbotopics/blob/main/examples/poliblog.csv)
    (reconstructed from `stm`'s preprocessed, stemmed `poliblog5k.docs`).

## 1–2. Corpus and model

The posts nest within six blogs, and we want to know how topic prevalence differs
by ideology. That covariate question makes the
[right model](../publishing/choosing-model.md) the STM.

```python
import csv, numpy as np, turbotopics as tt
from turbotopics import Corpus, stm

rows = list(csv.DictReader(open("examples/poliblog.csv")))
docs = [r["text"].split() for r in rows]          # already tokenized + stemmed by stm
corpus = Corpus.from_documents(docs, min_doc_freq=10, max_doc_fraction=0.5, rm_top=20)
# 2000 documents, vocabulary 2,612
```

## 3. Choose and justify K

```python
scan = tt.search_k(docs, ks=[10, 15, 20, 25], iterations=600)
for r in scan:
    print(f"K={r['k']:>2}  coherence={r['coherence']:.3f}  exclusivity={r['exclusivity']:.3f}")
```

We take `K = 15`, then confirm the substantive effects below survive `K ∈ {10, 20}`.

## 4. Validate

```python
model = tt.STM(num_topics=15, seed=1)
conservative = np.array([r["rating"] == "Conservative" for r in rows], float).reshape(-1, 1)
model.fit(docs, conservative, prevalence_names=["conservative"], em_iters=25)

# human validation
tests = tt.word_intrusion(model, n_words=5, seed=0)
# metric validation
frontier = tt.quality_frontier(model, n=10)        # coherence × exclusivity per topic
# stability under resampling
boot = tt.bootstrap_stability(docs, k=15, n_boot=30, iterations=600)
print("mean topic stability:", round(boot["mean"], 2))
```

The topics are readable: foreign policy (*isra, israel, iran*), the VP race
(*lieberman, mccain, biden*), the Obama–Wright story, social issues (*school,
abort, gay*), the financial crisis (*billion, market*).

## 5. Effects — and why clustering changes the answer

The posts are **nested in six blogs**. If we ignore that and trust ordinary
standard errors, we badly overstate our certainty. Estimate the ideology effect
on each topic with the method of composition, once naively and once clustered by
blog:

```python
draws = stm.posterior_theta_samples(model, nsims=25, seed=0)
blog  = np.array([r["blog"] for r in rows])

iid       = stm.estimate_effect(draws, conservative, feature_names=["conservative"])
clustered = stm.estimate_effect(draws, conservative, feature_names=["conservative"],
                                cluster=blog)
```

| Topic (FREX) | coef | SE (iid) | SE (**clustered**) | z (clustered) |
|--------------|-----:|---------:|-------------------:|--------------:|
| isra, israel, iran | **+0.055** | 0.006 | 0.010 | **+5.8** |
| wright, barack, obama | +0.044 | 0.006 | 0.020 | +2.2 |
| ballot, immigr, franken | +0.018 | 0.005 | 0.006 | +3.0 |
| rove, tortur, cheney | −0.065 | 0.006 | **0.034** | **−1.9** |
| lieberman, mccain, biden | −0.084 | 0.005 | **0.029** | −2.9 |

Clustering inflates the standard errors **three- to six-fold**. The Rove/torture
topic looks overwhelmingly liberal under iid errors (z ≈ −10) but is **not
significant** once we acknowledge that all those posts come from a handful of
blogs (z = −1.9). Conservatives reliably talk more about Israel/Iran and the
Obama–Wright story. The apparent torture effect does not survive honest
uncertainty.

!!! warning "Report the caveat"
    With only **six** clusters, cluster-robust inference is itself approximate
    (CR1 wants ~30+ clusters). Say so. A careful paper reports the clustered
    result *and* notes the small number of clusters. That is far more credible
    than using iid errors that assume 2,000 independent observations.

## 6. Report

Bounded inference (topic proportions live in `[0,1]`) via a fractional-logit
link, a topic table with FREX labels and prevalence, and the saved model. See
[Report and make reproducible](../publishing/reporting.md).

```python
glm = stm.estimate_effect(draws, conservative, feature_names=["conservative"],
                          cluster=blog, link="logit")
model.save("poliblog_stm.tt")
```
