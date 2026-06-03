# Gadarian: a survey experiment

This is the canonical STM vignette (Roberts, Stewart & Tingley), and it carries
the **model-choice** and **experimental-effect** load of the
[publishing workflow](../publishing/index.md). The data are 341 open-ended
responses from a survey experiment (Gadarian & Albertson): half of respondents
were primed to feel **anxious** about immigration (`treatment = 1`), half not. The
question is whether the prime changed *what people wrote about*.

!!! info "Focus of this example"
    Model choice (an experiment ⟶ STM) · effect estimation by the method of
    composition. The randomized design means responses are independent, so no
    clustering is needed, unlike [Poliblog](poliblog.md). For validation and
    nested-data clustering see [Poliblog](poliblog.md); for corpus building see
    [Du Bois](dubois.md).

    Data: [`examples/gadarian.csv`](https://github.com/nealcaren/turbotopics/blob/main/examples/gadarian.csv) ·
    full script: [`examples/stm_vignette.py`](https://github.com/nealcaren/turbotopics/blob/main/examples/stm_vignette.py)

## Why STM

The design is a randomized experiment with a single binary covariate. We want to
know how that covariate moves **topic prevalence**, with a valid hypothesis test.
That is precisely what the [Structural Topic Model](../publishing/choosing-model.md)
is for: prevalence regressed on `treatment`, plus the method of composition for
honest standard errors.

```python
import csv, numpy as np, turbotopics as tt
from turbotopics import tokenize, stm

rows = list(csv.DictReader(open("examples/gadarian.csv")))
docs = [tokenize(r["open.ended.response"], stopwords=stop, min_length=3) for r in rows]
treatment = np.array([float(r["treatment"]) for r in rows]).reshape(-1, 1)

model = tt.STM(num_topics=3, seed=1)
model.fit(docs, treatment, prevalence_names=["treatment"], em_iters=40)
```

A small `K` is appropriate here: short responses, a simple design, and a
theoretically motivated handful of frames.

## The treatment effect

```python
draws = stm.posterior_theta_samples(model, nsims=30, seed=0)
effects = stm.estimate_effect(draws, treatment, feature_names=["treatment"])
labels = stm.label_topics(model.topic_word, model.vocabulary, n=6)
for t, e in enumerate(effects):
    d = e.as_dict()["treatment"]
    print(f"T{t}: coef={d['coef']:+.3f}  z={d['z']:+.1f}  "
          f"[{', '.join(w for w, _ in labels[t]['frex'][:5])}]")
```

```
T0: coef=+0.121  z=+3.9  [benefits, using, fact, never, issue]
T2: coef=-0.076  z=-2.4  [difficult, away, process, low, looking]
```

The anxiety prime **significantly raises** prevalence of the threat/benefits-of-
control frame and **lowers** a more procedural frame. This is the substantive
finding of the original study. Because treatment was randomized and each
respondent contributes one independent response, ordinary method-of-composition
standard errors are appropriate; no clustering is needed.

## Robustness

The [full vignette script](https://github.com/nealcaren/turbotopics/blob/main/examples/stm_vignette.py)
adds the rest of the reviewer-proof apparatus on this dataset: FREX labels,
`findThoughts` representative responses, the topic-correlation network, and a
`searchK` check. It recovers the published treatment effect and is guarded by the
test suite.
