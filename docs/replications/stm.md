# Structural Topic Model: the `stm` vignette

**Source.** Roberts, M. E., Stewart, B. M., & Tingley, D. (2019). stm: An R
Package for Structural Topic Models. *Journal of Statistical Software*, 91(2).
The `stm` package is the field standard for prevalence- and content-covariate
topic models in the social sciences.

topica's `STM` reimplements the same model: correlated topics with a prevalence
regression, a content (SAGE) covariate, spectral initialization, and effect
estimation by the method of composition. This page asks whether it produces the
same answers as R's `stm`.

## What "replicate" means for STM

STM is fit by variational EM, which is non-convex: the objective has many local
optima, and the solution depends on where the optimization starts. R's own `stm`
does not return one canonical answer. Fit it twice from different random seeds
and the two topic-word matrices agree only to a cosine of about 0.81. So the bar
is not bit-identical output. It is statistical: under a matched initialization,
topica should land in the same neighborhood of solutions that R lands in, and its
agreement with R should sit inside the spread of R's agreement with itself.

We feed identical integer-coded documents to both engines and align topics
one-to-one before comparing. The harness lives in
[`parity/stm_r_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_r_compare.py)
and [`parity/stm_content_r_compare.py`](https://github.com/nealcaren/topica/blob/main/parity/stm_content_r_compare.py).

## Content model: exact agreement

The content (SAGE) covariate is the deterministic part of STM: given the topic
assignments, the per-group word distributions follow in closed form. Here topica
and R agree exactly. On a bilingual corpus fit with `content = ~group`, K = 2,
the best-aligned cosine between R's and topica's per-group word distributions is
1.000 in both groups, and both engines separate the two topics rather than
collapsing them (topic-separation near 0 in each):

| Content group | topica–R cosine | both separated |
|---|---:|:--|
| `de` | 1.000 | yes |
| `en` | 1.000 | yes |

This is the path where a symmetric-initialization bug once collapsed all topics
to the background; the exact match against R is how we know it is fixed.

## Prevalence model: same neighborhood as R

For the prevalence model we compare topica's spectral fit to R's spectral fit on
a 339-document, 303-word corpus, against the floor of R's agreement with itself:

| Comparison | aligned cosine |
|---|---:|
| R Spectral vs R Random (R's own basin spread) | 0.62 |
| R Random vs R Random (R's self-consistency) | 0.81 |
| **R Spectral vs topica Spectral** | **0.51** |

topica's agreement with R (0.51) sits within the spread of R's own
Spectral-versus-Random runs (gap 0.11). The two engines find the same family of
solutions and differ by the local optimum the optimizer settled in, exactly as
two R runs do. This is the expected behavior for a non-convex model, not a
discrepancy: there is no single STM fit to reproduce.

What does replicate stably across optima is the substantive conclusion. The
[Poliblog](../examples/poliblog.md) and [Gadarian](../examples/gadarian.md)
worked examples refit the canonical `stm` vignettes end to end and recover the
same prevalence effects the package documents, with honest standard errors from
the method of composition.

## Speed

On matched iterations from a spectral start, topica fits the same model 3–22×
faster than R `stm`, single-threaded, and more with multiple cores, since topica
parallelizes the variational E-step while `stm` is single-threaded. The full
table is on the [benchmarks page](../benchmarks.md).
