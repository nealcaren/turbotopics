# Benchmarks

How fast topica is, measured honestly. Every number here is fit time only (model
construction and import excluded), on fixed-seed synthetic corpora, on one
machine (Apple M-series, 14 cores), and reproducible with the command shown.
Speed depends on corpus size, vocabulary, the number of topics, and hardware, so
read these as orders of magnitude, not guarantees.

## STM vs R `stm`

This is the comparison that matters for social scientists. R's `stm` is the
field standard, and a fit you wait minutes for in R runs in seconds in topica.
Both engines run the **same** number of EM iterations from a spectral
initialization (R with `emtol=0` so it does not stop early), so this measures
per-iteration cost, not time to convergence.

**All cores** (topica parallelizes the variational E-step; R `stm` is single-threaded):

| docs | vocab | K | topica | R `stm` | speedup |
|-----:|------:|---:|-------:|--------:|--------:|
| 1,000 | 500 | 10 | 0.14s | 3.16s | **22.5×** |
| 2,000 | 2,000 | 10 | 0.49s | 6.60s | **13.5×** |
| 5,000 | 5,000 | 20 | 2.75s | 26.9s | **9.8×** |

**Single core** (apples-to-apples, `RAYON_NUM_THREADS=1`):

| docs | vocab | K | topica | R `stm` | speedup |
|-----:|------:|---:|-------:|--------:|--------:|
| 1,000 | 500 | 10 | 0.50s | 3.03s | **6.0×** |
| 2,000 | 2,000 | 10 | 1.44s | 6.51s | **4.5×** |
| 5,000 | 5,000 | 20 | 8.97s | 26.3s | **2.9×** |

So topica is roughly **3 to 6 times faster single-threaded and 10 to 23 times on
all cores**, and it produces the same fit (the content and prevalence models are
[validated against R `stm`](publishing/validation.md)). Reproduce:

```bash
python benchmarks/bench_stm.py                      # all cores
RAYON_NUM_THREADS=1 python benchmarks/bench_stm.py  # single core
```

!!! note "What this is, and is not"
    Per-iteration fit time, not time to convergence (the two engines may need a
    different number of iterations to converge). One machine, synthetic corpora.
    R `stm` is single-threaded by design; the all-cores column is topica's
    automatic parallelism, which is the speed you actually get.

## LDA: MALLET's algorithm without the JVM

topica's LDA binds RustMallet, David Mimno's Rust port of MALLET's SparseLDA
collapsed-Gibbs sampler, and reproduces its `train` CLI byte-for-byte; against
Java MALLET (a different RNG) it recovers the same topics (cosine 1.000). On fit
time it is roughly at parity with Java MALLET — and adds no JVM startup, unlike
the JVM samplers or pure-Python gensim.

Against [tomotopy](https://github.com/bab2min/tomotopy), a C++/SIMD library,
which one is faster turns on the number of topics K, because the two make
opposite algorithmic bets. tomotopy computes a dense topic distribution for every
token and vectorizes it with Eigen, which is fastest when K is small; topica
(like MALLET) uses a sparse sampler that visits only the topics a word actually
occupies, which wins when K is large. On a 3,500-document, 2,632-word corpus (200
Gibbs iterations, single core, fit time only) the two cross near K=200:

| K | topica | tomotopy | faster |
|---:|-------:|---------:|--------|
| 20 | 12.1s | 5.3s | tomotopy 2.3× |
| 50 | 13.4s | 7.4s | tomotopy 1.8× |
| 100 | 15.0s | 11.1s | tomotopy 1.35× |
| 200 | 18.1s | 19.1s | topica 1.06× |
| 400 | 23.8s | 35.1s | topica 1.48× |

topica's time barely moves as K grows, because the sparse sampler touches only
the topics each word occupies; tomotopy's rises with K, because it scores all K
every token. Fine-grained topic models, the large-K regime social scientists
often want, favor topica's sampler; small-K fits favor tomotopy's. Either way
topica also retains the MCMC posterior draws its uncertainty tooling needs
(`composition_theta`, `prevalence_ci`) at no measurable extra cost, which neither
MALLET nor tomotopy computes at all.

## Memory

topica holds the corpus and model state in compact native arrays, so it also fits
in a fraction of the reference tools' memory. Fitting the structural topic model
on the poliblog corpus, peak resident memory:

| docs | topica STM | R `stm` |
|-----:|-----------:|--------:|
| 2,000 | 287 MB | 1,463 MB |
| 3,500 | 375 MB | 1,504 MB |
| 5,000 | 463 MB | 1,737 MB |

About a quarter of R `stm`'s footprint, and the gap widens with corpus size.

## keyATM vs R `keyATM`

topica's keyATM reproduces the R package's keyword-assisted model and is
[validated against it](replications/keyatm-dynamic.md): the same keyword topics,
the same per-sweep asymmetric-α estimation, the same `model_fit` log-likelihood.
On speed it matches R's C++ sampler single-threaded and adds a
document-partitioned parallel sweep that R has no equivalent of. Same keywords,
same number of Gibbs sweeps, α learned each sweep on both sides; fit time only.

| docs | vocab | K | sweeps | topica (1 core) | topica (4 cores) | R `keyATM` |
|-----:|------:|---:|-------:|----------------:|-----------------:|-----------:|
| 2,000 | 2,632 | 10 | 1,000 | 25.9s | **12.1s** | 24.5s |

So topica is at parity with R single-threaded and about **2× faster on four
cores**. If you do not need the R-matching asymmetric prior, `estimate_alpha=False`
fixes a symmetric α and skips the per-sweep slice sampler for a further 15 to 20%
(more at larger K). This row, with the STM and LDA comparisons above, is
reproducible in one command:

```bash
python benchmarks/speed_vs_r.py
```

## Large-K sampling: SparseLDA vs LightLDA

[LightLDA](guides/models.md#sampler-choice-sparselda-vs-lightlda)'s
`O(1)`-per-token alias sampler is built for very large K. At the corpus sizes
typical of social science, SparseLDA stays faster, because its buckets remain
sparse; LightLDA's flatter scaling in K only pulls ahead past roughly K ≈ 1,000.
Use the default `sampler="sparse"` unless you have a specific large-K reason.

## Coherence

`c_v` and the other windowed coherence measures are computed in the Rust core,
counting only the word pairs within a topic's top-N rather than a full
vocabulary-by-vocabulary matrix. A 500-topic `c_v` that took minutes in a
pure-Python loop now takes a fraction of a second, which is what makes coherence
practical for model selection at large K.
