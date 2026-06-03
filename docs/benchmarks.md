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

topica's LDA is MALLET's SparseLDA collapsed-Gibbs sampler, reproduced
bit-for-bit (it matches MALLET's `train` output exactly). Against R, JVM MALLET,
and pure-Python gensim, that is a large speedup with no JVM startup.

Against [tomotopy](https://github.com/bab2min/tomotopy), a C++/SIMD library in
the same performance tier, plain LDA is a wash, and which one wins depends on
threading. 200 Gibbs iterations, fit time only.

**Single core** (exact, `num_threads=1` / `workers=1`): tomotopy's tighter inner
loop is about 20% ahead.

| docs | vocab | K | topica | tomotopy |
|------:|------:|---:|-------:|---------:|
| 2,000 | 1,000 | 20 | 1.84s | 1.59s |
| 5,000 | 2,000 | 50 | 6.58s | 5.37s |
| 10,000 | 3,000 | 50 | 14.2s | 11.3s |

**All cores** (both use approximate parallel Gibbs): topica's document-partitioned
parallelism scales better at these sizes, so it pulls even or slightly ahead.

| docs | vocab | K | topica | tomotopy |
|------:|------:|---:|-------:|---------:|
| 2,000 | 1,000 | 20 | 0.49s | 0.73s |
| 5,000 | 2,000 | 50 | 1.54s | 1.82s |
| 10,000 | 3,000 | 50 | 2.83s | 2.73s |

We report this straight: for plain LDA the two are interchangeable on speed.
topica's advantage is the STM, covariate-effect, and diagnostics stack built
around the sampler, not raw LDA throughput.

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
