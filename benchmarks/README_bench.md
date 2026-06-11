# topica benchmark harness

`bench.py` is the unified speed and memory benchmark for topica.  It runs three
model families (STM, keyATM, LDA) across several corpus sizes and thread counts,
measures both wall-clock fit time and peak resident set size (RSS) for each
engine, and produces a website table and two paper figures.

It also supports a same-model matrix (`--matrix`) that pits topica against
tomotopy 0.14 model-by-model at a fixed corpus size.

## Quick start

    python benchmarks/bench.py

This runs the full default sweep and writes three outputs:

| Output | Description |
|---|---|
| `benchmarks/bench_results.json` | Raw records (gitignored) |
| `benchmarks/website_table.md` | Markdown snippet for `docs/benchmarks.md` |
| `paper/fig_thread_scaling.pdf` | Thread-scaling speedup figure |
| `paper/fig_memory.pdf` | Peak RSS vs corpus size figure |

To re-render the outputs from a previous run without re-fitting:

    python benchmarks/bench.py --render

## Same-model matrix (tomotopy)

    python benchmarks/bench.py --matrix

Runs every matched model (LDA, CTM, DMR, HDP, PA, PT, SLDA, LabeledLDA) at
`MATRIX_SIZE` docs, single-threaded, and times topica against tomotopy for
each.  Writes:

| Output | Description |
|---|---|
| `benchmarks/matrix_results.json` | Raw records (gitignored) |
| `benchmarks/model_matrix.md` | Markdown table (gitignored) |

Smoke run for development:

    MATRIX_SIZE=600 MATRIX_K=10 MATRIX_ITERS=30 \
      python benchmarks/bench.py --matrix

### Models in the matrix

| Model | topica call | tomotopy call | Caveats |
|---|---|---|---|
| LDA | `LDA(num_topics=K, seed=1, optimize_interval=0, num_threads=1)` | `LDAModel(k=K, seed=1)` | Clean match |
| CTM | `CTM(num_topics=K).fit(docs, iters=ITERS, em_tol=0)` | `CTModel(k=K)` | topica uses Laplace/STM E-step; tomotopy uses mean-field; both are CTM variants |
| DMR | `DMR(num_topics=K, seed=1).fit(docs, features, iters=ITERS)` | `DMRModel(k=K, seed=1)` + `add_doc(words, metadata=str(r))` | topica takes a numeric covariate matrix; tomotopy takes string metadata |
| HDP | `HDP(seed=42).fit(docs, iters=ITERS)` | `HDPModel(seed=42)` | K is inferred by both; we match ITERS and report each library's discovered K |
| PA | `PA(num_super=K//2, num_sub=K, seed=1).fit(docs, iters=ITERS)` | `PAModel(k1=K//2, k2=K, seed=1)` | Clean match on super/sub counts |
| PT | `PT(num_topics=K, num_pseudo=100, seed=1).fit(docs, iters=ITERS)` | `PTModel(k=K, p=100, seed=1)` | Clean match |
| SLDA | `SupervisedLDA(num_topics=K, seed=1).fit(docs, y=rating, iters=ITERS)` | `SLDAModel(k=K, vars=['l'], seed=1)` + `add_doc(words, y=[r])` | topica: variational EM; tomotopy: Gibbs; same continuous response |
| LabeledLDA | `LabeledLDA(seed=1).fit(docs, doc_labels, iters=ITERS)` | `LLDAModel(k=2, seed=1)` + `add_doc(words, labels=lbls)` | 2 binary labels (Liberal/Conservative) from rating; K fixed by label count |

## Environment knobs

| Variable | Default | Meaning |
|---|---|---|
| `SIZES` | `2000,3500,5000` | Comma-separated corpus sizes (subsampled from poliblog5k) |
| `THREADS` | `1,2,4,8` | Thread counts for parallel Gibbs models (LDA, keyATM) |
| `STM_K` | `20` | Number of topics for STM |
| `STM_EM_ITERS` | `30` | EM iterations for STM |
| `KEYATM_K` | `10` | Number of topics for keyATM |
| `KEYATM_ITERS` | `1000` | Gibbs sweeps for keyATM |
| `LDA_K` | `20` | Number of topics for LDA |
| `LDA_ITERS` | `1000` | Gibbs iterations for LDA |
| `MATRIX_SIZE` | `3500` | Corpus size for same-model matrix |
| `MATRIX_K` | `20` | Number of topics for matrix models |
| `MATRIX_ITERS` | `500` | Iterations for matrix models |

Smoke run with small settings (for CI / development):

    SIZES=300,600 STM_EM_ITERS=3 KEYATM_ITERS=20 LDA_ITERS=30 THREADS=1,2 \
      python benchmarks/bench.py

## External dependencies

The harness auto-skips legs whose tools are absent:

- **R stm / R keyATM**: requires `Rscript` on PATH with the `stm`, `keyATM`,
  `quanteda`, and `jsonlite` packages installed.
- **Java MALLET**: requires `mallet` on PATH.
- **BERTopic clustering leg**: requires `bertopic` and `umap-learn` importable
  in the active Python environment.  When absent the leg prints a clean skip
  message and continues.  Published numbers for this leg must come from a
  machine where both packages are installed.
- **tomotopy matrix leg**: requires `tomotopy` importable.  Tested against
  tomotopy 0.14.0.

The poliblog5k corpus CSV (`benchmarks/poliblog5k_prepped.csv`) is generated
automatically on first run by calling `export_poliblog5k.R`.  It is gitignored;
the generator script is what we commit.

## How measurements work

### Peak RSS

Each topica fit runs in a child Python process wrapped with `/usr/bin/time`:

- macOS: `/usr/bin/time -l` reports peak RSS in bytes on a line containing
  "maximum resident set size"; we parse the leading integer and convert to MB.
- Linux: `/usr/bin/time -v` reports peak RSS in kibibytes on a line containing
  "Maximum resident set size (kbytes)"; we parse the trailing integer and
  convert to MB.

Running each fit in a subprocess means the number captures that fit's true
footprint, not the accumulated RSS of the parent.  The same `/usr/bin/time`
wrapping is applied to tomotopy subprocesses so RSS figures are
comparable across engines.

### BERTopic clustering-stage leg

The comparison is the clustering stage only: UMAP/PCA + HDBSCAN + c-TF-IDF.
Both topica and the reference BERTopic receive the same pre-built embedding
matrix so embedding generation time is excluded.  The shared embedding matrix
is a seeded random projection (Johnson-Lindenstrauss sketch, dim=384) of the
bag-of-words matrix; it is reproducible and requires no GPU or internet access.
This deliberately isolates clustering throughput from embedding quality.

### Published numbers

The numbers in the README and paper must come from a real (full-settings) run on
the maintainer's machine.  Smoke runs with reduced iteration counts are useful
for development only.  The results file and figures are gitignored so smoke
outputs are never accidentally committed as real results.
