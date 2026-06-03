"""Benchmark topica's STM fit time, optionally against the R ``stm`` package.

Measures wall-clock **fit time only** (import/startup excluded), at matched
``K`` / EM iterations / Spectral initialization, on synthetic corpora of varying
size and vocabulary. If ``Rscript`` with the ``stm`` package is on the PATH, the
identical integer-coded documents are also handed to R ``stm`` and the ratio is
reported; otherwise only topica numbers are printed.

Both engines run a fixed number of EM iterations from a Spectral init (R with
``max.em.its=iters, emtol=0`` so it does not stop early), so the comparison is
per-iteration cost, not time-to-convergence. R ``stm`` is single-threaded;
topica's variational E-step uses all cores by default — set
``RAYON_NUM_THREADS=1`` for an apples-to-apples single-core comparison.

Run::

    python benchmarks/bench_stm.py                      # topica on all cores
    RAYON_NUM_THREADS=1 python benchmarks/bench_stm.py  # single-threaded

The synthetic corpora are fixed-seed, so results are reproducible (timings vary
with hardware).
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import tempfile
import time

import numpy as np

from topica import STM

# (num_docs, vocab, num_topics) — a small/moderate/large-vocab sweep.
CONFIGS = [
    (1000, 500, 10),
    (2000, 2000, 10),
    (5000, 5000, 20),
]
EM_ITERS = 30
TOKENS_PER_DOC = 60


def synthetic_corpus(v, d, k_true, seed=0, length=TOKENS_PER_DOC):
    """A fixed-seed corpus of `d` docs over vocab `v` from `k_true` planted
    topics, with one prevalence covariate correlated with topic 0."""
    rng = np.random.default_rng(seed)
    beta = np.zeros((k_true, v))
    band = v // k_true
    for kk in range(k_true):
        cols = np.arange(kk * band, (kk + 1) * band) % v
        beta[kk, cols] = 1.0
        beta[kk] += 0.01
        beta[kk] /= beta[kk].sum()
    docs, cov = [], []
    for _ in range(d):
        theta = rng.dirichlet(np.ones(k_true) * 0.3)
        z = rng.choice(k_true, size=length, p=theta)
        docs.append([f"w{int(rng.choice(v, p=beta[zz]))}" for zz in z])
        cov.append(theta[0])
    return docs, np.array(cov).reshape(-1, 1)


def time_topica(docs, x, k, iters):
    t0 = time.perf_counter()
    m = STM(num_topics=k, init="spectral", seed=1)
    m.fit(docs, x, prevalence_names=["cov"], em_iters=iters)
    return time.perf_counter() - t0


def r_stm_available():
    if shutil.which("Rscript") is None:
        return False
    try:
        out = subprocess.run(
            ["Rscript", "-e", 'cat(requireNamespace("stm", quietly=TRUE))'],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return out.stdout.strip().endswith("TRUE")


_R_DRIVER = r"""
suppressMessages(library(stm))
lines <- readLines(file.path(dir, "docs.txt")); toks <- strsplit(lines, " ")
vocab <- sort(unique(unlist(toks))); vmap <- setNames(seq_along(vocab), vocab)
documents <- lapply(toks, function(d) {
  tb <- table(d); idx <- as.integer(vmap[names(tb)]); o <- order(idx)
  matrix(as.integer(rbind(idx[o], as.integer(tb)[o])), nrow = 2)
})
meta <- read.csv(file.path(dir, "meta.csv"))
el <- system.time(
  fit <- stm(documents, vocab, K = K, prevalence = ~cov, data = meta,
             init.type = "Spectral", max.em.its = ITERS, emtol = 0, verbose = FALSE)
)["elapsed"]
cat(sprintf("ELAPSED %f\n", el))
"""


def time_r_stm(docs, x, k, iters):
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "docs.txt"), "w") as f:
            f.write("\n".join(" ".join(doc) for doc in docs) + "\n")
        with open(os.path.join(d, "meta.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cov"])
            for c in x[:, 0]:
                w.writerow([c])
        script = f'dir <- "{d}"\nK <- {k}\nITERS <- {iters}\n' + _R_DRIVER
        out = subprocess.run(
            ["Rscript", "-e", script], capture_output=True, text=True, timeout=1800
        )
        for line in out.stdout.splitlines():
            if line.startswith("ELAPSED"):
                return float(line.split()[1])
        raise RuntimeError(f"R stm failed:\n{out.stdout}\n{out.stderr}")


def main():
    threads = os.environ.get("RAYON_NUM_THREADS", f"all ({os.cpu_count()} cores)")
    have_r = r_stm_available()
    print(f"topica threads: {threads};  EM iterations: {EM_ITERS};  "
          f"R stm: {'available' if have_r else 'not found (topica only)'}\n")
    header = f"{'docs':>6} {'vocab':>6} {'K':>3} | {'topica':>12}"
    if have_r:
        header += f" {'R stm':>10} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    for d, v, k in CONFIGS:
        docs, x = synthetic_corpus(v, d, k_true=max(k, 20))
        tt = time_topica(docs, x, k, EM_ITERS)
        row = f"{d:>6} {v:>6} {k:>3} | {tt:>10.2f}s"
        if have_r:
            rt = time_r_stm(docs, x, k, EM_ITERS)
            row += f" {rt:>8.2f}s {rt / tt:>7.1f}x"
        print(row)


if __name__ == "__main__":
    main()
