"""Unified speed + memory benchmark for topica vs R/Java references.

Runs three model families (STM, keyATM, LDA) across several corpus sizes
(subsamples of poliblog5k) and thread counts, measuring both wall-clock fit
time and peak resident set size (RSS) for each engine.  Optionally runs a
BERTopic clustering-stage comparison if bertopic and umap are importable.

Usage
-----
    python benchmarks/bench.py            # full default sweep
    python benchmarks/bench.py --render   # render outputs from a previous run

Env knobs
---------
    SIZES          corpus sizes to sweep (default 2000,3500,5000)
    THREADS        thread counts for parallel Gibbs models (default 1,2,4,8)
    STM_K          number of topics for STM (default 20)
    STM_EM_ITERS   EM iterations for STM (default 30)
    KEYATM_K       number of topics for keyATM (default 10)
    KEYATM_ITERS   Gibbs sweeps for keyATM (default 1000)
    LDA_K          number of topics for LDA (default 20)
    LDA_ITERS      Gibbs iterations for LDA (default 1000)

Notes
-----
Peak RSS is measured by spawning each fit as a child process wrapped with
/usr/bin/time so the measurement captures only that fit's memory footprint,
not the parent process's accumulated RSS.

The BERTopic leg compares the CLUSTERING STAGE ONLY: UMAP/PCA + HDBSCAN +
c-TF-IDF.  Both topica and the reference BERTopic receive the same pre-built
embedding matrix (a seeded random projection of the bag-of-words matrix,
dim=384), so embedding generation time is excluded and the comparison is
purely clustering throughput.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

HERE = Path(__file__).parent.resolve()
ROOT = HERE.parent
PREPPED_CSV = HERE / "poliblog5k_prepped.csv"
RESULTS_JSON = HERE / "bench_results.json"
PAPER_DIR = ROOT / "paper"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "parity"))

import speed_vs_r as SV  # noqa: E402 — reuse _rscript, _R_STM, _R_KEYATM
import keyatm_r_compare as KA  # noqa: E402 — reuse KEYWORD_SETS

# ---------------------------------------------------------------------------
# Env-var knobs
# ---------------------------------------------------------------------------

SIZES = [int(x) for x in os.environ.get("SIZES", "2000,3500,5000").split(",")]
THREADS = [int(x) for x in os.environ.get("THREADS", "1,2,4,8").split(",")]
STM_K = int(os.environ.get("STM_K", "20"))
STM_EM_ITERS = int(os.environ.get("STM_EM_ITERS", "30"))
KEYATM_K = int(os.environ.get("KEYATM_K", "10"))
KEYATM_ITERS = int(os.environ.get("KEYATM_ITERS", "1000"))
LDA_K = int(os.environ.get("LDA_K", "20"))
LDA_ITERS = int(os.environ.get("LDA_ITERS", "1000"))
MIN_DF = 3
EMBED_DIM = 384  # random-projection embedding dimension for BERTopic leg

PYTHON = sys.executable  # the venv python


# ---------------------------------------------------------------------------
# Peak-RSS measurement
# ---------------------------------------------------------------------------

def peak_rss_mb(argv: list[str]) -> tuple[str, float]:
    """Run argv wrapped with /usr/bin/time and return (stdout, peak_rss_mb).

    On macOS /usr/bin/time -l prints peak RSS in bytes (line contains
    "maximum resident set size").  On Linux /usr/bin/time -v prints it in
    kibibytes (line contains "Maximum resident set size (kbytes)").
    """
    if sys.platform == "darwin":
        cmd = ["/usr/bin/time", "-l"] + argv
    else:
        cmd = ["/usr/bin/time", "-v"] + argv

    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = proc.stdout

    # /usr/bin/time writes its report to stderr on both platforms.
    stderr = proc.stderr

    if proc.returncode != 0:
        # Surface the child's error but do not abort — return NaN for RSS.
        combined = (stdout + "\n" + stderr).strip()
        if "FIT_TIME" not in stdout:
            raise RuntimeError(
                f"Subprocess failed (rc={proc.returncode}):\n{combined}"
            )

    rss_mb = float("nan")
    if sys.platform == "darwin":
        for line in stderr.splitlines():
            lo = line.lower()
            if "maximum resident set size" in lo:
                # macOS format: "  1234567  maximum resident set size"
                try:
                    rss_mb = int(line.split()[0]) / (1024 * 1024)
                except (ValueError, IndexError):
                    pass
                break
    else:
        for line in stderr.splitlines():
            if "Maximum resident set size" in line:
                try:
                    rss_mb = int(line.split()[-1]) / 1024  # KB -> MB
                except (ValueError, IndexError):
                    pass
                break

    return stdout, rss_mb


def _rscript_rss(r_body: str, timeout: int = 3600) -> tuple[float, float]:
    """Run an R script body wrapped with /usr/bin/time.

    The body must print a line starting with ``R_TIME <seconds>``.
    Returns (r_time_seconds, peak_rss_mb).
    """
    script = r_body
    if sys.platform == "darwin":
        cmd = ["/usr/bin/time", "-l", "Rscript", "-e", script]
    else:
        cmd = ["/usr/bin/time", "-v", "Rscript", "-e", script]

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    stdout = proc.stdout
    stderr = proc.stderr

    # Check for R-level errors.
    if proc.returncode != 0 or "Error" in stderr.split("/usr/bin/time")[0]:
        # Tolerate "Error in" from R but only if R_TIME is present.
        lines_out = [l for l in stdout.splitlines() if l.startswith("R_TIME")]
        if not lines_out:
            raise RuntimeError(f"R failed:\n{stdout}\n{stderr}")

    r_time_lines = [l for l in stdout.splitlines() if l.startswith("R_TIME")]
    if not r_time_lines:
        raise RuntimeError(f"R did not print R_TIME:\n{stdout}\n{stderr}")
    r_time = float(r_time_lines[0].split()[1])

    rss_mb = float("nan")
    if sys.platform == "darwin":
        for line in stderr.splitlines():
            lo = line.lower()
            if "maximum resident set size" in lo:
                try:
                    rss_mb = int(line.split()[0]) / (1024 * 1024)
                except (ValueError, IndexError):
                    pass
                break
    else:
        for line in stderr.splitlines():
            if "Maximum resident set size" in line:
                try:
                    rss_mb = int(line.split()[-1]) / 1024
                except (ValueError, IndexError):
                    pass
                break

    return r_time, rss_mb


# ---------------------------------------------------------------------------
# Corpus helpers
# ---------------------------------------------------------------------------

def _ensure_csv() -> None:
    """Generate poliblog5k_prepped.csv via R if it does not exist."""
    if PREPPED_CSV.exists():
        return
    r_script = str(HERE / "export_poliblog5k.R")
    print(f"  generating {PREPPED_CSV.name} via R ...", flush=True)
    proc = subprocess.run(
        ["Rscript", r_script, str(PREPPED_CSV)],
        capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0 or not PREPPED_CSV.exists():
        raise RuntimeError(
            f"export_poliblog5k.R failed:\n{proc.stdout}\n{proc.stderr}"
        )
    print(f"  {proc.stdout.strip()}", flush=True)


def _load_full() -> tuple[list[list[str]], np.ndarray, np.ndarray]:
    rows = list(csv.DictReader(open(PREPPED_CSV, newline="")))
    toks = [r["text"].split() for r in rows]
    rating = np.array([1.0 if r["rating"] == "Liberal" else 0.0 for r in rows])
    day = np.array([float(r["day"]) for r in rows])
    return toks, rating, day


def _subsample(
    toks: list[list[str]],
    rating: np.ndarray,
    day: np.ndarray,
    n: int,
    seed: int = 0,
) -> tuple[list[list[str]], np.ndarray, np.ndarray, set[str]]:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(toks), size=min(n, len(toks)), replace=False)
    sub = [toks[i] for i in idx]

    df: Counter = Counter()
    for d in sub:
        df.update(set(d))
    vocab = {w for w, c in df.items() if c >= MIN_DF}

    docs: list[list[str]] = []
    keep: list[bool] = []
    for d in sub:
        dd = [w for w in d if w in vocab]
        keep.append(len(dd) > 0)
        if dd:
            docs.append(dd)

    keep_arr = np.array(keep)
    return docs, rating[idx][keep_arr], day[idx][keep_arr], vocab


# ---------------------------------------------------------------------------
# STM benchmark
# ---------------------------------------------------------------------------

def _bench_stm(
    docs: list[list[str]],
    rating: np.ndarray,
    day: np.ndarray,
) -> dict:
    from topica.stm import spline

    spline_basis, _ = spline(day, df=10)
    X = np.column_stack([rating, spline_basis])
    feat = ["ratingLiberal"] + [f"day_s{j}" for j in range(spline_basis.shape[1])]
    design = np.column_stack([np.ones(len(docs)), X])

    with tempfile.TemporaryDirectory() as d:
        vdocs_path = os.path.join(d, "vdocs.txt")
        design_path = os.path.join(d, "design.csv")
        docs_pkl = os.path.join(d, "docs.json")

        with open(vdocs_path, "w") as f:
            f.write("\n".join(" ".join(doc) for doc in docs) + "\n")
        with open(design_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["intercept"] + feat)
            w.writerows(design.tolist())

        # Save docs + X for subprocess.
        json.dump(
            {"docs": docs, "X": X.tolist(), "feat": feat},
            open(docs_pkl, "w"),
        )

        # R reference with RSS.
        r_body = (
            f'dir <- "{d}"\nKVAL <- {STM_K}\nNITERS <- {STM_EM_ITERS}\n'
            + SV._R_STM
        )
        r_time, ref_rss = _rscript_rss(r_body)

        # topica STM in subprocess for clean RSS measurement.
        py_script = (
            "import json, sys, time\n"
            f"sys.path.insert(0, {str(ROOT / 'python')!r})\n"
            f"d = json.load(open({docs_pkl!r}))\n"
            "import topica, numpy as np\n"
            "docs = d['docs']; feat = d['feat']\n"
            "X = np.array(d['X'])\n"
            f"t0 = time.perf_counter()\n"
            f"topica.STM(num_topics={STM_K}, init='spectral').fit("
            f"    docs, X, prevalence_names=feat,"
            f"    iters={STM_EM_ITERS}, em_tol=0.0)\n"
            "print('FIT_TIME', time.perf_counter()-t0)\n"
        )
        stdout, tt_rss = peak_rss_mb([PYTHON, "-c", py_script])
        tt_time = float(
            [l for l in stdout.splitlines() if l.startswith("FIT_TIME")][0].split()[1]
        )

    return {
        "topica_time": tt_time,
        "ref_time": r_time,
        "topica_rss_mb": tt_rss,
        "ref_rss_mb": ref_rss,
    }


# ---------------------------------------------------------------------------
# keyATM benchmark
# ---------------------------------------------------------------------------

def _bench_keyatm(
    docs: list[list[str]],
    vocab: set[str],
    threads: int,
) -> dict:
    kws = {
        name: [w for w in ws if w in vocab]
        for name, ws in KA.KEYWORD_SETS.items()
    }
    kws = {name: ws for name, ws in kws.items() if ws}

    with tempfile.TemporaryDirectory() as d:
        vdocs_path = os.path.join(d, "vdocs.txt")
        kw_path = os.path.join(d, "keywords.json")
        docs_pkl = os.path.join(d, "docs.json")

        with open(vdocs_path, "w") as f:
            f.write("\n".join(" ".join(doc) for doc in docs) + "\n")
        json.dump(kws, open(kw_path, "w"))
        json.dump({"docs": docs, "kws": kws}, open(docs_pkl, "w"))

        nreg = KEYATM_K - len(kws)
        r_body = (
            f'dir <- "{d}"\nNREG <- {nreg}\nNITERS <- {KEYATM_ITERS}\n'
            + SV._R_KEYATM
        )
        r_time, ref_rss = _rscript_rss(r_body)

        py_script = (
            "import json, sys, time\n"
            f"sys.path.insert(0, {str(ROOT / 'python')!r})\n"
            f"d = json.load(open({docs_pkl!r}))\n"
            "import topica\n"
            "docs = d['docs']; kws = d['kws']\n"
            f"t0 = time.perf_counter()\n"
            f"topica.KeyATM(kws, num_topics={KEYATM_K}, seed=1).fit("
            f"    docs, iters={KEYATM_ITERS}, num_threads={threads})\n"
            "print('FIT_TIME', time.perf_counter()-t0)\n"
        )
        stdout, tt_rss = peak_rss_mb([PYTHON, "-c", py_script])
        tt_time = float(
            [l for l in stdout.splitlines() if l.startswith("FIT_TIME")][0].split()[1]
        )

    return {
        "threads": threads,
        "topica_time": tt_time,
        "ref_time": r_time,
        "topica_rss_mb": tt_rss,
        "ref_rss_mb": ref_rss,
    }


# ---------------------------------------------------------------------------
# LDA benchmark
# ---------------------------------------------------------------------------

def _bench_lda(
    docs: list[list[str]],
    threads: int,
) -> dict:
    """Time topica LDA vs. Java MALLET at a given thread count.

    The MALLET reference always runs single-threaded; the thread count here
    applies only to topica.  The MALLET leg is skipped when mallet is absent.
    """
    mallet = shutil.which("mallet")

    with tempfile.TemporaryDirectory() as d:
        docs_pkl = os.path.join(d, "docs.json")
        json.dump({"docs": docs}, open(docs_pkl, "w"))

        ref_time: float | None = None
        ref_rss: float | None = None

        if mallet and threads == min(THREADS):
            # Run the MALLET reference ONCE (at the first thread count so we
            # do not repeat it for every thread sweep entry).
            txt = os.path.join(d, "tok.txt")
            with open(txt, "w") as f:
                for i, toks in enumerate(docs):
                    f.write(f"doc{i}\t{' '.join(toks)}\n")
            mal = os.path.join(d, "tok.mallet")
            subprocess.run(
                [
                    mallet, "import-file",
                    "--input", txt, "--output", mal,
                    "--keep-sequence",
                    "--token-regex", r"\S+",
                    "--line-regex", r"^(\S+)\t(.*)$",
                    "--name", "1", "--data", "2", "--label", "0",
                ],
                check=True, capture_output=True, text=True,
            )
            mallet_cmd = [
                mallet, "train-topics",
                "--input", mal,
                "--num-topics", str(LDA_K),
                "--num-iterations", str(LDA_ITERS),
                "--random-seed", "1",
                "--optimize-interval", "0",
                "--num-threads", "1",
            ]
            stdout_m, ref_rss_val = peak_rss_mb(mallet_cmd)
            # MALLET does not print FIT_TIME; we parse its own timing line or
            # fall back to wall-clock via /usr/bin/time wrapping.
            # Actually we time it by measuring the elapsed time ourselves
            # since mallet stdout doesn't have a timing marker in this form.
            # Re-run WITHOUT /usr/bin/time for the actual timing.
            t0 = time.perf_counter()
            subprocess.run(mallet_cmd, check=True, capture_output=True, text=True)
            ref_time = time.perf_counter() - t0
            ref_rss = ref_rss_val

        py_script = (
            "import json, sys, time\n"
            f"sys.path.insert(0, {str(ROOT / 'python')!r})\n"
            f"d = json.load(open({docs_pkl!r}))\n"
            "import topica\n"
            "docs = d['docs']\n"
            f"t0 = time.perf_counter()\n"
            f"topica.LDA(num_topics={LDA_K}, seed=1, optimize_interval=0,"
            f"           num_threads={threads}).fit(docs, iters={LDA_ITERS})\n"
            "print('FIT_TIME', time.perf_counter()-t0)\n"
        )
        stdout, tt_rss = peak_rss_mb([PYTHON, "-c", py_script])
        tt_time = float(
            [l for l in stdout.splitlines() if l.startswith("FIT_TIME")][0].split()[1]
        )

    return {
        "threads": threads,
        "topica_time": tt_time,
        "ref_time": ref_time,
        "topica_rss_mb": tt_rss,
        "ref_rss_mb": ref_rss,
    }


# ---------------------------------------------------------------------------
# BERTopic clustering-stage benchmark
# ---------------------------------------------------------------------------

def _bertopic_available() -> bool:
    try:
        import importlib.util
        return (
            importlib.util.find_spec("bertopic") is not None
            and importlib.util.find_spec("umap") is not None
        )
    except Exception:
        return False


def _build_embedding_matrix(
    docs: list[list[str]], vocab: set[str], dim: int, seed: int = 42
) -> np.ndarray:
    """Build a reproducible (n_docs, dim) float32 embedding matrix.

    We project the bag-of-words matrix with a seeded random Gaussian
    projection (Johnson-Lindenstrauss sketch).  This produces a deterministic
    dense embedding without any external model, so the BERTopic clustering-
    stage timing is reproducible and requires no GPU or internet access.
    This isolates clustering throughput from embedding generation time; the
    README notes this explicitly.
    """
    sorted_vocab = sorted(vocab)
    v2i = {w: i for i, w in enumerate(sorted_vocab)}
    V = len(sorted_vocab)

    # Build sparse BOW as a dense matrix (n x V).
    bow = np.zeros((len(docs), V), dtype=np.float32)
    for i, d in enumerate(docs):
        for w in d:
            if w in v2i:
                bow[i, v2i[w]] += 1.0

    # L2-normalise rows.
    norms = np.linalg.norm(bow, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    bow /= norms

    # Random projection.
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((V, dim)).astype(np.float32) / np.sqrt(dim)
    emb = bow @ proj
    # L2-normalise again.
    norms2 = np.linalg.norm(emb, axis=1, keepdims=True)
    norms2[norms2 == 0] = 1.0
    emb /= norms2
    return emb


def _bench_bertopic(
    docs: list[list[str]],
    vocab: set[str],
) -> dict | None:
    if not _bertopic_available():
        print("  BERTopic leg: skipped — bertopic/umap not importable", flush=True)
        return None

    emb = _build_embedding_matrix(docs, vocab, EMBED_DIM)
    flat_docs = [" ".join(d) for d in docs]

    with tempfile.TemporaryDirectory() as d:
        emb_path = os.path.join(d, "emb.npy")
        docs_path = os.path.join(d, "docs.json")
        np.save(emb_path, emb)
        json.dump(flat_docs, open(docs_path, "w"))

        # topica BERTopic clustering stage.
        py_topica = (
            "import json, sys, time, numpy as np\n"
            f"sys.path.insert(0, {str(ROOT / 'python')!r})\n"
            "import topica\n"
            f"docs = json.load(open({docs_path!r}))\n"
            f"emb = np.load({emb_path!r})\n"
            "t0 = time.perf_counter()\n"
            "topica.BERTopic().fit_transform(docs, emb)\n"
            "print('FIT_TIME', time.perf_counter()-t0)\n"
        )
        stdout_t, topica_rss = peak_rss_mb([PYTHON, "-c", py_topica])
        topica_time = float(
            [l for l in stdout_t.splitlines() if l.startswith("FIT_TIME")][0].split()[1]
        )

        # Reference BERTopic clustering stage (embedding_model=None so it
        # accepts precomputed embeddings, no sentence-transformer invoked).
        py_ref = (
            "import json, sys, time, numpy as np\n"
            f"docs = json.load(open({docs_path!r}))\n"
            f"emb = np.load({emb_path!r})\n"
            "from bertopic import BERTopic\n"
            "t0 = time.perf_counter()\n"
            "BERTopic(embedding_model=None).fit_transform(docs, emb)\n"
            "print('FIT_TIME', time.perf_counter()-t0)\n"
        )
        stdout_r, ref_rss = peak_rss_mb([PYTHON, "-c", py_ref])
        ref_time = float(
            [l for l in stdout_r.splitlines() if l.startswith("FIT_TIME")][0].split()[1]
        )

    return {
        "topica_time": topica_time,
        "ref_time": ref_time,
        "topica_rss_mb": topica_rss,
        "ref_rss_mb": ref_rss,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep() -> list[dict]:
    _ensure_csv()
    toks, rating_full, day_full = _load_full()
    print(f"poliblog5k: {len(toks)} docs total | sizes={SIZES} threads={THREADS}\n",
          flush=True)

    records: list[dict] = []

    for n in SIZES:
        docs, rating, day, vocab = _subsample(toks, rating_full, day_full, n)
        nd = len(docs)
        nv = len(vocab)
        print(f"== N~{n} ({nd} docs, {nv} vocab) ==", flush=True)

        # --- STM ---
        print("  STM vs R stm ...", flush=True)
        stm_res = _bench_stm(docs, rating, day)
        rec = {
            "n_docs": nd, "vocab": nv,
            "model": "stm", "threads": 1,
            "topica_time": stm_res["topica_time"],
            "ref_time": stm_res["ref_time"],
            "topica_rss_mb": stm_res["topica_rss_mb"],
            "ref_rss_mb": stm_res["ref_rss_mb"],
        }
        records.append(rec)
        print(
            f"    topica {stm_res['topica_time']:.1f}s  "
            f"R {stm_res['ref_time']:.1f}s  "
            f"speedup {stm_res['ref_time']/stm_res['topica_time']:.1f}x  "
            f"topica RSS {stm_res['topica_rss_mb']:.0f} MB  "
            f"R RSS {stm_res['ref_rss_mb']:.0f} MB",
            flush=True,
        )

        # --- keyATM (thread sweep) ---
        print("  keyATM vs R keyATM (thread sweep) ...", flush=True)
        ref_time_ka: float | None = None
        ref_rss_ka: float | None = None
        for t in THREADS:
            ka_res = _bench_keyatm(docs, vocab, t)
            # Capture ref numbers from the first thread-count entry.
            if ref_time_ka is None:
                ref_time_ka = ka_res["ref_time"]
                ref_rss_ka = ka_res["ref_rss_mb"]
            rec = {
                "n_docs": nd, "vocab": nv,
                "model": "keyatm", "threads": t,
                "topica_time": ka_res["topica_time"],
                "ref_time": ka_res["ref_time"],
                "topica_rss_mb": ka_res["topica_rss_mb"],
                "ref_rss_mb": ka_res["ref_rss_mb"],
            }
            records.append(rec)
            print(
                f"    threads={t}  topica {ka_res['topica_time']:.1f}s  "
                f"R {ka_res['ref_time']:.1f}s  "
                f"speedup {ka_res['ref_time']/ka_res['topica_time']:.1f}x  "
                f"RSS {ka_res['topica_rss_mb']:.0f} MB",
                flush=True,
            )

        # --- LDA (thread sweep) ---
        print("  LDA vs MALLET (thread sweep) ...", flush=True)
        mallet_ok = shutil.which("mallet") is not None
        if not mallet_ok:
            print("    MALLET not found — ref column will be null", flush=True)
        for t in THREADS:
            lda_res = _bench_lda(docs, t)
            rec = {
                "n_docs": nd, "vocab": nv,
                "model": "lda", "threads": t,
                "topica_time": lda_res["topica_time"],
                "ref_time": lda_res["ref_time"],
                "topica_rss_mb": lda_res["topica_rss_mb"],
                "ref_rss_mb": lda_res["ref_rss_mb"],
            }
            records.append(rec)
            ref_str = (
                f"MALLET {lda_res['ref_time']:.1f}s  "
                if lda_res["ref_time"] is not None
                else "MALLET n/a  "
            )
            print(
                f"    threads={t}  topica {lda_res['topica_time']:.1f}s  "
                + ref_str
                + f"RSS {lda_res['topica_rss_mb']:.0f} MB",
                flush=True,
            )

        # --- BERTopic (clustering stage only) ---
        print("  BERTopic clustering stage ...", flush=True)
        bt_res = _bench_bertopic(docs, vocab)
        if bt_res is not None:
            rec = {
                "n_docs": nd, "vocab": nv,
                "model": "bertopic", "threads": 1,
                "topica_time": bt_res["topica_time"],
                "ref_time": bt_res["ref_time"],
                "topica_rss_mb": bt_res["topica_rss_mb"],
                "ref_rss_mb": bt_res["ref_rss_mb"],
            }
            records.append(rec)
            print(
                f"    topica {bt_res['topica_time']:.1f}s  "
                f"ref {bt_res['ref_time']:.1f}s  "
                f"speedup {bt_res['ref_time']/bt_res['topica_time']:.1f}x  "
                f"RSS {bt_res['topica_rss_mb']:.0f} MB",
                flush=True,
            )

        print(flush=True)

    json.dump(records, open(RESULTS_JSON, "w"), indent=2)
    print(f"Results written to {RESULTS_JSON}", flush=True)
    return records


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render_website(records: list[dict] | None = None) -> None:
    """Render a markdown table for docs/benchmarks.md."""
    if records is None:
        records = json.load(open(RESULTS_JSON))

    import io

    # Build a lookup: (n_docs, model, threads) -> record
    by_key: dict[tuple, dict] = {}
    for r in records:
        by_key[(r["n_docs"], r["model"], r["threads"])] = r

    # Collect unique sizes and models (ordered).
    sizes = sorted({r["n_docs"] for r in records})
    models = []
    for m in ("stm", "keyatm", "lda", "bertopic"):
        if any(r["model"] == m for r in records):
            models.append(m)

    buf = io.StringIO()
    buf.write("## topica benchmark: speed vs reference (wall-clock)\n\n")
    buf.write(
        "Corpus: poliblog5k subsampled to each size (seeded, reproducible). "
        "Speedup = reference time / topica single-thread time. "
        "All timings exclude model loading.\n\n"
    )

    for model in models:
        model_records = [r for r in records if r["model"] == model]
        if not model_records:
            continue

        buf.write(f"### {model.upper()}\n\n")

        threads_for_model = sorted({r["threads"] for r in model_records})
        has_ref = any(r.get("ref_time") is not None for r in model_records)

        # Header row.
        header = ["n_docs", "vocab"]
        if has_ref:
            header += ["ref (s)", "topica 1-thread (s)", "1-thread speedup"]
        else:
            header += ["topica 1-thread (s)"]
        for t in threads_for_model:
            if t > 1:
                header += [f"topica {t}-thread (s)", f"{t}-thread speedup"]
        header += ["topica RSS (MB)"]
        if has_ref:
            header += ["ref RSS (MB)"]

        buf.write("| " + " | ".join(header) + " |\n")
        buf.write("|" + "|".join(["---"] * len(header)) + "|\n")

        for sz in sizes:
            row_records = {
                r["threads"]: r
                for r in model_records
                if r["n_docs"] == sz
            }
            if not row_records:
                continue

            r1 = row_records.get(min(threads_for_model))
            if r1 is None:
                continue

            row: list[str] = [str(r1["n_docs"]), str(r1["vocab"])]
            if has_ref:
                ref_t = r1.get("ref_time")
                row.append(f"{ref_t:.1f}" if ref_t is not None else "n/a")
                row.append(f"{r1['topica_time']:.1f}")
                row.append(
                    f"{ref_t/r1['topica_time']:.1f}x"
                    if ref_t is not None else "n/a"
                )
            else:
                row.append(f"{r1['topica_time']:.1f}")

            for t in threads_for_model:
                if t > 1:
                    rt = row_records.get(t)
                    if rt:
                        row.append(f"{rt['topica_time']:.1f}")
                        ref_t2 = rt.get("ref_time") or (r1.get("ref_time"))
                        row.append(
                            f"{ref_t2/rt['topica_time']:.1f}x"
                            if ref_t2 is not None else "n/a"
                        )
                    else:
                        row += ["n/a", "n/a"]

            row.append(f"{r1['topica_rss_mb']:.0f}")
            if has_ref:
                ref_rss = r1.get("ref_rss_mb")
                row.append(f"{ref_rss:.0f}" if ref_rss is not None else "n/a")

            buf.write("| " + " | ".join(row) + " |\n")

        buf.write("\n")

    table_md = buf.getvalue()
    out_path = HERE / "website_table.md"
    out_path.write_text(table_md)
    print(table_md)
    print(f"Website table written to {out_path}", flush=True)


def render_thread_figure(records: list[dict] | None = None) -> None:
    """Render paper figure 1: thread-scaling speedup vs thread count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if records is None:
        records = json.load(open(RESULTS_JSON))

    PAPER_DIR.mkdir(exist_ok=True)
    out_path = PAPER_DIR / "fig_thread_scaling.pdf"

    # Use the largest corpus size for the thread-scaling figure.
    sizes = sorted({r["n_docs"] for r in records})
    max_size = sizes[-1]

    fig, ax = plt.subplots(figsize=(6, 4))

    ylabel = "Speedup vs single-thread reference"
    for model, label, color in [
        ("lda", "LDA vs MALLET", "steelblue"),
        ("keyatm", "keyATM vs R keyATM", "firebrick"),
    ]:
        recs = [r for r in records if r["model"] == model and r["n_docs"] == max_size]
        if not recs:
            continue
        recs_sorted = sorted(recs, key=lambda r: r["threads"])
        threads = [r["threads"] for r in recs_sorted]
        # Speedup = ref_time / topica_time (using ref from single-thread entry).
        ref_t = next((r["ref_time"] for r in recs_sorted if r["ref_time"] is not None), None)
        if ref_t is None:
            # No reference available; use 1-thread topica as baseline.
            base = recs_sorted[0]["topica_time"]
            speedups = [base / r["topica_time"] for r in recs_sorted]
            ylabel = "Thread speedup (vs 1-thread topica)"
        else:
            speedups = [ref_t / r["topica_time"] for r in recs_sorted]

        ax.plot(threads, speedups, marker="o", label=label, color=color)

    ax.set_xlabel("Threads")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Thread scaling at N={max_size} docs")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(THREADS)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Thread-scaling figure written to {out_path}", flush=True)


def render_memory_figure(records: list[dict] | None = None) -> None:
    """Render paper figure 2: peak RSS vs corpus size for STM."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if records is None:
        records = json.load(open(RESULTS_JSON))

    PAPER_DIR.mkdir(exist_ok=True)
    out_path = PAPER_DIR / "fig_memory.pdf"

    stm_recs = sorted(
        [r for r in records if r["model"] == "stm"],
        key=lambda r: r["n_docs"],
    )

    sizes = [r["n_docs"] for r in stm_recs]
    topica_rss = [r["topica_rss_mb"] for r in stm_recs]
    ref_rss = [r.get("ref_rss_mb") for r in stm_recs]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, topica_rss, marker="o", label="topica STM", color="steelblue")
    if any(v is not None for v in ref_rss):
        valid = [(s, v) for s, v in zip(sizes, ref_rss) if v is not None]
        ax.plot(
            [s for s, _ in valid],
            [v for _, v in valid],
            marker="s", linestyle="--", label="R stm", color="firebrick",
        )

    ax.set_xlabel("Corpus size (docs)")
    ax.set_ylabel("Peak RSS (MB)")
    ax.set_title("Memory: topica STM vs R stm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Memory figure written to {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="topica unified benchmark")
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render outputs from an existing bench_results.json without running fits",
    )
    args = parser.parse_args()

    if args.render:
        if not RESULTS_JSON.exists():
            print(f"No results file found at {RESULTS_JSON}; run without --render first.")
            sys.exit(1)
        records = json.load(open(RESULTS_JSON))
    else:
        records = run_sweep()

    render_website(records)
    render_thread_figure(records)
    render_memory_figure(records)
    print("\nAll outputs written.", flush=True)


if __name__ == "__main__":
    main()
