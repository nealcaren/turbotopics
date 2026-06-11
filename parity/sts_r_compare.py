"""Cross-implementation validation: topica's STS vs the authors' reference R
implementation of the Structural Topic and Sentiment-Discourse model (Chen &
Mankad 2024, *Management Science*).

Both are the same model — Laplace variational EM over the 2K-1 prevalence +
sentiment-discourse latent, with the topic-word distribution modulated by a
continuous per-document sentiment. They are independent implementations, so
validation is statistical: fit both on the SAME tokenized corpus and ask whether
they recover the same topics.

This uses the authors' published immigration fit (Application 1, K=3) from their
replication package: R regenerates the exact corpus (``textProcessor`` +
``prepDocuments`` on the ``gadarian`` data bundled with ``stm``) and reads the
fitted ``immigration_results.RDS``, exporting its neutral-sentiment topic-word
matrix β = softmax(m + κ^(t)). topica then fits STS on that same corpus and we
align the two β matrices. Both engines use the same deterministic anchor-word
initialization, so they should land on closely matching topics despite different
κ estimators (the reference uses glmnet lasso; topica uses a ridge-penalized
Newton Poisson fit).

Point the script at the replication package via ``STS_REPL_DIR`` (default
``~/Downloads/mnsc.2022.00261``). Skips (exit 0) if Rscript, the ``stm`` package,
or the replication results are unavailable. Run directly:

    python parity/sts_r_compare.py
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from stm_r_compare import _best_alignment_cosine, _read_r_beta

REPL_DIR = os.environ.get(
    "STS_REPL_DIR", os.path.expanduser("~/Downloads/mnsc.2022.00261")
)


def available() -> tuple[bool, str]:
    """Whether Rscript, ``stm``, and the replication results are all present."""
    if shutil.which("Rscript") is None:
        return False, "Rscript not on PATH"
    rds = os.path.join(REPL_DIR, "Results", "immigration_results.RDS")
    if not os.path.isfile(rds):
        return False, f"replication results not found (set STS_REPL_DIR); looked in {REPL_DIR}"
    try:
        out = subprocess.run(
            ["Rscript", "-e", 'cat(requireNamespace("stm", quietly=TRUE))'],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.SubprocessError, OSError):
        return False, "could not run Rscript"
    if not out.stdout.strip().endswith("TRUE"):
        return False, "the 'stm' R package is not installed"
    return True, ""


# R driver: regenerate the gadarian corpus exactly as the paper does, read the
# published STS fit, and export (a) the corpus as token lists, (b) the metadata,
# and (c) the reference neutral-sentiment topic-word matrix.
_R_DRIVER = r"""
suppressMessages(library(stm))
repl <- Sys.getenv("STS_REPL_DIR")
temp <- textProcessor(documents = gadarian$open.ended.response, metadata = gadarian, verbose = FALSE)
out  <- prepDocuments(temp$documents, temp$vocab, temp$meta, verbose = FALSE)
fit  <- readRDS(file.path(repl, "Results", "immigration_results.RDS"))

if (length(fit$vocab) != length(out$vocab) || !all(fit$vocab == out$vocab)) {
  cat("vocab-mismatch\n"); quit(status = 0)
}

# Corpus as token lists: vocab word repeated by its count, one doc per line.
toks <- sapply(out$documents, function(m) paste(rep(out$vocab[m[1, ]], m[2, ]), collapse = " "))
writeLines(toks, file.path(dir, "docs.txt"))

meta <- out$meta
write.csv(data.frame(treatment = meta$treatment, pid_rep = meta$pid_rep),
          file.path(dir, "meta.csv"), row.names = FALSE)

# Reference topic-word at the mean sentiment (print.topWords' "Avg alpha" view):
# softmax(m_v + κ^(t)_{·,k} + κ^(s)_{·,k}·mean_d α^(s)_{d,k}). STS can park the
# topic signal in κ^(s) when the sentiment seed correlates with content, so the
# representative topic is at the mean α^(s), not at zero.
K <- ncol(fit$kappa$kappa_t)
as_mean <- apply(fit$alpha[, 1:K + K - 1, drop = FALSE], 2, mean)
beta <- t(sapply(1:K, function(k) {
  e <- exp(fit$mv + fit$kappa$kappa_t[, k] + fit$kappa$kappa_s[, k] * as_mean[k]); e / sum(e)
}))
colnames(beta) <- out$vocab
write.csv(beta, file.path(dir, "r_sts_beta.csv"), row.names = FALSE)
writeLines(out$vocab, file.path(dir, "r_vocab.txt"))
cat("ok\n")
"""


def run(verbose: bool = True) -> dict:
    """Fit topica STS on the published immigration corpus and compare its
    topic-word matrix to the reference. Returns alignment metrics."""
    ok, why = available()
    if not ok:
        raise RuntimeError(why)

    from topica import STS

    with tempfile.TemporaryDirectory() as d:
        env = {**os.environ, "STS_REPL_DIR": REPL_DIR}
        script = f'dir <- "{d}"\n' + _R_DRIVER
        proc = subprocess.run(
            ["Rscript", "-e", script], capture_output=True, text=True, timeout=600, env=env
        )
        if "vocab-mismatch" in proc.stdout:
            raise RuntimeError(
                "the regenerated gadarian vocabulary does not match the fitted RDS "
                "(likely a different stm/SnowballC version); cannot compare on a shared vocab"
            )
        if "ok" not in proc.stdout:
            raise RuntimeError(f"R driver failed:\n{proc.stdout}\n{proc.stderr}")

        docs = [line.split() for line in open(os.path.join(d, "docs.txt")) if line.strip()]
        with open(os.path.join(d, "meta.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        treatment = np.array([float(r["treatment"]) for r in rows])
        pid = np.array([float(r["pid_rep"]) for r in rows])
        r_vocab = open(os.path.join(d, "r_vocab.txt")).read().split()
        r_beta = _read_r_beta(os.path.join(d, "r_sts_beta.csv"), r_vocab)

    # topica STS on the same corpus, same K, same design (treatment + pid_rep +
    # their interaction for prevalence; treatment as the sentiment seed).
    X = np.column_stack([treatment, pid, treatment * pid])
    model = STS(num_topics=3, init="spectral")
    model.fit(
        docs, sentiment_seed=treatment.tolist(), prevalence=X,
        prevalence_names=["treatment", "pid_rep", "treatment:pid_rep"], iters=40,
    )
    tt_vocab = list(model.vocabulary)
    tt_beta_raw = np.asarray(model.topic_word)

    # Align topica β onto R's vocab order for a like-for-like comparison.
    tt_idx = {w: i for i, w in enumerate(tt_vocab)}
    tt_beta = np.zeros((tt_beta_raw.shape[0], len(r_vocab)))
    for j, w in enumerate(r_vocab):
        if w in tt_idx:
            tt_beta[:, j] = tt_beta_raw[:, tt_idx[w]]

    cosine = _best_alignment_cosine(r_beta, tt_beta)

    # Top-word agreement of the aligned topics — the parameterization-robust
    # signal a researcher reads. The reference's lasso κ makes its β far more
    # peaked than topica's ridge κ, which depresses full-distribution cosine even
    # when the same words top each topic, so report both.
    jaccard = _aligned_top_word_jaccard(r_beta, tt_beta, topn=10)

    # Chance baseline: align topica to a vocabulary-permuted reference. Real
    # correspondence must clearly beat this; an absolute cosine threshold would be
    # arbitrary given the different κ regularizers.
    rng = np.random.default_rng(0)
    chance = float(np.mean([
        _best_alignment_cosine(r_beta[:, rng.permutation(r_beta.shape[1])], tt_beta)
        for _ in range(5)
    ]))

    metrics = {
        "topic_cosine": cosine,
        "top_word_jaccard": jaccard,
        "chance_cosine": chance,
        "vocab_size": len(r_vocab),
        "n_docs": len(docs),
        "num_topics": 3,
    }
    if verbose:
        print(f"docs={len(docs)}  vocab={len(r_vocab)}  K=3")
        print(f"topica-vs-reference best-alignment cosine: {cosine:.3f}  (chance {chance:.3f})")
        print(f"aligned top-10 word Jaccard:               {jaccard:.3f}")
    return metrics


def _aligned_top_word_jaccard(a, b, topn=10):
    """Mean top-`topn` Jaccard overlap of the best one-to-one topic alignment
    between two K×V topic-word matrices (aligned by cosine)."""
    from topica.validation import _hungarian

    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    cost = 1.0 - an @ bn.T
    scores = []
    for i, j in _hungarian(cost):
        sa = set(np.argsort(a[i])[::-1][:topn])
        sb = set(np.argsort(b[j])[::-1][:topn])
        scores.append(len(sa & sb) / len(sa | sb))
    return float(np.mean(scores))


def main() -> int:
    ok, why = available()
    if not ok:
        print(f"skipping STS R parity: {why}")
        return 0
    m = run()
    # The two engines recover the same topics, but with a different κ regularizer
    # (reference lasso vs topica ridge), so validation is topic-level, not
    # bit-level: the alignment must clearly beat a vocabulary-permuted chance
    # baseline, and the aligned topics must share top words well above chance.
    assert m["topic_cosine"] > m["chance_cosine"] + 0.15, (
        f"topic alignment {m['topic_cosine']:.3f} not clearly above chance "
        f"{m['chance_cosine']:.3f}"
    )
    assert m["top_word_jaccard"] > 0.15, (
        f"aligned top-word overlap too low: {m['top_word_jaccard']:.3f}"
    )
    print("OK: topica STS recovers the reference immigration topics.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
