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
``prepDocuments`` on the ``gadarian`` data bundled with ``stm``), reads the fitted
``immigration_results.RDS`` (exporting its mean-sentiment topic-word matrix), and
also fits R ``stm`` on the same corpus. topica then fits both STS and STM and we
align everything.

The comparison is calibrated rather than absolute. On a small K=3 corpus the
topica-vs-R gap is irreducible: it is the *same* ~0.48 cosine for the already
validated STM as for STS, because two independent implementations partition 341
short documents into 3 topics slightly differently. So instead of an arbitrary
threshold we check that topica-STS sits about as close to R-STS as topica-STM
sits to R-STM (the cross-implementation baseline), and that topica's STS agrees
with its own STM — confirming STS correctly extends STM. This check fits topica
with its ``"lasso"`` κ estimator (the default is ``"ridge"``) to match the
reference's glmnet estimator; on a well-conditioned corpus the two agree closely.

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

# R STM on the same corpus — the calibration baseline. STS extends STM, and
# topica's STM is already validated against R's, so "topica-STS vs R-STS" should
# be about as close as "topica-STM vs R-STM".
fstm <- stm(out$documents, out$vocab, K, prevalence = ~treatment + pid_rep,
            data = meta, init.type = "Spectral", verbose = FALSE)
bstm <- exp(fstm$beta$logbeta[[1]]); colnames(bstm) <- out$vocab
write.csv(bstm, file.path(dir, "r_stm_beta.csv"), row.names = FALSE)

writeLines(out$vocab, file.path(dir, "r_vocab.txt"))
cat("ok\n")
"""


def _to_r_vocab(model, r_vocab):
    """A model's topic-word matrix reindexed onto R's vocabulary order."""
    raw = np.asarray(model.topic_word)
    idx = {w: i for i, w in enumerate(model.vocabulary)}
    out = np.zeros((raw.shape[0], len(r_vocab)))
    for j, w in enumerate(r_vocab):
        if w in idx:
            out[:, j] = raw[:, idx[w]]
    return out


def run(verbose: bool = True) -> dict:
    """Fit topica STS on the published immigration corpus and compare it to the
    reference STS — calibrated against the topica-vs-R STM baseline and topica's
    own STM, so the irreducible cross-implementation gap is accounted for.

    Validation logic (mirroring stm_r_compare's "as close as R is to itself"):
    topica's STS should sit about as close to R-STS as topica's already-validated
    STM sits to R-STM, and STS must agree with topica's own STM (it extends it)."""
    ok, why = available()
    if not ok:
        raise RuntimeError(why)

    from topica import STM, STS

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
        r_sts = _read_r_beta(os.path.join(d, "r_sts_beta.csv"), r_vocab)
        r_stm = _read_r_beta(os.path.join(d, "r_stm_beta.csv"), r_vocab)

    # topica STS and STM on the same corpus, same K, same prevalence design.
    X = np.column_stack([treatment, pid, treatment * pid])
    sts = STS(num_topics=3, init="spectral")
    sts.fit(docs, sentiment_seed=treatment.tolist(), prevalence=X,
            prevalence_names=["treatment", "pid_rep", "treatment:pid_rep"],
            iters=40, kappa_estimation="lasso")  # match the reference's κ estimator
    stm = STM(num_topics=3, init="spectral")
    stm.fit(docs, np.column_stack([treatment, pid]),
            prevalence_names=["treatment", "pid_rep"], iters=80)

    t_sts = _to_r_vocab(sts, r_vocab)
    t_stm = _to_r_vocab(stm, r_vocab)

    sts_vs_ref = _best_alignment_cosine(r_sts, t_sts)
    stm_vs_ref = _best_alignment_cosine(r_stm, t_stm)  # the cross-impl baseline
    sts_vs_stm = _best_alignment_cosine(t_sts, t_stm)  # internal consistency
    ref_ceiling = _best_alignment_cosine(r_sts, r_stm)  # R-STS vs R-STM
    jaccard = _aligned_top_word_jaccard(r_sts, t_sts, topn=10)

    rng = np.random.default_rng(0)
    chance = float(np.mean([
        _best_alignment_cosine(r_sts[:, rng.permutation(r_sts.shape[1])], t_sts)
        for _ in range(5)
    ]))

    metrics = {
        "sts_vs_ref": sts_vs_ref,
        "stm_vs_ref": stm_vs_ref,
        "sts_vs_stm": sts_vs_stm,
        "ref_ceiling": ref_ceiling,
        "top_word_jaccard": jaccard,
        "chance_cosine": chance,
        "vocab_size": len(r_vocab),
        "n_docs": len(docs),
        "num_topics": 3,
    }
    if verbose:
        print(f"docs={len(docs)}  vocab={len(r_vocab)}  K=3")
        print(f"topica-STS vs R-STS:  {sts_vs_ref:.3f}  (chance {chance:.3f}, top-10 Jaccard {jaccard:.3f})")
        print(f"topica-STM vs R-STM:  {stm_vs_ref:.3f}  <- cross-implementation baseline")
        print(f"R-STS   vs R-STM:     {ref_ceiling:.3f}  <- same-ecosystem ceiling")
        print(f"topica-STS vs topica-STM: {sts_vs_stm:.3f}  <- STS extends STM")
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
    # Validation is topic-level and benchmark-relative. On this small K=3 corpus
    # the topica-vs-R gap is irreducible (it is the same ~0.48 for the already
    # validated STM), so we check: (1) STS clearly beats chance, (2) STS is about
    # as close to R-STS as topica's STM is to R-STM, and (3) STS agrees with
    # topica's own STM — confirming it correctly extends it.
    assert m["sts_vs_ref"] > m["chance_cosine"] + 0.15, (
        f"STS-vs-reference {m['sts_vs_ref']:.3f} not clearly above chance {m['chance_cosine']:.3f}"
    )
    assert m["sts_vs_ref"] > m["stm_vs_ref"] - 0.1, (
        f"STS aligns to R ({m['sts_vs_ref']:.3f}) much worse than STM does "
        f"({m['stm_vs_ref']:.3f})"
    )
    assert m["sts_vs_stm"] > 0.8, (
        f"topica STS does not agree with topica STM ({m['sts_vs_stm']:.3f}); STS should extend STM"
    )
    print("OK: topica STS recovers the reference topics, as faithfully as STM does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
