"""Deterministic helper parity against R ``stm``.

The fitted STM model itself is non-convex, so fit parity is statistical. These
post-hoc helpers are deterministic given fixed matrices, so they should match R
``stm`` much more tightly.
"""

from __future__ import annotations

import csv
import shutil
import subprocess

import numpy as np
import pytest

from topica import stm


pytestmark = pytest.mark.parity


PHI = np.array(
    [
        [0.07456885, 0.01461855, 0.03145210, 0.03846849, 0.01146723, 0.29374495, 0.53567985],
        [0.03200643, 0.10301953, 0.17102731, 0.18408400, 0.07549548, 0.33745050, 0.09691676],
        [0.18688626, 0.19903125, 0.29301674, 0.03597393, 0.12396142, 0.06811504, 0.09301537],
    ],
    dtype=float,
)
VOCAB = ["a", "b", "c", "d", "e", "f", "g"]
THETA = np.array(
    [
        [0.70, 0.20, 0.10],
        [0.62, 0.28, 0.10],
        [0.15, 0.70, 0.15],
        [0.10, 0.63, 0.27],
        [0.20, 0.16, 0.64],
        [0.28, 0.12, 0.60],
    ],
    dtype=float,
)
X = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])


def _r_stm_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    proc = subprocess.run(
        ["Rscript", "-e", 'cat(requireNamespace("stm", quietly=TRUE))'],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return proc.stdout.strip().endswith("TRUE")


def _write_matrix(path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, delimiter=",", fmt="%.17g")


def _read_string_matrix(path) -> list[list[str]]:
    with open(path, newline="") as f:
        return [row for row in csv.reader(f)]


def _read_numeric_matrix(path) -> np.ndarray:
    with open(path, newline="") as f:
        return np.array([[float(x) for x in row] for row in csv.reader(f)], dtype=float)


@pytest.fixture(scope="module")
def r_stm_helper_outputs(tmp_path_factory):
    if not _r_stm_available():
        pytest.skip("Rscript with the 'stm' package not available")

    d = tmp_path_factory.mktemp("r-stm-helper")
    _write_matrix(d / "phi.csv", PHI)
    _write_matrix(d / "theta.csv", THETA)
    _write_matrix(d / "x.csv", X[:, None])
    (d / "vocab.txt").write_text("\n".join(VOCAB) + "\n")

    script = f"""
    suppressMessages(library(stm))
    dir <- "{d}"
    phi <- as.matrix(read.csv(file.path(dir, "phi.csv"), header=FALSE))
    theta <- as.matrix(read.csv(file.path(dir, "theta.csv"), header=FALSE))
    x <- as.numeric(read.csv(file.path(dir, "x.csv"), header=FALSE)[[1]])
    vocab <- readLines(file.path(dir, "vocab.txt"))
    logbeta <- log(phi)
    colnames(logbeta) <- vocab

    frex_idx <- calcfrex(logbeta, w=0.5)
    write.table(frex_idx, file.path(dir, "calcfrex.csv"), sep=",",
                row.names=FALSE, col.names=FALSE)

    obj <- list(beta=list(logbeta=list(logbeta)), settings=list(dim=list(K=nrow(phi))),
                vocab=vocab, theta=theta)
    class(obj) <- "STM"
    labels <- labelTopics(obj, n=4, frexweight=0.5)
    write.table(labels$prob, file.path(dir, "label_prob.csv"), sep=",",
                row.names=FALSE, col.names=FALSE)
    write.table(labels$frex, file.path(dir, "label_frex.csv"), sep=",",
                row.names=FALSE, col.names=FALSE)
    write.table(labels$score, file.path(dir, "label_score.csv"), sep=",",
                row.names=FALSE, col.names=FALSE)

    tc <- topicCorr(obj, method="simple", cutoff=0.05, verbose=FALSE)
    write.table(tc$cor, file.path(dir, "topic_cor.csv"), sep=",",
                row.names=FALSE, col.names=FALSE)
    write.table(tc$posadj, file.path(dir, "topic_adj.csv"), sep=",",
                row.names=FALSE, col.names=FALSE)

    meta <- data.frame(x=x)
    eff <- estimateEffect(1:ncol(theta) ~ x, obj, metadata=meta, uncertainty="None")
    coef <- do.call(rbind, lapply(eff$parameters, function(p) p[[1]]$est))
    se <- do.call(rbind, lapply(eff$parameters, function(p) sqrt(diag(p[[1]]$vcov))))
    write.table(coef, file.path(dir, "effect_coef.csv"), sep=",",
                row.names=FALSE, col.names=FALSE)
    write.table(se, file.path(dir, "effect_se.csv"), sep=",",
                row.names=FALSE, col.names=FALSE)
    """
    proc = subprocess.run(["Rscript", "-e", script], capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        pytest.skip(f"R stm helper script failed:\n{proc.stderr[-2000:]}")

    return {
        "calcfrex": _read_numeric_matrix(d / "calcfrex.csv").astype(int),
        "label_prob": _read_string_matrix(d / "label_prob.csv"),
        "label_frex": _read_string_matrix(d / "label_frex.csv"),
        "label_score": _read_string_matrix(d / "label_score.csv"),
        "topic_cor": _read_numeric_matrix(d / "topic_cor.csv"),
        "topic_adj": _read_numeric_matrix(d / "topic_adj.csv").astype(int),
        "effect_coef": _read_numeric_matrix(d / "effect_coef.csv"),
        "effect_se": _read_numeric_matrix(d / "effect_se.csv"),
    }


def test_calcfrex_selected_words_match_r_stm(r_stm_helper_outputs) -> None:
    got = stm.frex(PHI, VOCAB, w=0.5, n=4)
    expected_idx = r_stm_helper_outputs["calcfrex"] - 1
    for topic in range(PHI.shape[0]):
        expected_words = [VOCAB[i] for i in expected_idx[:4, topic]]
        got_words = [word for word, _ in got[topic]]
        # FREX uses ECDF ranks, so ties are common. R and topica agree on the
        # selected tied words; their tie ordering is not a substantive contract.
        assert set(got_words) == set(expected_words)


def test_label_topics_prob_frex_score_words_match_r_stm(r_stm_helper_outputs) -> None:
    got = stm.label_topics(PHI, VOCAB, n=4)
    for topic, row in enumerate(r_stm_helper_outputs["label_prob"]):
        assert [word for word, _ in got[topic]["prob"]] == row
    # FREX can have exact ties under stm's ECDF scoring. Compare the selected
    # word set rather than tie order.
    for topic, row in enumerate(r_stm_helper_outputs["label_frex"]):
        assert {word for word, _ in got[topic]["frex"]} == set(row)
    for topic, row in enumerate(r_stm_helper_outputs["label_score"]):
        assert [word for word, _ in got[topic]["score"]] == row


def test_topic_correlation_simple_method_matches_r_stm(r_stm_helper_outputs) -> None:
    got = stm.topic_correlation(THETA, threshold=0.05)
    np.testing.assert_allclose(got.cor, r_stm_helper_outputs["topic_cor"], atol=1e-8)
    # R's posadj keeps a 1 on the diagonal; topica's public adjacency zeroes it.
    expected_adj = r_stm_helper_outputs["topic_adj"].copy()
    np.fill_diagonal(expected_adj, 0)
    np.testing.assert_array_equal(got.adjacency, expected_adj)


def test_estimate_effect_identity_none_matches_r_stm(r_stm_helper_outputs) -> None:
    got = stm.estimate_effect(THETA, X, feature_names=["x"])
    got_coef = np.vstack([effect.coef for effect in got])
    got_se = np.vstack([effect.se for effect in got])
    np.testing.assert_allclose(got_coef, r_stm_helper_outputs["effect_coef"], atol=1e-8)
    np.testing.assert_allclose(got_se, r_stm_helper_outputs["effect_se"], atol=1e-8)
