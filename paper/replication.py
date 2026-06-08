"""Reproduce every quantitative artifact in the topica paper.

Run from the repo root:

    VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python paper/replication.py
    # ... --quick   to skip the slow cross-implementation checks (no R/Java needed)

Three parts:

  1. The *spanning comparison* (Worked example / breadth). Several models from
     different families -- count-based (LDA, CTM, STM) and embedding-based
     (BERTopic on deterministic LSA vectors) -- are fit on the SAME corpus and then
     scored by the SAME diagnostics in ONE loop: the framework's central claim made
     concrete. Reproduces with only numpy/scikit-learn (no sentence-transformers).

  2. The STM covariate-effect forest plot (`fig_poliblog_effect.pdf`) with honest
     method-of-composition intervals.

  3. *Cross-implementation validation* -- the part that keeps the validation claims
     honest. We do not just assert that topica reproduces MALLET, R `stm`, and R
     `keyATM`; we RUN those reference packages on the same inputs and report the
     agreement, by driving the scripts in `parity/` and `benchmarks/`. Each is
     skipped (clearly, not silently) if its toolchain (the `mallet` CLI, or
     `Rscript` with `stm`/`keyATM`) is not installed -- so the script runs anywhere,
     and reports exactly which reference comparisons actually executed.
"""

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

import numpy as np

import topica
from topica import Corpus, stm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
K = 15
SEED = 1


def load_poliblog():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "examples", "poliblog.csv"))))
    docs = [r["text"].split() for r in rows]
    conservative = np.array(
        [r["rating"] == "Conservative" for r in rows], float
    ).reshape(-1, 1)
    return docs, conservative


def lsa_embeddings(docs, dim=50, seed=0):
    """Deterministic document vectors via TF-IDF + truncated SVD (LSA).

    Stands in for any sentence embedder so the embedding-based model is fully
    reproducible. topica takes the vectors; where they come from is the user's call.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    tfidf = TfidfVectorizer(tokenizer=lambda d: d, preprocessor=lambda d: d,
                            token_pattern=None).fit_transform(docs)
    svd = TruncatedSVD(n_components=dim, random_state=seed).fit_transform(tfidf)
    return normalize(svd)  # L2-normalize so HDBSCAN sees comparable distances


def score(model, texts):
    """Every model is scored by the SAME diagnostics -- this is the whole point."""
    coh = float(np.mean(topica.coherence(model, texts, coherence_type="c_v")))
    exc = float(np.mean(topica.exclusivity(model)))
    div = float(topica.topic_diversity(model))
    return coh, exc, div


def spanning_comparison(docs, conservative):
    print("=" * 72)
    print("Spanning comparison: different model families, one scoring loop")
    print("=" * 72)

    fits = []

    def run(name, build, fit):
        t0 = time.perf_counter()
        m = build()
        fit(m)
        dt = time.perf_counter() - t0
        coh, exc, div = score(m, docs)
        fits.append((name, m.num_topics, coh, exc, div, dt))
        print(f"  fit {name:10s} K={m.num_topics:<3d} {dt:5.1f}s")
        return m

    run("LDA", lambda: topica.LDA(num_topics=K, seed=SEED),
        lambda m: m.fit(docs, iterations=500))
    run("CTM", lambda: topica.CTM(num_topics=K, seed=SEED),
        lambda m: m.fit(docs, em_iters=25))
    run("STM", lambda: topica.STM(num_topics=K, seed=SEED),
        lambda m: m.fit(docs, conservative, prevalence_names=["conservative"], em_iters=25))
    try:
        emb = lsa_embeddings(docs)
        run("BERTopic", lambda: topica.BERTopic(reducer="pca", n_components=5,
                                                min_cluster_size=10, seed=42),
            lambda m: m.fit_transform(docs, emb))
    except Exception as e:  # embedding extra not available: report and continue
        print(f"  (embedding model skipped: {type(e).__name__}: {e})")

    print("\n  model       K   coherence(c_v)  exclusivity  diversity")
    print("  " + "-" * 56)
    for name, k, coh, exc, div, _ in fits:
        print(f"  {name:10s} {k:<3d}    {coh:8.3f}      {exc:7.3f}    {div:6.3f}")
    return fits


def model_tour(docs, conservative):
    """One construct->fit->read pattern, four extra capabilities.

    Beyond the LDA/CTM/STM/BERTopic of the spanning comparison, four more
    models -- each adding a distinct capability -- are driven through the SAME
    three steps and read through the SAME top_words contract: a model that
    infers its own K, one that conditions topics on document metadata, a
    short-text model, and one steered by keywords.
    """
    print("\n" + "=" * 72)
    print("Model tour: one pattern (construct -> fit -> top_words), many models")
    print("=" * 72)

    hdp = topica.HDP(eta=0.3, seed=1)                       # infers its own K
    hdp.fit(docs, iters=300)

    dmr = topica.DMR(num_topics=15, seed=1)                 # topics conditioned on metadata
    dmr.fit(docs, conservative, feature_names=["conservative"])

    pt = topica.PT(num_topics=15, num_pseudo=100, seed=1)   # short-text via pseudo-documents
    pt.fit(docs, iters=500)

    seeds = {"economy": ["economi", "tax", "job", "market"],
             "foreign": ["iraq", "war", "troop", "iran"]}
    keyatm = topica.KeyATM(seeds, num_topics=15, seed=1)    # steered by keywords
    keyatm.fit(docs, iters=500)

    for name, m in [("HDP", hdp), ("DMR", dmr), ("PT", pt), ("KeyATM", keyatm)]:
        words = ", ".join(w for w, _ in m.top_words(6)[0])
        print("%-7s K=%-2d | %s" % (name, m.num_topics, words))


def capability_demos(docs, conservative):
    """Four demos that turn the paper's asserted claims into shown output.

    Each block is reproduced verbatim by a CodeChunk in the paper:
      (1) determinism to the bit (Section 'Design'),
      (2) search_k for choosing K honestly (Section 'workflow'),
      (3) honest uncertainty -- a posterior-bearing model answers, an
          embedding clustering refuses (Section 'workflow'),
      (4) cross-model alignment -- do two model families find the same
          themes? (the question the introduction poses).
    """
    print("\n" + "=" * 72)
    print("Capability demos: claims made concrete")
    print("=" * 72)

    # (1) Determinism to the bit: the same seed gives the same fit, exactly.
    print("\n[determinism] same seed, two fits:")
    a = topica.STM(num_topics=10, seed=1)
    a.fit(docs, conservative, prevalence_names=["conservative"], em_iters=15)
    b = topica.STM(num_topics=10, seed=1)
    b.fit(docs, conservative, prevalence_names=["conservative"], em_iters=15)
    print("topic_word identical:", np.array_equal(a.topic_word, b.topic_word))
    print("doc_topic  identical:", np.array_equal(a.doc_topic, b.doc_topic))

    # (2) search_k: report the diagnostics across K; do not maximize them.
    print("\n[search_k] coherence rises as K falls -- so do not chase it:")
    grid = topica.search_k(docs, ks=[5, 10, 15, 20], model="lda", iterations=200)
    for r in grid:
        print("K=%-2d  coherence=%7.2f  exclusivity=%.3f"
              % (r["k"], r["coherence"], r["exclusivity"]))

    # (3) Honest uncertainty: STM has a posterior; a clustering of embeddings
    #     does not, and standard_errors says so rather than inventing a number.
    print("\n[honest uncertainty] same call, two models:")
    se = topica.standard_errors(a, X=conservative,
                                feature_names=["conservative"], nsims=20)
    e0 = se[0]
    print("STM      effect=%+.3f  se=%.3f  (method of composition)"
          % (e0.coef[1], e0.se[1]))
    emb = lsa_embeddings(docs)
    bert = topica.BERTopic(reducer="pca", n_components=5,
                           min_cluster_size=10, seed=42)
    bert.fit_transform(docs, emb)
    try:
        topica.standard_errors(bert, X=conservative,
                               feature_names=["conservative"], nsims=20)
    except ValueError as err:
        print(err)   # the message already names the model and the honest reason

    # (4) Cross-model alignment: the introduction asks whether a count-based
    #     model and a clustering of embeddings find the same themes. Answer it.
    print("\n[alignment] LDA topics matched to BERTopic topics:")
    lda = topica.LDA(num_topics=15, seed=1)
    lda.fit(docs, iterations=300)
    for ti, tj, sim in topica.align_topics(lda, bert):
        li = ", ".join(w for w, _ in lda.top_words(3)[ti])
        bj = ", ".join(w for w, _ in bert.top_words(3)[tj])
        print("LDA t%-2d [%s] ~ BERTopic t%d [%s]  cos=%.2f"
              % (ti, li, tj, bj, sim))


def planted_corpus(seed=0, num_docs=250, doc_len=12):
    """Five topics with disjoint vocabularies; each document is drawn from a
    single topic, so a correct model should recover the five exactly. This is
    the known-truth complement to the reference-implementation checks: not 'do
    we match R/Java?' but 'do we recover a structure we planted ourselves?'."""
    topics = ["alpha bravo charlie delta echo foxtrot".split(),
              "lion tiger bear wolf fox otter".split(),
              "guitar piano violin drums trumpet cello".split(),
              "mercury venus earth mars jupiter saturn".split(),
              "python rust java haskell scala ocaml".split()]
    rng = np.random.default_rng(seed)
    docs = [list(rng.choice(topics[rng.integers(0, len(topics))], doc_len))
            for _ in range(num_docs)]
    return docs, [set(t) for t in topics]


def recovery_demo():
    """Fit three model families to a planted corpus and measure recovery.

    'Purity' is the probability mass a fitted topic places on its single
    best-matching planted topic (1.0 = the topic is exactly one true topic);
    'recovered' counts how many of the five planted topics are the best match
    of some fitted topic. Perfect recovery is purity 1.0 and 5/5.
    """
    print("\n" + "=" * 72)
    print("Recovery of known topics: fit a corpus whose truth we planted")
    print("=" * 72)
    docs, true_sets = planted_corpus(seed=0)

    def recovery(model, true_sets):
        idx = {w: i for i, w in enumerate(model.vocabulary)}
        phi = model.topic_word
        mass = [[phi[k, [idx[w] for w in s if w in idx]].sum() for s in true_sets]
                for k in range(phi.shape[0])]
        purity = np.mean([max(row) for row in mass])
        recovered = len({int(np.argmax(row)) for row in mass})
        return phi.shape[0], purity, recovered

    # Three fixed-K inference families, then HDP, which is given no K at all.
    fits = [("LDA", topica.LDA(num_topics=5, seed=1), dict(iterations=400)),
            ("CTM", topica.CTM(num_topics=5, seed=1), dict(em_iters=40)),
            ("GSDMM", topica.GSDMM(num_topics=5, seed=1), dict(iters=50)),
            ("HDP", topica.HDP(eta=0.3, seed=1), dict(iters=300))]
    print("\nmodel  K   purity  recovered")
    for name, m, kw in fits:
        m.fit(docs, **kw)
        K, p, r = recovery(m, true_sets)
        print("%-5s  %d   %.3f   %d/5" % (name, K, p, r))


def effect_figure(docs, conservative):
    print("\n" + "=" * 72)
    print("STM covariate-effect figure (fig_poliblog_effect.pdf)")
    print("=" * 72)
    import topica.viz as viz

    corpus = Corpus.from_documents(docs, min_doc_freq=10, max_doc_fraction=0.5, rm_top=20)
    print(f"  {corpus.num_docs} docs, vocab {corpus.num_words}")

    model = topica.STM(num_topics=K, seed=SEED)
    model.fit(docs, conservative, prevalence_names=["conservative"], em_iters=25)

    labeled = stm.label_topics(model.topic_word, model.vocabulary, n=3)
    topica.set_topic_labels(
        model, {t: ", ".join(w for w, _ in labeled[t]["frex"]) for t in range(K)}
    )
    panel = viz.effect_plot(model, corpus, X=conservative,
                            feature_names=["conservative"], nsims=100)
    out = os.path.join(HERE, "fig_poliblog_effect.pdf")
    panel.to_png(out)
    print(f"  wrote {out}")

    # The one-call report: a composite figure off the shared contract that shows
    # the reporting layer (topics-by-prevalence, coherence/exclusivity, the
    # honest correlation, and prevalence by group).
    report = topica.plot_report(model, texts=docs, groups=conservative.ravel(),
                                n=8, figsize=(11, 7),
                                title="topica report: poliblog STM (K=15)")
    report_out = os.path.join(HERE, "fig_poliblog_report.pdf")
    report.savefig(report_out, bbox_inches="tight")
    print(f"  wrote {report_out}")

    df = panel.to_frame().sort_values("coef")
    print("\n  Per-topic effect of conservative rating on prevalence:")
    for _, r in df.iterrows():
        flag = "" if r["reliable"] else "  (unreliable)"
        print(f"    topic {int(r['topic']):2d} {r['label'][:30]:<30} "
              f"coef={r['coef']:+.4f}  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]{flag}")


def cross_implementation():
    """Run the reference packages themselves and report the agreement.

    Each entry names the script, the executables it needs, and a one-line note.
    Missing toolchains are reported, never hidden -- the paper's validation claims
    only stand for the rows that actually ran here.
    """
    print("\n" + "=" * 72)
    print("Cross-implementation validation: topica vs. the reference packages")
    print("=" * 72)

    checks = [
        ("LDA / LabeledLDA / DMR vs. Java MALLET", "parity/mallet_parity.py",
         ["mallet", "java"]),
        ("STM vs. R stm (poliblog vignette)", "parity/stm_poliblog_compare.py",
         ["Rscript"]),
        ("KeyATM vs. R keyATM", "parity/keyatm_r_compare.py", ["Rscript"]),
        ("Fit-time speedups vs. R stm / keyATM", "benchmarks/speed_vs_r.py",
         ["Rscript"]),
    ]

    ran, skipped = [], []
    for label, script, exes in checks:
        missing = [e for e in exes if shutil.which(e) is None]
        print(f"\n--- {label}\n    ({script})")
        if missing:
            print(f"    SKIPPED: not installed: {', '.join(missing)}")
            skipped.append(label)
            continue
        proc = subprocess.run([sys.executable, script], cwd=ROOT_DIR,
                              capture_output=True, text=True)
        out = (proc.stdout or "").strip().splitlines()
        for line in out[-12:]:            # the scripts print a short summary; echo its tail
            print("    " + line)
        if proc.returncode != 0:
            print(f"    (exit {proc.returncode}); stderr tail:")
            for line in (proc.stderr or "").strip().splitlines()[-4:]:
                print("    ! " + line)
        ran.append(label)

    print("\n" + "-" * 72)
    print(f"Reference comparisons run here: {len(ran)}/{len(checks)}.")
    if skipped:
        print("Skipped (install the toolchain to reproduce): " + "; ".join(skipped))


def main():
    ap = argparse.ArgumentParser(description="Reproduce the topica paper's artifacts.")
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow cross-implementation checks (no R/Java needed)")
    args = ap.parse_args()

    docs, conservative = load_poliblog()
    spanning_comparison(docs, conservative)
    model_tour(docs, conservative)
    capability_demos(docs, conservative)
    recovery_demo()
    effect_figure(docs, conservative)
    if not args.quick:
        cross_implementation()


if __name__ == "__main__":
    main()
