"""Generate the worked-example figure for the topica JSS paper.

Fits the structural topic model on the political-blog corpus (the canonical stm
example) and saves the per-topic covariate-effect forest plot. Run from the repo
root:

    VIRTUAL_ENV="$PWD/.venv-dev" .venv-dev/bin/python paper/make_figures.py

Writes paper/fig_poliblog_effect.pdf and prints the effects table so the numbers
in topica.tex Section 8 can be filled in from a reproducible source.
"""

import csv
import os

import numpy as np

import topica
from topica import Corpus, stm
import topica.viz as viz

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
K = 15
SEED = 1


def main():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "examples", "poliblog.csv"))))
    docs = [r["text"].split() for r in rows]
    corpus = Corpus.from_documents(docs, min_doc_freq=10, max_doc_fraction=0.5, rm_top=20)
    print(f"{corpus.num_docs} docs, vocab {corpus.num_words}")

    conservative = np.array(
        [r["rating"] == "Conservative" for r in rows], float
    ).reshape(-1, 1)

    model = topica.STM(num_topics=K, seed=SEED)
    model.fit(docs, conservative, prevalence_names=["conservative"], em_iters=25)

    # Label topics by their top FREX words (frequency-and-exclusivity), the stm
    # convention, so the figure reads substantively rather than by topic number.
    labeled = stm.label_topics(model.topic_word, model.vocabulary, n=3)
    topica.set_topic_labels(
        model, {t: ", ".join(w for w, _ in labeled[t]["frex"]) for t in range(K)}
    )

    # The results figure: prevalence difference by blog ideology, with honest
    # method-of-composition intervals.
    panel = viz.effect_plot(
        model, corpus, X=conservative, feature_names=["conservative"], nsims=100,
    )
    out = os.path.join(HERE, "fig_poliblog_effect.pdf")
    panel.to_png(out)
    print(f"wrote {out}")

    # Print the table so the prose can cite specific recovered effects.
    df = panel.to_frame().sort_values("coef")
    print("\nPer-topic effect of conservative rating on prevalence:")
    for _, r in df.iterrows():
        flag = "" if r["reliable"] else "  (unreliable)"
        print(f"  topic {int(r['topic']):2d} {r['label'][:32]:<32} "
              f"coef={r['coef']:+.4f}  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]{flag}")


if __name__ == "__main__":
    main()
