# paper/

A JSS-style software paper introducing topica, targeting arXiv (cs.CL / stat.CO)
as the preprint and the *Journal of Statistical Software* as the eventual home.

## Files

- `topica.tex` — the manuscript (JSS `article` class).
- `topica.bib` — BibTeX, grounded in `docs/citing.md`.

## Building

The paper uses the JSS document class. Download `jss.cls`, `jss.bst`, and
`jsslogo.jpg` from <https://www.jstatsoft.org/style> and place them in this
directory, then:

```bash
cd paper
pdflatex topica
bibtex topica
pdflatex topica
pdflatex topica
```

If you do not have `jss.cls` to hand, swap the first line of `topica.tex` for
`\documentclass{article}` plus `\usepackage{hyperref}` and define the JSS macros
(`\pkg`, `\proglang`, `\code`, `\fct`, `\Abstract`, `\Keywords`, `\Address`,
`CodeChunk`/`CodeInput`) as no-ops or simple wrappers for a quick proof build.

## Status

Drafted in full: abstract, introduction, related software (Table 1), design and
architecture, the model family (Table 2), the analysis workflow, validation, and
performance (Table 3). These sections use real numbers from `docs/benchmarks.md`
and `docs/replications/`.

Open TODOs before submission:

1. **Worked-example figure.** `topica.tex` Section 8 has the runnable code and a
   marked `TODO(figure)`. Run `topica.viz.effect_plot` on the real poliblog
   corpus, save `paper/fig_poliblog_effect.pdf`, and insert it with the caption
   noted in the source. Replace the qualitative "recovers the same pattern" with
   the specific recovered effects once generated.
2. **Comparison-table audit (Table 1).** The feature matrix is drawn from current
   knowledge of each tool; re-verify each cell against the tools' current releases
   before submission, especially the "partial" marks.
3. **Polish pass** with the writing-editor levels (sentence/word) once the figure
   lands and the section order is final.

## Conventions

Prose follows the repo register: no em dashes, agent-led "we", concrete over
hedged, no LLM filler. Quantitative claims trace to a script in `benchmarks/` or
`parity/`; do not introduce a number that cannot be reproduced.
