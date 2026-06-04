# keyATM toolkit

keyATM-specific workflow helpers live in `topica.keyatm`. The general post-hoc
diagnostics (top words, representative documents, coherence, pyLDAvis, covariate
effects, …) are model-agnostic and work on a fitted `KeyATM` directly: see the
[Diagnostics](diagnostics.md) and [STM toolkit](stm.md) pages.

The fitted model also exposes the convergence trace keyATM's `plot_modelfit`
reports, as `KeyATM.log_likelihood_history` — a list of `(iteration, per-token
log-likelihood)` pairs (perplexity is `exp(-log_likelihood)`).

::: topica.keyatm.top_topics

::: topica.keyatm.by_strata

::: topica.keyatm.visualize_keywords

::: topica.keyatm.refine_keywords
