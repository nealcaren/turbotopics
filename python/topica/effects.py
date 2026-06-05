"""Prevalence analysis over a model's document-topic proportions ``theta``.

These tools work on the ``theta`` of any topica model:

- :func:`estimate_effect` regresses each topic's prevalence on document
  covariates (OLS, or the method of composition when given posterior draws).
- :func:`by_strata` reports mean prevalence within each level of a covariate.
- :func:`top_topics` lists each document's most prevalent topics.

Uncertainty propagation needs *draws* of ``theta``. STM and CTM have a
logistic-normal posterior, so :func:`posterior_theta_samples` draws from it. A
Gibbs model (LDA, keyATM, SeededLDA, ...) has no such posterior, so
:func:`dirichlet_theta_samples` draws ``theta`` from each document's Dirichlet
conditional given its proportions and length.
"""

from __future__ import annotations

import numpy as np

from .stm import estimate_effect, posterior_theta_samples
from .keyatm import by_strata, top_topics

__all__ = [
    "estimate_effect",
    "posterior_theta_samples",
    "dirichlet_theta_samples",
    "by_strata",
    "top_topics",
]


def dirichlet_theta_samples(doc_topic, doc_lengths, *, nsims=25, seed=0, prior=0.0):
    """Draw `nsims` samples of the document-topic matrix θ for a Gibbs model.

    A collapsed-Gibbs model's `doc_topic` is the posterior mean of each
    document's θ given its token-topic assignments, where
    ``θ_d ~ Dirichlet(α + n_d)`` and ``(α + n_d) = doc_topic_d · (N_d + Σα)``.
    With the document length `N_d` we recover that Dirichlet and sample it, so the
    draws carry each document's within-document estimation uncertainty. Feed the
    result to :func:`estimate_effect` for method-of-composition standard errors on
    a model that has no logistic-normal posterior of its own.

    Parameters
    ----------
    doc_topic : array (num_docs, num_topics)
        The fitted θ (rows sum to one), e.g. ``model.doc_topic``.
    doc_lengths : array (num_docs,)
        Tokens per document (``[len(d) for d in docs]``). Longer documents give
        tighter draws, exactly as they pin θ more firmly in the model.
    nsims : int
        Number of θ draws.
    seed : int
        RNG seed.
    prior : float
        Extra concentration added to every document (a flat pseudo-count `Σα`
        spread over the topics). 0 uses the token counts alone.

    Returns
    -------
    array (nsims, num_docs, num_topics)
        Matches :func:`posterior_theta_samples`, ready for
        :func:`estimate_effect`.
    """
    theta = np.asarray(doc_topic, dtype=np.float64)
    lengths = np.asarray(doc_lengths, dtype=np.float64)
    if theta.ndim != 2:
        raise ValueError("doc_topic must be a 2-D (num_docs, num_topics) array")
    if lengths.shape != (theta.shape[0],):
        raise ValueError("doc_lengths must have one entry per document")
    if prior < 0:
        raise ValueError("prior must be >= 0")

    # Concentration α + n_d for each document; clip tiny values so the gamma
    # draws are well defined for topics a document never uses.
    conc = theta * (lengths[:, None] + prior) + prior / theta.shape[1]
    conc = np.clip(conc, 1e-6, None)

    rng = np.random.default_rng(seed)
    # Dirichlet via independent gammas, normalized — vectorized over draws/docs.
    g = rng.standard_gamma(conc[None, :, :], size=(nsims,) + conc.shape)
    return g / g.sum(axis=2, keepdims=True)
