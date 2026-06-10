"""Estimator conformance: registry, contract definition, and check helper.

Every topica estimator must expose a uniform interface organized into three tiers.
This module encodes that contract, the principled exemptions from it, and the
known gaps (fixable drift tracked as a burn-down worklist). It also provides
``check_conformance(model_or_class) -> list[str]``, which callers and the
conformance test use to verify a model against its required tier.

The three tiers
---------------
Tier 0  (floor, every estimator):
    ``fit(data, ...)`` — with the canonical iteration kwarg ``iters`` where the
    model is iterative.
    Properties/methods: ``topic_word``, ``doc_topic``, ``vocabulary``,
    ``num_topics``, ``topic_names``, ``doc_names``, ``top_words``,
    ``coherence``, ``save``, ``load``.

Tier 1  (generative models, i.e. ``model_family != "none"``):
    ``transform(docs) -> (n, num_topics)``.

Tier 2  (family-specific):
    ``model_family == "dirichlet"``  ->  ``alpha``, ``theta_draws``,
    ``doc_lengths``.
    ``model_family == "logistic_normal"``  ->  ``eta_mean``, ``eta_cov``.

Exemptions
----------
Some requirements are principled exemptions that will never be closed, because
the model's statistics differ structurally from the requirement. These are
recorded in :data:`EXEMPT` as ``(model_name, requirement) -> reason``.

Known gaps
----------
:data:`KNOWN_GAPS` records temporary non-conformance:
``(model_name, requirement) -> phase_note``. The conformance test treats these
as expected failures so the suite is green today while this map is the
burn-down worklist. The program ends when ``KNOWN_GAPS == {}``.
"""

from __future__ import annotations

import inspect

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Each entry: (name, factory_callable, expected_model_family)
# Factories are zero/low-arg lambdas; construction follows the same patterns
# used in tests/test_standard_errors.py::_FAMILY_REGISTRY.

import topica as _topica

REGISTRY: list[tuple[str, object, str]] = [
    # collapsed-Gibbs / Dirichlet doc-topic posterior
    ("LDA",          lambda: _topica.LDA(2),                                     "dirichlet"),
    ("DMR",          lambda: _topica.DMR(2),                                     "dirichlet"),
    ("SAGE",         lambda: _topica.SAGE(2),                                    "dirichlet"),
    ("PA",           lambda: _topica.PA(num_super=2, num_sub=4),                 "dirichlet"),
    ("PT",           lambda: _topica.PT(num_topics=2, num_pseudo=10),            "dirichlet"),
    ("HDP",          lambda: _topica.HDP(),                                      "dirichlet"),
    ("LabeledLDA",   lambda: _topica.LabeledLDA(),                               "dirichlet"),
    ("SupervisedLDA",lambda: _topica.SupervisedLDA(num_topics=2),                "dirichlet"),
    ("HLDA",         lambda: _topica.HLDA(),                                     "none"),
    ("DTM",          lambda: _topica.DTM(2),                                     "none"),
    ("KeyATM",       lambda: _topica.KeyATM({"a": ["x"]}, num_topics=2),         "dirichlet"),
    ("SeededLDA",    lambda: _topica.SeededLDA({"a": ["x"], "b": ["y"]}),        "dirichlet"),
    ("GSDMM",        lambda: _topica.GSDMM(num_topics=5),                        "none"),
    # logistic-normal (variational eta posterior)
    ("STM",          lambda: _topica.STM(2),                                     "logistic_normal"),
    ("CTM",          lambda: _topica.CTM(2),                                     "logistic_normal"),
    # neural / embedding-based — no theta posterior
    ("ETM",          lambda: _topica.ETM(2),                                     "none"),
    ("ProdLDA",      lambda: _topica.ProdLDA(2),                                 "none"),
    ("FASTopic",     lambda: _topica.FASTopic(2),                                "none"),
    # embedding-cluster — no generative word distribution
    ("BERTopic",     lambda: _topica.BERTopic(min_cluster_size=5),               "none"),
    ("Top2Vec",      lambda: _topica.Top2Vec(),                                  "none"),
    # EmbeddingLDA is EXCLUDED: it is a Python wrapper around SeededLDA (see
    # module-level note in python/topica/embedding.py). It delegates every
    # fitted-model getter to self._model via __getattr__, has no class-level
    # PyO3 attributes, and its __init__ requires embeddings and vocabulary, so
    # it cannot be constructed generically. The underlying SeededLDA (which is
    # in this registry) covers the contract. Treat EmbeddingLDA as a user-facing
    # convenience wrapper, not a first-class estimator, until it graduates to a
    # proper PyO3 binding with its own class attributes.
]

# ---------------------------------------------------------------------------
# Contract tiers
# ---------------------------------------------------------------------------

# Tier 0: every estimator
TIER0_ATTRS = [
    "topic_word",
    "doc_topic",
    "vocabulary",
    "num_topics",
    "topic_names",
    "doc_names",
    "top_words",
    "coherence",
    "save",
    "load",
]
TIER0_ITERS = "iters"   # canonical iteration kwarg name

# Tier 1: generative models (model_family != "none")
TIER1_ATTRS = ["transform"]

# Tier 2: family-specific
TIER2_DIRICHLET = ["alpha", "theta_draws", "doc_lengths"]
TIER2_LOGISTIC_NORMAL = ["eta_mean", "eta_cov"]

# ---------------------------------------------------------------------------
# Principled exemptions — PERMANENT, will not be closed
# ---------------------------------------------------------------------------
# Key: (model_name, requirement)   Value: human-readable reason

EXEMPT: dict[tuple[str, str], str] = {

    # --- HLDA: topic-tree model, not a flat K-topic model ---
    # doc_topic: HLDA assigns each document to a root-to-leaf *path*, not a
    # K-vector simplex; there is no static (D, K) theta matrix.
    ("HLDA", "doc_topic"):  "HLDA is a topic tree (nested CRP); documents have paths, not a (D,K) theta simplex",
    # num_topics: the number of active nodes is discovered, not fixed to a
    # scalar K.  The model exposes num_nodes instead.
    ("HLDA", "num_topics"): "HLDA has num_nodes (tree), not a fixed scalar K",
    # doc_names: no static doc rows -> no doc_names index.
    ("HLDA", "doc_names"):  "HLDA has no static per-document row index (tree paths, not theta)",
    # coherence: requires a (K, V) topic_word matrix; HLDA's topic_word is
    # node-indexed, not a flat K-topic array.
    ("HLDA", "coherence"):  "HLDA coherence requires a flat (K,V) topic_word; node-indexed tree structure is incompatible",
    # transform: no flat K-topic generative model -> cannot infer a (n, K) theta.
    ("HLDA", "transform"):  "HLDA has no flat K-topic generative distribution; held-out transform is undefined",
    # iters: HLDA does accept iters, so this is NOT exempted.
    # alpha/theta_draws/doc_lengths: family is 'none' so Tier 2 dirichlet
    # does not apply.

    # --- DTM: time-sliced model, no static single theta ---
    # doc_topic: DTM's theta is time-conditioned; there is no single static
    # (D, K) matrix.  Use topic_word(time) and word_evolution for trajectories.
    ("DTM", "doc_topic"):   "DTM is time-sliced; doc_topic is undefined without a time slice argument",
    # doc_names: follows from no static doc_topic rows.
    ("DTM", "doc_names"):   "DTM has no static per-document row index (time-sliced model)",
    # coherence: coherence needs a static (K, V) phi matrix; DTM's topic_word
    # varies by time slice (callable, not a property).
    ("DTM", "coherence"):   "DTM coherence requires a static (K,V) topic_word; time-varying phi is incompatible",
    # transform: DTM's EM is time-conditioned; there is no time-slice-free
    # held-out inference.
    ("DTM", "transform"):   "DTM has no time-slice-free held-out inference",

    # --- BERTopic / Top2Vec: embedding-cluster models ---
    # topic_names and doc_names: these models DO expose them (already present).
    # coherence: they do NOT expose it as a model method (no class attr).
    ("BERTopic", "coherence"): "BERTopic c-TF-IDF topic_word is not a probability; model-level coherence() method absent (use topica.coherence() externally)",
    ("Top2Vec",  "coherence"): "Top2Vec c-TF-IDF topic_word is not a probability; model-level coherence() method absent (use topica.coherence() externally)",
    ("BERTopic", "doc_names"): "BERTopic exposes labels (cluster ids) not a doc_names property; mapping to names is cluster-specific",
    ("Top2Vec",  "doc_names"): "Top2Vec exposes labels (cluster ids) not a doc_names property",
    # transform: both DO expose transform; NOT exempted.
    # Tier 2: family is 'none'; no Tier 2 applies.
    # iters: neither model is iterative in the Gibbs/EM sense; NOT exempted
    # (they simply do not have iters, which is a KNOWN_GAP not an exemption
    # because it is left open pending a decision on whether to standardize
    # embedding models' epochs as 'iters').

    # --- GSDMM: Dirichlet mixture (one topic per document) ---
    # topic_names: not present at class level.
    ("GSDMM", "topic_names"): "GSDMM is a mixture model (one topic per document); topic_names property absent",

    # --- SAGE / PA / PT / DMR: no transform ---
    # SAGE: anchored word counts; inference is not a simple projection.
    ("SAGE", "transform"):  "SAGE keyword-anchored model has no held-out transform",
    # PA: super/sub structure; no flat single-theta held-out path.
    ("PA",   "transform"):  "PA super/sub-topic model has no held-out transform",
    # PT: pseudo-topic prior; no held-out transform implemented.
    ("PT",   "transform"):  "PT pseudo-topic model has no held-out transform",
    # KeyATM: keyword-constrained; transform omitted by design.
    ("KeyATM", "transform"): "KeyATM keyword-constrained model has no held-out transform",
    # SeededLDA: seeded variant; transform omitted (same reasoning as KeyATM).
    ("SeededLDA", "transform"): "SeededLDA seeded model has no held-out transform",
}

# ---------------------------------------------------------------------------
# Known gaps — TEMPORARY, tracked as a burn-down worklist
# ---------------------------------------------------------------------------
# Key: (model_name, requirement)   Value: phase note

KNOWN_GAPS: dict[tuple[str, str], str] = {

    # --- Phase 2: rename iteration param to iters ---
    ("LDA",          "iters"): "phase 2: rename 'iterations' -> 'iters' in LDA.fit",
    ("DMR",          "iters"): "phase 2: rename 'iterations' -> 'iters' in DMR.fit",
    ("SAGE",         "iters"): "phase 2: rename 'iterations' -> 'iters' in SAGE.fit",
    ("LabeledLDA",   "iters"): "phase 2: rename 'iterations' -> 'iters' in LabeledLDA.fit",
    ("SupervisedLDA","iters"): "phase 2: rename 'em_iters' -> 'iters' in SupervisedLDA.fit",
    ("STM",          "iters"): "phase 2: rename 'em_iters' -> 'iters' in STM.fit",
    ("CTM",          "iters"): "phase 2: rename 'em_iters' -> 'iters' in CTM.fit",
    ("DTM",          "iters"): "phase 2: rename 'em_iters' -> 'iters' in DTM.fit",
    ("ETM",          "iters"): "phase 2: expose iters in ETM.fit (currently none)",
    ("ProdLDA",      "iters"): "phase 2: expose iters in ProdLDA.fit (currently none)",
    ("FASTopic",     "iters"): "phase 2: expose iters in FASTopic.fit (currently none)",
    ("BERTopic",     "iters"): "phase 2: expose iters in BERTopic.fit if applicable, or document non-iterative",
    ("Top2Vec",      "iters"): "phase 2: expose iters in Top2Vec.fit if applicable, or document non-iterative",

    # --- Phase 3: add topic_names to all models that lack it ---
    ("LDA",          "topic_names"): "phase 3: add topic_names property to LDA",
    ("DMR",          "topic_names"): "phase 3: add topic_names property to DMR",
    ("SAGE",         "topic_names"): "phase 3: add topic_names property to SAGE",
    ("PA",           "topic_names"): "phase 3: add topic_names property to PA",
    ("PT",           "topic_names"): "phase 3: add topic_names property to PT",
    ("HDP",          "topic_names"): "phase 3: add topic_names property to HDP",
    ("LabeledLDA",   "topic_names"): "phase 3: add topic_names property to LabeledLDA",
    ("SupervisedLDA","topic_names"): "phase 3: add topic_names property to SupervisedLDA",
    ("STM",          "topic_names"): "phase 3: add topic_names property to STM",
    ("CTM",          "topic_names"): "phase 3: add topic_names property to CTM",
    ("HLDA",         "topic_names"): "phase 3: add topic_names property to HLDA (tree nodes)",
    ("DTM",          "topic_names"): "phase 3: add topic_names property to DTM",

    # --- Phase 4: add coherence + save/load + doc_names to neural models ---
    ("ETM",          "doc_names"):  "phase 4: add doc_names property to ETM",
    ("ETM",          "coherence"):  "phase 4: add coherence method to ETM",
    ("ETM",          "save"):       "phase 4: add save method to ETM",
    ("ETM",          "load"):       "phase 4: add load method to ETM",
    ("FASTopic",     "doc_names"):  "phase 4: add doc_names property to FASTopic",
    ("FASTopic",     "coherence"):  "phase 4: add coherence method to FASTopic",
    ("FASTopic",     "save"):       "phase 4: add save method to FASTopic",
    ("FASTopic",     "load"):       "phase 4: add load method to FASTopic",
    ("ProdLDA",      "save"):       "phase 4: add save method to ProdLDA",
    ("ProdLDA",      "load"):       "phase 4: add load method to ProdLDA",

    # --- Phase 5: add theta_draws + doc_lengths to remaining Gibbs models ---
    ("DMR",          "theta_draws"):    "phase 5: add theta_draws to DMR (retained MCMC draws)",
    ("DMR",          "doc_lengths"):    "phase 5: add doc_lengths to DMR",
    ("SAGE",         "theta_draws"):    "phase 5: add theta_draws to SAGE",
    ("SAGE",         "doc_lengths"):    "phase 5: add doc_lengths to SAGE",
    ("PA",           "theta_draws"):    "phase 5: add theta_draws to PA",
    ("PA",           "doc_lengths"):    "phase 5: add doc_lengths to PA",
    ("PT",           "theta_draws"):    "phase 5: add theta_draws to PT",
    ("PT",           "doc_lengths"):    "phase 5: add doc_lengths to PT",
    ("HDP",          "theta_draws"):    "phase 5: add theta_draws to HDP",
    ("HDP",          "doc_lengths"):    "phase 5: add doc_lengths to HDP",
    ("LabeledLDA",   "theta_draws"):    "phase 5: add theta_draws to LabeledLDA",
    ("LabeledLDA",   "doc_lengths"):    "phase 5: add doc_lengths to LabeledLDA",
    ("SupervisedLDA","theta_draws"):    "phase 5: add theta_draws to SupervisedLDA",
    ("SupervisedLDA","doc_lengths"):    "phase 5: add doc_lengths to SupervisedLDA",
}

# ---------------------------------------------------------------------------
# check_conformance helper
# ---------------------------------------------------------------------------

def _accepts_kwarg(fn, name: str) -> bool:
    """Whether ``fn`` accepts the keyword argument ``name``."""
    try:
        params = inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def check_conformance(model_or_class) -> list[str]:
    """Check ``model_or_class`` against the topica estimator contract.

    Returns a list of violation strings. An empty list means the model
    satisfies every applicable tier requirement (or has a valid exemption).
    Does NOT look up KNOWN_GAPS or EXEMPT — it reports all raw violations so
    the conformance test can categorize them. Call this from the test or from
    your own CI after adding a new estimator.

    Parameters
    ----------
    model_or_class : an estimator instance or class.

    Returns
    -------
    list of str
        Each entry is a human-readable description of the violation, e.g.
        ``"missing class attribute: topic_names"``.
    """
    if isinstance(model_or_class, type):
        cls = model_or_class
        instance = None
    else:
        cls = type(model_or_class)
        instance = model_or_class

    name = cls.__name__
    violations: list[str] = []

    # Tier 0: class-level attribute checks
    for attr in TIER0_ATTRS:
        if not hasattr(cls, attr):
            violations.append(f"missing class attribute: {attr}")

    # Tier 0: iters kwarg
    fit_fn = getattr(cls, "fit", None)
    if fit_fn is None:
        violations.append("missing fit method")
    else:
        if not _accepts_kwarg(fit_fn, TIER0_ITERS):
            violations.append(f"fit() does not accept kwarg: {TIER0_ITERS}")

    # Determine family from an instance (works on unfitted instances per model_family docstring)
    if instance is None:
        # Construct a minimal instance from the registry if possible
        _instance = None
        for reg_name, factory, _ in REGISTRY:
            if reg_name == name:
                try:
                    _instance = factory()
                except Exception:
                    pass
                break
        family_instance = _instance if _instance is not None else cls.__new__(cls)
    else:
        family_instance = instance

    from .effects import model_family as _model_family
    try:
        family = _model_family(family_instance)
    except Exception:
        family = "none"

    # Tier 1: transform required for generative models
    if family != "none":
        if not hasattr(cls, "transform"):
            violations.append("missing class attribute: transform (Tier 1 generative model)")

    # Tier 2: family-specific
    if family == "dirichlet":
        for attr in TIER2_DIRICHLET:
            if not hasattr(cls, attr):
                violations.append(f"missing class attribute: {attr} (Tier 2 dirichlet)")
    elif family == "logistic_normal":
        for attr in TIER2_LOGISTIC_NORMAL:
            if not hasattr(cls, attr):
                violations.append(f"missing class attribute: {attr} (Tier 2 logistic_normal)")

    return violations
