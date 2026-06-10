"""Estimator-interface conformance test.

Every estimator in the topica registry must expose a three-tier interface.
Tier 0 is the universal floor (topic_word, doc_topic, vocabulary, num_topics,
topic_names, doc_names, top_words, coherence, save, load; and iters in fit).
Tier 1 applies to generative models (model_family != "none"): transform.
Tier 2 is family-specific (alpha/theta_draws/doc_lengths for dirichlet;
eta_mean/eta_cov for logistic_normal).

The suite is green today because principled exemptions (EXEMPT) and known
temporary gaps (KNOWN_GAPS) are recorded in topica.conformance. Violations
that are neither exempted nor tracked land as real test failures — that is
the enforcement mechanism for new estimators and regressions.
"""

import inspect

import pytest

import topica
from topica.conformance import (
    EXEMPT,
    KNOWN_GAPS,
    REGISTRY,
    TIER0_ATTRS,
    TIER0_ITERS,
    TIER1_ATTRS,
    TIER2_DIRICHLET,
    TIER2_LOGISTIC_NORMAL,
    _accepts_kwarg,
)


# ---------------------------------------------------------------------------
# Build (model, requirement) parametrize list
# ---------------------------------------------------------------------------
# We build the full cross-product of (model, requirement) pairs upfront so
# every cell is a named pytest parameter. Cells in EXEMPT skip; cells in
# KNOWN_GAPS xfail; everything else must pass.


def _all_requirements_for(name: str, family: str) -> list[str]:
    """The requirements that apply to a given (name, family) pair."""
    reqs = list(TIER0_ATTRS) + [TIER0_ITERS]
    if family != "none":
        reqs += TIER1_ATTRS
    if family == "dirichlet":
        reqs += TIER2_DIRICHLET
    elif family == "logistic_normal":
        reqs += TIER2_LOGISTIC_NORMAL
    return reqs


# Build flat list of (model_name, requirement, family, cls) for parametrize.
_PARAMS: list[tuple[str, str, str, type]] = []
for _reg_name, _factory, _family in REGISTRY:
    try:
        _inst = _factory()
    except Exception as _exc:
        pytest.fail(f"Registry factory for {_reg_name} failed: {_exc}")
    _cls = type(_inst)
    for _req in _all_requirements_for(_reg_name, _family):
        _PARAMS.append((_reg_name, _req, _family, _cls))


def _param_ids():
    return [f"{p[0]}-{p[1]}" for p in _PARAMS]


@pytest.mark.parametrize("model_name,requirement,family,cls", _PARAMS, ids=_param_ids())
def test_conformance(model_name: str, requirement: str, family: str, cls: type) -> None:
    """Each (estimator, requirement) cell must conform, be exempted, or be
    listed in KNOWN_GAPS.

    - EXEMPT  -> pytest.skip (principled, permanent)
    - KNOWN_GAPS -> pytest.xfail (expected failure; worklist item)
    - neither  -> the check must pass (a new gap or regression = hard failure)
    """
    key = (model_name, requirement)

    if key in EXEMPT:
        pytest.skip(f"exempted: {EXEMPT[key]}")

    if key in KNOWN_GAPS:
        pytest.xfail(f"known gap: {KNOWN_GAPS[key]}")

    # --- run the actual check ---
    _check_requirement(model_name, requirement, family, cls)


def _check_requirement(model_name: str, requirement: str, family: str, cls: type) -> None:
    """Assert that (model_name, requirement) is satisfied."""
    if requirement == TIER0_ITERS:
        fit_fn = getattr(cls, "fit", None)
        assert fit_fn is not None, f"{model_name}: no fit method"
        assert _accepts_kwarg(fit_fn, TIER0_ITERS), (
            f"{model_name}.fit() does not accept kwarg '{TIER0_ITERS}'"
        )
    elif requirement == "transform":
        assert hasattr(cls, "transform"), (
            f"{model_name} is missing class attribute 'transform' (Tier 1 generative)"
        )
    else:
        assert hasattr(cls, requirement), (
            f"{model_name} is missing class attribute '{requirement}'"
        )


# ---------------------------------------------------------------------------
# Guard: no undeclared gap
# ---------------------------------------------------------------------------

def test_no_unexpected_gaps() -> None:
    """Every (model, requirement) in KNOWN_GAPS must map to a real registry
    entry with a real requirement that actually fails today.

    If this test fails it means either:
    (a) a gap was closed (the entry should be removed from KNOWN_GAPS), or
    (b) the model was renamed or removed (update the registry), or
    (c) a gap was listed for a non-existent (model, requirement) combination
        (fix the typo in KNOWN_GAPS).
    """
    # Build lookup of (name, family) from registry
    registry_map: dict[str, str] = {name: fam for name, _, fam in REGISTRY}

    for (model_name, requirement), note in KNOWN_GAPS.items():
        assert model_name in registry_map, (
            f"KNOWN_GAPS entry ({model_name!r}, {requirement!r}) references a model "
            f"not in the registry. Remove this entry or add {model_name!r} to the "
            "registry."
        )
        family = registry_map[model_name]
        applicable = _all_requirements_for(model_name, family)
        assert requirement in applicable, (
            f"KNOWN_GAPS entry ({model_name!r}, {requirement!r}): this requirement "
            f"does not apply to {model_name} (family={family!r}). Remove it from "
            "KNOWN_GAPS."
        )
        # Verify the gap actually fails today (i.e. is not already fixed).
        # If the check passes, the gap is stale and should be removed.
        try:
            factory = next(f for n, f, _ in REGISTRY if n == model_name)
            cls = type(factory())
        except Exception:
            continue
        try:
            _check_requirement(model_name, requirement, family, cls)
            actually_passes = True
        except AssertionError:
            actually_passes = False

        assert not actually_passes, (
            f"KNOWN_GAPS entry ({model_name!r}, {requirement!r}) is STALE: the "
            f"check passes today (note: {note!r}). Remove it from KNOWN_GAPS."
        )


def test_exempt_entries_are_consistent() -> None:
    """Every EXEMPT entry must name a real registry model and an applicable
    requirement (or a requirement that would apply if the exemption were not
    there). This prevents typos in the exemption map."""
    registry_names = {name for name, _, _ in REGISTRY}
    for (model_name, requirement), reason in EXEMPT.items():
        assert model_name in registry_names, (
            f"EXEMPT entry ({model_name!r}, {requirement!r}) references a model not "
            f"in the registry."
        )
        # Requirement must be one of the known requirement names
        all_reqs = (
            TIER0_ATTRS
            + [TIER0_ITERS]
            + TIER1_ATTRS
            + TIER2_DIRICHLET
            + TIER2_LOGISTIC_NORMAL
        )
        assert requirement in all_reqs, (
            f"EXEMPT entry ({model_name!r}, {requirement!r}) names an unknown "
            f"requirement. Known: {all_reqs}"
        )
