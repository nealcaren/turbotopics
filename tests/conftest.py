"""Shared fixtures for the topica test suite."""

import importlib.util
import pathlib
import sys

import pytest

# ---------------------------------------------------------------------------
# Worktree module override
#
# The shared dev venv's editable install points to the main repo's Python
# files. In a git worktree, the modified Python sources live in the worktree
# directory, not the main repo checkout. After the initial ``import topica``
# (which loads the compiled ``._topica`` extension from the main repo), we
# patch specific names from ``topica.formulas`` and ``topica.stm`` so that
# tests exercise the in-progress changes without breaking module-identity
# assertions in other test files.
#
# Strategy: load each worktree module into a temporary namespace, then copy
# only the changed symbols into the existing (already-imported) modules.
# This preserves ``topica.stm is sys.modules["topica.stm"]`` identity while
# making the fixed functions/classes visible.
# ---------------------------------------------------------------------------

_WORKTREE_PYTHON = pathlib.Path(__file__).parent.parent / "python"


def _exec_worktree_module(name: str) -> "types.ModuleType | None":
    """Load the worktree source of ``name`` into a fresh module object and
    return it (the existing ``sys.modules[name]`` is NOT replaced).

    The module is given the real dotted name (not a ``_worktree_`` alias) so
    that relative imports inside it resolve against the already-loaded
    ``sys.modules`` entries and no ``__package__ != __spec__.parent`` warnings
    are raised.
    """
    import types

    rel = name.replace(".", "/") + ".py"
    src = _WORKTREE_PYTHON / rel
    if not src.exists():
        return None
    # Use the real module name so __spec__.parent matches __package__.
    spec = importlib.util.spec_from_file_location(name, src)
    mod = importlib.util.module_from_spec(spec)
    # Temporarily install under the real name so relative imports work, then
    # restore the original after exec so sys.modules is not permanently changed.
    original = sys.modules.get(name)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        # Restore; caller decides whether to keep or discard the worktree module.
        if original is not None:
            sys.modules[name] = original
        else:
            del sys.modules[name]
    return mod


# Import the installed topica first so _topica.abi3.so is resolved.
import topica  # noqa: E402

# ------------------------------------------------------------------
# Patch topica.formulas with the worktree version's changed symbols.
# ------------------------------------------------------------------
_wt_formulas = _exec_worktree_module("topica.formulas")
if _wt_formulas is not None:
    import topica.formulas as _formulas  # noqa: E402
    _formulas._KnotCapturingContext = _wt_formulas._KnotCapturingContext
    _formulas.design_matrix = _wt_formulas.design_matrix
    _formulas.design_matrix_predict = _wt_formulas.design_matrix_predict

# ------------------------------------------------------------------
# Patch topica.stm with the worktree version's changed symbols.
# The worktree stm.py imports from .formulas, so give it the patched
# formulas module.
# ------------------------------------------------------------------
if _wt_formulas is not None:
    import topica.stm as _stm_mod  # noqa: E402
    _wt_stm = _exec_worktree_module("topica.stm")
    if _wt_stm is not None:
        # Copy the changed names into the live stm module.
        _stm_mod._build_reference_rows = _wt_stm._build_reference_rows
        _stm_mod.predicted_prevalence = _wt_stm.predicted_prevalence
        _stm_mod.PredictedPrevalence = _wt_stm.PredictedPrevalence
        # Update the public package and effects references so all three
        # point at the same function/class object.
        topica.predicted_prevalence = _stm_mod.predicted_prevalence
        topica.PredictedPrevalence = _stm_mod.PredictedPrevalence
        topica.effects.predicted_prevalence = _stm_mod.predicted_prevalence
        topica.effects.PredictedPrevalence = _stm_mod.PredictedPrevalence

from topica import Corpus  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "parity: slow CLI-parity check; builds the release binaries and "
        "compares binding output against the `train` CLI byte-for-byte.",
    )


# ---------------------------------------------------------------------------
# Two-cluster toy corpus: animal words vs space words (~15 docs each)
# ---------------------------------------------------------------------------

ANIMAL_WORDS = ["cat", "dog", "fish", "cat", "dog"]
SPACE_WORDS = ["planet", "star", "moon", "rocket", "planet"]

ANIMAL_DOCS: list[list[str]] = [list(ANIMAL_WORDS) for _ in range(15)]
SPACE_DOCS: list[list[str]] = [list(SPACE_WORDS) for _ in range(15)]
TOY_DOCS: list[list[str]] = ANIMAL_DOCS + SPACE_DOCS  # 30 docs total


@pytest.fixture(scope="session")
def toy_docs() -> list[list[str]]:
    """30-document two-cluster token lists (15 animal + 15 space)."""
    return TOY_DOCS


@pytest.fixture(scope="session")
def toy_corpus() -> Corpus:
    """Corpus built from toy_docs with no frequency filtering."""
    return Corpus.from_documents(TOY_DOCS)
