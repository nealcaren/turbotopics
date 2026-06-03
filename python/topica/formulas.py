"""R-style formula interface for building prevalence/effect design matrices.

Wraps `formulaic` (an optional dependency) so social scientists can write
``"~ treatment * party + spline(year, df=3)"`` instead of hand-stitching
`one_hot` and `np.hstack`. The library's own restricted-cubic-spline (the same
one behind :func:`topica.spline`) is exposed inside formulas as ``spline(...)``,
so the basis matches what the STM toolkit uses elsewhere.
"""

from __future__ import annotations


def _formula_context():
    """Symbols available inside a formula beyond the data columns."""
    from .stm import spline as _spline

    def spline(x, df=4, knots=None):
        # formulaic names the columns spline(col, df=k)[0..]; we return just the
        # basis (drop the names tuple from topica.spline).
        return _spline(x, df=df, knots=knots)[0]

    return {"spline": spline}


def design_matrix(formula, data):
    """Build a design matrix from an R-style `formula` and a pandas `data` frame.

    Returns ``(X, feature_names)`` where ``X`` is a ``(n_rows, p)`` float array
    and ``feature_names`` are the column labels. The intercept that `formulaic`
    adds is stripped, because :func:`topica.estimate_effect` and the STM
    prevalence model add their own. Categorical columns become treatment-coded
    dummies; ``a * b`` / ``a:b`` expand interactions; ``spline(x, df=k)`` uses
    topica's restricted cubic spline.

    Requires the optional ``formulaic`` package.
    """
    try:
        from formulaic import model_matrix
    except ImportError as e:  # pragma: no cover - exercised via message
        raise ImportError(
            "The formula interface needs the optional `formulaic` package "
            "(pip install formulaic, or pip install \"topica[formula]\")."
        ) from e
    import pandas as pd

    mm = model_matrix(formula, data, context=_formula_context())
    frame = pd.DataFrame(mm)
    if "Intercept" in frame.columns:
        frame = frame.drop(columns=["Intercept"])
    X = frame.to_numpy(dtype=float)
    names = [str(c) for c in frame.columns]
    return X, names
