"""Honest dynamic prevalence (topica.dynamics): the extractor turns a fitted
model + per-document timestamps into per-period ILR prevalence with a
measurement covariance R_t, and the state-space VAR consumes it.

These check the plumbing and, crucially, the honesty property that motivates the
whole module: R_t must SHRINK as documents-per-period grows. (The statistical
payoff -- that propagating R_t fixes the attenuation and false-Granger problems
of the naive two-step -- is validated separately on synthetic data where ground
truth is known; here we check the integration against a real fitted model.)

Runnable two ways:
    /usr/bin/python3 -m pytest tests/test_dynamics.py        # if pytest present
    PYTHONPATH=python /usr/bin/python3 tests/test_dynamics.py # plain driver
"""

import numpy as np

import topica
from topica import dynamics

ECON = ["tax", "market", "trade", "fiscal", "budget", "deficit", "wage", "bank"]
SOC = ["abortion", "gay", "marriage", "church", "family", "prayer", "school", "moral"]


def _corpus(seed=0):
    """12 periods with a smooth economic->social drift, and DELIBERATELY uneven
    documents-per-period so we can check R_t responds to sample size."""
    rng = np.random.default_rng(seed)
    # few docs early, many docs late -> R_t should be larger early
    per_period = [12, 12, 20, 20, 40, 40, 80, 80, 150, 150, 250, 250]
    docs, periods = [], []
    for t in range(12):
        soc_share = 0.15 + 0.7 * (t / 11.0)          # smooth drift 0.15 -> 0.85
        for _ in range(per_period[t]):
            heavy = SOC if rng.random() < soc_share else ECON
            light = ECON if heavy is SOC else SOC
            docs.append(rng.choice(heavy, 9).tolist() + rng.choice(light, 3).tolist())
            periods.append(2000 + t)
    return docs, periods, per_period


def _fit():
    docs, periods, per_period = _corpus()
    m = topica.CTM(num_topics=2, seed=1)
    m.fit(docs)
    return m, periods, per_period


def test_ilr_roundtrip():
    V = dynamics._ilr_basis(5)
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.dirichlet(np.ones(5))
        back = dynamics._ilr_inv(dynamics._ilr(p, V), V)
        assert np.allclose(back, p, atol=1e-9)
    # basis columns orthonormal and zero-sum
    assert np.allclose(V.T @ V, np.eye(4), atol=1e-12)
    assert np.allclose(V.sum(axis=0), 0.0, atol=1e-12)


def test_extractor_shapes_and_psd():
    m, periods, _ = _fit()
    prev = dynamics.period_prevalence(m, periods, nsims=20, seed=0)
    T, K = 12, 2
    assert prev.num_periods == T
    assert prev.eta.shape == (T, K - 1)
    assert prev.R.shape == (T, K - 1, K - 1)
    assert prev.labels == [2000 + t for t in range(T)]
    # each R_t is symmetric PSD
    for t in range(T):
        assert np.allclose(prev.R[t], prev.R[t].T, atol=1e-10)
        assert np.linalg.eigvalsh(prev.R[t]).min() > -1e-10
    # prevalence back-transforms to valid simplex rows
    P = prev.prevalence()
    assert P.shape == (T, K)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-8)
    # reliability is a valid [0,1] score per period (and works for d==1)
    rel = prev.reliability()
    assert rel.shape == (T,)
    assert np.all((rel >= 0.0) & (rel <= 1.0))


def test_Rt_shrinks_with_more_documents():
    """The honesty property: measurement error must fall as documents accumulate."""
    m, periods, per_period = _fit()
    prev = dynamics.period_prevalence(m, periods, nsims=25, seed=0)
    traceR = np.array([np.trace(prev.R[t]) for t in range(prev.num_periods)])
    n = np.array(per_period, dtype=float)
    # strongly negative association between docs-per-period and measurement error
    r = np.corrcoef(np.log(n), np.log(traceR))[0, 1]
    assert r < -0.8, f"expected R_t to shrink with N_t (got corr {r:.2f})"
    # and the sparsest period is much noisier than the densest
    assert traceR[np.argmin(n)] > 3.0 * traceR[np.argmax(n)]


def test_composition_adds_uncertainty_over_point_estimate():
    """nsims>1 (topic-estimation uncertainty included) gives R_t at least as
    large as nsims=1 (sampling error only)."""
    m, periods, _ = _fit()
    prev1 = dynamics.period_prevalence(m, periods, nsims=1, seed=0)
    prevS = dynamics.period_prevalence(m, periods, nsims=25, seed=0)
    t1 = np.array([np.trace(prev1.R[t]) for t in range(prev1.num_periods)])
    tS = np.array([np.trace(prevS.R[t]) for t in range(prevS.num_periods)])
    assert np.mean(tS) >= np.mean(t1)


def test_var_fit_and_granger():
    m, periods, _ = _fit()
    prev = dynamics.period_prevalence(m, periods, nsims=20, seed=0)
    var = dynamics.fit_prevalence_var(prev)
    d = prev.num_topics - 1
    assert var.A.shape == (d, d)
    assert var.c.shape == (d,)
    assert var.Q.shape == (d, d)
    assert np.isfinite(var.loglik)
    # Granger test returns a well-formed verdict
    g = var.granger_test(cause=0, effect=0)   # d==1 here: self-persistence
    assert 0.0 <= g["p_value"] <= 1.0
    assert g["lr_stat"] >= 0.0
    # forecast / IRF shapes
    assert var.forecast(3).shape == (3, d)
    assert var.impulse_response(0, horizon=5).shape == (6, d)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
