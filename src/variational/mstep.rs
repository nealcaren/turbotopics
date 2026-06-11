use crate::linalg::{make_diagonally_dominant, spd_inverse};

/// Pooled ridge regression of per-document latent `λ` on covariates `x`.
///
/// Returns Γ as `f × n`, where `Γ[i][t]` is the coefficient of covariate `i`
/// for latent dimension `t`. Solves `(X'X + ridge·I) Γ = X'Λ` via Cholesky
/// with a diagonal-dominance fallback for near-singular designs.
pub fn fit_gamma_ridge(
    x: &[Vec<f64>],
    lambda: &[Vec<f64>],
    f: usize,
    n: usize,
    ridge: f64,
) -> Vec<Vec<f64>> {
    let mut xtx = vec![0.0f64; f * f];
    let mut xtl = vec![0.0f64; f * n];
    for (xd, ld) in x.iter().zip(lambda) {
        for i in 0..f {
            for j in 0..f {
                xtx[i * f + j] += xd[i] * xd[j];
            }
            for t in 0..n {
                xtl[i * n + t] += xd[i] * ld[t];
            }
        }
    }
    for i in 0..f {
        xtx[i * f + i] += ridge;
    }
    let inv = spd_inverse(&xtx, f).unwrap_or_else(|| {
        let mut a = xtx.clone();
        make_diagonally_dominant(&mut a, f);
        spd_inverse(&a, f).unwrap()
    });
    let mut gamma = vec![vec![0.0f64; n]; f];
    for i in 0..f {
        for t in 0..n {
            let mut s = 0.0;
            for j in 0..f {
                s += inv[i * f + j] * xtl[j * n + t];
            }
            gamma[i][t] = s;
        }
    }
    gamma
}
