//! Dirichlet-Multinomial Regression (DMR) topic model (Mimno & McCallum, 2008).
//!
//! DMR replaces LDA's fixed document-topic prior with a per-document prior that
//! is a log-linear function of document features:
//!
//! ```text
//!     α_{d,t} = exp(λ_t · x_d)
//! ```
//!
//! where `x_d` is document `d`'s feature vector and `λ_t` is a learned weight
//! vector for topic `t`. Sampling is ordinary SparseLDA with this per-document
//! prior (we reuse [`crate::sampler::sample_doc`] verbatim, recomputing the
//! smoothing mass per document); the weights `λ` are fit by maximizing the
//! penalized likelihood of the topic assignments via L-BFGS.
//!
//! This module provides the per-document-α sweep and the objective/gradient;
//! the optimizer and Python surface build on top of it.

use rand::Rng;

use crate::optimize::digamma;
use crate::sampler::sample_doc;

/// Stirling-series log Γ. Shifts the argument to z ≥ 10 before applying the
/// asymptotic series so the result (and, importantly for the optimizer, its
/// numerical derivative) is accurate to ~1e-10. This is a local copy used only
/// by the DMR objective; LDA's MALLET-matched log Γ lives in `output.rs`.
fn log_gamma(mut z: f64) -> f64 {
    const HALF_LOG_TWO_PI: f64 = 0.918_938_533_204_672_7;
    let mut shift = 0i32;
    while z < 10.0 {
        z += 1.0;
        shift += 1;
    }
    let mut result = HALF_LOG_TWO_PI + (z - 0.5) * z.ln() - z + 1.0 / (12.0 * z)
        - 1.0 / (360.0 * z * z * z)
        + 1.0 / (1260.0 * z * z * z * z * z);
    while shift > 0 {
        shift -= 1;
        z -= 1.0;
        result -= z.ln();
    }
    result
}

/// Per-document, per-topic prior `α_{d,t} = exp(λ_t · x_d)`.
///
/// `lambda` is `[num_topics][num_features]`, `features` is
/// `[num_docs][num_features]`; returns `[num_docs][num_topics]`.
pub fn compute_doc_alpha(lambda: &[Vec<f64>], features: &[Vec<f64>]) -> Vec<Vec<f64>> {
    features
        .iter()
        .map(|x| {
            lambda
                .iter()
                .map(|lt| {
                    let dot: f64 = lt.iter().zip(x).map(|(l, xi)| l * xi).sum();
                    dot.exp()
                })
                .collect()
        })
        .collect()
}

/// One Gibbs sweep with a per-document prior `doc_alpha[d]` (DMR).
///
/// Identical to [`crate::sampler::run_sweep`] except the smoothing mass and
/// per-topic coefficients are recomputed for each document's own α vector
/// before sampling that document.
#[allow(clippy::too_many_arguments)]
pub fn run_sweep_dmr<R: Rng>(
    type_topic_counts: &mut [Vec<u32>],
    tokens_per_topic: &mut [u32],
    doc_topics: &mut [Vec<u32>],
    docs: &[Vec<u32>],
    doc_alpha: &[Vec<f64>],
    beta: f64,
    beta_sum: f64,
    topic_mask: u32,
    topic_bits: u32,
    num_topics: usize,
    rng: &mut R,
) {
    let mut cached_coefficients = vec![0.0f64; num_topics];
    let mut local_topic_counts = vec![0u32; num_topics];
    let mut local_topic_index = vec![0u32; num_topics];
    let mut scored_positions = vec![0usize; num_topics];
    let mut scored_values = vec![0.0f64; num_topics];

    for doc_idx in 0..docs.len() {
        let alpha = &doc_alpha[doc_idx];

        // Recompute the smoothing-only mass and coefficients for this document.
        let mut smoothing_only_mass = 0.0f64;
        for t in 0..num_topics {
            let denom = tokens_per_topic[t] as f64 + beta_sum;
            smoothing_only_mass += alpha[t] * beta / denom;
            cached_coefficients[t] = alpha[t] / denom;
        }

        sample_doc(
            type_topic_counts,
            tokens_per_topic,
            doc_topics,
            alpha,
            beta,
            beta_sum,
            topic_mask,
            topic_bits,
            num_topics,
            &docs[doc_idx],
            doc_idx,
            rng,
            &mut smoothing_only_mass,
            &mut cached_coefficients,
            &mut local_topic_counts,
            &mut local_topic_index,
            &mut scored_positions,
            &mut scored_values,
        );
    }
}

/// Penalized DMR log-likelihood of the current topic counts and its gradient
/// w.r.t. `lambda`.
///
/// For each document `d` with topic counts `n_{d,t}` (total `N_d`):
/// ```text
///   L = Σ_d [ logΓ(α_{d,·}) − logΓ(N_d + α_{d,·})
///             + Σ_t ( logΓ(n_{d,t} + α_{d,t}) − logΓ(α_{d,t}) ) ]
///       − 1/(2σ²) Σ_{t,f} λ_{t,f}²
/// ```
/// with `α_{d,t} = exp(λ_t · x_d)` and `α_{d,·} = Σ_t α_{d,t}`. The gradient
/// uses `∂α_{d,t}/∂λ_{t,f} = α_{d,t} · x_{d,f}`.
///
/// `doc_topic_counts` is `[num_docs][num_topics]`. Returns `(value, gradient)`
/// where `gradient` matches the shape of `lambda` (`[num_topics][num_features]`).
pub fn dmr_objective_and_gradient(
    lambda: &[Vec<f64>],
    features: &[Vec<f64>],
    doc_topic_counts: &[Vec<u32>],
    num_topics: usize,
    num_features: usize,
    prior_variance: f64,
) -> (f64, Vec<Vec<f64>>) {
    let mut value = 0.0f64;
    let mut grad = vec![vec![0.0f64; num_features]; num_topics];

    let mut alpha = vec![0.0f64; num_topics];

    for (d, x) in features.iter().enumerate() {
        let mut alpha_sum = 0.0f64;
        for t in 0..num_topics {
            let dot: f64 = lambda[t].iter().zip(x).map(|(l, xi)| l * xi).sum();
            let a = dot.exp();
            alpha[t] = a;
            alpha_sum += a;
        }

        let counts = &doc_topic_counts[d];
        let n_d: u32 = counts.iter().sum();

        value += log_gamma(alpha_sum) - log_gamma(alpha_sum + n_d as f64);
        let dg_alpha_sum = digamma(alpha_sum);
        let dg_alpha_sum_n = digamma(alpha_sum + n_d as f64);

        for t in 0..num_topics {
            let a = alpha[t];
            let n = counts[t] as f64;
            value += log_gamma(a + n) - log_gamma(a);

            // ∂L/∂α_{d,t}, then chain through ∂α/∂λ = α · x.
            let dl_da = dg_alpha_sum - dg_alpha_sum_n + digamma(a + n) - digamma(a);
            let coef = dl_da * a;
            let gt = &mut grad[t];
            for f in 0..num_features {
                gt[f] += coef * x[f];
            }
        }
    }

    // Gaussian prior N(0, σ²) on every weight.
    let inv_var = 1.0 / prior_variance;
    for t in 0..num_topics {
        for f in 0..num_features {
            value -= 0.5 * inv_var * lambda[t][f] * lambda[t][f];
            grad[t][f] -= inv_var * lambda[t][f];
        }
    }

    (value, grad)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Minimize `f` (value + gradient) with limited-memory BFGS and a backtracking
/// Armijo line search. Compact by design: DMR re-optimizes frequently between
/// sampling sweeps, so a short history and iteration budget suffice.
pub fn lbfgs_minimize<F>(x0: Vec<f64>, mut f: F, max_iter: usize, history: usize, tol: f64) -> Vec<f64>
where
    F: FnMut(&[f64]) -> (f64, Vec<f64>),
{
    let n = x0.len();
    let mut x = x0;
    let (mut fx, mut g) = f(&x);

    let mut s_list: Vec<Vec<f64>> = Vec::new();
    let mut y_list: Vec<Vec<f64>> = Vec::new();
    let mut rho_list: Vec<f64> = Vec::new();

    for _ in 0..max_iter {
        if g.iter().map(|v| v * v).sum::<f64>().sqrt() < tol {
            break;
        }

        // Two-loop recursion for the search direction d = -H·g.
        let m = s_list.len();
        let mut q = g.clone();
        let mut alpha = vec![0.0f64; m];
        for i in (0..m).rev() {
            let a = rho_list[i] * dot(&s_list[i], &q);
            alpha[i] = a;
            for j in 0..n {
                q[j] -= a * y_list[i][j];
            }
        }
        let gamma = if m > 0 {
            let yy = dot(&y_list[m - 1], &y_list[m - 1]);
            if yy > 0.0 {
                dot(&s_list[m - 1], &y_list[m - 1]) / yy
            } else {
                1.0
            }
        } else {
            1.0
        };
        for v in q.iter_mut() {
            *v *= gamma;
        }
        for i in 0..m {
            let b = rho_list[i] * dot(&y_list[i], &q);
            for j in 0..n {
                q[j] += (alpha[i] - b) * s_list[i][j];
            }
        }
        let mut d: Vec<f64> = q.iter().map(|v| -v).collect();

        // Fall back to steepest descent if the direction isn't a descent one.
        if dot(&d, &g) >= 0.0 {
            d = g.iter().map(|v| -v).collect();
        }
        let dg = dot(&d, &g);

        // Backtracking Armijo line search.
        let mut step = 1.0;
        let mut x_new = x.clone();
        let (mut fx_new, mut g_new) = (fx, g.clone());
        loop {
            for j in 0..n {
                x_new[j] = x[j] + step * d[j];
            }
            let r = f(&x_new);
            fx_new = r.0;
            g_new = r.1;
            if fx_new <= fx + 1e-4 * step * dg || step < 1e-12 {
                break;
            }
            step *= 0.5;
        }

        // Curvature update (skip if it would break positive-definiteness).
        let s: Vec<f64> = (0..n).map(|j| x_new[j] - x[j]).collect();
        let y: Vec<f64> = (0..n).map(|j| g_new[j] - g[j]).collect();
        let sy = dot(&s, &y);
        if sy > 1e-10 {
            if s_list.len() == history {
                s_list.remove(0);
                y_list.remove(0);
                rho_list.remove(0);
            }
            rho_list.push(1.0 / sy);
            s_list.push(s);
            y_list.push(y);
        }

        let converged = (fx - fx_new).abs() < tol * (1.0 + fx.abs());
        x = x_new;
        fx = fx_new;
        g = g_new;
        if converged {
            break;
        }
    }
    x
}

/// Optimize `lambda` in place to maximize the penalized DMR likelihood for the
/// current topic counts (one L-BFGS run, used periodically during sampling).
pub fn optimize_lambda(
    lambda: &mut [Vec<f64>],
    features: &[Vec<f64>],
    doc_topic_counts: &[Vec<u32>],
    num_topics: usize,
    num_features: usize,
    prior_variance: f64,
    max_iter: usize,
) {
    let mut x0 = Vec::with_capacity(num_topics * num_features);
    for lt in lambda.iter() {
        x0.extend_from_slice(lt);
    }

    let x = lbfgs_minimize(
        x0,
        |flat| {
            let mut lam = vec![vec![0.0f64; num_features]; num_topics];
            for t in 0..num_topics {
                lam[t].copy_from_slice(&flat[t * num_features..(t + 1) * num_features]);
            }
            // We minimize, so negate the (maximization) objective and gradient.
            let (val, grad) = dmr_objective_and_gradient(
                &lam,
                features,
                doc_topic_counts,
                num_topics,
                num_features,
                prior_variance,
            );
            let mut g = Vec::with_capacity(num_topics * num_features);
            for gt in &grad {
                g.extend(gt.iter().map(|v| -v));
            }
            (-val, g)
        },
        max_iter,
        7,
        1e-5,
    );

    for t in 0..num_topics {
        lambda[t].copy_from_slice(&x[t * num_features..(t + 1) * num_features]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The analytic gradient must match a finite-difference estimate.
    #[test]
    fn gradient_matches_finite_difference() {
        let num_topics = 3;
        let num_features = 2;
        let lambda = vec![
            vec![0.1, -0.2],
            vec![-0.3, 0.4],
            vec![0.05, 0.15],
        ];
        let features = vec![
            vec![1.0, 0.5],
            vec![1.0, -1.0],
            vec![1.0, 2.0],
            vec![1.0, 0.0],
        ];
        let counts = vec![
            vec![3u32, 1, 0],
            vec![0u32, 2, 2],
            vec![1u32, 1, 5],
            vec![2u32, 0, 1],
        ];
        let sigma2 = 10.0;

        let (_, grad) = dmr_objective_and_gradient(
            &lambda, &features, &counts, num_topics, num_features, sigma2,
        );

        let eps = 1e-6;
        for t in 0..num_topics {
            for f in 0..num_features {
                let mut lp = lambda.clone();
                let mut lm = lambda.clone();
                lp[t][f] += eps;
                lm[t][f] -= eps;
                let (vp, _) =
                    dmr_objective_and_gradient(&lp, &features, &counts, num_topics, num_features, sigma2);
                let (vm, _) =
                    dmr_objective_and_gradient(&lm, &features, &counts, num_topics, num_features, sigma2);
                let numeric = (vp - vm) / (2.0 * eps);
                assert!(
                    (numeric - grad[t][f]).abs() < 1e-4,
                    "grad[{}][{}]: analytic {} vs numeric {}",
                    t, f, grad[t][f], numeric
                );
            }
        }
    }

    // L-BFGS should recover known feature effects from synthetic topic counts.
    #[test]
    fn lbfgs_recovers_synthetic_effects() {
        // Two features (intercept + one covariate), two topics. Construct counts
        // where topic 1 is strongly favored when the covariate is high.
        let num_topics = 2;
        let num_features = 2;
        let mut features = Vec::new();
        let mut counts = Vec::new();
        for i in 0..200 {
            let cov = if i % 2 == 0 { 1.0 } else { -1.0 };
            features.push(vec![1.0, cov]);
            // High covariate -> more topic 1; low -> more topic 0.
            if cov > 0.0 {
                counts.push(vec![2u32, 8]);
            } else {
                counts.push(vec![8u32, 2]);
            }
        }
        let mut lambda = vec![vec![0.0f64; num_features]; num_topics];
        optimize_lambda(&mut lambda, &features, &counts, num_topics, num_features, 100.0, 100);

        // The covariate weight should push topic 1 up and topic 0 down.
        let effect_topic1 = lambda[1][1] - lambda[0][1];
        assert!(
            effect_topic1 > 0.5,
            "expected positive covariate effect on topic 1, got {}",
            effect_topic1
        );
    }
}

