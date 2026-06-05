//! ETM: the Embedded Topic Model (Dieng, Ruiz & Blei 2020).
//!
//! ETM is LDA with two changes: the topic-word distribution is factored through
//! embeddings, and the document-topic prior is logistic-normal. Each topic `k` is
//! a point `alpha_k` in the embedding space, each word `v` is a point `rho_v`, and
//!
//! ```text
//!   beta_{k,v} = softmax_v( rho_v . alpha_k )
//! ```
//!
//! so semantically related words share probability mass even when a topic never
//! saw them. The reference fits ETM with a variational autoencoder; topica fits it
//! on the same variational-EM core as [`crate::ctm`]:
//!
//! - E-step: per document, the logistic-normal Laplace approximation
//!   [`crate::ctm::ctm_hpb`], with `beta` computed from the embeddings. This is
//!   CTM's E-step unchanged.
//! - M-step: update the topic embeddings `alpha` (the word embeddings `rho` are
//!   fixed to the pretrained vectors by default) by L-BFGS against the expected
//!   token-topic counts, then rebuild `beta`. The objective and gradient are the
//!   softmax-cross-entropy form already used for the SAGE content model, with the
//!   embedding inner product in place of the per-word deviations.
//!
//! Per-document optimization avoids the amortization gap of the VAE encoder, so
//! the fit is at least as accurate; the trade is the usual variational-EM one,
//! less scalability to very large corpora than minibatch SGD. A future option is
//! an amortized VAE inference path (an encoder plus reparameterized minibatch
//! SGD) for corpora too large for the per-document E-step; it would reproduce the
//! reference's scaling at some cost in per-document posterior accuracy.

use crate::dmr::lbfgs_minimize;

/// The topic-word matrix `beta[k][v] = softmax_v(rho_v . alpha_k)`.
///
/// `rho` is `V x E` (word embeddings), `alpha` is `K x E` (topic embeddings);
/// returns `K x V` rows that each sum to one.
pub fn softmax_beta(rho: &[Vec<f64>], alpha: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let v = rho.len();
    alpha
        .iter()
        .map(|ak| {
            let mut eta = vec![0.0f64; v];
            let mut max = f64::NEG_INFINITY;
            for (w, rv) in rho.iter().enumerate() {
                let dot: f64 = rv.iter().zip(ak).map(|(r, a)| r * a).sum();
                eta[w] = dot;
                if dot > max {
                    max = dot;
                }
            }
            let mut z = 0.0;
            for e in &eta {
                z += (e - max).exp();
            }
            let log_z = max + z.ln();
            eta.iter().map(|e| (e - log_z).exp()).collect()
        })
        .collect()
}

/// M-step: update the topic embeddings `alpha` in place to maximize the expected
/// complete-data log-likelihood of the embedding-factored `beta`, then return the
/// rebuilt `beta`.
///
/// `counts[k][v]` are the expected token-topic counts from the E-step (summed over
/// documents). For each topic the objective is the softmax cross-entropy
/// `sum_v counts[k][v] * (rho_v . alpha_k - log Z_k)` with a Gaussian prior on
/// `alpha`, and the gradient is `sum_v (counts[k][v] - n_k * beta_{k,v}) * rho_v`.
/// `rho` is held fixed (the pretrained word embeddings).
pub fn optimize_topic_embeddings(
    rho: &[Vec<f64>],
    alpha: &mut [Vec<f64>],
    counts: &[Vec<f64>],
    prior_variance: f64,
    max_iter: usize,
) -> Vec<Vec<f64>> {
    let k = alpha.len();
    let v = rho.len();
    let e = if k > 0 { alpha[0].len() } else { 0 };
    let nk: Vec<f64> = counts.iter().map(|row| row.iter().sum()).collect();
    let inv_var = 1.0 / prior_variance;

    let mut x0 = Vec::with_capacity(k * e);
    for ak in alpha.iter() {
        x0.extend_from_slice(ak);
    }

    let x = lbfgs_minimize(
        x0,
        |flat| {
            let mut value = 0.0f64;
            let mut grad = vec![0.0f64; flat.len()];
            for topic in 0..k {
                let a = &flat[topic * e..(topic + 1) * e];
                // eta_v = rho_v . alpha_k, with a log-sum-exp for numerical safety.
                let mut eta = vec![0.0f64; v];
                let mut max = f64::NEG_INFINITY;
                for (w, rv) in rho.iter().enumerate() {
                    let dot: f64 = rv.iter().zip(a).map(|(r, av)| r * av).sum();
                    eta[w] = dot;
                    if dot > max {
                        max = dot;
                    }
                }
                let mut z = 0.0;
                for ev in &eta {
                    z += (ev - max).exp();
                }
                let log_z = max + z.ln();
                let g = &mut grad[topic * e..(topic + 1) * e];
                for (w, rv) in rho.iter().enumerate() {
                    let n = counts[topic][w];
                    value += n * (eta[w] - log_z);
                    let beta = (eta[w] - log_z).exp();
                    let resid = n - nk[topic] * beta;
                    for (ge, &r) in g.iter_mut().zip(rv) {
                        *ge += resid * r;
                    }
                }
            }
            // Gaussian prior N(0, prior_variance) on every embedding coordinate.
            for (i, &xi) in flat.iter().enumerate() {
                value -= 0.5 * inv_var * xi * xi;
                grad[i] -= inv_var * xi;
            }
            // lbfgs minimizes, so negate the maximization objective and gradient.
            (-value, grad.iter().map(|gv| -gv).collect())
        },
        max_iter,
        7,
        1e-4,
    );

    for topic in 0..k {
        alpha[topic].copy_from_slice(&x[topic * e..(topic + 1) * e]);
    }
    softmax_beta(rho, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn kl(p: &[f64], q: &[f64]) -> f64 {
        p.iter()
            .zip(q)
            .map(|(&pi, &qi)| if pi > 0.0 { pi * (pi / qi.max(1e-12)).ln() } else { 0.0 })
            .sum()
    }

    #[test]
    fn softmax_beta_rows_are_distributions() {
        let rho = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let alpha = vec![vec![2.0, -1.0], vec![-1.0, 2.0]];
        let beta = softmax_beta(&rho, &alpha);
        assert_eq!(beta.len(), 2);
        for row in &beta {
            assert_eq!(row.len(), 3);
            assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
            assert!(row.iter().all(|&p| p > 0.0));
        }
    }

    // Given expected counts generated from a planted beta, the M-step should
    // recover topic embeddings whose beta matches the planted one.
    #[test]
    fn m_step_recovers_planted_beta() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let (k, v, e) = (3, 30, 8);
        // Random word embeddings (fixed) and planted topic embeddings.
        let rho: Vec<Vec<f64>> =
            (0..v).map(|_| (0..e).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect()).collect();
        let alpha_true: Vec<Vec<f64>> =
            (0..k).map(|_| (0..e).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect()).collect();
        let beta_true = softmax_beta(&rho, &alpha_true);
        // Expected counts: a large multiple of the planted beta, so the M-step
        // target is essentially beta_true.
        let counts: Vec<Vec<f64>> =
            beta_true.iter().map(|row| row.iter().map(|&p| p * 5000.0).collect()).collect();

        let mut alpha = vec![vec![0.0f64; e]; k]; // start from zero
        let beta_hat = optimize_topic_embeddings(&rho, &mut alpha, &counts, 100.0, 200);

        for t in 0..k {
            assert!(kl(&beta_true[t], &beta_hat[t]) < 1e-3, "topic {t} KL too large");
        }
    }
}
