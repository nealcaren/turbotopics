//! Structural Topic and Sentiment-Discourse model (Chen & Mankad 2024, *Management
//! Science*). STS extends STM/CTM with a per-document, per-topic *continuous*
//! sentiment-discourse latent `α^(s)` that modulates the topic-word distribution.
//!
//! This module ports the variational E-step from the authors' reference R/`sts`
//! implementation (`opt.alpha.R`, `estimateHessian.R`). The per-document latent is
//! the `2K-1` vector `α = [α^(p)_{1..K-1}, α^(s)_{1..K}]`: prevalence uses `K-1`
//! dimensions (the last topic is the logistic-normal reference category, exactly
//! as in CTM), and sentiment uses the full `K`. The prevalence sub-block of the
//! objective/gradient/Hessian is identical to [`crate::ctm`]; STS adds the
//! sentiment block and the dependence of `β` on `α^(s)`.
//!
//! The E-step maximizes the per-document log-posterior `f(α)` by quasi-Newton, then
//! takes the Laplace covariance as the inverse of `-∇²f` at the optimum.
//!
//! Generative form (Chen & Mankad eq. 2): for topic `k`,
//! `β_{k,v} ∝ exp(m_v + κ^(t)_{k,v} + κ^(s)_{k,v}·α^(s)_k)`.

/// Per-topic topic-word coefficients. `kappa_t[k]` and `kappa_s[k]` are each a
/// length-`V` vector (baseline and sentiment-discourse loadings for topic `k`).
pub struct Kappa {
    pub kappa_t: Vec<Vec<f64>>, // K × V
    pub kappa_s: Vec<Vec<f64>>, // K × V
}

/// `β_{k,·}` for one topic: softmax over the vocabulary of
/// `m_v + κ^(t)_{k,v} + κ^(s)_{k,v}·α^(s)_k`.
fn topic_beta(mv: &[f64], kt: &[f64], ks: &[f64], alpha_s: f64) -> Vec<f64> {
    let v = mv.len();
    let mut lin = vec![0.0f64; v];
    let mut mx = f64::NEG_INFINITY;
    for i in 0..v {
        lin[i] = mv[i] + kt[i] + ks[i] * alpha_s;
        if lin[i] > mx {
            mx = lin[i];
        }
    }
    let mut s = 0.0;
    for x in lin.iter_mut() {
        *x = (*x - mx).exp();
        s += *x;
    }
    for x in lin.iter_mut() {
        *x /= s;
    }
    lin
}

/// `exp([α^(p), 0])` — the K unnormalized topic weights (last topic is the
/// reference, weight `exp(0)=1`).
fn expeta(alpha_p: &[f64]) -> Vec<f64> {
    let mut e = Vec::with_capacity(alpha_p.len() + 1);
    for &x in alpha_p {
        e.push(x.exp());
    }
    e.push(1.0);
    e
}

/// Shared per-document quantities derived from `α`: the K topic betas, the
/// unnormalized weights `expeta`, and `θ = expeta/Σexpeta`.
struct DocBeta {
    beta: Vec<Vec<f64>>, // K × V
    expeta: Vec<f64>,    // K
    theta: Vec<f64>,     // K
}

fn doc_beta(alpha: &[f64], kappa: &Kappa, mv: &[f64], k: usize) -> DocBeta {
    let expeta = expeta(&alpha[..k - 1]);
    let sum_e: f64 = expeta.iter().sum();
    let theta: Vec<f64> = expeta.iter().map(|&e| e / sum_e).collect();
    let mut beta = Vec::with_capacity(k);
    for t in 0..k {
        beta.push(topic_beta(mv, &kappa.kappa_t[t], &kappa.kappa_s[t], alpha[k - 1 + t]));
    }
    DocBeta { beta, expeta, theta }
}

/// Per-document log-posterior `f(α)` (to MAXIMIZE), `opt.alpha.R::log_posterior_byDoc`.
///
/// `alpha` is length `2K-1`. `words`/`counts` are the document's word indices and
/// counts. `mu`/`siginv` are the variational prior mean and precision (length
/// `2K-1` and `(2K-1)²`).
pub fn sts_lhood(
    alpha: &[f64],
    kappa: &Kappa,
    mv: &[f64],
    words: &[usize],
    counts: &[f64],
    mu: &[f64],
    siginv: &[f64],
    k: usize,
) -> f64 {
    let n = 2 * k - 1;
    let db = doc_beta(alpha, kappa, mv, k);
    let sum_e: f64 = db.expeta.iter().sum();
    let ndoc: f64 = counts.iter().sum();

    // term2: multinomial log-likelihood of the words under the mixture θ·β.
    let mut term2 = 0.0;
    for (wi, &w) in words.iter().enumerate() {
        let mut s = 0.0;
        for t in 0..k {
            s += db.expeta[t] * db.beta[t][w];
        }
        term2 += counts[wi] * s.ln();
    }
    term2 -= ndoc * sum_e.ln();

    // term1: logistic-normal prior, -0.5 (α-μ)ᵀ Σ⁻¹ (α-μ).
    let mut quad = 0.0;
    for i in 0..n {
        let di = alpha[i] - mu[i];
        for j in 0..n {
            quad += di * siginv[i * n + j] * (alpha[j] - mu[j]);
        }
    }
    -0.5 * quad + term2
}

/// Responsibilities `beta_bar[wi][t] = expeta_t β_{t,w} / Σ_t expeta_t β_{t,w}`
/// for each document word, plus the column sums `Σ_v κ^(s)_{k,v} β_{k,v}` used by
/// the gradient and Hessian.
struct DocResp {
    beta_bar: Vec<Vec<f64>>, // ntok × K
    kappa_bar: Vec<Vec<f64>>, // K × V  (κ^(s) minus its β-weighted mean)
    ks_beta_mean: Vec<f64>,   // K
}

fn doc_resp(db: &DocBeta, kappa: &Kappa, words: &[usize], k: usize, v: usize) -> DocResp {
    let mut beta_bar = Vec::with_capacity(words.len());
    for &w in words {
        let mut row = vec![0.0f64; k];
        let mut denom = 0.0;
        for t in 0..k {
            row[t] = db.expeta[t] * db.beta[t][w];
            denom += row[t];
        }
        for t in 0..k {
            row[t] /= denom;
        }
        beta_bar.push(row);
    }

    let mut ks_beta_mean = vec![0.0f64; k];
    let mut kappa_bar = Vec::with_capacity(k);
    for t in 0..k {
        let mut m = 0.0;
        for i in 0..v {
            m += kappa.kappa_s[t][i] * db.beta[t][i];
        }
        ks_beta_mean[t] = m;
        kappa_bar.push(kappa.kappa_s[t].iter().map(|&x| x - m).collect());
    }
    DocResp { beta_bar, kappa_bar, ks_beta_mean }
}

/// Gradient of [`sts_lhood`] w.r.t. `α` (length `2K-1`),
/// `opt.alpha.R::lapl_grad_alpha_eta`.
pub fn sts_grad(
    alpha: &[f64],
    kappa: &Kappa,
    mv: &[f64],
    words: &[usize],
    counts: &[f64],
    mu: &[f64],
    siginv: &[f64],
    k: usize,
) -> Vec<f64> {
    let n = 2 * k - 1;
    let v = mv.len();
    let db = doc_beta(alpha, kappa, mv, k);
    let dr = doc_resp(&db, kappa, words, k, v);
    let ndoc: f64 = counts.iter().sum();

    // g1 (prevalence): expected topic counts minus N_d·θ, first K-1 entries.
    let mut g1 = vec![0.0f64; k];
    for (wi, _) in words.iter().enumerate() {
        for t in 0..k {
            g1[t] += counts[wi] * dr.beta_bar[wi][t];
        }
    }
    for t in 0..k {
        g1[t] -= ndoc * db.theta[t];
    }

    // g2 (sentiment): Σ_w c_w β̄_{w,k} (κ^(s)_{k,w} − mean_β κ^(s)_k).
    let mut g2 = vec![0.0f64; k];
    for (wi, &w) in words.iter().enumerate() {
        for t in 0..k {
            g2[t] += counts[wi] * dr.beta_bar[wi][t] * dr.kappa_bar[t][w];
        }
    }

    // grad f = [g1[0..K-1], g2] − Σ⁻¹(α−μ).
    let mut g = vec![0.0f64; n];
    for i in 0..(k - 1) {
        g[i] = g1[i];
    }
    for t in 0..k {
        g[k - 1 + t] = g2[t];
    }
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += siginv[i * n + j] * (alpha[j] - mu[j]);
        }
        g[i] -= s;
    }
    g
}

/// The precision matrix `-∇²f(α) = -H_data + Σ⁻¹` (row-major `(2K-1)²`),
/// `estimateHessian.R`. Positive-definite at the optimum; its inverse is the
/// Laplace (variational) covariance `ν_d`.
pub fn sts_precision(
    alpha: &[f64],
    kappa: &Kappa,
    mv: &[f64],
    words: &[usize],
    counts: &[f64],
    siginv: &[f64],
    k: usize,
) -> Vec<f64> {
    let n = 2 * k - 1;
    let v = mv.len();
    let db = doc_beta(alpha, kappa, mv, k);
    let dr = doc_resp(&db, kappa, words, k, v);
    let ndoc: f64 = counts.iter().sum();

    // g1, g2 again (the diagonal terms of the data Hessian blocks).
    let mut g1 = vec![0.0f64; k];
    let mut g2 = vec![0.0f64; k];
    for (wi, &w) in words.iter().enumerate() {
        for t in 0..k {
            g1[t] += counts[wi] * dr.beta_bar[wi][t];
            g2[t] += counts[wi] * dr.beta_bar[wi][t] * dr.kappa_bar[t][w];
        }
    }
    for t in 0..k {
        g1[t] -= ndoc * db.theta[t];
    }

    // h_pp = diag(g1) − Σ_w c_w β̄_a β̄_b + N_d θθᵀ          (K×K)
    // h_ps = diag(g2) − Σ_w c_w κ̄_a β̄_a β̄_b                (K×K, rows=sentiment)
    // h_ss = diag(S_a) − Σ_w c_w κ̄_a β̄_a κ̄_b β̄_b           (K×K)
    let mut h_pp = vec![0.0f64; k * k];
    let mut h_ps = vec![0.0f64; k * k];
    let mut h_ss = vec![0.0f64; k * k];

    for (wi, &w) in words.iter().enumerate() {
        let c = counts[wi];
        for a in 0..k {
            let ba = dr.beta_bar[wi][a];
            let kba = dr.kappa_bar[a][w];
            for b in 0..k {
                let bb = dr.beta_bar[wi][b];
                h_pp[a * k + b] -= c * ba * bb;
                h_ps[a * k + b] -= c * kba * ba * bb;
                h_ss[a * k + b] -= c * kba * ba * dr.kappa_bar[b][w] * bb;
            }
        }
    }
    for a in 0..k {
        for b in 0..k {
            h_pp[a * k + b] += ndoc * db.theta[a] * db.theta[b];
        }
        h_pp[a * k + a] += g1[a];
        h_ps[a * k + a] += g2[a];
    }
    // h_ss diagonal: S_a = Σ_w c_w β̄_a (κ̄_a² − Σ_v κ̄_{a,v} κ^(s)_{a,v} β_{a,v}).
    for a in 0..k {
        let mut kbks = 0.0; // Σ_v κ̄_{a,v} κ^(s)_{a,v} β_{a,v}
        for i in 0..v {
            kbks += dr.kappa_bar[a][i] * kappa.kappa_s[a][i] * db.beta[a][i];
        }
        let mut s_a = 0.0;
        for (wi, &w) in words.iter().enumerate() {
            s_a += counts[wi] * dr.beta_bar[wi][a] * (dr.kappa_bar[a][w] * dr.kappa_bar[a][w] - kbks);
        }
        h_ss[a * k + a] += s_a;
    }
    let _ = dr.ks_beta_mean; // (already folded into kappa_bar)

    // Assemble the (2K-1) data Hessian H, then return -H + Σ⁻¹.
    // Prevalence indices 0..K-1; sentiment indices (K-1)..(2K-2).
    let mut precision = vec![0.0f64; n * n];
    for i in 0..(k - 1) {
        for j in 0..(k - 1) {
            precision[i * n + j] = -h_pp[i * k + j];
        }
    }
    for a in 0..k {
        for b in 0..(k - 1) {
            // sentiment row a, prevalence col b
            precision[(k - 1 + a) * n + b] = -h_ps[a * k + b];
            // symmetric transpose
            precision[b * n + (k - 1 + a)] = -h_ps[a * k + b];
        }
    }
    for a in 0..k {
        for b in 0..k {
            precision[(k - 1 + a) * n + (k - 1 + b)] = -h_ss[a * k + b];
        }
    }
    for idx in 0..(n * n) {
        precision[idx] += siginv[idx];
    }
    precision
}

// ---------------------------------------------------------------------------
// Fitted model + EM driver (PR1: κ held fixed; the Poisson κ M-step is PR2)
// ---------------------------------------------------------------------------

use crate::dmr::lbfgs_minimize;
use crate::linalg::{cholesky, half_logdet, make_diagonally_dominant, spd_inverse, spd_inverse_from_chol};
use rand::Rng;

/// A fitted STS model. With the E-step done and `κ` held fixed, this carries the
/// per-document latent prevalence/sentiment, the prior regression `Γ` / covariance
/// `Σ`, and the (fixed) topic-word coefficients `κ`.
pub struct StsModel {
    pub k: usize,
    pub num_types: usize,
    pub alpha: Vec<Vec<f64>>,         // D × (2K-1): [α^(p)_{1..K-1}, α^(s)_{1..K}]
    pub nu: Vec<Vec<f64>>,            // D × (2K-1)²: Laplace covariance per doc
    pub gamma: Option<Vec<Vec<f64>>>, // F × (2K-1): prevalence+sentiment regression
    pub sigma: Vec<f64>,             // (2K-1)²
    pub kappa_t: Vec<Vec<f64>>,      // K × V
    pub kappa_s: Vec<Vec<f64>>,      // K × V
    pub mv: Vec<f64>,                // V
    pub bound_history: Vec<f64>,
    pub converged: bool,
    pub em_iters_run: usize,
}

impl StsModel {
    /// Per-document topic prevalence `θ = softmax([α^(p), 0])` (length K).
    pub fn doc_topics(&self) -> Vec<Vec<f64>> {
        let k = self.k;
        self.alpha
            .iter()
            .map(|a| {
                let e = expeta(&a[..k - 1]);
                let s: f64 = e.iter().sum();
                e.iter().map(|x| x / s).collect()
            })
            .collect()
    }

    /// Per-document topic sentiment-discourse `α^(s)` (length K).
    pub fn doc_sentiment(&self) -> Vec<Vec<f64>> {
        let k = self.k;
        self.alpha.iter().map(|a| a[k - 1..].to_vec()).collect()
    }

    /// Baseline topic-word distributions `β_{k,·}` at `α^(s)=0` (K × V).
    pub fn topic_word(&self) -> Vec<Vec<f64>> {
        (0..self.k)
            .map(|t| topic_beta(&self.mv, &self.kappa_t[t], &self.kappa_s[t], 0.0))
            .collect()
    }
}

fn doc_sparse(doc: &[u32]) -> (Vec<usize>, Vec<f64>) {
    let mut idx: Vec<usize> = doc.iter().map(|&w| w as usize).collect();
    idx.sort_unstable();
    idx.dedup();
    let mut counts = vec![0.0f64; idx.len()];
    let pos: std::collections::HashMap<usize, usize> =
        idx.iter().enumerate().map(|(i, &w)| (w, i)).collect();
    for &w in doc {
        counts[pos[&(w as usize)]] += 1.0;
    }
    (idx, counts)
}

/// Pooled ridge regression of the per-document latent `λ` on covariates `x`
/// (the `opt.mu` "Pooled" mode / [`crate::ctm`]'s `fit_gamma`): returns `Γ` as
/// `F × n`, with `Γ[i][t]` the coefficient of covariate `i` for latent `t`.
fn fit_gamma_ridge(x: &[Vec<f64>], lambda: &[Vec<f64>], f: usize, n: usize, ridge: f64) -> Vec<Vec<f64>> {
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

/// Fit the STS model with the topic-word coefficients `κ` held fixed at their
/// initialization (PR1). The E-step is the Laplace variational inference of
/// [`sts_lhood`]/[`sts_grad`]/[`sts_precision`]; the M-step updates `Γ` and `Σ`
/// exactly as in CTM/STM. `kappa_s_scale` sets the fixed sentiment loading
/// magnitude (`κ^(s)_{k,v} = scale·κ^(t)_{k,v}`).
pub fn fit_sts<R: Rng>(
    docs: &[Vec<u32>],
    num_topics: usize,
    num_types: usize,
    em_iters: usize,
    em_tol: f64,
    prevalence: Option<&[Vec<f64>]>,
    sentiment_seed: Option<&[f64]>,
    kappa_s_scale: f64,
    init_spectral: bool,
    rng: &mut R,
) -> StsModel {
    let k = num_topics;
    let n = 2 * k - 1;
    let d = docs.len();
    let v = num_types;
    let nf = prevalence.map(|x| x[0].len());

    let sparse: Vec<(Vec<usize>, Vec<f64>)> = docs.iter().map(|doc| doc_sparse(doc)).collect();

    // Baseline log word rates m_v.
    let mut freq = vec![1.0f64; v];
    let mut total = v as f64;
    for doc in docs {
        for &w in doc {
            freq[w as usize] += 1.0;
            total += 1.0;
        }
    }
    let mv: Vec<f64> = (0..v).map(|i| (freq[i] / total).ln()).collect();

    // β init (anchor-word spectral, else random), then κ_t = ln β − m so that at
    // α^(s)=0 the topics start at β; κ_s fixed proportional to κ_t.
    let beta = if init_spectral {
        crate::spectral::spectral_init(docs, k, v).unwrap_or_else(|| {
            let mut b = vec![vec![0.0f64; v]; k];
            for row in b.iter_mut() {
                let mut s = 0.0;
                for x in row.iter_mut() {
                    *x = 1.0 + rng.gen::<f64>();
                    s += *x;
                }
                for x in row.iter_mut() {
                    *x /= s;
                }
            }
            b
        })
    } else {
        let mut b = vec![vec![0.0f64; v]; k];
        for row in b.iter_mut() {
            let mut s = 0.0;
            for x in row.iter_mut() {
                *x = 1.0 + rng.gen::<f64>();
                s += *x;
            }
            for x in row.iter_mut() {
                *x /= s;
            }
        }
        b
    };
    let mut kappa_t = vec![vec![0.0f64; v]; k];
    let mut kappa_s = vec![vec![0.0f64; v]; k];
    for t in 0..k {
        for i in 0..v {
            kappa_t[t][i] = beta[t][i].max(1e-12).ln() - mv[i];
            kappa_s[t][i] = kappa_s_scale * kappa_t[t][i];
        }
    }
    let kappa = Kappa { kappa_t: kappa_t.clone(), kappa_s: kappa_s.clone() };

    // Latent init: prevalence at 0, sentiment at the centered seed (or 0).
    let mut alpha = vec![vec![0.0f64; n]; d];
    if let Some(seed) = sentiment_seed {
        let mean: f64 = seed.iter().sum::<f64>() / d as f64;
        for di in 0..d {
            for t in 0..k {
                alpha[di][k - 1 + t] = seed[di] - mean;
            }
        }
    }

    let mut gamma: Option<Vec<Vec<f64>>> = nf.map(|f| vec![vec![0.0f64; n]; f]);
    let mut mu_shared = vec![0.0f64; n];
    let mut sigma = vec![0.0f64; n * n];
    for i in 0..n {
        sigma[i * n + i] = 1.0;
    }
    let mut nu_store = vec![vec![0.0f64; n * n]; d];

    let doc_mu = |di: usize, gamma: &Option<Vec<Vec<f64>>>, mu_shared: &[f64]| -> Vec<f64> {
        match (prevalence, gamma) {
            (Some(x), Some(g)) => (0..n)
                .map(|t| x[di].iter().zip(g).map(|(xi, gr)| xi * gr[t]).sum())
                .collect(),
            _ => mu_shared.to_vec(),
        }
    };

    let mut bound_history = Vec::with_capacity(em_iters);
    let mut converged = false;
    let mut em_iters_run = 0usize;

    for em in 0..em_iters {
        em_iters_run = em + 1;
        let siginv = spd_inverse(&sigma, n).unwrap_or_else(|| {
            let mut s = sigma.clone();
            make_diagonally_dominant(&mut s, n);
            spd_inverse(&s, n).unwrap()
        });
        let entropy = match cholesky(&sigma, n) {
            Some(l) => half_logdet(&l, n),
            None => 0.0,
        };

        let mut total_bound = 0.0;
        for (di, (words, counts)) in sparse.iter().enumerate() {
            if words.is_empty() {
                continue;
            }
            let mu_d = doc_mu(di, &gamma, &mu_shared);
            let a_hat = lbfgs_minimize(
                alpha[di].clone(),
                |a| {
                    (
                        -sts_lhood(a, &kappa, &mv, words, counts, &mu_d, &siginv, k),
                        sts_grad(a, &kappa, &mv, words, counts, &mu_d, &siginv, k)
                            .iter()
                            .map(|g| -g)
                            .collect(),
                    )
                },
                100,
                7,
                1e-5,
            );
            alpha[di] = a_hat.clone();

            let mut prec = sts_precision(&a_hat, &kappa, &mv, words, counts, &siginv, k);
            let (nu_d, half_ld) = match cholesky(&prec, n) {
                Some(l) => (spd_inverse_from_chol(&l, n), half_logdet(&l, n)),
                None => {
                    make_diagonally_dominant(&mut prec, n);
                    let l = cholesky(&prec, n).expect("PD after diagonal dominance");
                    (spd_inverse_from_chol(&l, n), half_logdet(&l, n))
                }
            };
            nu_store[di] = nu_d;
            // bound = f(α̂) − 0.5·log|prec| − 0.5·log|Σ|  (standard Laplace ELBO;
            // ll − 0.5 quad collapses to sts_lhood at the optimum).
            let f_at = sts_lhood(&a_hat, &kappa, &mv, words, counts, &mu_d, &siginv, k);
            total_bound += f_at - half_ld - entropy;
        }
        bound_history.push(total_bound);

        if em_tol > 0.0 && bound_history.len() >= 2 {
            let prev = bound_history[bound_history.len() - 2];
            let rel = (total_bound - prev).abs() / (prev.abs() + 1e-12);
            if rel < em_tol {
                converged = true;
                break;
            }
        }

        // M-step: Γ (pooled ridge) or shared mean μ.
        if let (Some(x), Some(f)) = (prevalence, nf) {
            gamma = Some(fit_gamma_ridge(x, &alpha, f, n, 1e-6));
        } else {
            for i in 0..n {
                mu_shared[i] = alpha.iter().map(|a| a[i]).sum::<f64>() / d as f64;
            }
        }

        // Σ = (1/D)[ Σ_d ν_d + Σ_d (α_d − μ_d)(α_d − μ_d)ᵀ ].
        let mus: Vec<Vec<f64>> = (0..d).map(|di| doc_mu(di, &gamma, &mu_shared)).collect();
        for i in 0..n {
            for j in 0..n {
                let mut cross = 0.0;
                for di in 0..d {
                    cross += (alpha[di][i] - mus[di][i]) * (alpha[di][j] - mus[di][j]);
                }
                let nu_sum: f64 = nu_store.iter().map(|nu| nu[i * n + j]).sum();
                sigma[i * n + j] = (nu_sum + cross) / d as f64;
            }
        }
    }

    StsModel {
        k,
        num_types: v,
        alpha,
        nu: nu_store,
        gamma,
        sigma,
        kappa_t,
        kappa_s,
        mv,
        bound_history,
        converged,
        em_iters_run,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup() -> (usize, usize, Kappa, Vec<f64>, Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>) {
        // Small deterministic problem: K=3 topics, V=6 vocabulary.
        let k = 3usize;
        let v = 6usize;
        // Distinct kappa per topic so the sentiment block is non-degenerate.
        let mut kt = vec![vec![0.0; v]; k];
        let mut ks = vec![vec![0.0; v]; k];
        for t in 0..k {
            for i in 0..v {
                kt[t][i] = 0.10 * ((t * 7 + i * 3 % 5) as f64) - 0.5;
                ks[t][i] = 0.20 * (((i + 2 * t) % 4) as f64) - 0.3;
            }
        }
        let kappa = Kappa { kappa_t: kt, kappa_s: ks };
        let mv: Vec<f64> = (0..v).map(|i| -1.0 - 0.1 * i as f64).collect();
        let words = vec![0usize, 2, 3, 5];
        let counts = vec![3.0, 1.0, 4.0, 2.0];
        let n = 2 * k - 1;
        let mu: Vec<f64> = (0..n).map(|i| 0.05 * i as f64 - 0.1).collect();
        // A symmetric positive-definite precision: 2I + small off-diagonal.
        let mut siginv = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                siginv[i * n + j] = if i == j { 2.0 } else { 0.1 };
            }
        }
        (k, v, kappa, mv, words, counts, mu, siginv)
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let (k, _v, kappa, mv, words, counts, mu, siginv) = setup();
        let n = 2 * k - 1;
        let alpha: Vec<f64> = vec![0.3, -0.2, 0.4, -0.1, 0.25];
        let g = sts_grad(&alpha, &kappa, &mv, &words, &counts, &mu, &siginv, k);
        let eps = 1e-6;
        for i in 0..n {
            let mut ap = alpha.clone();
            let mut am = alpha.clone();
            ap[i] += eps;
            am[i] -= eps;
            let fp = sts_lhood(&ap, &kappa, &mv, &words, &counts, &mu, &siginv, k);
            let fm = sts_lhood(&am, &kappa, &mv, &words, &counts, &mu, &siginv, k);
            let fd = (fp - fm) / (2.0 * eps);
            assert!((g[i] - fd).abs() < 1e-5, "grad[{i}] {} vs fd {}", g[i], fd);
        }
    }

    #[test]
    fn precision_is_negative_hessian_of_f() {
        let (k, _v, kappa, mv, words, counts, mu, siginv) = setup();
        let n = 2 * k - 1;
        let alpha: Vec<f64> = vec![0.3, -0.2, 0.4, -0.1, 0.25];
        let p = sts_precision(&alpha, &kappa, &mv, &words, &counts, &siginv, k);
        // Central-difference the gradient: ∇²f_{ij} = d grad_i / d alpha_j.
        // The returned matrix is -∇²f, so it must equal -FD(grad).
        let eps = 1e-6;
        for j in 0..n {
            let mut ap = alpha.clone();
            let mut am = alpha.clone();
            ap[j] += eps;
            am[j] -= eps;
            let gp = sts_grad(&ap, &kappa, &mv, &words, &counts, &mu, &siginv, k);
            let gm = sts_grad(&am, &kappa, &mv, &words, &counts, &mu, &siginv, k);
            for i in 0..n {
                let hess_ij = (gp[i] - gm[i]) / (2.0 * eps);
                assert!(
                    (p[i * n + j] + hess_ij).abs() < 1e-4,
                    "precision[{i},{j}] {} vs -hess {}",
                    p[i * n + j],
                    -hess_ij
                );
            }
        }
    }

    #[test]
    fn symmetric_precision() {
        let (k, _v, kappa, mv, words, counts, _mu, siginv) = setup();
        let n = 2 * k - 1;
        let alpha: Vec<f64> = vec![0.3, -0.2, 0.4, -0.1, 0.25];
        let p = sts_precision(&alpha, &kappa, &mv, &words, &counts, &siginv, k);
        for i in 0..n {
            for j in 0..n {
                assert!((p[i * n + j] - p[j * n + i]).abs() < 1e-9);
            }
        }
    }

    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Two topics on disjoint vocabulary blocks; each document is drawn from one
    /// block, and the prevalence covariate is the block indicator.
    fn planted_corpus() -> (Vec<Vec<u32>>, Vec<Vec<f64>>, Vec<usize>, usize) {
        let v = 8usize;
        let mut rng = StdRng::seed_from_u64(0);
        let block_a: Vec<u32> = (0..4).collect();
        let block_b: Vec<u32> = (4..8).collect();
        let mut docs = Vec::new();
        let mut x = Vec::new();
        let mut truth = Vec::new();
        for d in 0..60 {
            let a = d % 2 == 0;
            let block = if a { &block_a } else { &block_b };
            let doc: Vec<u32> = (0..12).map(|_| block[rng.gen_range(0..block.len())]).collect();
            docs.push(doc);
            x.push(vec![1.0, if a { 0.0 } else { 1.0 }]); // intercept + indicator
            truth.push(if a { 0 } else { 1 });
        }
        (docs, x, truth, v)
    }

    #[test]
    fn em_bound_increases_and_recovers_topics() {
        let (docs, x, truth, v) = planted_corpus();
        let mut rng = StdRng::seed_from_u64(1);
        let m = fit_sts(&docs, 2, v, 40, 1e-6, Some(&x), None, 0.1, true, &mut rng);

        // The variational bound increases monotonically (allowing tiny slack).
        for w in m.bound_history.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "bound dropped: {} -> {}", w[0], w[1]);
        }

        // The two planted blocks are recovered as the two topics: each topic's
        // top words come from one block. Map topics to blocks by their heaviest
        // word, then check prevalence separates the document groups.
        let tw = m.topic_word();
        let top0 = (0..v).max_by(|&a, &b| tw[0][a].partial_cmp(&tw[0][b]).unwrap()).unwrap();
        let topic_for_block_a = if top0 < 4 { 0 } else { 1 };

        let theta = m.doc_topics();
        let mut correct = 0;
        for (d, th) in theta.iter().enumerate() {
            let dominant = if th[0] >= th[1] { 0 } else { 1 };
            let expected = if truth[d] == 0 { topic_for_block_a } else { 1 - topic_for_block_a };
            if dominant == expected {
                correct += 1;
            }
        }
        assert!(correct as f64 / theta.len() as f64 > 0.9, "only {correct}/60 docs separated");
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let (docs, x, _truth, v) = planted_corpus();
        let mut r1 = StdRng::seed_from_u64(1);
        let mut r2 = StdRng::seed_from_u64(1);
        let m1 = fit_sts(&docs, 2, v, 15, 0.0, Some(&x), None, 0.1, true, &mut r1);
        let m2 = fit_sts(&docs, 2, v, 15, 0.0, Some(&x), None, 0.1, true, &mut r2);
        for (a, b) in m1.alpha.iter().flatten().zip(m2.alpha.iter().flatten()) {
            assert!((a - b).abs() < 1e-12);
        }
        assert_eq!(m1.bound_history.len(), m2.bound_history.len());
    }
}
