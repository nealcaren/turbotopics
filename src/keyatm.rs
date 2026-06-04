//! keyATM: Keyword-Assisted Topic Model (base variant).
//!
//! Eshima, Imai & Sasaki (2024), "Keyword-Assisted Topic Models",
//! American Journal of Political Science 68(2):730–750.
//!
//! Like LDA, but the first `num_keyword_topics` topics are "keyword topics"
//! each with a researcher-supplied keyword list. A token in a keyword topic
//! is drawn either from that topic's keyword-only distribution (switch s=1,
//! prob π_k) or from the regular full-vocabulary distribution (switch s=0,
//! prob 1−π_k).  Remaining topics are regular LDA topics (no keywords, s=0
//! forced).
//!
//! ## Generative model (collapsed; integrate out θ, φ, φ̃, π)
//!
//! ```text
//! θ_d  ~ Dir(α·1_K)
//! φ_k  ~ Dir(β·1_V)                      [regular word dist, all topics]
//! φ̃_k ~ Dir(β_key·1_{L_k})               [keyword dist, keyword topics only]
//! π_k  ~ Beta(γ₁, γ₂)                    [keyword switch, keyword topics only]
//!
//! z_{d,i} ~ Categorical(θ_d)
//! s_{d,i} ~ Bernoulli(π_{z_{d,i}})       [only if z is a keyword topic AND
//!                                          w is a keyword of z; else s=0 forced]
//! w_{d,i} | z, s=0 ~ φ_z                 [regular draw]
//! w_{d,i} | z, s=1 ~ φ̃_z                 [keyword draw]
//! ```
//!
//! ## Collapsed Gibbs — joint (z, s) sample per token
//!
//! Counts:
//! - `ndk[d][k]` — doc-topic counts (D×K).
//! - `nkw[k][w]` — topic-word counts for s=0 tokens (K×V).
//! - `nk0[k]`    — total s=0 tokens in topic k.
//! - `nkx[k][j]` — keyword-only counts for s=1 tokens; j = position of word
//!                  in keywords[k] (K×L_k; zero-sized for non-keyword topics).
//! - `nk1[k]`    — total s=1 tokens in topic k.
//!
//! Unnormalised sampling weights (after removing token's current assignment):
//!
//! For every topic k, regular state s=0 (always allowed):
//! ```text
//! P(z=k,s=0) = (α + ndk[d][k])
//!            × (β + nkw[k][w]) / (V·β + nk0[k])
//!            × (γ₂ + nk0[k]) / (γ₁+γ₂ + nk0[k]+nk1[k])   if k is keyword topic
//!            × 1.0                                           if k is regular topic
//! ```
//!
//! Additionally, ONLY if k is a keyword topic AND w ∈ keywords[k], state s=1:
//! ```text
//! P(z=k,s=1) = (α + ndk[d][k])
//!            × (β_key + nkx[k][j]) / (L_k·β_key + nk1[k])
//!            × (γ₁ + nk1[k]) / (γ₁+γ₂ + nk0[k]+nk1[k])
//! ```

use rand::Rng;

// ---------------------------------------------------------------------------
// Token weighting (keyATM's "weighted LDA")
// ---------------------------------------------------------------------------

/// How each token is weighted in the count tables. keyATM weights tokens by a
/// function of corpus frequency so that frequent words count for less and rare,
/// informative words count for more (Eshima, Imai & Sasaki 2024, Section 3.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightScheme {
    /// Unweighted: every token contributes 1.0 (ordinary collapsed Gibbs).
    None,
    /// Information theory (keyATM default): `weight(w) = -log2(freq(w)/N)`, the
    /// word's surprisal in bits.
    InfoTheory,
    /// Inverse frequency: `weight(w) = N / freq(w)`.
    InvFreq,
}

/// Per-word weights from corpus frequencies, matching keyATM's `weights_*`.
/// `num_types` is the vocabulary size; words absent from the corpus get weight
/// 1.0 (they never appear in a token, so the value is immaterial).
pub fn compute_vocab_weights(
    docs: &[Vec<u32>],
    num_types: usize,
    scheme: WeightScheme,
) -> Vec<f64> {
    if scheme == WeightScheme::None {
        return vec![1.0; num_types];
    }
    let mut freq = vec![0.0f64; num_types];
    for doc in docs {
        for &w in doc {
            freq[w as usize] += 1.0;
        }
    }
    let total: f64 = freq.iter().sum();
    let ln2 = std::f64::consts::LN_2;
    (0..num_types)
        .map(|w| {
            let f = freq[w];
            if f <= 0.0 {
                return 1.0;
            }
            match scheme {
                WeightScheme::None => 1.0,
                WeightScheme::InfoTheory => -((f / total).ln()) / ln2,
                WeightScheme::InvFreq => total / f,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Model struct
// ---------------------------------------------------------------------------

/// Fitted keyATM Base model.
///
/// Stores the final Gibbs state; query it with the provided methods.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct KeyAtmModel {
    /// Vocabulary size V.
    pub num_types: usize,
    /// Total number of topics K (keyword + regular).
    pub num_topics: usize,
    /// Number of keyword topics (topics 0..num_keyword_topics).
    pub num_keyword_topics: usize,
    /// Dirichlet prior on doc-topic distributions (α).
    pub alpha: f64,
    /// Dirichlet prior on regular topic-word distributions (β).
    pub beta: f64,
    /// Dirichlet prior on keyword-only distributions (β_key).
    pub beta_key: f64,
    /// Beta(γ₁, γ₂) prior on keyword switch; γ₁ controls how much weight goes
    /// to the keyword distribution, γ₂ to the regular one.
    pub gamma1: f64,
    /// See `gamma1`.
    pub gamma2: f64,
    /// Keyword sets per topic; entries are word-id vecs, empty for non-keyword topics.
    pub keywords: Vec<Vec<usize>>,
    /// `L[k]` = `keywords[k].len()` (cached).
    pub l: Vec<usize>,
    /// `ndk[d][k]` — weighted doc-topic counts (D×K). Each token contributes its
    /// word's weight (see `vocab_weights`), so these are real-valued, not integer.
    pub ndk: Vec<Vec<f64>>,
    /// `nkw[k][w]` — weighted regular (s=0) topic-word counts (K×V).
    pub nkw: Vec<Vec<f64>>,
    /// `nk0[k]` — weighted total s=0 mass in topic k.
    pub nk0: Vec<f64>,
    /// `nkx[k][j]` — weighted keyword (s=1) count for the j-th keyword of topic k.
    /// Length of inner vec = L_k; empty for non-keyword topics.
    pub nkx: Vec<Vec<f64>>,
    /// `nk1[k]` — weighted total s=1 mass in topic k.
    pub nk1: Vec<f64>,
    /// Per-word weight applied to each token (keyATM's "weighted LDA"). All 1.0
    /// under `WeightScheme::None`; otherwise frequency-based (see `WeightScheme`).
    pub vocab_weights: Vec<f64>,
    /// Covariate model only: learned DMR coefficients `λ[k][f]` for the
    /// log-linear document-topic prior `α_{d,k} = exp(x_d · λ_k)`. `None` for the
    /// base (symmetric-α) model.
    pub lambda: Option<Vec<Vec<f64>>>,
    /// Covariate model only: the document feature matrix used at fit time
    /// (`[D][F]`), kept for held-out inference. `None` for the base model.
    pub features: Option<Vec<Vec<f64>>>,
    /// Dynamic model only: the fitted Chib (1998) change-point HMM over time
    /// segments. `None` for the base and covariate models.
    pub dynamic: Option<DynamicState>,
}

/// Fitted state of the keyATM **Dynamic** model (Eshima, Imai & Sasaki 2024,
/// Section 3.3), a Chib (1998) change-point hidden Markov model on topic
/// prevalence over time.
///
/// Documents are grouped into `num_time` ordered segments (one per timestamp).
/// Each segment is assigned to one of `num_states` latent states via a
/// left-to-right HMM (a segment stays in its state or advances to the next, so
/// the state sequence is non-decreasing and every state is visited). Each state
/// `r` owns its own document-topic Dirichlet prior `alphas[r]`, so topic
/// prevalence shifts at the estimated change points.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct DynamicState {
    /// Number of latent states S.
    pub num_states: usize,
    /// Number of time segments T (distinct timestamps).
    pub num_time: usize,
    /// `time_index[d]` — 0-based time segment of document d (length D).
    pub time_index: Vec<usize>,
    /// `alphas[r][k]` — per-state document-topic Dirichlet prior (S×K).
    pub alphas: Vec<Vec<f64>>,
    /// `r_est[t]` — latent state assigned to time segment t (length T).
    pub r_est: Vec<usize>,
    /// `p_est[r][r']` — left-to-right transition matrix (S×S).
    pub p_est: Vec<Vec<f64>>,
}

impl KeyAtmModel {
    /// Effective topic-word distribution for topic k, length V, sums to 1.
    ///
    /// For a keyword topic k, the distribution mixes the regular and keyword
    /// distributions by the learned switch rate:
    /// ```text
    /// π_k  = (γ₁ + nk1[k]) / (γ₁+γ₂ + nk0[k]+nk1[k])
    /// reg_k(w)  = (β + nkw[k][w]) / (V·β + nk0[k])
    /// key_k(w)  = (β_key + nkx[k][j]) / (L_k·β_key + nk1[k])   for w = keywords[k][j]
    ///           = 0                                               otherwise
    /// phi_k(w)  = (1−π_k)·reg_k(w) + π_k·key_k(w)
    /// ```
    /// For a regular topic: `phi_k(w) = reg_k(w)`.
    /// The result is renormalised to sum to 1.
    pub fn topic_word(&self, k: usize) -> Vec<f64> {
        let v = self.num_types;
        let beta = self.beta;

        let reg_denom = v as f64 * beta + self.nk0[k];

        let mut phi = (0..v)
            .map(|w| (beta + self.nkw[k][w]) / reg_denom)
            .collect::<Vec<f64>>();

        if k < self.num_keyword_topics && self.l[k] > 0 {
            let pi_num = self.gamma1 + self.nk1[k];
            let pi_den = self.gamma1 + self.gamma2 + self.nk0[k] + self.nk1[k];
            let pi_k = pi_num / pi_den;
            let one_minus_pi = 1.0 - pi_k;

            let key_denom = self.l[k] as f64 * self.beta_key + self.nk1[k];
            let beta_key = self.beta_key;

            // Scale the regular component.
            for p in &mut phi {
                *p *= one_minus_pi;
            }
            // Add the keyword component.
            for (j, &kw) in self.keywords[k].iter().enumerate() {
                phi[kw] += pi_k * (beta_key + self.nkx[k][j]) / key_denom;
            }
        }

        // Renormalise (handles floating-point drift).
        let sum: f64 = phi.iter().sum();
        for p in &mut phi {
            *p /= sum;
        }
        phi
    }

    /// Topic-word distributions for all K topics. Shape: K × V.
    pub fn topic_word_all(&self) -> Vec<Vec<f64>> {
        (0..self.num_topics).map(|k| self.topic_word(k)).collect()
    }

    /// Doc-topic distributions θ, shape D×K, rows sum to 1.
    ///
    /// Base model: `θ_{d,k} = (ndk[d][k] + α) / (N_d + K·α)`. Covariate model:
    /// the symmetric α is replaced by the per-document prior
    /// `α_{d,k} = exp(x_d · λ_k)`.
    pub fn doc_topic(&self) -> Vec<Vec<f64>> {
        // Dynamic model: each document's α is the prior of its time segment's
        // current HMM state, `α_{d} = alphas[r_est[time_index[d]]]`.
        if let Some(dyn_) = &self.dynamic {
            return self
                .ndk
                .iter()
                .enumerate()
                .map(|(d, row)| {
                    let a_row = &dyn_.alphas[dyn_.r_est[dyn_.time_index[d]]];
                    let a_sum: f64 = a_row.iter().sum();
                    let n_d: f64 = row.iter().sum::<f64>() + a_sum;
                    row.iter()
                        .zip(a_row.iter())
                        .map(|(&c, &a)| (c + a) / n_d)
                        .collect()
                })
                .collect();
        }

        // Covariate model: per-document, per-topic α from the regression.
        if let (Some(lambda), Some(features)) = (&self.lambda, &self.features) {
            let doc_alpha = crate::dmr::compute_doc_alpha(lambda, features);
            return self
                .ndk
                .iter()
                .zip(doc_alpha.iter())
                .map(|(row, a_row)| {
                    let a_sum: f64 = a_row.iter().sum();
                    let n_d: f64 = row.iter().sum::<f64>() + a_sum;
                    row.iter()
                        .zip(a_row.iter())
                        .map(|(&c, &a)| (c + a) / n_d)
                        .collect()
                })
                .collect();
        }

        let k = self.num_topics;
        let alpha = self.alpha;
        let k_alpha = k as f64 * alpha;

        self.ndk
            .iter()
            .map(|row| {
                let n_d: f64 = row.iter().sum::<f64>() + k_alpha;
                row.iter().map(|&c| (c + alpha) / n_d).collect()
            })
            .collect()
    }

    /// Learned per-topic keyword switch rate π_k (length K).
    ///
    /// π_k = (γ₁ + nk1[k]) / (γ₁+γ₂ + nk0[k]+nk1[k]) for keyword topics.
    /// 0.0 for regular topics.
    pub fn keyword_rate(&self) -> Vec<f64> {
        (0..self.num_topics)
            .map(|k| {
                if k < self.num_keyword_topics && self.l[k] > 0 {
                    let num = self.gamma1 + self.nk1[k];
                    let den = self.gamma1 + self.gamma2 + self.nk0[k] + self.nk1[k];
                    num / den
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Dynamic model: smoothed topic prevalence per time segment, shape T×K,
    /// rows sum to 1. For time segment t in state `r = r_est[t]`, the prevalence
    /// is the normalised state prior `alphas[r] / Σ_k alphas[r][k]` — the
    /// posterior mean topic proportion the HMM assigns to that period.
    ///
    /// Returns `None` for non-dynamic models.
    pub fn time_prevalence(&self) -> Option<Vec<Vec<f64>>> {
        let dyn_ = self.dynamic.as_ref()?;
        Some(
            dyn_.r_est
                .iter()
                .map(|&r| {
                    let a_row = &dyn_.alphas[r];
                    let s: f64 = a_row.iter().sum();
                    a_row.iter().map(|&a| a / s).collect()
                })
                .collect(),
        )
    }
}

// ---------------------------------------------------------------------------
// Sampler internals
// ---------------------------------------------------------------------------

/// Weighted categorical sample; `weights` need not be normalised.
/// Mirrors the `sample_index` in `gsdmm.rs`.
fn sample_index<R: Rng>(weights: &[f64], rng: &mut R) -> usize {
    let total: f64 = weights.iter().sum();
    let mut r = rng.gen::<f64>() * total;
    for (i, &w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

// ---------------------------------------------------------------------------
// Numeric helpers for the dynamic (HMM) sampler
// ---------------------------------------------------------------------------

/// Stirling-series log Γ; shifts the argument up to z ≥ 10 for accuracy (same
/// approximation `dtm.rs` uses, adequate for the small positive α and counts
/// the Dirichlet-multinomial marginal needs).
fn lgamma(mut z: f64) -> f64 {
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

/// Log density of Gamma(shape `a`, scale `b`), matching keyATM's `gammapdfln`:
/// `-a·ln(b) - lnΓ(a) + (a-1)·ln(x) - x/b`.
fn gammapdfln(x: f64, a: f64, b: f64) -> f64 {
    -a * b.ln() - lgamma(a) + (a - 1.0) * x.ln() - x / b
}

/// keyATM `shrinkp`: maps a positive α to (0, 1) via `x / (1 + x)`.
#[inline]
fn shrinkp(x: f64) -> f64 {
    x / (1.0 + x)
}

/// Standard-normal variate (Box–Muller).
fn sample_normal<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-300);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Gamma(shape, 1) variate (Marsaglia & Tsang; boosted for shape < 1).
fn sample_gamma<R: Rng>(shape: f64, rng: &mut R) -> f64 {
    if shape < 1.0 {
        let g = sample_gamma(shape + 1.0, rng);
        let u: f64 = rng.gen::<f64>().max(1e-300);
        return g * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = sample_normal(rng);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u: f64 = rng.gen::<f64>();
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v;
        }
        if u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// Beta(a, b) variate via two Gamma draws.
fn sample_beta<R: Rng>(a: f64, b: f64, rng: &mut R) -> f64 {
    let x = sample_gamma(a, rng);
    let y = sample_gamma(b, rng);
    if x + y <= 0.0 {
        0.5
    } else {
        x / (x + y)
    }
}

/// Inverted keyword index: for each word, the keyword topics that claim it.
///
/// The hot path needs, per token, the s=1 candidates (the keyword topics for
/// which the token's word is a keyword). Almost every word is a keyword of zero
/// topics, so `by_word[w]` is usually empty and the inner loop does no work,
/// instead of probing a per-topic hash map K times per token.
struct KeywordIndex {
    /// `by_word[w]` = `(topic, keyword_position)` for every keyword topic that
    /// lists word `w`. Empty for non-keyword words.
    by_word: Vec<Vec<(usize, usize)>>,
}

impl KeywordIndex {
    fn build(keywords: &[Vec<usize>], num_types: usize) -> Self {
        let mut by_word = vec![Vec::new(); num_types];
        for (k, kws) in keywords.iter().enumerate() {
            for (j, &w) in kws.iter().enumerate() {
                by_word[w].push((k, j));
            }
        }
        KeywordIndex { by_word }
    }

    /// Returns `Some(j)` if word `w` is the j-th keyword of topic `k`. Used only
    /// on the remove/re-add steps (once per token), so the short linear scan is
    /// cheaper than the hash map it replaces.
    #[inline]
    fn keyword_index(&self, k: usize, w: usize) -> Option<usize> {
        self.by_word[w].iter().find(|&&(t, _)| t == k).map(|&(_, j)| j)
    }
}

/// Read-only scalar parameters the token sampler needs, bundled so the
/// sequential and parallel paths share one signature.
#[derive(Clone, Copy)]
struct SampleParams {
    num_topics: usize,
    num_keyword_topics: usize,
    num_types: usize,
    beta: f64,
    beta_key: f64,
    gamma1: f64,
    gamma2: f64,
}

/// Resample the (z, s) assignment of a single token operating directly on the
/// count tables (not the whole model), so it can run against either the model's
/// own tables (sequential) or a worker's local clones (parallel). Removes the
/// token by its weight, samples a new (k, s), and re-adds it.
#[allow(clippy::too_many_arguments)]
#[inline]
fn resample_token_inner<R: Rng>(
    nkw: &mut [Vec<f64>],
    nk0: &mut [f64],
    nkx: &mut [Vec<f64>],
    nk1: &mut [f64],
    ndk_row: &mut [f64],
    l: &[usize],
    ki: &KeywordIndex,
    p: SampleParams,
    alpha_row: &[f64],
    w: usize,
    wt: f64,
    old_z: usize,
    old_s: u8,
    scratch: &mut [f64],
    rng: &mut R,
) -> (usize, u8) {
    let k = p.num_topics;
    let num_kw = p.num_keyword_topics;
    let beta = p.beta;
    let beta_key = p.beta_key;
    let gamma1 = p.gamma1;
    let gamma2 = p.gamma2;

    // --- Remove token from current counts (by its weight). ---
    ndk_row[old_z] -= wt;
    if old_s == 0 {
        nkw[old_z][w] -= wt;
        nk0[old_z] -= wt;
    } else {
        let j = ki.keyword_index(old_z, w).expect("old s=1 but w not a keyword");
        nkx[old_z][j] -= wt;
        nk1[old_z] -= wt;
    }

    // --- Build candidate weights into `scratch` (len 2*K). ---
    let v_beta = p.num_types as f64 * beta;
    let weights = &mut scratch[..2 * k];
    weights.fill(0.0);

    // s=0 candidate for every topic (always allowed). No keyword lookup here.
    for kk in 0..k {
        let ndk_val = ndk_row[kk] + alpha_row[kk];
        let nkw_val = nkw[kk][w] + beta;
        let nk0_val = nk0[kk];
        let reg_likelihood = nkw_val / (v_beta + nk0_val);
        let switch_factor_s0 = if kk < num_kw {
            let nk1_val = nk1[kk];
            (gamma2 + nk0_val) / (gamma1 + gamma2 + nk0_val + nk1_val)
        } else {
            1.0
        };
        weights[2 * kk] = ndk_val * reg_likelihood * switch_factor_s0;
    }

    // s=1 candidates only for the keyword topics that actually list `w`
    // (usually none, so this loop is empty for most tokens).
    for &(kk, j) in &ki.by_word[w] {
        let ndk_val = ndk_row[kk] + alpha_row[kk];
        let nk0_val = nk0[kk];
        let nk1_val = nk1[kk];
        let lk = l[kk] as f64;
        let key_likelihood = (nkx[kk][j] + beta_key) / (lk * beta_key + nk1_val);
        let switch_factor_s1 = (gamma1 + nk1_val) / (gamma1 + gamma2 + nk0_val + nk1_val);
        weights[2 * kk + 1] = ndk_val * key_likelihood * switch_factor_s1;
    }

    // --- Categorical sample ---
    let chosen = sample_index(weights, rng);
    let new_k = chosen / 2;
    let new_s = (chosen % 2) as u8;

    // --- Re-increment counts (by weight). ---
    ndk_row[new_k] += wt;
    if new_s == 0 {
        nkw[new_k][w] += wt;
        nk0[new_k] += wt;
    } else {
        let j = ki.keyword_index(new_k, w).expect("new s=1 but w not a keyword");
        nkx[new_k][j] += wt;
        nk1[new_k] += wt;
    }

    (new_k, new_s)
}

/// Scalar params from the model.
fn sample_params(model: &KeyAtmModel) -> SampleParams {
    SampleParams {
        num_topics: model.num_topics,
        num_keyword_topics: model.num_keyword_topics,
        num_types: model.num_types,
        beta: model.beta,
        beta_key: model.beta_key,
        gamma1: model.gamma1,
        gamma2: model.gamma2,
    }
}

/// One full sequential Gibbs sweep over all tokens in all documents.
fn sweep<R: Rng>(
    model: &mut KeyAtmModel,
    docs: &[Vec<u32>],
    assignments: &mut [Vec<(usize, u8)>],
    ki: &KeywordIndex,
    doc_alpha: &[Vec<f64>],
    rng: &mut R,
) {
    let p = sample_params(model);
    let mut scratch = vec![0.0f64; 2 * p.num_topics];
    // Disjoint &mut borrows of distinct model fields in a single call are allowed.
    for d in 0..docs.len() {
        let alpha_row = &doc_alpha[d];
        for pos in 0..docs[d].len() {
            let w = docs[d][pos] as usize;
            let wt = model.vocab_weights[w];
            let (old_z, old_s) = assignments[d][pos];
            let (new_z, new_s) = resample_token_inner(
                &mut model.nkw, &mut model.nk0, &mut model.nkx, &mut model.nk1,
                &mut model.ndk[d], &model.l, ki, p, alpha_row, w, wt, old_z, old_s,
                &mut scratch, rng,
            );
            assignments[d][pos] = (new_z, new_s);
        }
    }
}

/// Contiguous, near-even document ranges for `n_threads` workers.
fn partition_ranges(n_docs: usize, n_threads: usize) -> Vec<(usize, usize)> {
    let t = n_threads.max(1).min(n_docs.max(1));
    let base = n_docs / t;
    let rem = n_docs % t;
    let mut ranges = Vec::with_capacity(t);
    let mut start = 0;
    for i in 0..t {
        let len = base + usize::from(i < rem);
        ranges.push((start, start + len));
        start += len;
    }
    ranges
}

/// Approximate parallel Gibbs sweep (AD-LDA; Newman et al. 2009). Documents are
/// partitioned across workers; each worker clones the global topic-word tables
/// (`nkw`, `nk0`, `nkx`, `nk1`), samples its partition independently, then the
/// per-word deltas are reconciled as `final = Σ_w worker_w − (W−1)·original`.
/// Per-document tables (`ndk`, assignments) are partition-local, so they are
/// written straight back. Deterministic for a fixed `sweep_seed` and worker count.
fn parallel_sweep_keyatm(
    model: &mut KeyAtmModel,
    docs: &[Vec<u32>],
    assignments: &mut [Vec<(usize, u8)>],
    ki: &KeywordIndex,
    doc_alpha: &[Vec<f64>],
    num_threads: usize,
    sweep_seed: u64,
) {
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use rayon::prelude::*;

    let p = sample_params(model);
    let ranges = partition_ranges(docs.len(), num_threads);

    // Snapshots of the global tables for reconciliation.
    let orig_nkw = model.nkw.clone();
    let orig_nk0 = model.nk0.clone();
    let orig_nkx = model.nkx.clone();
    let orig_nk1 = model.nk1.clone();
    let l = &model.l;
    let vocab_weights = &model.vocab_weights;

    struct WorkerOut {
        nkw: Vec<Vec<f64>>,
        nk0: Vec<f64>,
        nkx: Vec<Vec<f64>>,
        nk1: Vec<f64>,
        start: usize,
        ndk: Vec<Vec<f64>>,
        asgn: Vec<Vec<(usize, u8)>>,
    }

    let outs: Vec<WorkerOut> = ranges
        .par_iter()
        .enumerate()
        .map(|(wid, &(start, end))| {
            let mut nkw = orig_nkw.clone();
            let mut nk0 = orig_nk0.clone();
            let mut nkx = orig_nkx.clone();
            let mut nk1 = orig_nk1.clone();
            let mut ndk: Vec<Vec<f64>> = model.ndk[start..end].to_vec();
            let mut asgn: Vec<Vec<(usize, u8)>> = assignments[start..end].to_vec();
            let mut scratch = vec![0.0f64; 2 * p.num_topics];
            let mut rng = ChaCha8Rng::seed_from_u64(
                sweep_seed ^ (wid as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );

            for (li, d) in (start..end).enumerate() {
                let alpha_row = &doc_alpha[d];
                for pos in 0..docs[d].len() {
                    let w = docs[d][pos] as usize;
                    let wt = vocab_weights[w];
                    let (old_z, old_s) = asgn[li][pos];
                    let (new_z, new_s) = resample_token_inner(
                        &mut nkw, &mut nk0, &mut nkx, &mut nk1, &mut ndk[li], l, ki, p,
                        alpha_row, w, wt, old_z, old_s, &mut scratch, &mut rng,
                    );
                    asgn[li][pos] = (new_z, new_s);
                }
            }
            WorkerOut { nkw, nk0, nkx, nk1, start, ndk, asgn }
        })
        .collect();

    // Write back partition-local tables (disjoint document ranges).
    for out in &outs {
        for (li, (row, a)) in out.ndk.iter().zip(out.asgn.iter()).enumerate() {
            model.ndk[out.start + li] = row.clone();
            assignments[out.start + li] = a.clone();
        }
    }

    // Reconcile the global tables: final = Σ_w worker_w − (W−1)·original,
    // clamping tiny negative floating-point residues to zero.
    let wm1 = (outs.len() as f64) - 1.0;
    let k = p.num_topics;
    for kk in 0..k {
        for w in 0..p.num_types {
            let sum: f64 = outs.iter().map(|o| o.nkw[kk][w]).sum();
            model.nkw[kk][w] = (sum - wm1 * orig_nkw[kk][w]).max(0.0);
        }
        let sum0: f64 = outs.iter().map(|o| o.nk0[kk]).sum();
        model.nk0[kk] = (sum0 - wm1 * orig_nk0[kk]).max(0.0);
        let sum1: f64 = outs.iter().map(|o| o.nk1[kk]).sum();
        model.nk1[kk] = (sum1 - wm1 * orig_nk1[kk]).max(0.0);
        for j in 0..model.nkx[kk].len() {
            let sumx: f64 = outs.iter().map(|o| o.nkx[kk][j]).sum();
            model.nkx[kk][j] = (sumx - wm1 * orig_nkx[kk][j]).max(0.0);
        }
    }
}

/// Run one sweep, choosing the exact sequential path or the approximate parallel
/// path by `num_threads`. The parallel path draws one seed from `rng` so it stays
/// deterministic for a fixed seed and worker count.
fn run_sweep<R: Rng>(
    model: &mut KeyAtmModel,
    docs: &[Vec<u32>],
    assignments: &mut [Vec<(usize, u8)>],
    ki: &KeywordIndex,
    doc_alpha: &[Vec<f64>],
    num_threads: usize,
    rng: &mut R,
) {
    if num_threads <= 1 {
        sweep(model, docs, assignments, ki, doc_alpha, rng);
    } else {
        let seed: u64 = rng.gen();
        parallel_sweep_keyatm(model, docs, assignments, ki, doc_alpha, num_threads, seed);
    }
}

// ---------------------------------------------------------------------------
// Public fit function
// ---------------------------------------------------------------------------

/// Fit a keyATM Base model by collapsed Gibbs sampling.
///
/// # Arguments
/// * `docs`               — corpus; each doc is a list of word-ids in `0..num_types`.
/// * `num_types`          — vocabulary size V.
/// * `num_topics`         — total number of topics K.
/// * `keywords`           — keyword sets, one per topic; length must equal K.
///                          The first `keywords.len()` non-empty entries define
///                          keyword topics; the rest are regular topics.
///                          Pass empty slices for regular topics.
/// * `alpha`              — symmetric Dirichlet prior on doc-topic dists.
/// * `beta`               — symmetric Dirichlet prior on regular topic-word dists.
/// * `beta_key`           — symmetric Dirichlet prior on keyword-only dists.
/// * `gamma1`, `gamma2`   — Beta prior on the keyword switch π_k.
///                          Higher γ₁ → more weight on keyword distribution.
/// * `iters`              — number of full Gibbs sweeps.
/// * `rng`                — random-number source (deterministic for fixed seed).
pub fn fit_keyatm<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    keywords: &[Vec<usize>],
    alpha: f64,
    beta: f64,
    beta_key: f64,
    gamma1: f64,
    gamma2: f64,
    iters: usize,
    weights: WeightScheme,
    num_threads: usize,
    rng: &mut R,
) -> KeyAtmModel {
    assert_eq!(
        keywords.len(),
        num_topics,
        "keywords length must equal num_topics"
    );

    let (mut model, mut assignments, ki) = init_state(
        docs, num_types, num_topics, keywords, alpha, beta, beta_key, gamma1, gamma2, weights, rng,
    );

    // Symmetric prior: every document-topic gets the same α.
    let doc_alpha = vec![vec![alpha; num_topics]; docs.len()];

    for _ in 0..iters {
        run_sweep(&mut model, docs, &mut assignments, &ki, &doc_alpha, num_threads, rng);
    }
    model
}

/// Shared initialisation for the base and covariate fits: build the keyword
/// index, seed token assignments (keyword tokens anchored to their keyword topic
/// with the switch on), and the count tables. Returns the model with
/// `lambda`/`features` unset.
#[allow(clippy::too_many_arguments)]
fn init_state<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    keywords: &[Vec<usize>],
    alpha: f64,
    beta: f64,
    beta_key: f64,
    gamma1: f64,
    gamma2: f64,
    weights: WeightScheme,
    rng: &mut R,
) -> (KeyAtmModel, Vec<Vec<(usize, u8)>>, KeywordIndex) {
    let d_count = docs.len();
    let v = num_types;
    let k = num_topics;
    let l: Vec<usize> = keywords.iter().map(|kws| kws.len()).collect();
    let num_keyword_topics = keywords.iter().filter(|kws| !kws.is_empty()).count();
    let ki = KeywordIndex::build(keywords, num_types);
    let vocab_weights = compute_vocab_weights(docs, v, weights);

    let mut word_keyword_topics: Vec<Vec<usize>> = vec![Vec::new(); v];
    for (kk, kws) in keywords.iter().enumerate() {
        for &w in kws {
            word_keyword_topics[w as usize].push(kk);
        }
    }

    // A token whose word is a keyword of some topic starts in that keyword topic
    // (random among ties) with the switch on (s=1); other tokens start in a
    // uniformly random topic with s=0. Each token contributes its word weight.
    let mut ndk = vec![vec![0.0f64; k]; d_count];
    let mut nkw = vec![vec![0.0f64; v]; k];
    let mut nk0 = vec![0.0f64; k];
    let mut nkx: Vec<Vec<f64>> = keywords.iter().map(|kws| vec![0.0f64; kws.len()]).collect();
    let mut nk1 = vec![0.0f64; k];

    let assignments: Vec<Vec<(usize, u8)>> = docs
        .iter()
        .enumerate()
        .map(|(d, doc)| {
            doc.iter()
                .map(|&word| {
                    let w = word as usize;
                    let wt = vocab_weights[w];
                    let cands = &word_keyword_topics[w];
                    let (z, s): (usize, u8) = if cands.is_empty() {
                        ((rng.gen::<f64>() * k as f64) as usize % k, 0)
                    } else {
                        let z = cands[(rng.gen::<f64>() * cands.len() as f64) as usize % cands.len()];
                        (z, 1)
                    };
                    ndk[d][z] += wt;
                    if s == 0 {
                        nkw[z][w] += wt;
                        nk0[z] += wt;
                    } else {
                        let j = ki.keyword_index(z, w).unwrap();
                        nkx[z][j] += wt;
                        nk1[z] += wt;
                    }
                    (z, s)
                })
                .collect()
        })
        .collect();

    let model = KeyAtmModel {
        num_types: v,
        num_topics: k,
        num_keyword_topics,
        alpha,
        beta,
        beta_key,
        gamma1,
        gamma2,
        keywords: keywords.to_vec(),
        l,
        ndk,
        nkw,
        nk0,
        nkx,
        nk1,
        vocab_weights,
        lambda: None,
        features: None,
        dynamic: None,
    };
    (model, assignments, ki)
}

/// Fit a keyATM **Covariate** model. The document-topic prior is a
/// Dirichlet-Multinomial regression on document features,
/// `α_{d,k} = exp(x_d · λ_k)` (Mimno & McCallum 2008), matching the keyATM R
/// package's covariate model. `λ` is re-estimated by L-BFGS every
/// `opt_interval` sweeps once past `burn_in`; the keyword (z, s) sampler is
/// otherwise identical to the base model.
#[allow(clippy::too_many_arguments)]
pub fn fit_keyatm_cov<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    keywords: &[Vec<usize>],
    features: &[Vec<f64>],
    num_features: usize,
    beta: f64,
    beta_key: f64,
    gamma1: f64,
    gamma2: f64,
    iters: usize,
    opt_interval: usize,
    burn_in: usize,
    prior_variance: f64,
    lbfgs_iters: usize,
    weights: WeightScheme,
    num_threads: usize,
    rng: &mut R,
) -> KeyAtmModel {
    assert_eq!(keywords.len(), num_topics, "keywords length must equal num_topics");
    assert_eq!(features.len(), docs.len(), "features rows must equal number of documents");

    // α is replaced by the covariate prior; pass a nominal 1.0 for the struct.
    let (mut model, mut assignments, ki) = init_state(
        docs, num_types, num_topics, keywords, 1.0, beta, beta_key, gamma1, gamma2, weights, rng,
    );

    let mut lambda = vec![vec![0.0f64; num_features]; num_topics];
    let mut doc_alpha = crate::dmr::compute_doc_alpha(&lambda, features);

    for it in 0..iters {
        run_sweep(&mut model, docs, &mut assignments, &ki, &doc_alpha, num_threads, rng);
        if opt_interval > 0 && it + 1 > burn_in && (it + 1 - burn_in) % opt_interval == 0 {
            crate::dmr::optimize_lambda(
                &mut lambda,
                features,
                &model.ndk,
                num_topics,
                num_features,
                prior_variance,
                lbfgs_iters,
            );
            doc_alpha = crate::dmr::compute_doc_alpha(&lambda, features);
        }
    }

    model.lambda = Some(lambda);
    model.features = Some(features.to_vec());
    model
}

// ---------------------------------------------------------------------------
// Dynamic model (Chib 1998 change-point HMM on topic prevalence)
// ---------------------------------------------------------------------------

/// Dirichlet-multinomial marginal log-likelihood of topic `k`'s contribution
/// over documents `[doc_start, doc_end]` under state prior `alpha`, plus the
/// Gamma(shape, scale) prior on `alpha[k]`. Matches keyATM's `alpha_loglik`.
fn dyn_alpha_loglik(
    alpha: &[f64],
    k: usize,
    doc_start: usize,
    doc_end: usize,
    ndk: &[Vec<f64>],
    doc_len: &[f64],
    prior_shape: f64,
    prior_scale: f64,
) -> f64 {
    let alpha_sum: f64 = alpha.iter().sum();
    let fixed = lgamma(alpha_sum) - lgamma(alpha[k]);
    let mut loglik = gammapdfln(alpha[k], prior_shape, prior_scale);
    for d in doc_start..=doc_end {
        loglik += fixed + lgamma(ndk[d][k] + alpha[k]) - lgamma(doc_len[d] + alpha_sum);
    }
    loglik
}

/// Pólya (Dirichlet-multinomial) log-likelihood of all documents in time
/// segment `[doc_start, doc_end]` under prior `alpha`. keyATM's `polyapdfln`.
fn dyn_polyapdfln(
    alpha: &[f64],
    doc_start: usize,
    doc_end: usize,
    ndk: &[Vec<f64>],
    doc_len: &[f64],
) -> f64 {
    let alpha_sum: f64 = alpha.iter().sum();
    let lg_alpha_sum = lgamma(alpha_sum);
    let lg_alpha: Vec<f64> = alpha.iter().map(|&a| lgamma(a)).collect();
    let mut loglik = 0.0;
    for d in doc_start..=doc_end {
        loglik += lg_alpha_sum - lgamma(doc_len[d] + alpha_sum);
        for (k, &a) in alpha.iter().enumerate() {
            loglik += lgamma(ndk[d][k] + a) - lg_alpha[k];
        }
    }
    loglik
}

/// Slice-sample every entry of one state's `alpha` vector in place, over the
/// documents `[doc_start, doc_end]` that currently belong to the state. Keyword
/// topics use the Gamma(`eta1`, `eta2`) prior, regular topics Gamma(`eta1_reg`,
/// `eta2_reg`). Mirrors keyATM's `sample_alpha_state`.
#[allow(clippy::too_many_arguments)]
fn dyn_sample_alpha_state<R: Rng>(
    alpha: &mut [f64],
    num_keyword_topics: usize,
    doc_start: usize,
    doc_end: usize,
    ndk: &[Vec<f64>],
    doc_len: &[f64],
    eta1: f64,
    eta2: f64,
    eta1_reg: f64,
    eta2_reg: f64,
    min_v: f64,
    max_v: f64,
    rng: &mut R,
) {
    const MAX_SHRINK_TIME: usize = 200;
    let num_topics = alpha.len();
    let order = shuffled_topic_ids(num_topics, rng);

    for &k in &order {
        let (shape, scale) = if k < num_keyword_topics {
            (eta1, eta2)
        } else {
            (eta1_reg, eta2_reg)
        };

        let keep = alpha[k];
        let store_loglik =
            dyn_alpha_loglik(alpha, k, doc_start, doc_end, ndk, doc_len, shape, scale);

        let mut start = min_v;
        let mut end = max_v;
        let previous_p = shrinkp(alpha[k]);
        let slice_ = store_loglik - 2.0 * (1.0 - previous_p).ln() + rng.gen::<f64>().max(1e-300).ln();

        for _ in 0..MAX_SHRINK_TIME {
            let new_p = start + (end - start) * rng.gen::<f64>();
            alpha[k] = new_p / (1.0 - new_p); // expandp
            let new_loglik =
                dyn_alpha_loglik(alpha, k, doc_start, doc_end, ndk, doc_len, shape, scale);
            let new_likelihood = new_loglik - 2.0 * (1.0 - new_p).ln();

            if slice_ < new_likelihood {
                break;
            } else if previous_p < new_p {
                end = new_p;
            } else if new_p < previous_p {
                start = new_p;
            } else {
                alpha[k] = keep;
                break;
            }
        }
    }
}

/// Fisher–Yates shuffle of `0..n` using `rng` (deterministic for a fixed seed).
fn shuffled_topic_ids<R: Rng>(n: usize, rng: &mut R) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.gen::<f64>() * (i as f64 + 1.0)) as usize % (i + 1);
        v.swap(i, j);
    }
    v
}

/// Fit a keyATM **Dynamic** model: a Chib (1998) change-point HMM over time
/// segments, each state carrying its own document-topic prior `alphas[r]`
/// (Eshima, Imai & Sasaki 2024, Section 3.3).
///
/// `time_index` gives the 0-based time segment of each document. Documents must
/// be sorted by time so segments are contiguous (validated). `num_states` is the
/// number of latent regimes the prevalence path is allowed to occupy.
///
/// The token (z, s) sampler is identical to the base model; on top of it each
/// sweep slice-samples the per-state `alphas`, then runs forward filtering /
/// backward sampling (FFBS) of the state path and resamples the left-to-right
/// transition matrix.
#[allow(clippy::too_many_arguments)]
pub fn fit_keyatm_dynamic<R: Rng>(
    docs: &[Vec<u32>],
    num_types: usize,
    num_topics: usize,
    keywords: &[Vec<usize>],
    time_index: &[usize],
    num_states: usize,
    beta: f64,
    beta_key: f64,
    gamma1: f64,
    gamma2: f64,
    eta1: f64,
    eta2: f64,
    eta1_reg: f64,
    eta2_reg: f64,
    iters: usize,
    weights: WeightScheme,
    num_threads: usize,
    rng: &mut R,
) -> KeyAtmModel {
    assert_eq!(keywords.len(), num_topics, "keywords length must equal num_topics");
    assert_eq!(time_index.len(), docs.len(), "time_index length must equal number of documents");
    assert!(num_states >= 1, "num_states must be at least 1");

    // Time segments must be contiguous and non-decreasing (docs sorted by time).
    for w in time_index.windows(2) {
        assert!(w[1] >= w[0], "documents must be sorted by time_index (non-decreasing)");
    }
    let num_time = time_index.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    assert!(num_time >= num_states, "num_time ({num_time}) must be >= num_states ({num_states})");

    // Document index ranges for each time segment.
    let mut time_doc_start = vec![0usize; num_time];
    let mut time_doc_end = vec![0usize; num_time];
    {
        let mut prev: i64 = -1;
        for (d, &t) in time_index.iter().enumerate() {
            if t as i64 != prev {
                time_doc_start[t] = d;
                prev = t as i64;
            }
        }
        for t in 0..num_time - 1 {
            time_doc_end[t] = time_doc_start[t + 1] - 1;
        }
        time_doc_end[num_time - 1] = docs.len() - 1;
    }

    let min_v = shrinkp(1e-9);
    let max_v = shrinkp(100.0);

    // α is replaced by the per-state HMM prior; pass a nominal 1.0 for the struct.
    let (mut model, mut assignments, ki) = init_state(
        docs, num_types, num_topics, keywords, 1.0, beta, beta_key, gamma1, gamma2, weights, rng,
    );
    let num_keyword_topics = model.num_keyword_topics;

    // Weighted document lengths (Σ token weights), matching the weighted counts
    // the Dirichlet-multinomial marginal (alpha_loglik / polyapdfln) consumes.
    let doc_len: Vec<f64> = docs
        .iter()
        .map(|d| d.iter().map(|&w| model.vocab_weights[w as usize]).sum())
        .collect();

    // --- Initialise HMM state ---
    // Per-state α, keyATM's 50/K start.
    let mut alphas = vec![vec![50.0 / num_topics as f64; num_topics]; num_states];

    // State path: contiguous near-even split, so every state holds >= 1 segment.
    let mut r_est = vec![0usize; num_time];
    {
        let base = num_time / num_states;
        let rem = num_time % num_states;
        let mut idx = 0;
        for r in 0..num_states {
            let cnt = base + usize::from(r < rem);
            for _ in 0..cnt {
                r_est[idx] = r;
                idx += 1;
            }
        }
    }

    // Left-to-right transition matrix: diagonal Beta(1,1), super-diagonal the
    // complement, last state absorbing.
    let mut p_est = vec![vec![0.0f64; num_states]; num_states];
    for r in 0..num_states - 1 {
        let pii = sample_beta(1.0, 1.0, rng);
        p_est[r][r] = pii;
        p_est[r][r + 1] = 1.0 - pii;
    }
    p_est[num_states - 1][num_states - 1] = 1.0;

    let mut r_count = vec![0usize; num_states];
    for &r in &r_est {
        r_count[r] += 1;
    }

    let mut prk = vec![vec![0.0f64; num_states]; num_time];

    for _ in 0..iters {
        // 1. Token (z, s) sweep with each doc's α tied to its segment's state.
        let doc_alpha: Vec<Vec<f64>> = time_index
            .iter()
            .map(|&t| alphas[r_est[t]].clone())
            .collect();
        run_sweep(&mut model, docs, &mut assignments, &ki, &doc_alpha, num_threads, rng);

        // 2. Slice-sample each state's α over the documents it currently owns.
        // States are contiguous in time; their document ranges are disjoint, so
        // the per-state slice sampling is independent and parallelises cleanly.
        // The α slice sampler dominates the per-sweep cost on large corpora.
        let ranges: Vec<(usize, usize)> = {
            let mut seg = 0usize;
            (0..num_states)
                .map(|r| {
                    let seg_start = seg;
                    let seg_end = seg + r_count[r] - 1;
                    seg = seg_end + 1;
                    (time_doc_start[seg_start], time_doc_end[seg_end])
                })
                .collect()
        };
        if num_threads > 1 {
            use rand_chacha::rand_core::SeedableRng;
            use rand_chacha::ChaCha8Rng;
            use rayon::prelude::*;
            let base: u64 = rng.gen();
            let ndk = &model.ndk;
            alphas
                .par_iter_mut()
                .zip(ranges.par_iter())
                .enumerate()
                .for_each(|(r, (alpha, &(d_start, d_end)))| {
                    let mut srng = ChaCha8Rng::seed_from_u64(
                        base ^ (r as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
                    );
                    dyn_sample_alpha_state(
                        alpha, num_keyword_topics, d_start, d_end, ndk, &doc_len,
                        eta1, eta2, eta1_reg, eta2_reg, min_v, max_v, &mut srng,
                    );
                });
        } else {
            for (r, &(d_start, d_end)) in ranges.iter().enumerate() {
                dyn_sample_alpha_state(
                    &mut alphas[r], num_keyword_topics, d_start, d_end, &model.ndk, &doc_len,
                    eta1, eta2, eta1_reg, eta2_reg, min_v, max_v, rng,
                );
            }
        }

        // 3. Forward filter: Prk[t][r] = p(state_t = r | y_{1..t}). The Pólya
        // likelihoods logfy[t][r] are independent across (t, r); precompute them
        // (in parallel when threaded — this does not change results, only speed).
        let logfy_all: Vec<Vec<f64>> = if num_threads > 1 {
            use rayon::prelude::*;
            let ndk = &model.ndk;
            (1..num_time)
                .into_par_iter()
                .map(|t| {
                    (0..num_states)
                        .map(|r| dyn_polyapdfln(&alphas[r], time_doc_start[t], time_doc_end[t], ndk, &doc_len))
                        .collect()
                })
                .collect()
        } else {
            (1..num_time)
                .map(|t| {
                    (0..num_states)
                        .map(|r| dyn_polyapdfln(&alphas[r], time_doc_start[t], time_doc_end[t], &model.ndk, &doc_len))
                        .collect()
                })
                .collect()
        };

        for v in prk[0].iter_mut() {
            *v = 0.0;
        }
        prk[0][0] = 1.0; // first segment is always state 0
        for t in 1..num_time {
            let logfy = &logfy_all[t - 1];
            // rt_k[r] = Σ_j Prk[t-1][j] · P[j][r]
            let rt_k: Vec<f64> = (0..num_states)
                .map(|r| (0..num_states).map(|j| prk[t - 1][j] * p_est[j][r]).sum())
                .collect();
            // Normalise log(rt_k[r]) + logfy[r] over the non-zero entries.
            let mut log_unnorm = vec![f64::NEG_INFINITY; num_states];
            for r in 0..num_states {
                if rt_k[r] > 0.0 {
                    log_unnorm[r] = rt_k[r].ln() + logfy[r];
                }
            }
            let m = log_unnorm.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let logsum = m + log_unnorm.iter().map(|&v| (v - m).exp()).sum::<f64>().ln();
            for r in 0..num_states {
                prk[t][r] = if rt_k[r] > 0.0 {
                    (log_unnorm[r] - logsum).exp()
                } else {
                    0.0
                };
            }
        }

        // 4. Backward sample the state path; last segment is the final state.
        for c in r_count.iter_mut() {
            *c = 0;
        }
        r_est[num_time - 1] = num_states - 1;
        r_count[num_states - 1] += 1;
        for t in (0..num_time - 1).rev() {
            let next = r_est[t + 1];
            let probs: Vec<f64> = (0..num_states).map(|r| prk[t][r] * p_est[r][next]).collect();
            let s = sample_index(&probs, rng);
            r_est[t] = s;
            r_count[s] += 1;
        }

        // 5. Resample the transition matrix from the new path.
        for r in 0..num_states - 1 {
            let pii = sample_beta(r_count[r] as f64, 2.0, rng);
            p_est[r][r] = pii;
            p_est[r][r + 1] = 1.0 - pii;
        }
    }

    model.dynamic = Some(DynamicState {
        num_states,
        num_time,
        time_index: time_index.to_vec(),
        alphas,
        r_est,
        p_est,
    });
    model
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand_chacha::rand_core::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    /// Build a small keyword-annotated corpus and verify that keyword topics
    /// are pulled toward their designated keyword blocks.
    ///
    /// Vocabulary layout:
    /// - Block A: words 0..10
    /// - Block B: words 10..20
    /// - Block C: words 20..30
    ///
    /// Topic 0 keywords: [0, 1]  (block A words)
    /// Topic 1 keywords: [10, 11] (block B words)
    /// Topic 2: regular (no keywords)
    ///
    /// Docs: 100 from block A, 100 from block B, 100 from block C, each with
    /// a small amount of noise from a different block.
    #[test]
    fn keywords_steer_topics() {
        let block_size = 10usize;
        let num_blocks = 3usize;
        let v = num_blocks * block_size; // V = 30

        // 100 docs per block, each doc = 5 tokens cycling through block words +
        // 1 noise token from a different block.
        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut labels: Vec<usize> = Vec::new(); // ground-truth block
        for b in 0..num_blocks {
            let offset = b * block_size;
            for d in 0..100usize {
                let mut doc: Vec<u32> = (0..5)
                    .map(|i| (offset + (i + d) % block_size) as u32)
                    .collect();
                // one noise token from the next block
                let noise_block = (b + 1) % num_blocks;
                doc.push((noise_block * block_size + d % block_size) as u32);
                docs.push(doc);
                labels.push(b);
            }
        }

        let keywords: Vec<Vec<usize>> = vec![
            vec![0, 1],   // topic 0 → block A keywords
            vec![10, 11], // topic 1 → block B keywords
            vec![],       // topic 2 → regular
        ];

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let model = fit_keyatm(
            &docs, v, 3, &keywords, 0.1, 0.1, 0.5, 1.0, 1.0, 200, WeightScheme::None, 1, &mut rng,
        );

        // Helper: block with max probability mass in topic_word(k).
        let dominant_block = |k: usize| -> usize {
            let phi = model.topic_word(k);
            (0..num_blocks)
                .max_by(|&ba, &bb| {
                    let sa: f64 = (0..block_size).map(|i| phi[ba * block_size + i]).sum();
                    let sb: f64 = (0..block_size).map(|i| phi[bb * block_size + i]).sum();
                    sa.partial_cmp(&sb).unwrap()
                })
                .unwrap()
        };

        // Topic 0 should be dominated by block A (block 0).
        assert_eq!(
            dominant_block(0),
            0,
            "topic 0 (keyword=block A) should be dominated by block A words"
        );
        // Topic 1 should be dominated by block B (block 1).
        assert_eq!(
            dominant_block(1),
            1,
            "topic 1 (keyword=block B) should be dominated by block B words"
        );
    }

    /// Every `keyword_rate()` value must lie in [0, 1], and regular topics
    /// must return exactly 0.0.
    #[test]
    fn keyword_rate_in_unit_interval() {
        let v = 30usize;
        let docs: Vec<Vec<u32>> = (0..60usize)
            .map(|d| (0..4).map(|i| ((i + d) % v) as u32).collect())
            .collect();

        let keywords: Vec<Vec<usize>> = vec![
            vec![0, 1, 2],  // keyword topic
            vec![10, 11],   // keyword topic
            vec![],         // regular
            vec![],         // regular
        ];

        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let model = fit_keyatm(
            &docs, v, 4, &keywords, 0.1, 0.1, 0.5, 1.0, 1.0, 50, WeightScheme::None, 1, &mut rng,
        );

        let rates = model.keyword_rate();
        assert_eq!(rates.len(), 4);

        // Keyword topics must be in [0, 1].
        for k in 0..2 {
            assert!(
                rates[k] >= 0.0 && rates[k] <= 1.0,
                "keyword_rate[{k}] = {} not in [0,1]",
                rates[k]
            );
        }
        // Regular topics must be exactly 0.
        for k in 2..4 {
            assert_eq!(rates[k], 0.0, "keyword_rate[{k}] should be 0 for regular topic");
        }
    }

    /// Shape and normalisation invariants.
    #[test]
    fn shapes_and_normalisation() {
        let v = 20usize;
        let docs: Vec<Vec<u32>> = (0..40usize)
            .map(|d| (0..3).map(|i| ((i + d) % v) as u32).collect())
            .collect();

        let keywords: Vec<Vec<usize>> = vec![
            vec![0, 1, 2], // keyword topic
            vec![],        // regular
            vec![],        // regular
        ];

        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let model = fit_keyatm(
            &docs, v, 3, &keywords, 0.5, 0.1, 0.5, 1.0, 1.0, 30, WeightScheme::None, 1, &mut rng,
        );

        let d = docs.len();
        let k = 3usize;

        // topic_word(k) length = V, sums to 1.
        for kk in 0..k {
            let phi = model.topic_word(kk);
            assert_eq!(phi.len(), v, "topic_word({kk}) length should be num_types");
            let s: f64 = phi.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-10,
                "topic_word({kk}) sums to {s:.12}, expected 1.0"
            );
        }

        // doc_topic() is D×K, rows sum to 1.
        let theta = model.doc_topic();
        assert_eq!(theta.len(), d, "doc_topic() should have D rows");
        for (dd, row) in theta.iter().enumerate() {
            assert_eq!(row.len(), k, "doc_topic() row {dd} should have K columns");
            let s: f64 = row.iter().sum();
            assert!(
                (s - 1.0).abs() < 1e-10,
                "doc_topic() row {dd} sums to {s:.12}, expected 1.0"
            );
        }
    }

    /// Dynamic model: a corpus whose topic mix changes halfway through time
    /// should produce a monotone state path that flips at the change point, and
    /// the per-state α should reflect the shift in prevalence.
    #[test]
    fn dynamic_recovers_change_point() {
        // Vocab: block A = 0..6 (economic), block B = 6..12 (social).
        // Keyword topic 0 -> A, topic 1 -> B.
        let keywords: Vec<Vec<usize>> = vec![vec![0, 1, 2], vec![6, 7, 8]];
        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut time_index: Vec<usize> = Vec::new();
        let num_time = 10usize;
        for t in 0..num_time {
            // First half mostly block A; second half mostly block B.
            let b_heavy = t >= 5;
            for d in 0..30usize {
                let (heavy_off, light_off) = if b_heavy { (6, 0) } else { (0, 6) };
                let mut doc: Vec<u32> = (0..6)
                    .map(|i| (heavy_off + (i + d) % 6) as u32)
                    .collect();
                doc.push((light_off + d % 6) as u32);
                docs.push(doc);
                time_index.push(t);
            }
        }

        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let model = fit_keyatm_dynamic(
            &docs, 12, 2, &keywords, &time_index, 2, 0.01, 0.1, 1.0, 1.0, 1.0, 1.0,
            2.0, 1.0, 300, WeightScheme::None, 1, &mut rng,
        );

        let dyn_ = model.dynamic.as_ref().expect("dynamic state present");
        // State path is non-decreasing and visits both states.
        for w in dyn_.r_est.windows(2) {
            assert!(w[1] >= w[0], "state path must be non-decreasing");
        }
        assert_eq!(dyn_.r_est[0], 0);
        assert_eq!(dyn_.r_est[num_time - 1], 1);

        // Smoothed prevalence: social topic (1) should rise in the later state.
        let tp = model.time_prevalence().unwrap();
        let early = tp[0][1];
        let late = tp[num_time - 1][1];
        assert!(late - early > 0.3, "social prevalence should rise: {early} -> {late}");
    }

    /// Dynamic fits with the same seed must be identical.
    #[test]
    fn dynamic_deterministic_for_fixed_seed() {
        let keywords: Vec<Vec<usize>> = vec![vec![0, 1], vec![6, 7]];
        let mut docs: Vec<Vec<u32>> = Vec::new();
        let mut time_index: Vec<usize> = Vec::new();
        for t in 0..6usize {
            for d in 0..15usize {
                let off = if t >= 3 { 6 } else { 0 };
                docs.push((0..5).map(|i| (off + (i + d) % 6) as u32).collect());
                time_index.push(t);
            }
        }
        let mut r1 = ChaCha8Rng::seed_from_u64(9);
        let mut r2 = ChaCha8Rng::seed_from_u64(9);
        let m1 = fit_keyatm_dynamic(
            &docs, 12, 2, &keywords, &time_index, 2, 0.01, 0.1, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 100, WeightScheme::None, 1, &mut r1,
        );
        let m2 = fit_keyatm_dynamic(
            &docs, 12, 2, &keywords, &time_index, 2, 0.01, 0.1, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 100, WeightScheme::None, 1, &mut r2,
        );
        assert_eq!(
            m1.dynamic.as_ref().unwrap().r_est,
            m2.dynamic.as_ref().unwrap().r_est
        );
        assert_eq!(m1.doc_topic(), m2.doc_topic());
    }

    /// Two fits with the same seed must be bit-for-bit identical.
    #[test]
    fn deterministic_for_fixed_seed() {
        let v = 15usize;
        let docs: Vec<Vec<u32>> = (0..30usize)
            .map(|d| (0..4).map(|i| ((i + d) % v) as u32).collect())
            .collect();

        let keywords: Vec<Vec<usize>> = vec![
            vec![0, 1],    // keyword topic
            vec![5, 6, 7], // keyword topic
            vec![],        // regular
        ];

        let mut r1 = ChaCha8Rng::seed_from_u64(77);
        let mut r2 = ChaCha8Rng::seed_from_u64(77);

        let m1 = fit_keyatm(&docs, v, 3, &keywords, 0.1, 0.1, 0.5, 1.0, 1.0, 40, WeightScheme::None, 1, &mut r1);
        let m2 = fit_keyatm(&docs, v, 3, &keywords, 0.1, 0.1, 0.5, 1.0, 1.0, 40, WeightScheme::None, 1, &mut r2);

        assert_eq!(
            m1.topic_word_all(),
            m2.topic_word_all(),
            "topic_word_all() differs between two identical-seed runs"
        );
        assert_eq!(
            m1.doc_topic(),
            m2.doc_topic(),
            "doc_topic() differs between two identical-seed runs"
        );
    }
}
