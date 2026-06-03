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

use std::collections::HashMap;

use rand::Rng;

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
    /// `ndk[d][k]` — doc-topic counts (D×K).
    pub ndk: Vec<Vec<u32>>,
    /// `nkw[k][w]` — regular (s=0) topic-word counts (K×V).
    pub nkw: Vec<Vec<u32>>,
    /// `nk0[k]` — total s=0 token count in topic k.
    pub nk0: Vec<u32>,
    /// `nkx[k][j]` — keyword (s=1) count for the j-th keyword of topic k.
    /// Length of inner vec = L_k; empty for non-keyword topics.
    pub nkx: Vec<Vec<u32>>,
    /// `nk1[k]` — total s=1 token count in topic k.
    pub nk1: Vec<u32>,
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

        let reg_denom = v as f64 * beta + self.nk0[k] as f64;

        let mut phi = (0..v)
            .map(|w| (beta + self.nkw[k][w] as f64) / reg_denom)
            .collect::<Vec<f64>>();

        if k < self.num_keyword_topics && self.l[k] > 0 {
            let pi_num = self.gamma1 + self.nk1[k] as f64;
            let pi_den = self.gamma1 + self.gamma2 + self.nk0[k] as f64 + self.nk1[k] as f64;
            let pi_k = pi_num / pi_den;
            let one_minus_pi = 1.0 - pi_k;

            let key_denom = self.l[k] as f64 * self.beta_key + self.nk1[k] as f64;
            let beta_key = self.beta_key;

            // Scale the regular component.
            for p in &mut phi {
                *p *= one_minus_pi;
            }
            // Add the keyword component.
            for (j, &kw) in self.keywords[k].iter().enumerate() {
                phi[kw] += pi_k * (beta_key + self.nkx[k][j] as f64) / key_denom;
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
    /// θ_{d,k} = (ndk[d][k] + α) / (N_d + K·α).
    pub fn doc_topic(&self) -> Vec<Vec<f64>> {
        let k = self.num_topics;
        let alpha = self.alpha;
        let k_alpha = k as f64 * alpha;

        self.ndk
            .iter()
            .map(|row| {
                let n_d: f64 = row.iter().map(|&c| c as f64).sum::<f64>() + k_alpha;
                row.iter().map(|&c| (c as f64 + alpha) / n_d).collect()
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
                    let num = self.gamma1 + self.nk1[k] as f64;
                    let den = self.gamma1
                        + self.gamma2
                        + self.nk0[k] as f64
                        + self.nk1[k] as f64;
                    num / den
                } else {
                    0.0
                }
            })
            .collect()
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

/// Per-topic precomputed keyword lookup structures.
struct KeywordIndex {
    /// `is_keyword[k]` maps word-id → index in keywords[k] (None if not a keyword).
    lookup: Vec<HashMap<usize, usize>>,
}

impl KeywordIndex {
    fn build(keywords: &[Vec<usize>]) -> Self {
        let lookup = keywords
            .iter()
            .map(|kws| {
                kws.iter()
                    .enumerate()
                    .map(|(j, &w)| (w, j))
                    .collect::<HashMap<usize, usize>>()
            })
            .collect();
        KeywordIndex { lookup }
    }

    /// Returns `Some(j)` if word `w` is the j-th keyword of topic `k`.
    #[inline]
    fn keyword_index(&self, k: usize, w: usize) -> Option<usize> {
        self.lookup[k].get(&w).copied()
    }
}

/// Resample the (z, s) assignment of a single token in doc `d` at position
/// `pos` in-place (collapsed Gibbs step).
///
/// Removes the token from its current counts, builds the full candidate
/// (k, s) weight vector, samples one state, and re-increments.
fn resample_token<R: Rng>(
    model: &mut KeyAtmModel,
    ki: &KeywordIndex,
    d: usize,
    w: usize,
    old_z: usize,
    old_s: u8,
    rng: &mut R,
) -> (usize, u8) {
    let k = model.num_topics;
    let v = model.num_types;
    let alpha = model.alpha;
    let beta = model.beta;
    let beta_key = model.beta_key;
    let gamma1 = model.gamma1;
    let gamma2 = model.gamma2;
    let num_kw = model.num_keyword_topics;

    // --- Remove token from current counts ---
    model.ndk[d][old_z] -= 1;
    if old_s == 0 {
        model.nkw[old_z][w] -= 1;
        model.nk0[old_z] -= 1;
    } else {
        // old_s == 1: find j for keyword index
        let j = ki.keyword_index(old_z, w).expect("old s=1 but w not a keyword");
        model.nkx[old_z][j] -= 1;
        model.nk1[old_z] -= 1;
    }

    // --- Build candidate weights ---
    // Upper bound: 2*K candidates (each topic has s=0; keyword topics that
    // contain w also have s=1). We collect into a small Vec of (k, s, weight).
    let v_beta = v as f64 * beta;

    // We'll store weights for each state in a flat vec:
    //   For k in 0..K: index 2*k   => (k, s=0)
    //                  index 2*k+1 => (k, s=1) — only non-zero if eligible
    let mut weights = vec![0.0f64; 2 * k];

    for kk in 0..k {
        let ndk_val = model.ndk[d][kk] as f64 + alpha;
        let nkw_val = model.nkw[kk][w] as f64 + beta;
        let nk0_val = model.nk0[kk] as f64;
        let nk1_val = model.nk1[kk] as f64;

        // s=0 state (always allowed for every topic).
        let reg_likelihood = nkw_val / (v_beta + nk0_val);
        let switch_factor_s0 = if kk < num_kw {
            (gamma2 + nk0_val) / (gamma1 + gamma2 + nk0_val + nk1_val)
        } else {
            1.0
        };
        weights[2 * kk] = ndk_val * reg_likelihood * switch_factor_s0;

        // s=1 state: only if kk is a keyword topic AND w is in keywords[kk].
        if kk < num_kw {
            if let Some(j) = ki.keyword_index(kk, w) {
                let lk = model.l[kk] as f64;
                let key_likelihood =
                    (model.nkx[kk][j] as f64 + beta_key) / (lk * beta_key + nk1_val);
                let switch_factor_s1 =
                    (gamma1 + nk1_val) / (gamma1 + gamma2 + nk0_val + nk1_val);
                weights[2 * kk + 1] = ndk_val * key_likelihood * switch_factor_s1;
            }
        }
    }

    // --- Categorical sample ---
    let chosen = sample_index(&weights, rng);
    let new_k = chosen / 2;
    let new_s = (chosen % 2) as u8;

    // --- Re-increment counts ---
    model.ndk[d][new_k] += 1;
    if new_s == 0 {
        model.nkw[new_k][w] += 1;
        model.nk0[new_k] += 1;
    } else {
        let j = ki.keyword_index(new_k, w).expect("new s=1 but w not a keyword");
        model.nkx[new_k][j] += 1;
        model.nk1[new_k] += 1;
    }

    (new_k, new_s)
}

/// One full Gibbs sweep over all tokens in all documents.
fn sweep<R: Rng>(
    model: &mut KeyAtmModel,
    docs: &[Vec<u32>],
    assignments: &mut Vec<Vec<(usize, u8)>>,
    ki: &KeywordIndex,
    rng: &mut R,
) {
    for d in 0..docs.len() {
        let doc = &docs[d];
        for pos in 0..doc.len() {
            let w = doc[pos] as usize;
            let (old_z, old_s) = assignments[d][pos];
            let (new_z, new_s) = resample_token(model, ki, d, w, old_z, old_s, rng);
            assignments[d][pos] = (new_z, new_s);
        }
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
    rng: &mut R,
) -> KeyAtmModel {
    assert_eq!(
        keywords.len(),
        num_topics,
        "keywords length must equal num_topics"
    );

    let d_count = docs.len();
    let v = num_types;
    let k = num_topics;

    // Derive derived counts.
    let l: Vec<usize> = keywords.iter().map(|kws| kws.len()).collect();
    let num_keyword_topics = keywords.iter().filter(|kws| !kws.is_empty()).count();
    // Keyword topics are the first `num_keyword_topics` entries; verify they
    // are all at the front (the contract from the caller).
    // We use `num_keyword_topics` purely as an upper bound for keyword-topic
    // checks: any topic k < num_keyword_topics with l[k]==0 is also "regular".

    // Build keyword index.
    let ki = KeywordIndex::build(keywords);

    // For each word, which topics hold it as a keyword (for seeded init).
    let mut word_keyword_topics: Vec<Vec<usize>> = vec![Vec::new(); v];
    for (kk, kws) in keywords.iter().enumerate() {
        for &w in kws {
            word_keyword_topics[w as usize].push(kk);
        }
    }

    // --- Initialisation ---
    // A token whose word is a keyword of some topic starts in that keyword topic
    // (random among ties) with the keyword switch on (s=1); this anchors keyword
    // topics to their keywords and bootstraps the keyword distribution. Other
    // tokens start in a uniformly random topic with s=0. Without this seeding a
    // keyword word would land in its keyword topic only 1/K of the time and the
    // switch would never engage.
    let mut ndk = vec![vec![0u32; k]; d_count];
    let mut nkw = vec![vec![0u32; v]; k];
    let mut nk0 = vec![0u32; k];
    let mut nkx: Vec<Vec<u32>> = keywords.iter().map(|kws| vec![0u32; kws.len()]).collect();
    let mut nk1 = vec![0u32; k];

    let mut assignments: Vec<Vec<(usize, u8)>> = docs
        .iter()
        .enumerate()
        .map(|(d, doc)| {
            doc.iter()
                .map(|&word| {
                    let w = word as usize;
                    let cands = &word_keyword_topics[w];
                    let (z, s): (usize, u8) = if cands.is_empty() {
                        ((rng.gen::<f64>() * k as f64) as usize % k, 0)
                    } else {
                        let z = cands[(rng.gen::<f64>() * cands.len() as f64) as usize % cands.len()];
                        (z, 1) // word is a keyword of z -> start with the switch on
                    };
                    ndk[d][z] += 1;
                    if s == 0 {
                        nkw[z][w] += 1;
                        nk0[z] += 1;
                    } else {
                        let j = ki.keyword_index(z, w).unwrap();
                        nkx[z][j] += 1;
                        nk1[z] += 1;
                    }
                    (z, s)
                })
                .collect()
        })
        .collect();

    let mut model = KeyAtmModel {
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
    };

    // --- Gibbs sweeps ---
    for _ in 0..iters {
        sweep(&mut model, docs, &mut assignments, &ki, rng);
    }

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
            &docs, v, 3, &keywords, 0.1, 0.1, 0.5, 1.0, 1.0, 200, &mut rng,
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
            &docs, v, 4, &keywords, 0.1, 0.1, 0.5, 1.0, 1.0, 50, &mut rng,
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
            &docs, v, 3, &keywords, 0.5, 0.1, 0.5, 1.0, 1.0, 30, &mut rng,
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

        let m1 = fit_keyatm(&docs, v, 3, &keywords, 0.1, 0.1, 0.5, 1.0, 1.0, 40, &mut r1);
        let m2 = fit_keyatm(&docs, v, 3, &keywords, 0.1, 0.1, 0.5, 1.0, 1.0, 40, &mut r2);

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
