//! Collapsed Variational Bayes, zeroth-order (CVB0) inference for LDA
//! (Asuncion, Welling, Smyth & Teh, *UAI* 2009, "On Smoothing and Inference for
//! Topic Models").
//!
//! A deterministic alternative to collapsed Gibbs. Instead of a hard topic per
//! token, each *(document, word-type)* cell keeps a soft responsibility vector
//! `γ` over topics, updated from **expected** counts:
//!
//! ```text
//! γ_{d,w,k} ∝ (E[n_dk]^{-dw} + α_k) · (E[n_wk]^{-dw} + β) / (E[n_k]^{-dw} + β̄)
//! ```
//!
//! the same factored conditional as CGS, mean-field instead of sampled. So it is
//! reproducible (no seed dependence beyond the initial γ), has no burn-in, and
//! tends to match or beat CGS on held-out perplexity. The cost is `O(K)` per
//! cell (γ is dense), so CVB0 does **not** have the sparse samplers' large-K
//! speed; it competes on quality and determinism.
//!
//! Memory is kept to `O(#unique (doc,word) cells · K)` by sharing one γ across
//! all tokens of the same word-type in a document (the standard efficient CVB0;
//! those tokens have an identical conditional), rather than one γ per token.

use rand::Rng;

use crate::corpus::Corpus;
use crate::model::TopicModel;

/// A CVB0 inference engine over per-(document, word-type) responsibilities.
pub struct Cvb0 {
    pub num_topics: usize,
    pub num_types: usize,
    pub alpha: Vec<f64>,
    pub alpha_sum: f64,
    pub beta: f64,
    pub beta_sum: f64,

    /// `cells[d]` = the unique `(word_id, count)` pairs in document `d`.
    cells: Vec<Vec<(u32, u32)>>,
    /// `gamma[d][i]` = the length-`K` responsibility vector for cell `i` of doc `d`.
    gamma: Vec<Vec<Vec<f64>>>,

    /// Expected document-topic counts `E[n_dk]`.
    n_dk: Vec<Vec<f64>>,
    /// Expected word-topic counts `E[n_wk]`.
    n_wk: Vec<Vec<f64>>,
    /// Expected topic counts `E[n_k]`.
    n_k: Vec<f64>,
    /// Document lengths (token counts), for θ.
    doc_len: Vec<f64>,
}

impl Cvb0 {
    /// Initialize from `corpus` with a random γ per cell (the only stochastic
    /// step; everything after is deterministic).
    pub fn new<R: Rng>(
        corpus: &Corpus,
        num_topics: usize,
        alpha: &[f64],
        beta: f64,
        rng: &mut R,
    ) -> Cvb0 {
        let num_types = corpus.num_types();
        let beta_sum = beta * num_types as f64;
        let alpha_sum: f64 = alpha.iter().sum();
        let k = num_topics;

        let mut n_dk = vec![vec![0.0f64; k]; corpus.docs.len()];
        let mut n_wk = vec![vec![0.0f64; k]; num_types];
        let mut n_k = vec![0.0f64; k];
        let mut cells: Vec<Vec<(u32, u32)>> = Vec::with_capacity(corpus.docs.len());
        let mut gamma: Vec<Vec<Vec<f64>>> = Vec::with_capacity(corpus.docs.len());
        let mut doc_len = vec![0.0f64; corpus.docs.len()];

        for (d, doc) in corpus.docs.iter().enumerate() {
            // Collapse the document to unique (word, count) cells.
            let mut counts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
            for &w in doc {
                *counts.entry(w).or_insert(0) += 1;
            }
            doc_len[d] = doc.len() as f64;
            let mut dcells: Vec<(u32, u32)> = counts.into_iter().collect();
            dcells.sort_unstable(); // deterministic cell order
            let mut dgamma: Vec<Vec<f64>> = Vec::with_capacity(dcells.len());

            for &(w, c) in &dcells {
                // Random initial responsibilities, normalized.
                let mut g = vec![0.0f64; k];
                let mut s = 0.0;
                for x in g.iter_mut() {
                    *x = rng.gen::<f64>() + 1e-6;
                    s += *x;
                }
                let cf = c as f64;
                for (t, x) in g.iter_mut().enumerate() {
                    *x /= s;
                    n_dk[d][t] += cf * *x;
                    n_wk[w as usize][t] += cf * *x;
                    n_k[t] += cf * *x;
                }
                dgamma.push(g);
            }
            cells.push(dcells);
            gamma.push(dgamma);
        }

        Cvb0 {
            num_topics: k,
            num_types,
            alpha: alpha.to_vec(),
            alpha_sum,
            beta,
            beta_sum,
            cells,
            gamma,
            n_dk,
            n_wk,
            n_k,
            doc_len,
        }
    }

    /// One deterministic CVB0 sweep over every cell. Returns the mean absolute
    /// change in γ across all cells, a convergence signal (→ 0 as the
    /// responsibilities settle), so the caller can early-stop.
    pub fn sweep(&mut self) -> f64 {
        let k = self.num_topics;
        let beta = self.beta;
        let beta_sum = self.beta_sum;
        let uniform = 1.0 / k as f64;
        let mut total_change = 0.0f64;
        let mut n_cells = 0usize;
        let mut old = vec![0.0f64; k];

        for d in 0..self.cells.len() {
            for i in 0..self.cells[d].len() {
                let (w, c) = self.cells[d][i];
                let (w, cf) = (w as usize, c as f64);

                // Snapshot the old γ and remove this cell's mass (`-dw` counts).
                for t in 0..k {
                    let g = self.gamma[d][i][t];
                    old[t] = g;
                    self.n_dk[d][t] -= cf * g;
                    self.n_wk[w][t] -= cf * g;
                    self.n_k[t] -= cf * g;
                }

                // Recompute γ ∝ (n_dk+α)(n_wk+β)/(n_k+β̄).
                let mut sum = 0.0f64;
                for t in 0..k {
                    let val = (self.n_dk[d][t] + self.alpha[t]) * (self.n_wk[w][t] + beta)
                        / (self.n_k[t] + beta_sum);
                    self.gamma[d][i][t] = if val > 0.0 { val } else { 0.0 };
                    sum += self.gamma[d][i][t];
                }

                // Normalize, accumulate |Δγ|, and add the new mass back.
                for t in 0..k {
                    let new = if sum > 0.0 { self.gamma[d][i][t] / sum } else { uniform };
                    self.gamma[d][i][t] = new;
                    total_change += (new - old[t]).abs();
                    self.n_dk[d][t] += cf * new;
                    self.n_wk[w][t] += cf * new;
                    self.n_k[t] += cf * new;
                }
                n_cells += 1;
            }
        }
        if n_cells == 0 {
            0.0
        } else {
            total_change / n_cells as f64
        }
    }

    /// Smoothed topic-word φ `(E[n_wk]+β)/(E[n_k]+β̄)`, accumulated into
    /// `acc[word][topic]` (matches the MH samplers' orientation).
    pub fn phi_into(&self, acc: &mut [Vec<f64>]) {
        for w in 0..self.num_types {
            for t in 0..self.num_topics {
                let denom = self.n_k[t] + self.beta_sum;
                acc[w][t] += (self.n_wk[w][t] + self.beta) / denom;
            }
        }
    }

    /// Smoothed doc-topic θ `(E[n_dk]+α)/(n_d+α_sum)`, into `acc[doc][topic]`.
    pub fn theta_into(&self, acc: &mut [Vec<f64>]) {
        for d in 0..self.cells.len() {
            let denom = self.doc_len[d] + self.alpha_sum;
            for t in 0..self.num_topics {
                acc[d][t] += (self.n_dk[d][t] + self.alpha[t]) / denom;
            }
        }
    }

    /// Pack into a [`TopicModel`] via the MAP hard assignment (argmax γ per
    /// token) so the rest of the codebase (coherence, save/load, held-out
    /// inference) is reused. The φ/θ point estimates above are the soft CVB0
    /// solution and are what `fit` returns; this backs the archival/diagnostic
    /// machinery that expects integer counts.
    pub fn to_topic_model(&self, corpus: &Corpus) -> TopicModel {
        let mut model =
            TopicModel::new(self.num_topics, self.alpha_sum, self.beta, self.num_types);
        model.alpha.copy_from_slice(&self.alpha);
        model.alpha_sum = self.alpha_sum;
        model.beta = self.beta;
        model.beta_sum = self.beta_sum;

        // Per-(doc,word) argmax topic; assign every token of that cell to it.
        let doc_topics: Vec<Vec<u32>> = corpus
            .docs
            .iter()
            .enumerate()
            .map(|(d, doc)| {
                // Map word -> argmax topic for this doc's cells.
                let mut best: std::collections::HashMap<u32, u32> =
                    std::collections::HashMap::new();
                for (i, &(w, _)) in self.cells[d].iter().enumerate() {
                    let g = &self.gamma[d][i];
                    let mut bt = 0usize;
                    for t in 1..self.num_topics {
                        if g[t] > g[bt] {
                            bt = t;
                        }
                    }
                    best.insert(w, bt as u32);
                }
                doc.iter().map(|&w| best[&w]).collect()
            })
            .collect();
        model.initialize_from_assignments(corpus, doc_topics);
        model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus::Corpus;
    use rand::SeedableRng;
    use rand_pcg::Pcg64Mcg;

    fn planted(n_blocks: usize, wpb: usize, n_docs: usize) -> (Corpus, usize) {
        let v = n_blocks * wpb;
        let docs: Vec<Vec<u32>> = (0..n_docs)
            .map(|d| {
                let b = d % n_blocks;
                let block: Vec<u32> = (b * wpb..(b + 1) * wpb).map(|w| w as u32).collect();
                let mut doc = block.clone();
                doc.extend(block);
                doc
            })
            .collect();
        let corpus = Corpus {
            id_to_word: (0..v).map(|i| format!("w{i}")).collect(),
            doc_names: (0..n_docs).map(|i| format!("d{i}")).collect(),
            doc_labels: vec![String::new(); n_docs],
            doc_freqs: vec![0u32; v],
            total_freqs: vec![0u32; v],
            docs,
        };
        (corpus, n_blocks)
    }

    #[test]
    fn recovers_planted_blocks() {
        let wpb = 5;
        let (corpus, n_blocks) = planted(4, wpb, 200);
        let k = n_blocks;
        let v = corpus.num_types();
        let alpha = vec![0.1f64; k];
        let mut rng = Pcg64Mcg::seed_from_u64(1);
        let mut m = Cvb0::new(&corpus, k, &alpha, 0.01, &mut rng);
        for _ in 0..100 {
            m.sweep();
        }
        let mut phi = vec![vec![0.0f64; k]; v]; // [word][topic]
        m.phi_into(&mut phi);
        let mut covered = std::collections::HashSet::new();
        for t in 0..k {
            let mut idx: Vec<usize> = (0..v).collect();
            idx.sort_by(|&a, &b| phi[b][t].partial_cmp(&phi[a][t]).unwrap());
            let top: std::collections::HashSet<usize> = idx[..wpb].iter().copied().collect();
            for b in 0..n_blocks {
                let block: std::collections::HashSet<usize> = (b * wpb..(b + 1) * wpb).collect();
                if block.is_subset(&top) {
                    covered.insert(b);
                }
            }
        }
        assert_eq!(covered.len(), n_blocks, "only recovered {covered:?}");
    }

    #[test]
    fn deterministic_for_seed() {
        let (corpus, _) = planted(3, 4, 90);
        let alpha = vec![0.1f64; 3];
        let run = || {
            let mut rng = Pcg64Mcg::seed_from_u64(7);
            let mut m = Cvb0::new(&corpus, 3, &alpha, 0.01, &mut rng);
            for _ in 0..40 {
                m.sweep();
            }
            let mut th = vec![vec![0.0f64; 3]; corpus.docs.len()];
            m.theta_into(&mut th);
            th
        };
        let a = run();
        let b = run();
        for (ra, rb) in a.iter().zip(b.iter()) {
            for (x, y) in ra.iter().zip(rb.iter()) {
                assert!((x - y).abs() < 1e-12);
            }
        }
    }
}
