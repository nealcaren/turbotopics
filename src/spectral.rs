//! Spectral (anchor-word) initialization for the variational models — the
//! deterministic topic-word initializer used by STM's default `init.type =
//! "Spectral"`. Based on Arora et al. (2013), "A Practical Algorithm for Topic
//! Modeling with Provable Guarantees".
//!
//! The idea: build the word-word co-occurrence matrix Q, find K "anchor words"
//! (words that are near-exclusive to one topic) by greedy farthest-point
//! selection, then recover each word's topic loadings as a non-negative convex
//! combination of the anchors. The result is a deterministic β init that gives
//! stable, reproducible topic solutions (no random seed).
//!
//! The dense V×V co-occurrence makes this best for moderate vocabularies (after
//! pruning); callers fall back to random init when it is unavailable.

use std::collections::HashMap;

/// Build the row-normalized co-occurrence matrix `Q̄` (V×V) and the word
/// marginals `p`. Returns `None` if the corpus is too small/degenerate.
fn cooccurrence(docs: &[Vec<u32>], v: usize) -> Option<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut q = vec![vec![0.0f64; v]; v];
    let mut used_docs = 0usize;
    for doc in docs {
        let n = doc.len();
        if n < 2 {
            continue;
        }
        used_docs += 1;
        let mut wc: HashMap<usize, f64> = HashMap::new();
        for &w in doc {
            *wc.entry(w as usize).or_insert(0.0) += 1.0;
        }
        let words: Vec<(usize, f64)> = wc.into_iter().collect();
        let norm = 1.0 / (n as f64 * (n as f64 - 1.0));
        for &(wi, ci) in &words {
            for &(wj, cj) in &words {
                let val = if wi == wj { ci * (ci - 1.0) } else { ci * cj };
                q[wi][wj] += val * norm;
            }
        }
    }
    if used_docs == 0 {
        return None;
    }
    for row in &mut q {
        for x in row.iter_mut() {
            *x /= used_docs as f64;
        }
    }
    let p: Vec<f64> = q.iter().map(|r| r.iter().sum()).collect();
    // Row-normalize → Q̄.
    let mut qbar = q;
    for row in qbar.iter_mut() {
        let s: f64 = row.iter().sum();
        if s > 0.0 {
            for x in row.iter_mut() {
                *x /= s;
            }
        }
    }
    Some((qbar, p))
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Greedy farthest-point anchor selection (Gram-Schmidt residuals). Only words
/// with marginal `p_w >= min_p` are candidates (avoids rare-word anchors).
fn fast_anchor_words(qbar: &[Vec<f64>], p: &[f64], k: usize, min_p: f64) -> Option<Vec<usize>> {
    let v = qbar.len();
    let candidates: Vec<usize> = (0..v).filter(|&w| p[w] >= min_p).collect();
    if candidates.len() < k {
        return None;
    }

    // First anchor: candidate with the largest row norm.
    let mut anchors = Vec::with_capacity(k);
    let first = *candidates
        .iter()
        .max_by(|&&a, &&b| dot(&qbar[a], &qbar[a]).partial_cmp(&dot(&qbar[b], &qbar[b])).unwrap())
        .unwrap();
    anchors.push(first);

    // Orthonormal basis of selected anchor rows; track each candidate's residual.
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut residual: Vec<Vec<f64>> = candidates.iter().map(|&w| qbar[w].clone()).collect();

    let add_basis = |basis: &mut Vec<Vec<f64>>, residual: &mut Vec<Vec<f64>>, anchor_pos: usize| {
        // Gram-Schmidt: orthonormalize the anchor's residual, then project it out.
        let mut b = residual[anchor_pos].clone();
        let norm = dot(&b, &b).sqrt();
        if norm > 1e-12 {
            for x in b.iter_mut() {
                *x /= norm;
            }
            for r in residual.iter_mut() {
                let proj = dot(r, &b);
                for (ri, bi) in r.iter_mut().zip(&b) {
                    *ri -= proj * bi;
                }
            }
            basis.push(b);
        }
    };
    let first_pos = candidates.iter().position(|&w| w == first).unwrap();
    add_basis(&mut basis, &mut residual, first_pos);

    while anchors.len() < k {
        // Pick the candidate with the largest residual norm.
        let (best_pos, _) = residual
            .iter()
            .enumerate()
            .filter(|(pos, _)| !anchors.contains(&candidates[*pos]))
            .map(|(pos, r)| (pos, dot(r, r)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())?;
        if dot(&residual[best_pos], &residual[best_pos]) < 1e-12 {
            return None; // can't find K independent anchors
        }
        anchors.push(candidates[best_pos]);
        add_basis(&mut basis, &mut residual, best_pos);
    }
    Some(anchors)
}

/// Recover the K×V topic-word matrix from the anchors (Arora "recoverL2" via
/// exponentiated gradient): each word's topic loadings are the non-negative,
/// sum-to-one weights best reconstructing its Q̄ row from the anchor rows.
fn recover(qbar: &[Vec<f64>], p: &[f64], anchors: &[usize], k: usize, v: usize) -> Vec<Vec<f64>> {
    // Anchor Gram matrix G (K×K) and, per word, b_k = <anchor_k, word>.
    let anchor_rows: Vec<&Vec<f64>> = anchors.iter().map(|&a| &qbar[a]).collect();
    let mut g = vec![vec![0.0f64; k]; k];
    for a in 0..k {
        for b in 0..k {
            g[a][b] = dot(anchor_rows[a], anchor_rows[b]);
        }
    }

    // A[word][topic] = C[word][topic] * p[word]; later normalized per topic.
    let mut a_mat = vec![vec![0.0f64; k]; v];
    for w in 0..v {
        let bvec: Vec<f64> = (0..k).map(|t| dot(anchor_rows[t], &qbar[w])).collect();
        // Exponentiated gradient on the simplex: min Cᵀ G C − 2 Cᵀ b.
        let mut c = vec![1.0 / k as f64; k];
        let eta = 50.0;
        for _ in 0..120 {
            let gc: Vec<f64> = (0..k).map(|i| (0..k).map(|j| g[i][j] * c[j]).sum()).collect();
            // grad_i = 2(gc_i − b_i)
            let mut total = 0.0;
            for i in 0..k {
                c[i] *= (-eta * 2.0 * (gc[i] - bvec[i])).exp();
                if !c[i].is_finite() {
                    c[i] = 0.0;
                }
                total += c[i];
            }
            if total > 0.0 {
                for ci in c.iter_mut() {
                    *ci /= total;
                }
            } else {
                c = vec![1.0 / k as f64; k];
            }
        }
        for t in 0..k {
            a_mat[w][t] = c[t] * p[w];
        }
    }

    // β_{k,w} = A[w][k] normalized over words (+ tiny smoothing).
    let mut beta = vec![vec![1e-8f64; v]; k];
    for t in 0..k {
        let mut col = 0.0;
        for w in 0..v {
            col += a_mat[w][t];
        }
        if col <= 0.0 {
            // Degenerate topic — uniform.
            for w in 0..v {
                beta[t][w] = 1.0 / v as f64;
            }
            continue;
        }
        for w in 0..v {
            beta[t][w] = (a_mat[w][t] + 1e-8) / (col + v as f64 * 1e-8);
        }
    }
    beta
}

/// Deterministic anchor-word initialization of the K×V topic-word matrix.
/// Returns `None` when it is not applicable (corpus too small, fewer candidate
/// words than topics, or degenerate co-occurrence) so the caller can fall back.
pub fn spectral_init(docs: &[Vec<u32>], k: usize, v: usize) -> Option<Vec<Vec<f64>>> {
    if v < k {
        return None;
    }
    let (qbar, p) = cooccurrence(docs, v)?;
    // Require anchors to have at least a small marginal mass.
    let min_p = 0.0;
    let anchors = fast_anchor_words(&qbar, &p, k, min_p)?;
    Some(recover(&qbar, &p, &anchors, k, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_planted_topics() {
        // Three disjoint-vocabulary topics; each doc draws from one. Spectral
        // init should put each topic's mass on one vocabulary block.
        let blocks = [[0u32, 1, 2], [3, 4, 5], [6, 7, 8]];
        let mut docs = Vec::new();
        for i in 0..150 {
            let b = blocks[i % 3];
            docs.push(vec![b[0], b[1], b[2], b[0], b[1], b[2]]);
        }
        let beta = spectral_init(&docs, 3, 9).expect("spectral init");
        // Each topic concentrates on one block (its top-3 words are one block).
        for t in 0..3 {
            let mut idx: Vec<usize> = (0..9).collect();
            idx.sort_by(|&a, &b| beta[t][b].partial_cmp(&beta[t][a]).unwrap());
            let top: std::collections::HashSet<usize> = idx[..3].iter().copied().collect();
            let is_block = blocks
                .iter()
                .any(|blk| blk.iter().all(|&w| top.contains(&(w as usize))));
            assert!(is_block, "topic {} top words not a single block: {:?}", t, &idx[..3]);
        }
    }
}
