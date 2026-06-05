//! Density clustering for the embedding-model branch (Top2Vec, BERTopic).
//!
//! A thin wrapper over `petal-clustering`'s HDBSCAN so the rest of topica keeps
//! working in plain `Vec<Vec<f64>>` and never sees `ndarray` directly. HDBSCAN
//! is the clustering stage both Top2Vec and BERTopic run after reducing the
//! document embeddings: it finds clusters of varying density and leaves sparse
//! points unassigned (the "outlier" topic, conventionally label `-1`).

use ndarray::Array2;
use petal_clustering::{Fit, HDbscan};

/// Cluster `points` (row-major; each row is one embedding vector) with HDBSCAN.
///
/// Returns one label per point: cluster ids are a dense `0..n_clusters` range
/// (assigned in a deterministic order), and noise points get `-1`, matching the
/// HDBSCAN / BERTopic outlier convention. `min_cluster_size` is the smallest
/// group that counts as a topic; `min_samples` controls how conservative the
/// density estimate is (larger = more points called noise).
pub fn hdbscan_labels(points: &[Vec<f64>], min_cluster_size: usize, min_samples: usize) -> Vec<i64> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    let dim = points[0].len();
    assert!(
        points.iter().all(|r| r.len() == dim),
        "all points must share the same dimensionality"
    );
    let flat: Vec<f64> = points.iter().flat_map(|r| r.iter().copied()).collect();
    let array = Array2::from_shape_vec((n, dim), flat).expect("point matrix shape");

    let mut model: HDbscan<f64, _> = HDbscan::default();
    model.min_cluster_size = min_cluster_size.max(2);
    model.min_samples = min_samples.max(1);
    let (clusters, _outliers, _scores) = model.fit(&array, None);

    // Map petal's arbitrary cluster keys to a dense, deterministic 0..k.
    let mut ids: Vec<usize> = clusters.keys().copied().collect();
    ids.sort_unstable();
    let mut labels = vec![-1i64; n];
    for (new_id, old_id) in ids.into_iter().enumerate() {
        for &idx in &clusters[&old_id] {
            labels[idx] = new_id as i64;
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    // Two tight, well-separated blobs should fall into two distinct clusters.
    // HDBSCAN may legitimately call a few border points noise, so we assert on
    // each blob's majority label rather than demanding every point be assigned.
    #[test]
    fn separates_two_blobs() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut pts = Vec::new();
        for _ in 0..30 {
            pts.push(vec![rng.gen::<f64>() * 0.3, rng.gen::<f64>() * 0.3]); // near origin
        }
        for _ in 0..30 {
            pts.push(vec![5.0 + rng.gen::<f64>() * 0.3, 5.0 + rng.gen::<f64>() * 0.3]); // near (5,5)
        }
        let labels = hdbscan_labels(&pts, 5, 2);

        let majority = |slice: &[i64]| {
            let mut counts = std::collections::HashMap::new();
            for &l in slice {
                if l >= 0 {
                    *counts.entry(l).or_insert(0) += 1;
                }
            }
            counts.into_iter().max_by_key(|&(_, c)| c).map(|(l, c)| (l, c))
        };
        let (a, na) = majority(&labels[..30]).expect("blob 1 has a cluster");
        let (b, nb) = majority(&labels[30..]).expect("blob 2 has a cluster");
        assert!(a != b, "blobs share a label: {labels:?}");
        assert!(na >= 24 && nb >= 24, "blobs too fragmented: {labels:?}");
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(hdbscan_labels(&[], 4, 2).is_empty());
    }
}
