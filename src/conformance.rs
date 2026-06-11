//! Rust-side estimator conformance — the analog of `python/topica/conformance.py`.
//! Catches contract gaps in `cargo test --lib`, at the source, before the Python
//! layer or a release. The full multi-model registry and EXEMPT set are populated
//! in a later step; this file establishes the check.

use crate::estimator::{Estimator, ModelFamily, DirichletModel};
use crate::variational::LogisticNormalModel;

/// Check a fitted model against the Tier-0 contract and its family's Tier-2
/// requirements. Returns a list of violation messages (empty = conforming).
///
/// Note: logistic-normal Tier-2 (`eta_mean`/`eta_cov`) is checked through the
/// `LogisticNormalModel` trait by the typed helper [`check_logistic_normal`],
/// since `&dyn Estimator` cannot be downcast to it; a registered logistic-normal
/// model is wired through that helper in the conformance test.
#[allow(dead_code)]
pub fn check_conformance(m: &dyn Estimator) -> Vec<String> {
    let mut v = Vec::new();
    let k = m.num_topics();
    let tw = m.topic_word();
    if tw.len() != k {
        v.push(format!("topic_word has {} rows but num_topics is {}", tw.len(), k));
    }
    if matches!(m.model_family(), ModelFamily::Dirichlet | ModelFamily::LogisticNormal) {
        let theta = m.doc_topic();
        for (d, row) in theta.iter().enumerate() {
            let s: f64 = row.iter().sum();
            if (s - 1.0).abs() > 1e-6 {
                v.push(format!("doc_topic row {d} sums to {s}, expected 1"));
                break;
            }
        }
    }
    v
}

/// Tier-2 shape check for Dirichlet (collapsed-Gibbs) models: alpha length,
/// doc_lengths length, and theta_draws nesting.
#[allow(dead_code)]
pub fn check_dirichlet(m: &dyn DirichletModel) -> Vec<String> {
    let mut v = Vec::new();
    let k = m.num_topics();
    let d = m.doc_topic().len();
    if m.alpha().len() != k {
        v.push(format!("alpha has len {} but num_topics is {}", m.alpha().len(), k));
    }
    if m.doc_lengths().len() != d {
        v.push(format!("doc_lengths has len {} but doc_topic has {} rows", m.doc_lengths().len(), d));
    }
    let draws = m.theta_draws();
    if !draws.is_empty() {
        for (s, draw) in draws.iter().enumerate() {
            if draw.len() != d {
                v.push(format!("theta_draws[{s}] has {} rows but doc_topic has {d}", draw.len()));
                break;
            }
            for (di, row) in draw.iter().enumerate() {
                if row.len() != k {
                    v.push(format!("theta_draws[{s}][{di}] has len {} but num_topics is {k}", row.len()));
                    break;
                }
            }
        }
    }
    v
}

/// Tier-2 shape check for logistic-normal models: eta_mean is (D, eta_dim) and
/// eta_cov is (D, eta_dim²) with a consistent eta_dim.
#[allow(dead_code)]
pub fn check_logistic_normal(m: &dyn LogisticNormalModel) -> Vec<String> {
    let mut v = Vec::new();
    let dim = m.eta_dim();
    for (d, row) in m.eta_mean().iter().enumerate() {
        if row.len() != dim {
            v.push(format!("eta_mean row {d} has len {} != eta_dim {dim}", row.len()));
            break;
        }
    }
    for (d, row) in m.eta_cov().iter().enumerate() {
        if row.len() != dim * dim {
            v.push(format!("eta_cov row {d} has len {} != eta_dim² {}", row.len(), dim * dim));
            break;
        }
    }
    v
}
