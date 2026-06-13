//! A small backend trait shared by the Metropolis-Hastings LDA samplers
//! (LightLDA, WarpLDA). Each owns its own dense sampling state and packs back
//! into a [`TopicModel`] when training finishes, so a single generic `fit` loop
//! can drive any of them. Adding a new MH sampler — or reusing one of these in
//! another count-based model — is then a matter of implementing this trait and
//! calling the shared runner, rather than copying the training loop.
//!
//! The default SparseLDA sampler is deliberately *not* behind this trait: it
//! mutates a [`TopicModel`] in place, supports the convergence-tol trace and the
//! document-partitioned parallel sweep, and carries the CLI byte-parity
//! guarantee (the Python binding matches the bundled `train` CLI byte-for-byte),
//! so it keeps its dedicated path.

use rand_pcg::Pcg64Mcg;

use crate::corpus::Corpus;
use crate::model::TopicModel;

/// An MH LDA sampler driven by the shared training loop. Methods mirror the
/// inherent ones on [`crate::lightlda::LightLda`] / [`crate::warplda::WarpLda`];
/// the trait fixes the RNG to `Pcg64Mcg` (the type the Python `LDA` path uses)
/// so it stays object-simple and monomorphizes cleanly.
pub trait MhSampler {
    /// One full sampling sweep over every token.
    fn sweep(&mut self, corpus: &Corpus, rng: &mut Pcg64Mcg);
    /// Replace the hyperparameters after an optimization step.
    fn set_hyper(&mut self, alpha: &[f64], beta: f64);
    /// Accumulate a smoothed φ snapshot into `acc[word][topic]`.
    fn phi_into(&self, acc: &mut [Vec<f64>]);
    /// Accumulate a smoothed θ snapshot into `acc[doc][topic]`.
    fn theta_into(&self, corpus: &Corpus, acc: &mut [Vec<f64>]);
    /// Pack the sampler state into a [`TopicModel`] for the shared downstream
    /// machinery (optimization, φ/θ, save/load, held-out inference).
    fn to_topic_model(&self) -> TopicModel;
}
