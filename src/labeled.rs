//! Labeled LDA (Ramage et al., 2009).
//!
//! A supervised topic model: each document carries a set of labels, each label
//! corresponds to one topic, and a document's tokens may only be assigned to
//! the topics in that document's label set. Sampling is ordinary collapsed
//! Gibbs restricted to the allowed topics — because label sets are usually
//! small, iterating them directly is both correct and fast (no need for the
//! SparseLDA three-bucket machinery).
//!
//! Documents with an empty label set are treated as unconstrained (all topics
//! allowed), which lets a corpus mix labeled and unlabeled documents.

use rand::Rng;

use crate::model::TopicModel;

/// Initialize a model for Labeled LDA: each token gets a random topic drawn
/// from its document's allowed-topic set.
pub fn initialize_labeled<R: Rng>(
    model: &mut TopicModel,
    docs: &[Vec<u32>],
    allowed: &[Vec<usize>],
    rng: &mut R,
) {
    let num_types = model.num_types;
    let num_topics = model.num_topics;

    let mut type_totals = vec![0usize; num_types];
    for doc in docs {
        for &w in doc {
            type_totals[w as usize] += 1;
        }
    }
    model.type_topic_counts = type_totals
        .iter()
        .map(|&total| vec![0u32; num_topics.min(total)])
        .collect();
    model.tokens_per_topic = vec![0u32; num_topics];

    model.doc_topics = docs
        .iter()
        .enumerate()
        .map(|(d, doc)| {
            let allow = &allowed[d];
            doc.iter()
                .map(|_| {
                    let t = if allow.is_empty() {
                        rng.gen_range(0..num_topics)
                    } else {
                        allow[rng.gen_range(0..allow.len())]
                    };
                    t as u32
                })
                .collect()
        })
        .collect();

    for (d, doc) in docs.iter().enumerate() {
        for (pos, &w) in doc.iter().enumerate() {
            let t = model.doc_topics[d][pos] as usize;
            model.tokens_per_topic[t] += 1;
            model.increment_type_topic(w as usize, t);
        }
    }
}

/// One restricted Gibbs sweep: every token resamples only among the topics
/// allowed for its document.
pub fn run_sweep_labeled<R: Rng>(
    model: &mut TopicModel,
    docs: &[Vec<u32>],
    allowed: &[Vec<usize>],
    rng: &mut R,
) {
    let num_topics = model.num_topics;
    let beta = model.beta;
    let beta_sum = model.beta_sum;

    let all_topics: Vec<usize> = (0..num_topics).collect();
    let mut local = vec![0u32; num_topics];
    let mut scores: Vec<f64> = Vec::with_capacity(num_topics);

    for d in 0..docs.len() {
        let allow: &[usize] = if allowed[d].is_empty() {
            &all_topics
        } else {
            &allowed[d]
        };

        for t in local.iter_mut() {
            *t = 0;
        }
        for &t in &model.doc_topics[d] {
            local[t as usize] += 1;
        }

        let doc = &docs[d];
        for pos in 0..doc.len() {
            let w = doc[pos] as usize;
            let old = model.doc_topics[d][pos] as usize;

            model.decrement_type_topic(w, old);
            model.tokens_per_topic[old] -= 1;
            local[old] -= 1;

            scores.clear();
            let mut total = 0.0f64;
            for &t in allow {
                let n_wt = model.get_type_topic_count(w, t) as f64;
                let s = (local[t] as f64 + model.alpha[t]) * (n_wt + beta)
                    / (model.tokens_per_topic[t] as f64 + beta_sum);
                scores.push(s);
                total += s;
            }

            let mut r = rng.gen::<f64>() * total;
            let mut chosen = allow[allow.len() - 1];
            for (i, &t) in allow.iter().enumerate() {
                r -= scores[i];
                if r <= 0.0 {
                    chosen = t;
                    break;
                }
            }

            model.increment_type_topic(w, chosen);
            model.tokens_per_topic[chosen] += 1;
            local[chosen] += 1;
            model.doc_topics[d][pos] = chosen as u32;
        }
    }
}
