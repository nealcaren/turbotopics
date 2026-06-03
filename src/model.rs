use rand::Rng;
use crate::corpus::Corpus;

#[derive(Clone)]
pub struct TopicModel {
    pub num_topics: usize,
    pub num_types: usize,
    // Bitmask and shift for packing (count, topic) into a u32.
    // Entry = (count << topic_bits) | topic
    // Sorted descending by value (so descending by count), zeroes terminate.
    pub topic_mask: u32,
    pub topic_bits: u32,

    pub alpha: Vec<f64>,
    pub alpha_sum: f64,
    pub beta: f64,
    pub beta_sum: f64,

    // For each word type: sorted Vec<u32> of packed (count, topic) pairs.
    pub type_topic_counts: Vec<Vec<u32>>,
    // Total tokens assigned to each topic across all documents.
    pub tokens_per_topic: Vec<u32>,
    // Per-document topic assignments, parallel to corpus.docs.
    pub doc_topics: Vec<Vec<u32>>,
}

impl TopicModel {
    pub fn new(num_topics: usize, alpha_sum: f64, beta: f64, num_types: usize) -> Self {
        // Compute the topic_mask and topic_bits.
        // We need enough bits to represent any topic index.
        // MALLET uses: if power-of-2, mask = num_topics-1; else next higher power-of-2 mask.
        let (topic_mask, topic_bits) = if num_topics.is_power_of_two() {
            let mask = (num_topics - 1) as u32;
            let bits = mask.count_ones();
            (mask, bits)
        } else {
            let mask = (num_topics as u32).next_power_of_two() - 1;
            let bits = mask.count_ones();
            (mask, bits)
        };

        let alpha = vec![alpha_sum / num_topics as f64; num_topics];
        let beta_sum = beta * num_types as f64;

        TopicModel {
            num_topics,
            num_types,
            topic_mask,
            topic_bits,
            alpha,
            alpha_sum,
            beta,
            beta_sum,
            type_topic_counts: Vec::new(),
            tokens_per_topic: vec![0; num_topics],
            doc_topics: Vec::new(),
        }
    }

    pub fn initialize<R: Rng>(&mut self, corpus: &Corpus, rng: &mut R) {
        // Count total occurrences of each word type so we can size the inner vecs.
        let mut type_totals = vec![0usize; self.num_types];
        for doc in &corpus.docs {
            for &word_id in doc {
                type_totals[word_id as usize] += 1;
            }
        }

        // Allocate type_topic_counts: size = min(num_topics, type_total) per word.
        self.type_topic_counts = type_totals
            .iter()
            .map(|&total| vec![0u32; self.num_topics.min(total)])
            .collect();

        // Random initial topic assignments.
        self.doc_topics = corpus
            .docs
            .iter()
            .map(|doc| {
                doc.iter()
                    .map(|_| rng.gen_range(0..self.num_topics) as u32)
                    .collect()
            })
            .collect();

        // Build type_topic_counts and tokens_per_topic from initial assignments.
        // Collect (word_id, topic) pairs first to avoid borrow conflict.
        let assignments: Vec<(usize, usize)> = corpus
            .docs
            .iter()
            .zip(self.doc_topics.iter())
            .flat_map(|(doc, topics)| {
                doc.iter().map(|&w| w as usize).zip(topics.iter().map(|&t| t as usize))
            })
            .collect();

        for (word_id, topic) in assignments {
            self.tokens_per_topic[topic] += 1;
            self.increment_type_topic(word_id, topic);
        }
    }

    /// Increment the count for (word_id, topic) in type_topic_counts,
    /// maintaining descending sort.
    pub fn increment_type_topic(&mut self, word_id: usize, topic: usize) {
        let counts = &mut self.type_topic_counts[word_id];
        let topic = topic as u32;

        // Find the existing entry for this topic, or the first empty slot.
        let mut index = 0;
        while index < counts.len() && counts[index] > 0 {
            if counts[index] & self.topic_mask == topic {
                break;
            }
            index += 1;
        }

        if index == counts.len() {
            // This shouldn't happen if we sized correctly, but grow if needed.
            counts.push(0);
        }

        let current_count = counts[index] >> self.topic_bits;
        counts[index] = ((current_count + 1) << self.topic_bits) | topic;

        // Bubble up to maintain descending order.
        while index > 0 && counts[index] > counts[index - 1] {
            counts.swap(index, index - 1);
            index -= 1;
        }
    }



    /// Decrement the count for (word_id, topic) in type_topic_counts,
    /// maintaining descending sort (mirror of `increment_type_topic`, used by
    /// the Labeled-LDA restricted sampler). No-op if the entry is absent.
    pub fn decrement_type_topic(&mut self, word_id: usize, topic: usize) {
        let counts = &mut self.type_topic_counts[word_id];
        let topic = topic as u32;

        let mut index = 0;
        while index < counts.len() && counts[index] > 0 {
            if counts[index] & self.topic_mask == topic {
                break;
            }
            index += 1;
        }
        if index >= counts.len() || counts[index] == 0 {
            return; // not present
        }

        let new_count = (counts[index] >> self.topic_bits) - 1;
        counts[index] = if new_count == 0 {
            0
        } else {
            (new_count << self.topic_bits) | topic
        };

        // Bubble down: a smaller value (or a zero) sinks past larger entries.
        while index + 1 < counts.len() && counts[index] < counts[index + 1] {
            counts.swap(index, index + 1);
            index += 1;
        }
    }

    /// Look up the count for (word_id, topic). Returns 0 if not present.
    pub fn get_type_topic_count(&self, word_id: usize, topic: usize) -> u32 {
        let counts = &self.type_topic_counts[word_id];
        let topic = topic as u32;
        for &entry in counts {
            if entry == 0 {
                break;
            }
            if entry & self.topic_mask == topic {
                return entry >> self.topic_bits;
            }
        }
        0
    }
}
