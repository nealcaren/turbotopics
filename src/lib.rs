pub mod coherence;
pub mod corpus;
pub mod ctm;
pub mod dmr;
pub mod dtm;
pub mod gsdmm;
pub mod hdp;
pub mod hlda;
pub mod keyatm;
pub mod labeled;
pub mod lightlda;
pub mod linalg;
pub mod model;
pub mod optimize;
pub mod output;
pub mod pa;
pub mod pt;
pub mod sage;
pub mod sampler;
pub mod seeded;
pub mod slda;
pub mod spectral;

// Embedding-native model branch (Top2Vec/BERTopic/...): clustering pipeline over
// user-supplied embeddings. Behind the `embeddings` feature (implied by `python`).
// reduce -> cluster -> represent are the three pipeline stages.
#[cfg(feature = "embeddings")]
pub mod cluster;
#[cfg(feature = "embeddings")]
pub mod reduce;
#[cfg(feature = "embeddings")]
pub mod represent;
#[cfg(feature = "embeddings")]
pub mod top2vec;

#[cfg(feature = "python")]
mod python;
