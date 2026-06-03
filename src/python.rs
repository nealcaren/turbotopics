//! PyO3 bindings: a Pythonic `LDA` + `Corpus` surface over the SparseLDA core.
//!
//! The compiled module is exposed to Python as `topica._topica`
//! (see pyproject.toml). A thin pure-Python package re-exports it.
//!
//! Design notes:
//!  * The heavy Gibbs sampling runs inside `Python::allow_threads`, so other
//!    Python threads keep running during training.
//!  * `LDA.fit` ports the averaging loop from `src/bin/train.rs` (the only
//!    pipeline logic that lived in the binary rather than the library), so the
//!    Python results match the `train` CLI exactly for a given seed.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use numpy::ndarray::{Array1, Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray2, ToPyArray};

use crate::dmr;
use crate::dtm;
use crate::gsdmm;
use crate::hdp;
use crate::keyatm;
use crate::seeded;
use crate::hlda;
use crate::pa;
use crate::pt;
use crate::slda;
use crate::labeled;
use crate::sage;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use regex::Regex;

use crate::corpus::{self, InputFormat, LoadOptions};
use crate::model::TopicModel;
use crate::{coherence as coh, ctm, lightlda, optimize, output, sampler};

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn io_err(e: std::io::Error) -> PyErr {
    PyIOError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Model serialization (save / load)
// ---------------------------------------------------------------------------

/// Serializable form of an ndarray `Array2` (shape + row-major data).
#[derive(serde::Serialize, serde::Deserialize)]
struct Arr2 {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}
/// Serializable form of an ndarray `Array3`.
#[derive(serde::Serialize, serde::Deserialize)]
struct Arr3 {
    d0: usize,
    d1: usize,
    d2: usize,
    data: Vec<f64>,
}

fn arr2_opt(a: &Option<Array2<f64>>) -> Option<Arr2> {
    a.as_ref().map(|m| Arr2 { rows: m.nrows(), cols: m.ncols(), data: m.iter().copied().collect() })
}
fn arr2_back(s: Option<Arr2>) -> Option<Array2<f64>> {
    s.map(|a| Array2::from_shape_vec((a.rows, a.cols), a.data).unwrap())
}
fn arr3_opt(a: &Option<Array3<f64>>) -> Option<Arr3> {
    a.as_ref().map(|m| {
        let d = m.dim();
        Arr3 { d0: d.0, d1: d.1, d2: d.2, data: m.iter().copied().collect() }
    })
}
fn arr3_back(s: Option<Arr3>) -> Option<Array3<f64>> {
    s.map(|a| Array3::from_shape_vec((a.d0, a.d1, a.d2), a.data).unwrap())
}
fn arr1_opt(a: &Option<Array1<f64>>) -> Option<Vec<f64>> {
    a.as_ref().map(|m| m.to_vec())
}
fn arr1_back(s: Option<Vec<f64>>) -> Option<Array1<f64>> {
    s.map(Array1::from)
}

fn write_state<S: serde::Serialize>(path: &str, state: &S) -> PyResult<()> {
    let bytes = bincode::serialize(state)
        .map_err(|e| PyValueError::new_err(format!("serialization failed: {e}")))?;
    std::fs::write(path, bytes).map_err(io_err)
}
fn read_state<S: serde::de::DeserializeOwned>(path: &str) -> PyResult<S> {
    let bytes = std::fs::read(path).map_err(io_err)?;
    bincode::deserialize(&bytes)
        .map_err(|e| PyValueError::new_err(format!("not a valid topica model file: {e}")))
}

// Per-model serializable snapshots (ndarray fields stored as Arr2/Arr3/Vec).
#[derive(serde::Serialize, serde::Deserialize)]
struct LdaState {
    num_topics: usize, alpha_sum: Option<f64>, beta: f64, optimize_interval: usize,
    burn_in: usize, seed: u64, num_threads: usize, fitted: bool,
    phi: Option<Arr2>, theta: Option<Arr2>, model: Option<TopicModel>,
    corpus: Option<corpus::Corpus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct DmrState {
    num_topics: usize, beta: f64, optimize_interval: usize, burn_in: usize, seed: u64,
    prior_variance: f64, lbfgs_iters: usize, fitted: bool,
    phi: Option<Arr2>, theta: Option<Arr2>, feature_effects: Option<Arr2>,
    feature_names: Vec<String>, corpus: Option<corpus::Corpus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct LabeledState {
    alpha: f64, beta: f64, seed: u64, fitted: bool, num_topics: usize,
    phi: Option<Arr2>, theta: Option<Arr2>, label_vocab: Vec<String>,
    corpus: Option<corpus::Corpus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct SageState {
    num_topics: usize, alpha: f64, prior_variance: f64, optimize_interval: usize,
    burn_in: usize, seed: u64, lbfgs_iters: usize, fitted: bool, num_groups: usize,
    beta: Vec<Vec<f64>>, theta: Option<Arr2>, group_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct CtmState {
    num_topics: usize, sigma_shrink: f64, seed: u64, init_spectral: bool, fitted: bool,
    beta: Option<Arr2>, theta: Option<Arr2>, corr: Option<Arr2>,
    eta_mean: Option<Arr2>, eta_cov: Option<Arr3>,
    #[serde(default)] mu: Vec<f64>, #[serde(default)] sigma: Vec<f64>,
    corpus: Option<corpus::Corpus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct StmState {
    num_topics: usize, sigma_shrink: f64, seed: u64, init_spectral: bool, fitted: bool,
    beta: Option<Arr2>, theta: Option<Arr2>, corr: Option<Arr2>,
    eta_mean: Option<Arr2>, eta_cov: Option<Arr3>, gamma: Option<Arr2>,
    feature_names: Vec<String>, content_beta: Option<Vec<Vec<Vec<f64>>>>,
    #[serde(default)] mu: Vec<f64>, #[serde(default)] sigma: Vec<f64>,
    group_names: Vec<String>, corpus: Option<corpus::Corpus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct HdpState {
    alpha: f64, gamma: f64, eta: f64, seed: u64, resample_conc: bool, fitted: bool,
    num_topics: usize, learned_alpha: f64, learned_gamma: f64,
    beta: Option<Arr2>, theta: Option<Arr2>, corpus: Option<corpus::Corpus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct DtmState {
    num_topics: usize, alpha: f64, chain_variance: f64, obs_variance: f64, seed: u64,
    fitted: bool, num_times: usize, bound: f64,
    topic_words: Option<Vec<Vec<Vec<f64>>>>, corpus: Option<corpus::Corpus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct SldaState {
    num_topics: usize, alpha: f64, seed: u64, fitted: bool, sigma2: f64,
    eta: Option<Vec<f64>>, beta: Option<Arr2>, theta: Option<Arr2>,
    log_beta: Option<Vec<Vec<f64>>>, corpus: Option<corpus::Corpus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct PtState {
    num_topics: usize, num_pseudo: usize, alpha: f64, beta: f64, seed: u64, fitted: bool,
    phi: Option<Arr2>, theta: Option<Arr2>, corpus: Option<corpus::Corpus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct GsdmmState {
    k_max: usize, alpha: f64, beta: f64, seed: u64, fitted: bool, num_used: usize,
    phi: Option<Arr2>, theta: Option<Arr2>, doc_cluster: Vec<usize>,
    corpus: Option<corpus::Corpus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct SeededState {
    num_topics: usize, alpha: f64, beta: f64, weight: f64, seed: u64, fitted: bool,
    topic_names: Vec<String>, phi: Option<Arr2>, theta: Option<Arr2>,
    corpus: Option<corpus::Corpus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct KeyAtmState {
    num_topics: usize, alpha: f64, beta: f64, beta_keyword: f64, gamma1: f64, gamma2: f64,
    seed: u64, fitted: bool, topic_names: Vec<String>, keyword_rate: Vec<f64>,
    phi: Option<Arr2>, theta: Option<Arr2>, corpus: Option<corpus::Corpus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct PaState {
    num_super: usize, num_sub: usize, alpha: f64, beta: f64, seed: u64, fitted: bool,
    phi: Option<Arr2>, theta: Option<Arr2>, super_sub: Option<Arr2>,
    corpus: Option<corpus::Corpus>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct HldaState {
    depth: usize, gamma: f64, eta: f64, alpha: f64, seed: u64, fitted: bool,
    num_nodes: usize, node_topic_word: Option<Arr2>, node_levels: Vec<usize>,
    node_parents: Vec<i64>, doc_paths: Vec<Vec<usize>>, corpus: Option<corpus::Corpus>,
}

// ---------------------------------------------------------------------------
// Corpus building from in-memory tokenised documents
// ---------------------------------------------------------------------------

/// Build a `corpus::Corpus` from already-tokenised documents.
///
/// Mirrors the vocab-construction and frequency-filtering logic of
/// `corpus::load_text_file`, minus the regex tokenisation/lowercasing — the
/// caller owns tokenisation here.
#[allow(clippy::too_many_arguments)]
fn build_corpus_from_docs(
    docs_in: Vec<Vec<String>>,
    doc_names_in: Option<Vec<String>>,
    doc_labels_in: Option<Vec<String>>,
    stopwords: HashSet<String>,
    min_doc_freq: u32,
    max_doc_fraction: f64,
    min_cf: u32,
    rm_top: usize,
) -> PyResult<corpus::Corpus> {
    let n = docs_in.len();
    if let Some(names) = &doc_names_in {
        if names.len() != n {
            return Err(PyValueError::new_err(format!(
                "doc_names has {} entries but there are {} documents",
                names.len(),
                n
            )));
        }
    }
    if let Some(labels) = &doc_labels_in {
        if labels.len() != n {
            return Err(PyValueError::new_err(format!(
                "doc_labels has {} entries but there are {} documents",
                labels.len(),
                n
            )));
        }
    }

    let mut vocab: HashMap<String, usize> = HashMap::new();
    let mut id_to_word: Vec<String> = Vec::new();
    let mut total_freqs: Vec<u32> = Vec::new();
    let mut docs: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut per_doc_type_sets: Vec<HashSet<usize>> = Vec::with_capacity(n);

    for tokens in &docs_in {
        let mut token_ids: Vec<u32> = Vec::with_capacity(tokens.len());
        let mut seen: HashSet<usize> = HashSet::new();
        for tok in tokens {
            if stopwords.contains(tok) {
                continue;
            }
            let id = if let Some(&eid) = vocab.get(tok) {
                eid
            } else {
                let new_id = id_to_word.len();
                vocab.insert(tok.clone(), new_id);
                id_to_word.push(tok.clone());
                total_freqs.push(0);
                new_id
            };
            total_freqs[id] += 1;
            token_ids.push(id as u32);
            seen.insert(id);
        }
        docs.push(token_ids);
        per_doc_type_sets.push(seen);
    }

    let doc_names: Vec<String> = doc_names_in
        .unwrap_or_else(|| (0..n).map(|i| format!("doc_{}", i)).collect());
    let doc_labels: Vec<String> = doc_labels_in.unwrap_or_else(|| vec![String::new(); n]);

    let num_types = id_to_word.len();
    let num_docs = docs.len();

    let mut doc_freqs = vec![0u32; num_types];
    for set in &per_doc_type_sets {
        for &id in set {
            doc_freqs[id] += 1;
        }
    }

    // Frequency filtering. `min_doc_freq`/`max_doc_fraction` prune by document
    // frequency; `min_cf` prunes by collection (total) frequency; `rm_top` drops
    // the most frequent words by collection frequency (tomotopy's min_df/min_cf/
    // rm_top). The top-`rm_top` set is by total frequency, ties broken by id.
    let max_df = (num_docs as f64 * max_doc_fraction).ceil() as u32;
    let drop_top: HashSet<usize> = if rm_top > 0 {
        let mut order: Vec<usize> = (0..num_types).collect();
        order.sort_by(|&a, &b| total_freqs[b].cmp(&total_freqs[a]).then(a.cmp(&b)));
        order.into_iter().take(rm_top).collect()
    } else {
        HashSet::new()
    };
    let keep: Vec<bool> = (0..num_types)
        .map(|id| {
            doc_freqs[id] >= min_doc_freq
                && doc_freqs[id] <= max_df
                && total_freqs[id] >= min_cf
                && !drop_top.contains(&id)
        })
        .collect();

    if keep.iter().all(|&k| k) {
        return Ok(corpus::Corpus {
            id_to_word,
            docs,
            doc_names,
            doc_labels,
            doc_freqs,
            total_freqs,
        });
    }

    // Remap surviving vocabulary to a dense id range.
    let mut remap: Vec<Option<usize>> = vec![None; num_types];
    let mut new_id_to_word: Vec<String> = Vec::new();
    let mut new_doc_freqs: Vec<u32> = Vec::new();
    let mut new_total_freqs: Vec<u32> = Vec::new();
    for id in 0..num_types {
        if keep[id] {
            remap[id] = Some(new_id_to_word.len());
            new_id_to_word.push(id_to_word[id].clone());
            new_doc_freqs.push(doc_freqs[id]);
            new_total_freqs.push(total_freqs[id]);
        }
    }

    let new_docs: Vec<Vec<u32>> = docs
        .into_iter()
        .map(|doc| {
            doc.into_iter()
                .filter_map(|id| remap[id as usize].map(|r| r as u32))
                .collect()
        })
        .collect();

    // Drop documents emptied by pruning, keeping names/labels aligned.
    let mut final_docs: Vec<Vec<u32>> = Vec::new();
    let mut final_names: Vec<String> = Vec::new();
    let mut final_labels: Vec<String> = Vec::new();
    for ((doc, name), label) in new_docs
        .into_iter()
        .zip(doc_names.into_iter())
        .zip(doc_labels.into_iter())
    {
        if !doc.is_empty() {
            final_docs.push(doc);
            final_names.push(name);
            final_labels.push(label);
        }
    }

    Ok(corpus::Corpus {
        id_to_word: new_id_to_word,
        docs: final_docs,
        doc_names: final_names,
        doc_labels: final_labels,
        doc_freqs: new_doc_freqs,
        total_freqs: new_total_freqs,
    })
}

// ---------------------------------------------------------------------------
// Corpus pyclass
// ---------------------------------------------------------------------------

/// A preprocessed, integer-encoded document collection.
///
/// Build one from already-tokenised documents with
/// :meth:`Corpus.from_documents`, from a raw text file with
/// :meth:`Corpus.from_text_file`, or load a binary corpus written by the
/// ``preprocess`` CLI with :meth:`Corpus.load`.
#[pyclass(module = "topica")]
#[derive(Clone)]
pub struct Corpus {
    inner: corpus::Corpus,
}

#[pymethods]
impl Corpus {
    /// Build a corpus from pre-tokenised documents.
    ///
    /// `documents` is a sequence of token lists. Optional `doc_names` /
    /// `doc_labels` (each the same length as `documents`) attach an id and a
    /// label to every document. `stopwords` are dropped. Vocabulary is pruned by
    /// `min_doc_freq` (minimum document frequency) and `max_doc_fraction`
    /// (maximum fraction of documents), by `min_cf` (minimum collection/total
    /// frequency), and by `rm_top` (drop the N most frequent words) — matching
    /// tomotopy's `min_df` / `min_cf` / `rm_top`.
    #[staticmethod]
    #[pyo3(signature = (documents, *, doc_names=None, doc_labels=None,
                        stopwords=None, min_doc_freq=1, max_doc_fraction=1.0,
                        min_cf=0, rm_top=0))]
    #[allow(clippy::too_many_arguments)]
    fn from_documents(
        documents: Vec<Vec<String>>,
        doc_names: Option<Vec<String>>,
        doc_labels: Option<Vec<String>>,
        stopwords: Option<Vec<String>>,
        min_doc_freq: u32,
        max_doc_fraction: f64,
        min_cf: u32,
        rm_top: usize,
    ) -> PyResult<Self> {
        let stop: HashSet<String> = stopwords.unwrap_or_default().into_iter().collect();
        let inner = build_corpus_from_docs(
            documents,
            doc_names,
            doc_labels,
            stop,
            min_doc_freq,
            max_doc_fraction,
            min_cf,
            rm_top,
        )?;
        Ok(Corpus { inner })
    }

    /// Load and tokenise a raw text file (MALLET-style), matching the
    /// ``preprocess`` CLI.
    ///
    /// `format` is ``"plain"`` (one document per line) or ``"tsv"``. In plain
    /// mode, `id_field=True` treats the first whitespace token as the doc id.
    /// In tsv mode, `id_column`/`label_column`/`text_column` select columns
    /// (`label_column=None` disables labels).
    #[staticmethod]
    #[pyo3(signature = (path, *, format="plain", id_field=false,
                        id_column=0, label_column=1, text_column=2,
                        token_regex=None, stopwords=None,
                        min_doc_freq=1, max_doc_fraction=1.0))]
    #[allow(clippy::too_many_arguments)]
    fn from_text_file(
        path: &str,
        format: &str,
        id_field: bool,
        id_column: usize,
        label_column: Option<usize>,
        text_column: usize,
        token_regex: Option<String>,
        stopwords: Option<Vec<String>>,
        min_doc_freq: u32,
        max_doc_fraction: f64,
    ) -> PyResult<Self> {
        let fmt = match format {
            "plain" => InputFormat::Plain { id_field },
            "tsv" => InputFormat::Tsv {
                id_column,
                label_column,
                text_column,
            },
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown format {:?} (use 'plain' or 'tsv')",
                    other
                )))
            }
        };
        let stop: HashSet<String> = stopwords.unwrap_or_default().into_iter().collect();
        let opts = LoadOptions {
            format: fmt,
            token_regex: token_regex.unwrap_or_else(|| corpus::DEFAULT_TOKEN_REGEX.to_string()),
            stopwords: stop,
            min_doc_freq,
            max_doc_fraction,
        };
        let inner = corpus::load_text_file(Path::new(path), &opts).map_err(io_err)?;
        Ok(Corpus { inner })
    }

    /// Load a binary corpus file written by the ``preprocess`` CLI or
    /// :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = corpus::load_corpus(Path::new(path)).map_err(io_err)?;
        Ok(Corpus { inner })
    }

    /// Write this corpus to a binary file (the ``preprocess`` format), so it
    /// can be reused by the CLI tools or reloaded with :meth:`load`.
    fn save(&self, path: &str) -> PyResult<()> {
        corpus::save_corpus(&self.inner, Path::new(path)).map_err(io_err)
    }

    #[getter]
    fn num_docs(&self) -> usize {
        self.inner.num_docs()
    }

    #[getter]
    fn num_words(&self) -> usize {
        self.inner.num_types()
    }

    #[getter]
    fn total_tokens(&self) -> usize {
        self.inner.total_tokens()
    }

    #[getter]
    fn vocabulary(&self) -> Vec<String> {
        self.inner.id_to_word.clone()
    }

    #[getter]
    fn doc_names(&self) -> Vec<String> {
        self.inner.doc_names.clone()
    }

    #[getter]
    fn doc_labels(&self) -> Vec<String> {
        self.inner.doc_labels.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Corpus(num_docs={}, num_words={}, total_tokens={})",
            self.inner.num_docs(),
            self.inner.num_types(),
            self.inner.total_tokens()
        )
    }
}

// ---------------------------------------------------------------------------
// LDA pyclass
// ---------------------------------------------------------------------------

/// SparseLDA topic model (the MALLET algorithm).
///
/// Construct with the hyperparameters, then call :meth:`fit` on a
/// :class:`Corpus` or a list of token lists. After fitting, the estimated
/// distributions are available as :attr:`topic_word` (φ) and
/// :attr:`doc_topic` (θ).
#[pyclass(module = "topica")]
pub struct LDA {
    num_topics: usize,
    alpha_sum: Option<f64>,
    beta: f64,
    optimize_interval: usize,
    burn_in: usize,
    seed: u64,
    num_threads: usize,
    // Sampling backend: false = SparseLDA (MALLET), true = LightLDA alias-MH.
    light: bool,
    mh_steps: usize,

    // Populated after fit().
    fitted: bool,
    phi: Option<Array2<f64>>,   // (num_topics, num_words)
    theta: Option<Array2<f64>>, // (num_docs, num_topics)
    model: Option<TopicModel>,
    corpus: Option<corpus::Corpus>,
}

impl LDA {
    /// Transpose the accumulated φ/θ snapshots into the conventional matrix
    /// orientation and store the fitted state. Shared by both sampler paths.
    #[allow(clippy::too_many_arguments)]
    fn finalize_fit(
        &mut self,
        num_topics: usize,
        num_types: usize,
        num_docs: usize,
        acc_phi: Vec<Vec<f64>>,
        acc_theta: Vec<Vec<f64>>,
        model: TopicModel,
        corpus: corpus::Corpus,
    ) {
        // phi: transpose (word, topic) -> (topic, word).
        let mut phi = Array2::<f64>::zeros((num_topics, num_types));
        for (w, row) in acc_phi.iter().enumerate() {
            for (t, &v) in row.iter().enumerate() {
                phi[[t, w]] = v;
            }
        }
        let mut theta = Array2::<f64>::zeros((num_docs, num_topics));
        for (d, row) in acc_theta.iter().enumerate() {
            for (t, &v) in row.iter().enumerate() {
                theta[[d, t]] = v;
            }
        }
        self.phi = Some(phi);
        self.theta = Some(theta);
        self.model = Some(model);
        self.corpus = Some(corpus);
        self.fitted = true;
    }

    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }

    /// Top-`n` word ids per topic, descending by φ.
    fn top_word_ids(&self, n: usize) -> Vec<Vec<usize>> {
        let phi = self.phi.as_ref().unwrap();
        let num_words = phi.shape()[1];
        (0..self.num_topics)
            .map(|t| {
                let mut idx: Vec<usize> = (0..num_words).collect();
                idx.sort_by(|&a, &b| phi[[t, b]].partial_cmp(&phi[[t, a]]).unwrap());
                idx.truncate(n);
                idx
            })
            .collect()
    }

    /// Map held-out documents (a `Corpus` or `list[list[str]]`) to trained
    /// vocabulary ids, dropping out-of-vocabulary tokens. Returns
    /// `(docs_as_ids, num_tokens_scored, num_oov_dropped)`.
    fn map_heldout(
        &self,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<Vec<usize>>, usize, usize)> {
        let trained = self.corpus.as_ref().unwrap();
        let index: HashMap<&str, usize> = trained
            .id_to_word
            .iter()
            .enumerate()
            .map(|(i, w)| (w.as_str(), i))
            .collect();

        let str_docs: Vec<Vec<String>> = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
                .docs
                .iter()
                .map(|doc| {
                    doc.iter()
                        .map(|&wid| c.inner.id_to_word[wid as usize].clone())
                        .collect()
                })
                .collect()
        } else {
            data.extract::<Vec<Vec<String>>>().map_err(|_| {
                PyValueError::new_err("expected a Corpus or a list of token lists (list[list[str]])")
            })?
        };

        let mut out = Vec::with_capacity(str_docs.len());
        let mut n_tokens = 0usize;
        let mut n_oov = 0usize;
        for doc in &str_docs {
            let mut ids = Vec::with_capacity(doc.len());
            for tok in doc {
                match index.get(tok.as_str()) {
                    Some(&id) => {
                        ids.push(id);
                        n_tokens += 1;
                    }
                    None => n_oov += 1,
                }
            }
            out.push(ids);
        }
        Ok((out, n_tokens, n_oov))
    }
}

#[pymethods]
impl LDA {
    /// Create an unfitted model.
    ///
    /// `alpha_sum` is the total document-topic Dirichlet mass (default:
    /// `num_topics`, i.e. 1.0 per topic). `beta` is the per-word topic-word
    /// prior. With `optimize_interval > 0`, α and β are re-estimated every
    /// that-many iterations once past `burn_in`.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha_sum=None, beta=0.01,
                        optimize_interval=50, burn_in=200, seed=42, num_threads=1,
                        sampler="sparse", mh_steps=2))]
    fn new(
        num_topics: usize,
        alpha_sum: Option<f64>,
        beta: f64,
        optimize_interval: usize,
        burn_in: usize,
        seed: u64,
        num_threads: usize,
        sampler: &str,
        mh_steps: usize,
    ) -> PyResult<Self> {
        if num_topics == 0 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        if beta <= 0.0 {
            return Err(PyValueError::new_err("beta must be > 0"));
        }
        let light = match sampler {
            "sparse" | "mallet" => false,
            "lightlda" | "light" | "alias" => true,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown sampler {other:?}; expected \"sparse\" or \"lightlda\""
                )))
            }
        };
        if light && mh_steps == 0 {
            return Err(PyValueError::new_err("mh_steps must be >= 1 for the lightlda sampler"));
        }
        Ok(LDA {
            num_topics,
            alpha_sum,
            beta,
            optimize_interval,
            burn_in,
            seed,
            num_threads: num_threads.max(1),
            light,
            mh_steps,
            fitted: false,
            phi: None,
            theta: None,
            model: None,
            corpus: None,
        })
    }

    /// Run Gibbs sampling on `data`, then average `num_samples` snapshots
    /// (taken `sample_interval` iterations apart) into the final φ/θ estimates.
    ///
    /// `data` may be a :class:`Corpus` or a list of token lists (each a list of
    /// strings). When a token-list is passed, an internal corpus is built with
    /// no frequency filtering — build a :class:`Corpus` explicitly for that.
    ///
    /// `progress`, if given, is called as ``progress(iteration, ll_per_token)``
    /// every `progress_interval` iterations during the main loop.
    #[pyo3(signature = (data, *, iterations=1000, num_samples=5, sample_interval=25,
                        progress=None, progress_interval=50))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        iterations: usize,
        num_samples: usize,
        sample_interval: usize,
        progress: Option<PyObject>,
        progress_interval: usize,
    ) -> PyResult<()> {
        // Accept either a Corpus or a list[list[str]].
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err(
                    "fit() expects a Corpus or a list of token lists (list[list[str]])",
                )
            })?;
            build_corpus_from_docs(docs, None, None, HashSet::new(), 1, 1.0, 0, 0)?
        };

        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }

        let num_topics = self.num_topics;
        let num_types = corpus.num_types();
        let num_docs = corpus.num_docs();
        let alpha_sum = self.alpha_sum.unwrap_or(num_topics as f64);
        let total_tokens = corpus.total_tokens().max(1) as f64;

        let mut model = TopicModel::new(num_topics, alpha_sum, self.beta, num_types);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        model.initialize(&corpus, &mut rng);

        let optimize_interval = self.optimize_interval;
        let burn_in = self.burn_in;
        let num_threads = self.num_threads;
        let seed_base = self.seed;
        let light = self.light;
        let mh_steps = self.mh_steps;
        let beta = self.beta;

        // LightLDA path: alias-MH sampling on dense count tables, packed back
        // into a TopicModel at the end. Separate from the SparseLDA path below so
        // the well-tested MALLET sampler is left untouched.
        if light {
            let (acc_phi, acc_theta, model) = py.allow_threads(move || {
                let alpha0 = vec![alpha_sum / num_topics as f64; num_topics];
                let mut ls = lightlda::LightLda::new(&corpus, num_topics, &alpha0, beta, &mut rng);
                ls.mh_steps = mh_steps;

                for iter in 1..=iterations {
                    ls.sweep(&corpus, &mut rng);
                    if optimize_interval > 0 && iter > burn_in && iter % optimize_interval == 0 {
                        let mut m = ls.to_topic_model();
                        optimize::optimize_alpha(&mut m, &corpus);
                        optimize::optimize_beta(&mut m);
                        ls.set_hyper(&m.alpha, m.beta);
                    }
                    if let Some(cb) = &progress {
                        if progress_interval > 0 && iter % progress_interval == 0 {
                            let m = ls.to_topic_model();
                            let ll = output::model_log_likelihood(&m, &corpus) / total_tokens;
                            Python::with_gil(|py| {
                                let _ = cb.call1(py, (iter, ll));
                            });
                        }
                    }
                }

                let mut acc_phi = vec![vec![0.0f64; num_topics]; num_types];
                let mut acc_theta = vec![vec![0.0f64; num_topics]; num_docs];
                for _ in 0..num_samples {
                    for _ in 0..sample_interval {
                        ls.sweep(&corpus, &mut rng);
                    }
                    ls.phi_into(&mut acc_phi);
                    ls.theta_into(&corpus, &mut acc_theta);
                }
                let n = (num_samples.max(1)) as f64;
                for row in acc_phi.iter_mut() {
                    for v in row.iter_mut() { *v /= n; }
                }
                for row in acc_theta.iter_mut() {
                    for v in row.iter_mut() { *v /= n; }
                }
                let model = ls.to_topic_model();
                (acc_phi, acc_theta, (model, corpus))
            });
            let (model, corpus) = model;
            self.finalize_fit(num_topics, num_types, num_docs, acc_phi, acc_theta, model, corpus);
            return Ok(());
        }

        // Heavy loop runs with the GIL released; the progress callback briefly
        // re-acquires it. allow_threads returns the owned model + accumulators.
        let (acc_phi, acc_theta, model) = py.allow_threads(move || {
            // One Gibbs sweep: exact sequential path when single-threaded,
            // approximate parallel sampling otherwise. `sweep` seeds the
            // per-worker RNGs so parallel runs are deterministic.
            let mut sweep: u64 = 0;
            let mut do_sweep =
                |model: &mut TopicModel, rng: &mut ChaCha8Rng| {
                    sweep += 1;
                    if num_threads <= 1 {
                        sampler::run_iteration(model, &corpus, rng);
                    } else {
                        let s = seed_base
                            .wrapping_add(sweep.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        parallel_sweep(model, &corpus.docs, num_threads, s);
                    }
                };

            // ---- main training loop (ports src/bin/train.rs) ----
            for iter in 1..=iterations {
                do_sweep(&mut model, &mut rng);

                if optimize_interval > 0 && iter > burn_in && iter % optimize_interval == 0 {
                    optimize::optimize_alpha(&mut model, &corpus);
                    optimize::optimize_beta(&mut model);
                }

                if let Some(cb) = &progress {
                    if progress_interval > 0 && iter % progress_interval == 0 {
                        let ll = output::model_log_likelihood(&model, &corpus) / total_tokens;
                        Python::with_gil(|py| {
                            let _ = cb.call1(py, (iter, ll));
                        });
                    }
                }
            }

            // ---- sampling phase: average num_samples smoothed snapshots ----
            let mut acc_phi = vec![vec![0.0f64; num_topics]; num_types];
            let mut acc_theta = vec![vec![0.0f64; num_topics]; num_docs];

            for _ in 0..num_samples {
                for _ in 0..sample_interval {
                    do_sweep(&mut model, &mut rng);
                }
                accumulate_phi(&model, &mut acc_phi);
                accumulate_theta(&model, &corpus, &mut acc_theta);
            }

            let n = (num_samples.max(1)) as f64;
            for row in acc_phi.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }
            for row in acc_theta.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }

            // Return the corpus too (move it back out for storage).
            (acc_phi, acc_theta, (model, corpus))
        });
        let (model, corpus) = model;
        self.finalize_fit(num_topics, num_types, num_docs, acc_phi, acc_theta, model, corpus);
        Ok(())
    }

    /// Topic-word probability matrix φ, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Document-topic probability matrix θ, shape ``(num_docs, num_topics)``.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// The vocabulary: word for each column of :attr:`topic_word`.
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    /// Document ids, parallel to the rows of :attr:`doc_topic`.
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    /// Per-topic α after (optional) optimisation, shape ``(num_topics,)``.
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let a = Array1::from(self.model.as_ref().unwrap().alpha.clone());
        Ok(a.to_pyarray_bound(py))
    }

    /// The (optimised) symmetric β.
    #[getter]
    fn beta(&self) -> PyResult<f64> {
        self.require_fitted()?;
        Ok(self.model.as_ref().unwrap().beta)
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// Top `n` words per topic as ``(word, probability)`` pairs.
    ///
    /// Returns a list of `n`-length lists (one per topic), or — when `topic`
    /// is given — just that topic's list.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let phi = self.phi.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let num_words = vocab.len();

        let one_topic = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err(format!(
                    "topic {} out of range (num_topics={})",
                    t, self.num_topics
                )));
            }
            let mut idx: Vec<usize> = (0..num_words).collect();
            idx.sort_by(|&a, &b| phi[[t, b]].partial_cmp(&phi[[t, a]]).unwrap());
            let items: Vec<Bound<'py, PyTuple>> = idx
                .iter()
                .take(n)
                .map(|&w| PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), phi[[t, w]].into_py(py)]))
                .collect();
            Ok(PyList::new_bound(py, items))
        };

        match topic {
            Some(t) => Ok(one_topic(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one_topic).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// MALLET-formula model log-likelihood of the final sampler state.
    fn log_likelihood(&self) -> PyResult<f64> {
        self.require_fitted()?;
        Ok(output::model_log_likelihood(
            self.model.as_ref().unwrap(),
            self.corpus.as_ref().unwrap(),
        ))
    }

    /// Write topic-word probabilities to a TSV file (the ``train`` CLI format).
    fn save_topic_word(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        let phi = self.phi.as_ref().unwrap();
        let corpus = self.corpus.as_ref().unwrap();
        // Re-orient to [word][topic] as write_topic_word_matrix expects.
        let phi_wt: Vec<Vec<f64>> = (0..corpus.num_types())
            .map(|w| (0..self.num_topics).map(|t| phi[[t, w]]).collect())
            .collect();
        output::write_topic_word_matrix(&phi_wt, corpus, Path::new(path)).map_err(io_err)
    }

    /// Write document-topic probabilities to a TSV file (the ``train`` CLI format).
    fn save_doc_topic(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        let theta = self.theta.as_ref().unwrap();
        let corpus = self.corpus.as_ref().unwrap();
        let theta_dt: Vec<Vec<f64>> = (0..corpus.num_docs())
            .map(|d| (0..self.num_topics).map(|t| theta[[d, t]]).collect())
            .collect();
        output::write_doc_topic_matrix(&theta_dt, corpus, Path::new(path)).map_err(io_err)
    }

    /// Held-out evaluation via the Wallach et al. (2009) left-to-right
    /// estimator (the method MALLET's ``evaluate-topics`` uses).
    ///
    /// `data` is a held-out :class:`Corpus` or `list[list[str]]`; its tokens are
    /// matched to the training vocabulary by string (out-of-vocabulary tokens
    /// are dropped). Returns a dict with `log_likelihood` (total held-out log
    /// P(data)), `perplexity` (``exp(-LL / num_tokens)``, lower is better),
    /// `num_tokens` (scored), and `num_oov` (dropped). Cost grows with the
    /// square of document length, so keep `num_particles` modest.
    #[pyo3(signature = (data, *, num_particles=10, seed=None))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        num_particles: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.require_fitted()?;
        if num_particles == 0 {
            return Err(PyValueError::new_err("num_particles must be >= 1"));
        }
        let (docs, n_tokens, n_oov) = self.map_heldout(data)?;
        let model = self.model.as_ref().unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(seed.unwrap_or(self.seed));

        let ll = py.allow_threads(move || {
            let mut total = 0.0;
            for doc in &docs {
                total += left_to_right_doc(model, doc, num_particles, &mut rng);
            }
            total
        });

        let perplexity = if n_tokens > 0 {
            (-ll / n_tokens as f64).exp()
        } else {
            f64::NAN
        };

        let d = PyDict::new_bound(py);
        d.set_item("log_likelihood", ll)?;
        d.set_item("perplexity", perplexity)?;
        d.set_item("num_tokens", n_tokens)?;
        d.set_item("num_oov", n_oov)?;
        Ok(d)
    }

    /// Held-out perplexity (lower is better) — convenience wrapper over
    /// :meth:`evaluate`. See `evaluate` for `data`/`num_particles` semantics.
    #[pyo3(signature = (data, *, num_particles=10, seed=None))]
    fn perplexity<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        num_particles: usize,
        seed: Option<u64>,
    ) -> PyResult<f64> {
        let d = self.evaluate(py, data, num_particles, seed)?;
        d.get_item("perplexity")?.unwrap().extract()
    }

    /// UMass topic coherence for each topic, shape ``(num_topics,)``.
    ///
    /// Intrinsic (no external corpus): for each topic's top-`n` words,
    /// `Σ_{i>j} log[(codoc(w_i,w_j)+1)/docfreq(w_j)]` over the training corpus.
    /// Higher (closer to 0) is more coherent. `numpy.mean(...)` gives the
    /// usual single-number summary.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = self.top_word_ids(n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Per-topic diagnostics (MALLET-style), one dict per topic, suitable for
    /// `pandas.DataFrame(model.diagnostics())`.
    ///
    /// Keys: `topic`, `tokens` (assignments to the topic), `coherence` (UMass),
    /// `exclusivity` (mean top-word share of φ vs. other topics; higher = more
    /// distinctive), `effective_words` (`exp(H(φ_t))`; lower = more focused),
    /// `rank1_docs` (documents whose dominant topic is this one), `alpha`, and
    /// `top_words`.
    #[pyo3(signature = (n=10))]
    fn diagnostics<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyList>> {
        self.require_fitted()?;
        let phi = self.phi.as_ref().unwrap();
        let theta = self.theta.as_ref().unwrap();
        let corpus = self.corpus.as_ref().unwrap();
        let model = self.model.as_ref().unwrap();
        let vocab = &corpus.id_to_word;
        let num_words = phi.shape()[1];
        let num_docs = theta.shape()[0];

        let tops = self.top_word_ids(n);
        let coh = umass_coherence(corpus, &tops);

        // Column sums of φ for the exclusivity denominator.
        let mut col_sum = vec![0.0f64; num_words];
        for t in 0..self.num_topics {
            for (w, c) in col_sum.iter_mut().enumerate() {
                *c += phi[[t, w]];
            }
        }

        // Rank-1 (dominant-topic) document counts.
        let mut rank1 = vec![0usize; self.num_topics];
        for d in 0..num_docs {
            let mut best = 0usize;
            let mut best_v = theta[[d, 0]];
            for t in 1..self.num_topics {
                if theta[[d, t]] > best_v {
                    best_v = theta[[d, t]];
                    best = t;
                }
            }
            rank1[best] += 1;
        }

        let list = PyList::empty_bound(py);
        for t in 0..self.num_topics {
            let topn = &tops[t];

            let mut excl = 0.0;
            for &w in topn {
                if col_sum[w] > 0.0 {
                    excl += phi[[t, w]] / col_sum[w];
                }
            }
            if !topn.is_empty() {
                excl /= topn.len() as f64;
            }

            let rowsum: f64 = (0..num_words).map(|w| phi[[t, w]]).sum();
            let mut h = 0.0;
            if rowsum > 0.0 {
                for w in 0..num_words {
                    let p = phi[[t, w]] / rowsum;
                    if p > 0.0 {
                        h -= p * p.ln();
                    }
                }
            }
            let effective_words = h.exp();

            let words: Vec<String> = topn.iter().map(|&w| vocab[w].clone()).collect();

            let d = PyDict::new_bound(py);
            d.set_item("topic", t)?;
            d.set_item("tokens", model.tokens_per_topic[t])?;
            d.set_item("coherence", coh[t])?;
            d.set_item("exclusivity", excl)?;
            d.set_item("effective_words", effective_words)?;
            d.set_item("rank1_docs", rank1[t])?;
            d.set_item("alpha", model.alpha[t])?;
            d.set_item("top_words", words)?;
            list.append(d)?;
        }
        Ok(list)
    }

    /// Infer document-topic distributions for *new, unseen* documents under the
    /// fitted model (sklearn-style `transform`). `data` is a :class:`Corpus` or
    /// `list[list[str]]`; tokens are matched to the training vocabulary by
    /// string (OOV dropped). A document with no in-vocabulary tokens gets the
    /// prior θ. Returns an array of shape ``(num_new_docs, num_topics)`` whose
    /// rows sum to 1.
    #[pyo3(signature = (data, *, iterations=100, burn_in=10, num_samples=10,
                        sample_interval=5, seed=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        iterations: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let (docs, _n, _oov) = self.map_heldout(data)?;
        let model = self.model.as_ref().unwrap();
        let k = self.num_topics;
        let mut rng = ChaCha8Rng::seed_from_u64(seed.unwrap_or(self.seed));

        let thetas: Vec<Vec<f64>> = py.allow_threads(move || {
            docs.iter()
                .map(|d| {
                    infer_doc(model, d, iterations, burn_in, num_samples, sample_interval, &mut rng)
                })
                .collect()
        });

        let mut arr = Array2::<f64>::zeros((thetas.len(), k));
        for (i, row) in thetas.iter().enumerate() {
            for (t, &v) in row.iter().enumerate() {
                arr[[i, t]] = v;
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// The `n` training documents most strongly associated with `topic`, as
    /// ``(doc_name, weight)`` pairs sorted by descending θ for that topic.
    #[pyo3(signature = (topic, n=10))]
    fn top_documents<'py>(
        &self,
        py: Python<'py>,
        topic: usize,
        n: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        self.require_fitted()?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic {} out of range (num_topics={})",
                topic, self.num_topics
            )));
        }
        let theta = self.theta.as_ref().unwrap();
        let names = &self.corpus.as_ref().unwrap().doc_names;
        let num_docs = theta.shape()[0];

        let mut idx: Vec<usize> = (0..num_docs).collect();
        idx.sort_by(|&a, &b| theta[[b, topic]].partial_cmp(&theta[[a, topic]]).unwrap());
        let items: Vec<Bound<'py, PyTuple>> = idx
            .iter()
            .take(n)
            .map(|&d| {
                PyTuple::new_bound(
                    py,
                    &[names[d].clone().into_py(py), theta[[d, topic]].into_py(py)],
                )
            })
            .collect();
        Ok(PyList::new_bound(py, items))
    }

    /// Pairwise Jensen-Shannon divergence between topic-word distributions,
    /// shape ``(num_topics, num_topics)`` (base 2, in [0, 1]; 0 on the diagonal).
    /// Low off-diagonal values flag near-duplicate topics.
    #[getter]
    fn topic_divergence<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let phi = self.phi.as_ref().unwrap();
        let k = self.num_topics;
        let w = phi.shape()[1];

        // Normalize each topic row to a distribution over words.
        let rows: Vec<Vec<f64>> = (0..k)
            .map(|t| {
                let s: f64 = (0..w).map(|i| phi[[t, i]]).sum();
                (0..w).map(|i| phi[[t, i]] / s).collect()
            })
            .collect();

        let mut arr = Array2::<f64>::zeros((k, k));
        for a in 0..k {
            for b in (a + 1)..k {
                let d = js_divergence(&rows[a], &rows[b]);
                arr[[a, b]] = d;
                arr[[b, a]] = d;
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// The `n` training documents most similar to document `doc` (by index),
    /// as ``(doc_name, divergence)`` pairs sorted by ascending Jensen-Shannon
    /// divergence of their document-topic distributions.
    #[pyo3(signature = (doc, n=10))]
    fn similar_documents<'py>(
        &self,
        py: Python<'py>,
        doc: usize,
        n: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        self.require_fitted()?;
        let theta = self.theta.as_ref().unwrap();
        let names = &self.corpus.as_ref().unwrap().doc_names;
        let num_docs = theta.shape()[0];
        let k = self.num_topics;
        if doc >= num_docs {
            return Err(PyValueError::new_err(format!(
                "doc {} out of range (num_docs={})",
                doc, num_docs
            )));
        }

        let target: Vec<f64> = (0..k).map(|t| theta[[doc, t]]).collect();
        let mut scored: Vec<(usize, f64)> = (0..num_docs)
            .filter(|&d| d != doc)
            .map(|d| {
                let q: Vec<f64> = (0..k).map(|t| theta[[d, t]]).collect();
                (d, js_divergence(&target, &q))
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let items: Vec<Bound<'py, PyTuple>> = scored
            .iter()
            .take(n)
            .map(|&(d, div)| {
                PyTuple::new_bound(py, &[names[d].clone().into_py(py), div.into_py(py)])
            })
            .collect();
        Ok(PyList::new_bound(py, items))
    }

    /// Save the fitted model to `path` (compact binary). Reload with `LDA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &LdaState {
            num_topics: self.num_topics, alpha_sum: self.alpha_sum, beta: self.beta,
            optimize_interval: self.optimize_interval, burn_in: self.burn_in, seed: self.seed,
            num_threads: self.num_threads, fitted: self.fitted,
            phi: arr2_opt(&self.phi), theta: arr2_opt(&self.theta),
            model: self.model.clone(), corpus: self.corpus.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: LdaState = read_state(path)?;
        Ok(LDA {
            num_topics: s.num_topics, alpha_sum: s.alpha_sum, beta: s.beta,
            optimize_interval: s.optimize_interval, burn_in: s.burn_in, seed: s.seed,
            num_threads: s.num_threads, light: false, mh_steps: 2, fitted: s.fitted,
            phi: arr2_back(s.phi), theta: arr2_back(s.theta), model: s.model, corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "LDA(num_topics={}, beta={}, fitted={})",
            self.num_topics, self.beta, self.fitted
        )
    }
}

// ---------------------------------------------------------------------------
// Averaging helpers (ported from src/bin/train.rs)
// ---------------------------------------------------------------------------

/// Snapshot the current smoothed topic-word distribution into `acc[word][topic]`.
fn accumulate_phi(m: &TopicModel, acc: &mut [Vec<f64>]) {
    for word_id in 0..m.num_types {
        for topic in 0..m.num_topics {
            let count = m.get_type_topic_count(word_id, topic);
            let denom = m.tokens_per_topic[topic] as f64 + m.beta_sum;
            acc[word_id][topic] += (count as f64 + m.beta) / denom;
        }
    }
}

/// Snapshot the current smoothed document-topic distribution into `acc[doc][topic]`.
fn accumulate_theta(m: &TopicModel, c: &corpus::Corpus, acc: &mut [Vec<f64>]) {
    let mut counts = vec![0u32; m.num_topics];
    for doc_idx in 0..c.num_docs() {
        for t in counts.iter_mut() {
            *t = 0;
        }
        for &t in &m.doc_topics[doc_idx] {
            counts[t as usize] += 1;
        }
        let doc_len = c.docs[doc_idx].len() as f64;
        let denom = doc_len + m.alpha_sum;
        for t in 0..m.num_topics {
            acc[doc_idx][t] += (counts[t] as f64 + m.alpha[t]) / denom;
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel Gibbs sampling (MALLET-style approximate distributed sampling)
// ---------------------------------------------------------------------------

/// Split `n` items into up to `parts` contiguous, balanced ranges.
fn partition_ranges(n: usize, parts: usize) -> Vec<(usize, usize)> {
    let parts = parts.max(1).min(n.max(1));
    let base = n / parts;
    let rem = n % parts;
    let mut ranges = Vec::with_capacity(parts);
    let mut start = 0;
    for i in 0..parts {
        let len = base + if i < rem { 1 } else { 0 };
        ranges.push((start, start + len));
        start += len;
    }
    ranges
}

struct WorkerOut {
    ttc: Vec<Vec<u32>>,
    tpt: Vec<u32>,
    start: usize,
    dt: Vec<Vec<u32>>,
}

/// One approximate-parallel Gibbs sweep. Documents are partitioned across
/// `num_threads` workers; each samples its slice against a private copy of the
/// topic-word counts (so workers don't see each other's within-sweep updates),
/// then the per-worker count changes are reconciled exactly into the global
/// model. Token bookkeeping stays consistent (each token belongs to exactly one
/// worker); only the sampling distribution is approximated. `sweep_seed` makes
/// the result deterministic for a fixed `num_threads`.
fn parallel_sweep(model: &mut TopicModel, docs: &[Vec<u32>], num_threads: usize, sweep_seed: u64) {
    let k = model.num_topics;
    let mask = model.topic_mask;
    let bits = model.topic_bits;
    let beta = model.beta;
    let beta_sum = model.beta_sum;
    let v = model.num_types;
    let ranges = partition_ranges(docs.len(), num_threads);

    // Snapshot for reconciliation, and clone the shared read-only inputs.
    let original_ttc = model.type_topic_counts.clone();
    let original_tpt = model.tokens_per_topic.clone();
    let alpha = model.alpha.clone();
    let dt_all = &model.doc_topics;

    // --- Workers: each samples its document partition independently. ---
    let outs: Vec<WorkerOut> = ranges
        .par_iter()
        .enumerate()
        .map(|(wid, &(start, end))| {
            let mut ttc = original_ttc.clone();
            let mut tpt = original_tpt.clone();
            let mut dt: Vec<Vec<u32>> = dt_all[start..end].to_vec();
            let mut rng = ChaCha8Rng::seed_from_u64(
                sweep_seed ^ (wid as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            );
            sampler::run_sweep(
                &mut ttc,
                &mut tpt,
                &mut dt,
                &docs[start..end],
                &alpha,
                beta,
                beta_sum,
                mask,
                bits,
                k,
                &mut rng,
            );
            WorkerOut { ttc, tpt, start, dt }
        })
        .collect();

    // --- Reconcile topic-word counts. Each worker started from `original` and
    // only changed its own documents' tokens, so the exact global state is
    //     final = Σ_w worker_w − (W−1)·original
    // computed densely per word (in parallel, reusing a per-thread accumulator
    // to avoid per-word allocation). Re-encode into the packed layout. ---
    let wm1 = (outs.len() as i64) - 1;
    let new_ttc: Vec<Vec<u32>> = (0..v)
        .into_par_iter()
        .map_init(
            || vec![0i64; k],
            |acc, w| {
                for a in acc.iter_mut() {
                    *a = 0;
                }
                for out in &outs {
                    for &e in &out.ttc[w] {
                        if e == 0 {
                            break;
                        }
                        acc[(e & mask) as usize] += (e >> bits) as i64;
                    }
                }
                for &e in &original_ttc[w] {
                    if e == 0 {
                        break;
                    }
                    acc[(e & mask) as usize] -= wm1 * (e >> bits) as i64;
                }
                let mut entries: Vec<u32> = (0..k)
                    .filter(|&t| acc[t] > 0)
                    .map(|t| ((acc[t] as u32) << bits) | (t as u32))
                    .collect();
                entries.sort_unstable_by(|a, b| b.cmp(a));
                let len = original_ttc[w].len();
                entries.resize(len.max(entries.len()), 0);
                entries
            },
        )
        .collect();
    model.type_topic_counts = new_ttc;

    // --- Reconcile tokens-per-topic the same way. ---
    let mut tpt: Vec<i64> = original_tpt.iter().map(|&c| c as i64).collect();
    for out in &outs {
        for t in 0..k {
            tpt[t] += out.tpt[t] as i64 - original_tpt[t] as i64;
        }
    }
    model.tokens_per_topic = tpt.iter().map(|&c| c.max(0) as u32).collect();

    // --- Write each worker's updated topic assignments back into place. ---
    for out in outs {
        let start = out.start;
        for (i, row) in out.dt.into_iter().enumerate() {
            model.doc_topics[start + i] = row;
        }
    }
}

// ---------------------------------------------------------------------------
// Held-out evaluation: Wallach et al. (2009) left-to-right estimator
// ---------------------------------------------------------------------------

/// Dense per-topic φ(word | topic) vectors for each distinct word in `doc`,
/// under the fixed trained model. Decoding each word's packed sparse entries
/// once removes the repeated O(K) linear scans of `get_type_topic_count` from
/// the inference/eval inner loops — an exact (value-identical) speedup that
/// matters once K is large.
fn build_phi_cache(model: &TopicModel, doc: &[usize]) -> HashMap<usize, Vec<f64>> {
    let k = model.num_topics;
    let denom: Vec<f64> = (0..k)
        .map(|t| model.beta_sum + model.tokens_per_topic[t] as f64)
        .collect();

    let mut cache: HashMap<usize, Vec<f64>> = HashMap::new();
    for &w in doc {
        cache.entry(w).or_insert_with(|| {
            let mut dense = vec![0u32; k];
            for &entry in &model.type_topic_counts[w] {
                if entry == 0 {
                    break;
                }
                let t = (entry & model.topic_mask) as usize;
                dense[t] = entry >> model.topic_bits;
            }
            (0..k)
                .map(|t| (model.beta + dense[t] as f64) / denom[t])
                .collect()
        });
    }
    cache
}

/// Sample an index in proportion to non-negative `weights` summing to `total`.
fn sample_categorical<R: Rng>(weights: &[f64], total: f64, rng: &mut R) -> usize {
    let mut u = rng.gen::<f64>() * total;
    for (i, &w) in weights.iter().enumerate() {
        u -= w;
        if u <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

/// Estimate log P(doc) under a fixed trained model with the left-to-right
/// estimator. `doc` holds trained-vocabulary word ids (OOV already dropped).
///
/// Per token position, the probability is averaged across `num_particles`
/// particles (each a left-to-right pass that resamples earlier positions);
/// the document log-likelihood is the sum of the logs of those averages.
/// The trained topic-word counts stay fixed — only per-document topic counts
/// evolve.
fn left_to_right_doc<R: Rng>(
    model: &TopicModel,
    doc: &[usize],
    num_particles: usize,
    rng: &mut R,
) -> f64 {
    let k = model.num_topics;
    let n = doc.len();
    if n == 0 {
        return 0.0;
    }
    let alpha = &model.alpha;
    let alpha_sum = model.alpha_sum;
    let phi_cache = build_phi_cache(model, doc);

    let mut word_prob = vec![0.0f64; n]; // accumulated across particles
    let mut weights = vec![0.0f64; k];

    for _ in 0..num_particles {
        let mut local = vec![0u32; k]; // per-document topic counts
        let mut z = vec![0usize; n]; // topic assigned to each position this pass

        for pos in 0..n {
            // Resample the topic of every earlier token given the current state.
            for prev in 0..pos {
                let phi = &phi_cache[&doc[prev]];
                local[z[prev]] -= 1;
                let mut total = 0.0;
                for t in 0..k {
                    let val = (alpha[t] + local[t] as f64) * phi[t];
                    weights[t] = val;
                    total += val;
                }
                let t_new = sample_categorical(&weights, total, rng);
                z[prev] = t_new;
                local[t_new] += 1;
            }

            // Score the current token: p(w_pos) = (Σ_t weight_t)/(alpha_sum+pos).
            let phi = &phi_cache[&doc[pos]];
            let mut total = 0.0;
            for t in 0..k {
                let val = (alpha[t] + local[t] as f64) * phi[t];
                weights[t] = val;
                total += val;
            }
            word_prob[pos] += total / (alpha_sum + pos as f64);

            // Sample this token's topic and fold it into the local counts.
            let t_new = sample_categorical(&weights, total, rng);
            z[pos] = t_new;
            local[t_new] += 1;
        }
    }

    let mut ll = 0.0;
    let r = num_particles as f64;
    for p in &word_prob {
        ll += (p / r).ln();
    }
    ll
}

/// UMass coherence for each topic's top-word list (descending by probability).
/// `C = Σ_{i>j} log[(codoc(w_i,w_j)+1) / docfreq(w_j)]`, intrinsic to the
/// training corpus.
fn umass_coherence(corpus: &corpus::Corpus, tops: &[Vec<usize>]) -> Vec<f64> {
    use std::collections::HashSet;

    let relevant: HashSet<usize> = tops.iter().flatten().copied().collect();
    let mut codoc: HashMap<(usize, usize), u32> = HashMap::new();

    for doc in &corpus.docs {
        let mut present: Vec<usize> = doc
            .iter()
            .map(|&w| w as usize)
            .filter(|w| relevant.contains(w))
            .collect();
        present.sort_unstable();
        present.dedup();
        for a in 0..present.len() {
            for b in (a + 1)..present.len() {
                *codoc.entry((present[a], present[b])).or_insert(0) += 1;
            }
        }
    }

    tops.iter()
        .map(|top| {
            let mut score = 0.0;
            for i in 1..top.len() {
                for j in 0..i {
                    let (wi, wj) = (top[i], top[j]); // wj is the more probable word
                    let key = if wi < wj { (wi, wj) } else { (wj, wi) };
                    let co = *codoc.get(&key).unwrap_or(&0) as f64;
                    let dfj = corpus.doc_freqs[wj].max(1) as f64;
                    score += ((co + 1.0) / dfj).ln();
                }
            }
            score
        })
        .collect()
}

/// Infer a document-topic distribution for a *new* document under a fixed
/// trained model (the MALLET TopicInferencer approach): run Gibbs over the
/// document's tokens, sampling each topic from
/// `(alpha_t + n_{t,doc}) * (beta + N_{w,t})/(beta_sum + tokens_per_topic_t)`
/// while the trained topic-word counts stay frozen, then average θ snapshots.
fn infer_doc<R: Rng>(
    model: &TopicModel,
    doc: &[usize],
    iterations: usize,
    burn_in: usize,
    num_samples: usize,
    sample_interval: usize,
    rng: &mut R,
) -> Vec<f64> {
    let k = model.num_topics;
    let alpha = &model.alpha;
    let alpha_sum = model.alpha_sum;
    let n = doc.len();

    let mut theta = vec![0.0f64; k];
    if n == 0 {
        // No in-vocabulary tokens: fall back to the prior.
        for t in 0..k {
            theta[t] = alpha[t] / alpha_sum;
        }
        return theta;
    }

    let phi_cache = build_phi_cache(model, doc);

    let mut local = vec![0u32; k];
    let mut z = vec![0usize; n];
    for i in 0..n {
        let t = rng.gen_range(0..k);
        z[i] = t;
        local[t] += 1;
    }

    let mut weights = vec![0.0f64; k];
    let mut samples_taken = 0usize;
    for iter in 1..=iterations {
        for i in 0..n {
            let phi = &phi_cache[&doc[i]];
            local[z[i]] -= 1;
            let mut total = 0.0;
            for t in 0..k {
                let v = (alpha[t] + local[t] as f64) * phi[t];
                weights[t] = v;
                total += v;
            }
            let t_new = sample_categorical(&weights, total, rng);
            z[i] = t_new;
            local[t_new] += 1;
        }
        if iter > burn_in
            && samples_taken < num_samples
            && (iter - burn_in) % sample_interval.max(1) == 0
        {
            let denom = n as f64 + alpha_sum;
            for t in 0..k {
                theta[t] += (local[t] as f64 + alpha[t]) / denom;
            }
            samples_taken += 1;
        }
    }

    if samples_taken == 0 {
        let denom = n as f64 + alpha_sum;
        for t in 0..k {
            theta[t] = (local[t] as f64 + alpha[t]) / denom;
        }
    } else {
        for t in theta.iter_mut() {
            *t /= samples_taken as f64;
        }
    }
    theta
}

/// Collapsed-Gibbs inference of a held-out document's topic proportions θ
/// against a *fixed* normalized topic-word matrix `phi` (K rows, each a
/// distribution over the vocabulary) and a Dirichlet prior `alpha` (length K).
/// This is the model-agnostic counterpart to [`infer_doc`]: any Gibbs model
/// that exposes a normalized topic-word matrix can reuse it for `transform`.
fn infer_theta_gibbs<R: Rng>(
    phi: &[Vec<f64>],
    alpha: &[f64],
    doc: &[usize],
    iterations: usize,
    burn_in: usize,
    num_samples: usize,
    sample_interval: usize,
    rng: &mut R,
) -> Vec<f64> {
    let k = phi.len();
    let alpha_sum: f64 = alpha.iter().sum();
    let n = doc.len();
    let mut theta = vec![0.0f64; k];
    if n == 0 {
        for t in 0..k {
            theta[t] = alpha[t] / alpha_sum;
        }
        return theta;
    }

    // Per-token phi column (probability of this word under each topic).
    let cols: Vec<Vec<f64>> = doc.iter().map(|&w| (0..k).map(|t| phi[t][w]).collect()).collect();

    let mut local = vec![0u32; k];
    let mut z = vec![0usize; n];
    for i in 0..n {
        let t = rng.gen_range(0..k);
        z[i] = t;
        local[t] += 1;
    }

    let mut weights = vec![0.0f64; k];
    let mut samples_taken = 0usize;
    for iter in 1..=iterations {
        for i in 0..n {
            let col = &cols[i];
            local[z[i]] -= 1;
            let mut total = 0.0;
            for t in 0..k {
                let v = (alpha[t] + local[t] as f64) * col[t];
                weights[t] = v;
                total += v;
            }
            let t_new = sample_categorical(&weights, total, rng);
            z[i] = t_new;
            local[t_new] += 1;
        }
        if iter > burn_in
            && samples_taken < num_samples
            && (iter - burn_in) % sample_interval.max(1) == 0
        {
            let denom = n as f64 + alpha_sum;
            for t in 0..k {
                theta[t] += (local[t] as f64 + alpha[t]) / denom;
            }
            samples_taken += 1;
        }
    }

    if samples_taken == 0 {
        let denom = n as f64 + alpha_sum;
        for t in 0..k {
            theta[t] = (local[t] as f64 + alpha[t]) / denom;
        }
    } else {
        for t in theta.iter_mut() {
            *t /= samples_taken as f64;
        }
    }
    theta
}

/// Batch wrapper for [`infer_theta_gibbs`]: maps new docs to ids, runs the
/// sampler per document (parallel, seeded deterministically per doc), and
/// returns a ``(num_docs, K)`` array. `alpha` is the length-K prior.
fn transform_gibbs<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    id_to_word: &[String],
    phi: &Array2<f64>,
    alpha: &[f64],
    iterations: usize,
    burn_in: usize,
    num_samples: usize,
    sample_interval: usize,
    base_seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let docs = docs_to_ids(data, id_to_word)?;
    let docs_usize: Vec<Vec<usize>> =
        docs.iter().map(|d| d.iter().map(|&w| w as usize).collect()).collect();
    let phi_rows: Vec<Vec<f64>> = phi.outer_iter().map(|r| r.to_vec()).collect();
    let alpha_v = alpha.to_vec();
    let k = phi_rows.len();

    let rows: Vec<Vec<f64>> = py.allow_threads(|| {
        docs_usize
            .par_iter()
            .enumerate()
            .map(|(i, d)| {
                let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(i as u64));
                infer_theta_gibbs(
                    &phi_rows, &alpha_v, d, iterations, burn_in, num_samples, sample_interval,
                    &mut rng,
                )
            })
            .collect()
    });

    let mut arr = Array2::<f64>::zeros((rows.len(), k));
    for (i, row) in rows.iter().enumerate() {
        for (t, &v) in row.iter().enumerate() {
            arr[[i, t]] = v;
        }
    }
    Ok(arr.to_pyarray_bound(py))
}

/// Jensen-Shannon divergence (base 2, in [0, 1]) between two distributions.
fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    let mut d = 0.0;
    for i in 0..p.len() {
        let m = 0.5 * (p[i] + q[i]);
        if p[i] > 0.0 && m > 0.0 {
            d += 0.5 * p[i] * (p[i] / m).log2();
        }
        if q[i] > 0.0 && m > 0.0 {
            d += 0.5 * q[i] * (q[i] / m).log2();
        }
    }
    d.max(0.0)
}

// ---------------------------------------------------------------------------
// DMR: Dirichlet-Multinomial Regression topic model
// ---------------------------------------------------------------------------

/// Per-document topic-count vectors `[num_docs][num_topics]`.
fn doc_topic_counts(doc_topics: &[Vec<u32>], k: usize) -> Vec<Vec<u32>> {
    doc_topics
        .iter()
        .map(|topics| {
            let mut c = vec![0u32; k];
            for &t in topics {
                c[t as usize] += 1;
            }
            c
        })
        .collect()
}

/// Convert a `Vec<Vec<f64>>` (rows) into an `Array2`.
fn vecs_to_arr2(rows: &[Vec<f64>]) -> Array2<f64> {
    let r = rows.len();
    let c = if r > 0 { rows[0].len() } else { 0 };
    let mut a = Array2::<f64>::zeros((r, c));
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            a[[i, j]] = v;
        }
    }
    a
}

/// Shared `top_words(n, topic=None)` implementation over a φ matrix: returns a
/// list of `(word, prob)` for one topic, or a list of those lists for all.
fn topic_words_helper<'py>(
    py: Python<'py>,
    beta: &Array2<f64>,
    vocab: &[String],
    num_topics: usize,
    n: usize,
    topic: Option<usize>,
) -> PyResult<Bound<'py, PyAny>> {
    let tops = top_word_ids_phi(beta, num_topics, n);
    let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
        if t >= num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        let items: Vec<Bound<'py, PyTuple>> = tops[t]
            .iter()
            .map(|&w| PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), beta[[t, w]].into_py(py)]))
            .collect();
        Ok(PyList::new_bound(py, items))
    };
    match topic {
        Some(t) => Ok(one(t)?.into_any()),
        None => {
            let all: Vec<Bound<'py, PyList>> = (0..num_topics).map(one).collect::<PyResult<_>>()?;
            Ok(PyList::new_bound(py, all).into_any())
        }
    }
}

/// Top-`n` word ids per topic from a (num_topics, num_words) φ matrix.
fn top_word_ids_phi(phi: &Array2<f64>, num_topics: usize, n: usize) -> Vec<Vec<usize>> {
    let w = phi.shape()[1];
    (0..num_topics)
        .map(|t| {
            let mut idx: Vec<usize> = (0..w).collect();
            idx.sort_by(|&a, &b| phi[[t, b]].partial_cmp(&phi[[t, a]]).unwrap());
            idx.truncate(n);
            idx
        })
        .collect()
}

/// Parse a feature matrix (a 2D numpy float array or a list of float lists)
/// into `[num_docs][num_features]`.
fn parse_features(data: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f64>>> {
    if let Ok(arr) = data.extract::<PyReadonlyArray2<f64>>() {
        let a = arr.as_array();
        let (rows, cols) = (a.shape()[0], a.shape()[1]);
        return Ok((0..rows)
            .map(|i| (0..cols).map(|j| a[[i, j]]).collect())
            .collect());
    }
    data.extract::<Vec<Vec<f64>>>().map_err(|_| {
        PyValueError::new_err("features must be a 2D float array or a list of float lists")
    })
}

/// Dirichlet-Multinomial Regression topic model (Mimno & McCallum, 2008).
///
/// Like :class:`LDA`, but the per-document topic prior is a log-linear function
/// of document features: ``α_{d,t} = exp(λ_t · x_d)``. After fitting, the
/// learned weights are available as :attr:`feature_effects` — how each covariate
/// shifts each topic's prevalence.
#[pyclass(module = "topica")]
pub struct DMR {
    num_topics: usize,
    beta: f64,
    optimize_interval: usize,
    burn_in: usize,
    seed: u64,
    prior_variance: f64,
    lbfgs_iters: usize,

    fitted: bool,
    phi: Option<Array2<f64>>,            // (num_topics, num_words)
    theta: Option<Array2<f64>>,          // (num_docs, num_topics)
    feature_effects: Option<Array2<f64>>, // (num_topics, num_features)
    feature_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
}

impl DMR {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl DMR {
    /// Create an unfitted DMR model. `prior_variance` is the Gaussian prior
    /// variance σ² on the feature weights λ (smaller = stronger shrinkage);
    /// `lbfgs_iters` caps the L-BFGS steps per optimization round.
    #[new]
    #[pyo3(signature = (num_topics, *, beta=0.01, optimize_interval=50,
                        burn_in=200, seed=42, prior_variance=1.0, lbfgs_iters=20))]
    fn new(
        num_topics: usize,
        beta: f64,
        optimize_interval: usize,
        burn_in: usize,
        seed: u64,
        prior_variance: f64,
        lbfgs_iters: usize,
    ) -> PyResult<Self> {
        if num_topics == 0 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        if beta <= 0.0 {
            return Err(PyValueError::new_err("beta must be > 0"));
        }
        if prior_variance <= 0.0 {
            return Err(PyValueError::new_err("prior_variance must be > 0"));
        }
        Ok(DMR {
            num_topics,
            beta,
            optimize_interval,
            burn_in,
            seed,
            prior_variance,
            lbfgs_iters,
            fitted: false,
            phi: None,
            theta: None,
            feature_effects: None,
            feature_names: Vec::new(),
            corpus: None,
        })
    }

    /// Fit the model. `data` is a :class:`Corpus` or `list[list[str]]`;
    /// `features` is a `(num_docs, F)` numpy array or list of float lists (an
    /// intercept column is prepended automatically). `feature_names` (length F)
    /// names the columns; an "intercept" name is prepended.
    #[pyo3(signature = (data, features, *, feature_names=None, iterations=1000,
                        num_samples=5, sample_interval=25, progress=None, progress_interval=50))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        features: &Bound<'_, PyAny>,
        feature_names: Option<Vec<String>>,
        iterations: usize,
        num_samples: usize,
        sample_interval: usize,
        progress: Option<PyObject>,
        progress_interval: usize,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }

        let raw = parse_features(features)?;
        if raw.len() != corpus.num_docs() {
            return Err(PyValueError::new_err(format!(
                "features has {} rows but corpus has {} documents",
                raw.len(),
                corpus.num_docs()
            )));
        }
        let f_in = raw.first().map(|r| r.len()).unwrap_or(0);
        if raw.iter().any(|r| r.len() != f_in) {
            return Err(PyValueError::new_err("all feature rows must have the same length"));
        }
        if let Some(names) = &feature_names {
            if names.len() != f_in {
                return Err(PyValueError::new_err(format!(
                    "feature_names has {} entries but features has {} columns",
                    names.len(),
                    f_in
                )));
            }
        }

        // Prepend an intercept column.
        let nf = f_in + 1;
        let feats: Vec<Vec<f64>> = raw
            .iter()
            .map(|x| {
                let mut v = Vec::with_capacity(nf);
                v.push(1.0);
                v.extend_from_slice(x);
                v
            })
            .collect();
        let mut names = vec!["intercept".to_string()];
        names.extend(
            feature_names.unwrap_or_else(|| (0..f_in).map(|i| format!("feature_{}", i)).collect()),
        );

        let k = self.num_topics;
        let num_types = corpus.num_types();
        let num_docs = corpus.num_docs();
        let total_tokens = corpus.total_tokens().max(1) as f64;

        // λ starts at zero -> α ≡ 1 (symmetric) before optimization kicks in.
        let mut lambda = vec![vec![0.0f64; nf]; k];
        let mut model = TopicModel::new(k, k as f64, self.beta, num_types);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        model.initialize(&corpus, &mut rng);

        let optimize_interval = self.optimize_interval;
        let burn_in = self.burn_in;
        let prior_variance = self.prior_variance;
        let lbfgs_iters = self.lbfgs_iters;

        let (acc_phi, acc_theta, feat_eff, model, corpus) = py.allow_threads(move || {
            for iter in 1..=iterations {
                let doc_alpha = dmr::compute_doc_alpha(&lambda, &feats);
                dmr::run_sweep_dmr(
                    &mut model.type_topic_counts,
                    &mut model.tokens_per_topic,
                    &mut model.doc_topics,
                    &corpus.docs,
                    &doc_alpha,
                    model.beta,
                    model.beta_sum,
                    model.topic_mask,
                    model.topic_bits,
                    k,
                    &mut rng,
                );

                if optimize_interval > 0 && iter > burn_in && iter % optimize_interval == 0 {
                    let dtc = doc_topic_counts(&model.doc_topics, k);
                    dmr::optimize_lambda(
                        &mut lambda, &feats, &dtc, k, nf, prior_variance, lbfgs_iters,
                    );
                }

                if let Some(cb) = &progress {
                    if progress_interval > 0 && iter % progress_interval == 0 {
                        let dtc = doc_topic_counts(&model.doc_topics, k);
                        let (ll, _) = dmr::dmr_objective_and_gradient(
                            &lambda, &feats, &dtc, k, nf, prior_variance,
                        );
                        let llpt = ll / total_tokens;
                        Python::with_gil(|py| {
                            let _ = cb.call1(py, (iter, llpt));
                        });
                    }
                }
            }

            // Sampling phase: λ is now fixed, so α per doc is fixed too.
            let doc_alpha = dmr::compute_doc_alpha(&lambda, &feats);
            let mut acc_phi = vec![vec![0.0f64; k]; num_types];
            let mut acc_theta = vec![vec![0.0f64; k]; num_docs];

            for _ in 0..num_samples {
                for _ in 0..sample_interval {
                    dmr::run_sweep_dmr(
                        &mut model.type_topic_counts,
                        &mut model.tokens_per_topic,
                        &mut model.doc_topics,
                        &corpus.docs,
                        &doc_alpha,
                        model.beta,
                        model.beta_sum,
                        model.topic_mask,
                        model.topic_bits,
                        k,
                        &mut rng,
                    );
                }
                accumulate_phi(&model, &mut acc_phi);
                // DMR θ uses the per-document prior.
                let counts = doc_topic_counts(&model.doc_topics, k);
                for d in 0..num_docs {
                    let asum: f64 = doc_alpha[d].iter().sum();
                    let denom = corpus.docs[d].len() as f64 + asum;
                    for t in 0..k {
                        acc_theta[d][t] += (counts[d][t] as f64 + doc_alpha[d][t]) / denom;
                    }
                }
            }

            let n = num_samples.max(1) as f64;
            for row in acc_phi.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }
            for row in acc_theta.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }

            (acc_phi, acc_theta, lambda, model, corpus)
        });
        let _ = model;

        let mut phi = Array2::<f64>::zeros((k, num_types));
        for (w, row) in acc_phi.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                phi[[t, w]] = val;
            }
        }
        let mut theta = Array2::<f64>::zeros((num_docs, k));
        for (d, row) in acc_theta.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[d, t]] = val;
            }
        }
        let mut fe = Array2::<f64>::zeros((k, nf));
        for (t, row) in feat_eff.iter().enumerate() {
            for (f, &val) in row.iter().enumerate() {
                fe[[t, f]] = val;
            }
        }

        self.phi = Some(phi);
        self.theta = Some(theta);
        self.feature_effects = Some(fe);
        self.feature_names = names;
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix φ, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Learned feature weights λ, shape ``(num_topics, num_features)`` — how
    /// each feature (column 0 is the intercept) shifts each topic's log-prior.
    /// Positive ⇒ the feature raises that topic's prevalence.
    #[getter]
    fn feature_effects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.feature_effects.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Feature names aligned with the columns of :attr:`feature_effects`
    /// (``"intercept"`` first).
    #[getter]
    fn feature_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.feature_names.clone())
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// Top `n` words per topic as ``(word, probability)`` pairs (all topics, or
    /// one when `topic` is given).
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let phi = self.phi.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tops = top_word_ids_phi(phi, self.num_topics, n);

        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err(format!(
                    "topic {} out of range (num_topics={})",
                    t, self.num_topics
                )));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| {
                    PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), phi[[t, w]].into_py(py)])
                })
                .collect();
            Ok(PyList::new_bound(py, items))
        };

        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Infer topic proportions θ for *new* documents by collapsed Gibbs against
    /// the fitted topic-word matrix. `data` is a :class:`Corpus` or
    /// `list[list[str]]`; OOV tokens are dropped. `features` (optional, a
    /// ``(num_docs, F)`` covariate array matching training, no intercept) sets
    /// each document's Dirichlet prior `α_d = exp(Xγ)`; if omitted the
    /// intercept-only baseline prior is used. Returns ``(num_docs, num_topics)``.
    #[pyo3(signature = (data, features=None, *, iterations=100, burn_in=10,
                        num_samples=10, sample_interval=5, seed=None))]
    #[allow(clippy::too_many_arguments)]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        features: Option<PyReadonlyArray2<f64>>,
        iterations: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let k = self.num_topics;
        let eff = self.feature_effects.as_ref().unwrap(); // (K, F) incl. intercept at col 0
        let nf = eff.shape()[1];
        let id_to_word = &self.corpus.as_ref().unwrap().id_to_word;
        let docs = docs_to_ids(data, id_to_word)?;
        let docs_usize: Vec<Vec<usize>> =
            docs.iter().map(|d| d.iter().map(|&w| w as usize).collect()).collect();
        let phi_rows: Vec<Vec<f64>> =
            self.phi.as_ref().unwrap().outer_iter().map(|r| r.to_vec()).collect();

        // Per-document Dirichlet prior α_d = exp(Xγ); intercept is column 0.
        let alphas: Vec<Vec<f64>> = match &features {
            Some(x) => {
                let x = x.as_array();
                if x.shape()[0] != docs_usize.len() {
                    return Err(PyValueError::new_err(
                        "features rows must match number of documents",
                    ));
                }
                if x.shape()[1] + 1 != nf {
                    return Err(PyValueError::new_err(format!(
                        "features must have {} columns (the {} training covariates, no intercept)",
                        nf - 1,
                        nf - 1
                    )));
                }
                (0..docs_usize.len())
                    .map(|d| {
                        (0..k)
                            .map(|t| {
                                let mut s = eff[[t, 0]];
                                for f in 1..nf {
                                    s += eff[[t, f]] * x[[d, f - 1]];
                                }
                                s.exp()
                            })
                            .collect()
                    })
                    .collect()
            }
            None => {
                let base: Vec<f64> = (0..k).map(|t| eff[[t, 0]].exp()).collect();
                vec![base; docs_usize.len()]
            }
        };

        let base_seed = seed.unwrap_or(self.seed);
        let rows: Vec<Vec<f64>> = py.allow_threads(|| {
            docs_usize
                .par_iter()
                .zip(alphas.par_iter())
                .enumerate()
                .map(|(i, (d, alpha))| {
                    let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(i as u64));
                    infer_theta_gibbs(
                        &phi_rows, alpha, d, iterations, burn_in, num_samples, sample_interval,
                        &mut rng,
                    )
                })
                .collect()
        });
        let mut arr = Array2::<f64>::zeros((rows.len(), k));
        for (i, row) in rows.iter().enumerate() {
            for (t, &v) in row.iter().enumerate() {
                arr[[i, t]] = v;
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// Save the fitted model to `path` (compact binary). Reload with `DMR.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &DmrState {
            num_topics: self.num_topics, beta: self.beta,
            optimize_interval: self.optimize_interval, burn_in: self.burn_in, seed: self.seed,
            prior_variance: self.prior_variance, lbfgs_iters: self.lbfgs_iters, fitted: self.fitted,
            phi: arr2_opt(&self.phi), theta: arr2_opt(&self.theta),
            feature_effects: arr2_opt(&self.feature_effects),
            feature_names: self.feature_names.clone(), corpus: self.corpus.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: DmrState = read_state(path)?;
        Ok(DMR {
            num_topics: s.num_topics, beta: s.beta, optimize_interval: s.optimize_interval,
            burn_in: s.burn_in, seed: s.seed, prior_variance: s.prior_variance,
            lbfgs_iters: s.lbfgs_iters, fitted: s.fitted,
            phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            feature_effects: arr2_back(s.feature_effects),
            feature_names: s.feature_names, corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!("DMR(num_topics={}, fitted={})", self.num_topics, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// Labeled LDA
// ---------------------------------------------------------------------------

/// Supervised topic model (Ramage et al., 2009): each document carries a set of
/// labels, each label is a topic, and a document's tokens are constrained to its
/// labels' topics. The number of topics is the number of distinct labels.
///
/// Documents with an empty label set are treated as unconstrained (all topics).
#[pyclass(module = "topica")]
pub struct LabeledLDA {
    alpha: f64,
    beta: f64,
    seed: u64,

    fitted: bool,
    num_topics: usize,
    phi: Option<Array2<f64>>,
    theta: Option<Array2<f64>>,
    label_vocab: Vec<String>,
    corpus: Option<corpus::Corpus>,
}

impl LabeledLDA {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl LabeledLDA {
    /// Create an unfitted model. `alpha` is the (symmetric) per-topic prior
    /// over a document's allowed topics.
    #[new]
    #[pyo3(signature = (*, alpha=0.1, beta=0.01, seed=42))]
    fn new(alpha: f64, beta: f64, seed: u64) -> PyResult<Self> {
        if alpha <= 0.0 {
            return Err(PyValueError::new_err("alpha must be > 0"));
        }
        if beta <= 0.0 {
            return Err(PyValueError::new_err("beta must be > 0"));
        }
        Ok(LabeledLDA {
            alpha,
            beta,
            seed,
            fitted: false,
            num_topics: 0,
            phi: None,
            theta: None,
            label_vocab: Vec::new(),
            corpus: None,
        })
    }

    /// Fit the model. `data` is a :class:`Corpus` or `list[list[str]]`;
    /// `labels` is a list (one per document) of label lists. The topic set is
    /// the union of all labels (or `label_names`, which also fixes topic order).
    /// An empty label list leaves that document unconstrained.
    #[pyo3(signature = (data, labels, *, label_names=None, iterations=1000,
                        num_samples=5, sample_interval=25, progress=None, progress_interval=50))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        labels: Vec<Vec<String>>,
        label_names: Option<Vec<String>>,
        iterations: usize,
        num_samples: usize,
        sample_interval: usize,
        progress: Option<PyObject>,
        progress_interval: usize,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?
        };
        let num_docs = corpus.num_docs();
        if num_docs == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        if labels.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "labels has {} entries but corpus has {} documents",
                labels.len(),
                num_docs
            )));
        }

        // Topic vocabulary: provided order, or the sorted union of all labels.
        let label_vocab: Vec<String> = match label_names {
            Some(n) => n,
            None => {
                let mut set: HashSet<String> = HashSet::new();
                for ls in &labels {
                    for l in ls {
                        set.insert(l.clone());
                    }
                }
                let mut v: Vec<String> = set.into_iter().collect();
                v.sort();
                v
            }
        };
        if label_vocab.is_empty() {
            return Err(PyValueError::new_err("no labels found; provide labels or label_names"));
        }
        let k = label_vocab.len();
        let index: HashMap<&str, usize> = label_vocab
            .iter()
            .enumerate()
            .map(|(i, l)| (l.as_str(), i))
            .collect();

        let allowed: Vec<Vec<usize>> = labels
            .iter()
            .map(|ls| {
                let mut v: Vec<usize> =
                    ls.iter().filter_map(|l| index.get(l.as_str()).copied()).collect();
                v.sort_unstable();
                v.dedup();
                v
            })
            .collect();

        let num_types = corpus.num_types();
        let total_tokens = corpus.total_tokens().max(1) as f64;
        let alpha_sum = self.alpha * k as f64;
        let mut model = TopicModel::new(k, alpha_sum, self.beta, num_types);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        labeled::initialize_labeled(&mut model, &corpus.docs, &allowed, &mut rng);

        let (acc_phi, acc_theta, model, corpus) = py.allow_threads(move || {
            for iter in 1..=iterations {
                labeled::run_sweep_labeled(&mut model, &corpus.docs, &allowed, &mut rng);
                if let Some(cb) = &progress {
                    if progress_interval > 0 && iter % progress_interval == 0 {
                        let ll = output::model_log_likelihood(&model, &corpus) / total_tokens;
                        Python::with_gil(|py| {
                            let _ = cb.call1(py, (iter, ll));
                        });
                    }
                }
            }

            let all_topics: Vec<usize> = (0..k).collect();
            let mut acc_phi = vec![vec![0.0f64; k]; num_types];
            let mut acc_theta = vec![vec![0.0f64; k]; num_docs];
            for _ in 0..num_samples {
                for _ in 0..sample_interval {
                    labeled::run_sweep_labeled(&mut model, &corpus.docs, &allowed, &mut rng);
                }
                accumulate_phi(&model, &mut acc_phi);
                let counts = doc_topic_counts(&model.doc_topics, k);
                for d in 0..num_docs {
                    let allow: &[usize] = if allowed[d].is_empty() {
                        &all_topics
                    } else {
                        &allowed[d]
                    };
                    let asum: f64 = allow.iter().map(|&t| model.alpha[t]).sum();
                    let denom = corpus.docs[d].len() as f64 + asum;
                    for &t in allow {
                        acc_theta[d][t] += (counts[d][t] as f64 + model.alpha[t]) / denom;
                    }
                }
            }

            let n = num_samples.max(1) as f64;
            for row in acc_phi.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }
            for row in acc_theta.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }
            (acc_phi, acc_theta, model, corpus)
        });
        let _ = model;

        let mut phi = Array2::<f64>::zeros((k, num_types));
        for (w, row) in acc_phi.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                phi[[t, w]] = val;
            }
        }
        let mut theta = Array2::<f64>::zeros((num_docs, k));
        for (d, row) in acc_theta.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[d, t]] = val;
            }
        }

        self.num_topics = k;
        self.phi = Some(phi);
        self.theta = Some(theta);
        self.label_vocab = label_vocab;
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix φ, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; for each
    /// document only its label topics are non-zero, and rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// The label name for each topic, in topic (column) order.
    #[getter]
    fn labels(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.label_vocab.clone())
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[getter]
    fn num_topics(&self) -> PyResult<usize> {
        self.require_fitted()?;
        Ok(self.num_topics)
    }

    /// Top `n` words for one topic (by label name or index) or all topics.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let phi = self.phi.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tops = top_word_ids_phi(phi, self.num_topics, n);

        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err(format!(
                    "topic {} out of range (num_topics={})",
                    t, self.num_topics
                )));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| {
                    PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), phi[[t, w]].into_py(py)])
                })
                .collect();
            Ok(PyList::new_bound(py, items))
        };

        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Infer label (topic) proportions θ for *new* documents by collapsed Gibbs
    /// against the fitted topic-word matrix, treating every label as available
    /// (unsupervised inference). `data` is a :class:`Corpus` or
    /// `list[list[str]]`; OOV tokens are dropped. Returns ``(num_docs,
    /// num_topics)``; columns align with :attr:`labels`.
    #[pyo3(signature = (data, *, iterations=100, burn_in=10, num_samples=10,
                        sample_interval=5, seed=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        iterations: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let alpha = vec![self.alpha; self.num_topics];
        transform_gibbs(
            py, data, &self.corpus.as_ref().unwrap().id_to_word, self.phi.as_ref().unwrap(),
            &alpha, iterations, burn_in, num_samples, sample_interval,
            seed.unwrap_or(self.seed),
        )
    }

    /// Save the fitted model to `path`. Reload with `LabeledLDA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &LabeledState {
            alpha: self.alpha, beta: self.beta, seed: self.seed, fitted: self.fitted,
            num_topics: self.num_topics, phi: arr2_opt(&self.phi), theta: arr2_opt(&self.theta),
            label_vocab: self.label_vocab.clone(), corpus: self.corpus.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: LabeledState = read_state(path)?;
        Ok(LabeledLDA {
            alpha: s.alpha, beta: s.beta, seed: s.seed, fitted: s.fitted,
            num_topics: s.num_topics, phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            label_vocab: s.label_vocab, corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "LabeledLDA(num_topics={}, fitted={})",
            self.num_topics, self.fitted
        )
    }
}

// ---------------------------------------------------------------------------
// SAGE: content-covariate topic model
// ---------------------------------------------------------------------------

/// Parse a per-document group covariate (list of strings or ints) to strings.
fn parse_groups(obj: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(v) = obj.extract::<Vec<String>>() {
        return Ok(v);
    }
    if let Ok(v) = obj.extract::<Vec<i64>>() {
        return Ok(v.iter().map(|x| x.to_string()).collect());
    }
    Err(PyValueError::new_err("groups must be a list of strings or ints"))
}

/// Content-covariate topic model (SAGE / the STM content model).
///
/// Topics are shared, but each topic's word distribution varies by a
/// document-level **group** covariate, so you can read how a topic is worded
/// differently across groups. Construct, then :meth:`fit` on documents plus a
/// per-document group label.
#[pyclass(module = "topica")]
pub struct SAGE {
    num_topics: usize,
    alpha: f64,
    prior_variance: f64,
    optimize_interval: usize,
    burn_in: usize,
    seed: u64,
    lbfgs_iters: usize,

    fitted: bool,
    num_groups: usize,
    beta: Vec<Vec<f64>>, // [K*G][V]
    theta: Option<Array2<f64>>,
    group_names: Vec<String>,
    corpus: Option<corpus::Corpus>,
}

impl SAGE {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }

    /// β for (topic, group) averaged over groups → a plain (K, V) topic-word.
    fn topic_marginal(&self) -> Array2<f64> {
        let k = self.num_topics;
        let g = self.num_groups;
        let v = self.corpus.as_ref().unwrap().num_types();
        let mut out = Array2::<f64>::zeros((k, v));
        for kk in 0..k {
            for gg in 0..g {
                let cell = &self.beta[kk * g + gg];
                for vv in 0..v {
                    out[[kk, vv]] += cell[vv] / g as f64;
                }
            }
        }
        out
    }
}

#[pymethods]
impl SAGE {
    /// Create an unfitted model. `alpha` is the symmetric document-topic prior;
    /// `prior_variance` is the Gaussian prior on the κ content deviations.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha=0.1, prior_variance=1.0,
                        optimize_interval=50, burn_in=100, seed=42, lbfgs_iters=20))]
    fn new(
        num_topics: usize,
        alpha: f64,
        prior_variance: f64,
        optimize_interval: usize,
        burn_in: usize,
        seed: u64,
        lbfgs_iters: usize,
    ) -> PyResult<Self> {
        if num_topics == 0 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        if prior_variance <= 0.0 {
            return Err(PyValueError::new_err("prior_variance must be > 0"));
        }
        Ok(SAGE {
            num_topics,
            alpha,
            prior_variance,
            optimize_interval,
            burn_in,
            seed,
            lbfgs_iters,
            fitted: false,
            num_groups: 0,
            beta: Vec::new(),
            theta: None,
            group_names: Vec::new(),
            corpus: None,
        })
    }

    /// Fit the model. `data` is a :class:`Corpus` or `list[list[str]]`;
    /// `groups` is a per-document group label (strings or ints), one per
    /// document. `group_names` fixes the group order (defaults to sorted union).
    #[pyo3(signature = (data, groups, *, group_names=None, iterations=1000,
                        num_samples=5, sample_interval=25, progress=None, progress_interval=50))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        groups: &Bound<'_, PyAny>,
        group_names: Option<Vec<String>>,
        iterations: usize,
        num_samples: usize,
        sample_interval: usize,
        progress: Option<PyObject>,
        progress_interval: usize,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?
        };
        let num_docs = corpus.num_docs();
        if num_docs == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }

        let groups_str = parse_groups(groups)?;
        if groups_str.len() != num_docs {
            return Err(PyValueError::new_err(format!(
                "groups has {} entries but corpus has {} documents",
                groups_str.len(),
                num_docs
            )));
        }

        let group_vocab: Vec<String> = match group_names {
            Some(n) => n,
            None => {
                let mut set: HashSet<String> = groups_str.iter().cloned().collect();
                let mut v: Vec<String> = set.drain().collect();
                v.sort();
                v
            }
        };
        let gindex: HashMap<&str, usize> = group_vocab
            .iter()
            .enumerate()
            .map(|(i, g)| (g.as_str(), i))
            .collect();
        let groups_idx: Vec<usize> = groups_str
            .iter()
            .map(|g| {
                gindex
                    .get(g.as_str())
                    .copied()
                    .ok_or_else(|| PyValueError::new_err(format!("group {:?} not in group_names", g)))
            })
            .collect::<PyResult<_>>()?;

        let k = self.num_topics;
        let group_n = group_vocab.len();
        let num_types = corpus.num_types();
        let alpha = self.alpha;
        let alpha_sum = alpha * k as f64;
        let total_tokens = corpus.total_tokens().max(1) as f64;

        let mut model = sage::SageModel::new(k, group_n, num_types, alpha, self.prior_variance);
        model.set_background(&corpus.docs);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        model.initialize(&corpus.docs, &groups_idx, &mut rng);

        let optimize_interval = self.optimize_interval;
        let burn_in = self.burn_in;
        let lbfgs_iters = self.lbfgs_iters;

        let (beta, acc_theta, corpus) = py.allow_threads(move || {
            for iter in 1..=iterations {
                sage::run_sweep_sage(&mut model, &corpus.docs, &groups_idx, &mut rng);
                if optimize_interval > 0 && iter > burn_in && iter % optimize_interval == 0 {
                    sage::optimize_kappa(&mut model, lbfgs_iters);
                }
                if let Some(cb) = &progress {
                    if progress_interval > 0 && iter % progress_interval == 0 {
                        // Data log-likelihood under the current β, per token.
                        let mut ll = 0.0;
                        for c in 0..(k * group_n) {
                            for v in 0..num_types {
                                let n = model.counts[c][v] as f64;
                                if n > 0.0 {
                                    ll += n * model.beta[c][v].max(1e-300).ln();
                                }
                            }
                        }
                        let llpt = ll / total_tokens;
                        Python::with_gil(|py| {
                            let _ = cb.call1(py, (iter, llpt));
                        });
                    }
                }
            }
            sage::optimize_kappa(&mut model, lbfgs_iters); // final β refresh

            let mut acc_theta = vec![vec![0.0f64; k]; num_docs];
            for _ in 0..num_samples {
                for _ in 0..sample_interval {
                    sage::run_sweep_sage(&mut model, &corpus.docs, &groups_idx, &mut rng);
                }
                let counts = doc_topic_counts(&model.doc_topics, k);
                for d in 0..num_docs {
                    let denom = corpus.docs[d].len() as f64 + alpha_sum;
                    for t in 0..k {
                        acc_theta[d][t] += (counts[d][t] as f64 + alpha) / denom;
                    }
                }
            }
            let n = num_samples.max(1) as f64;
            for row in acc_theta.iter_mut() {
                for v in row.iter_mut() {
                    *v /= n;
                }
            }
            (model.beta.clone(), acc_theta, corpus)
        });

        let mut theta = Array2::<f64>::zeros((num_docs, k));
        for (d, row) in acc_theta.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[d, t]] = val;
            }
        }

        self.num_groups = group_n;
        self.beta = beta;
        self.theta = Some(theta);
        self.group_names = group_vocab;
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// Topic-word distributions per group, shape ``(num_topics, num_groups, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        self.require_fitted()?;
        let k = self.num_topics;
        let g = self.num_groups;
        let v = self.corpus.as_ref().unwrap().num_types();
        let mut arr = Array3::<f64>::zeros((k, g, v));
        for kk in 0..k {
            for gg in 0..g {
                let cell = &self.beta[kk * g + gg];
                for vv in 0..v {
                    arr[[kk, gg, vv]] = cell[vv];
                }
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// Group-averaged topic-word matrix, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word_marginal<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.topic_marginal().to_pyarray_bound(py))
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Group names, in the index order used by :attr:`topic_word`'s second axis.
    #[getter]
    fn groups(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.group_names.clone())
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    #[getter]
    fn num_groups(&self) -> PyResult<usize> {
        self.require_fitted()?;
        Ok(self.num_groups)
    }

    /// Top `n` words for a topic. With `group` (name or index) given, uses that
    /// group's word distribution; otherwise the group-averaged distribution.
    #[pyo3(signature = (topic, *, group=None, n=10))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        topic: usize,
        group: Option<&Bound<'py, PyAny>>,
        n: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        self.require_fitted()?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err(format!(
                "topic {} out of range (num_topics={})",
                topic, self.num_topics
            )));
        }
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let dist: Vec<f64> = match group {
            Some(gobj) => {
                let gi = self.resolve_group(gobj)?;
                self.beta[topic * self.num_groups + gi].clone()
            }
            None => {
                let m = self.topic_marginal();
                (0..vocab.len()).map(|v| m[[topic, v]]).collect()
            }
        };
        let mut idx: Vec<usize> = (0..vocab.len()).collect();
        idx.sort_by(|&a, &b| dist[b].partial_cmp(&dist[a]).unwrap());
        let items: Vec<Bound<'py, PyTuple>> = idx
            .iter()
            .take(n)
            .map(|&v| PyTuple::new_bound(py, &[vocab[v].clone().into_py(py), dist[v].into_py(py)]))
            .collect();
        Ok(PyList::new_bound(py, items))
    }

    /// Words that most distinguish how `topic` is worded in `group_a` vs
    /// `group_b`, by log-ratio of the two groups' word probabilities. Returns
    /// ``(word, log_ratio)`` — positive favours `group_a`.
    #[pyo3(signature = (topic, group_a, group_b, n=10))]
    fn word_contrast<'py>(
        &self,
        py: Python<'py>,
        topic: usize,
        group_a: &Bound<'py, PyAny>,
        group_b: &Bound<'py, PyAny>,
        n: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        self.require_fitted()?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        let ga = self.resolve_group(group_a)?;
        let gb = self.resolve_group(group_b)?;
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let a = &self.beta[topic * self.num_groups + ga];
        let b = &self.beta[topic * self.num_groups + gb];
        let ratio: Vec<f64> = (0..vocab.len())
            .map(|v| (a[v].max(1e-300) / b[v].max(1e-300)).ln())
            .collect();
        let mut idx: Vec<usize> = (0..vocab.len()).collect();
        idx.sort_by(|&x, &y| ratio[y].partial_cmp(&ratio[x]).unwrap());
        let items: Vec<Bound<'py, PyTuple>> = idx
            .iter()
            .take(n)
            .map(|&v| PyTuple::new_bound(py, &[vocab[v].clone().into_py(py), ratio[v].into_py(py)]))
            .collect();
        Ok(PyList::new_bound(py, items))
    }

    /// UMass topic coherence per topic (group-averaged), shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(&self.topic_marginal(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `SAGE.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &SageState {
            num_topics: self.num_topics, alpha: self.alpha, prior_variance: self.prior_variance,
            optimize_interval: self.optimize_interval, burn_in: self.burn_in, seed: self.seed,
            lbfgs_iters: self.lbfgs_iters, fitted: self.fitted, num_groups: self.num_groups,
            beta: self.beta.clone(), theta: arr2_opt(&self.theta),
            group_names: self.group_names.clone(), corpus: self.corpus.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: SageState = read_state(path)?;
        Ok(SAGE {
            num_topics: s.num_topics, alpha: s.alpha, prior_variance: s.prior_variance,
            optimize_interval: s.optimize_interval, burn_in: s.burn_in, seed: s.seed,
            lbfgs_iters: s.lbfgs_iters, fitted: s.fitted, num_groups: s.num_groups,
            beta: s.beta, theta: arr2_back(s.theta), group_names: s.group_names, corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "SAGE(num_topics={}, num_groups={}, fitted={})",
            self.num_topics, self.num_groups, self.fitted
        )
    }
}

impl SAGE {
    /// Resolve a group given as a name (str) or an index (int) to its index.
    fn resolve_group(&self, obj: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(i) = obj.extract::<usize>() {
            if i < self.num_groups {
                return Ok(i);
            }
            return Err(PyValueError::new_err(format!(
                "group index {} out of range (num_groups={})",
                i, self.num_groups
            )));
        }
        if let Ok(s) = obj.extract::<String>() {
            return self
                .group_names
                .iter()
                .position(|g| g == &s)
                .ok_or_else(|| PyValueError::new_err(format!("unknown group {:?}", s)));
        }
        Err(PyValueError::new_err("group must be a name (str) or index (int)"))
    }
}

// ---------------------------------------------------------------------------
// CTM: Correlated Topic Model (STM's logistic-normal variational core)
// ---------------------------------------------------------------------------

/// Extract the per-document variational posterior of η from a fitted CTM/STM:
/// the means λ (D × K-1) and covariances ν (D × K-1 × K-1). These define the
/// logistic-normal posterior `η_d ~ N(λ_d, ν_d)` used for sampling θ draws
/// (method-of-composition uncertainty).
/// Map new documents (a `Corpus` or `list[list[str]]`) onto the training
/// vocabulary, dropping out-of-vocabulary tokens. Tokens are lowercased to
/// match the corpus loader. Returns one `Vec<u32>` of word-ids per document.
fn docs_to_ids(
    data: &Bound<'_, PyAny>,
    id_to_word: &[String],
) -> PyResult<Vec<Vec<u32>>> {
    let word_to_id: HashMap<&str, u32> = id_to_word
        .iter()
        .enumerate()
        .map(|(i, w)| (w.as_str(), i as u32))
        .collect();
    let str_docs: Vec<Vec<String>> = if let Ok(c) = data.extract::<Corpus>() {
        c.inner
            .docs
            .iter()
            .map(|d| d.iter().map(|&w| c.inner.id_to_word[w as usize].clone()).collect())
            .collect()
    } else {
        data.extract().map_err(|_| {
            PyValueError::new_err("transform() expects a Corpus or a list of token lists")
        })?
    };
    Ok(str_docs
        .into_iter()
        .map(|doc| {
            doc.iter()
                .filter_map(|t| word_to_id.get(t.to_lowercase().as_str()).copied())
                .collect()
        })
        .collect())
}

/// Run the CTM/STM variational E-step inference for a batch of documents,
/// returning their topic proportions θ as a ``(num_docs, K)`` array. Parallel
/// over documents; the per-doc result is independent so order is preserved.
fn infer_theta_batch(
    py: Python<'_>,
    beta: &[Vec<f64>],
    mu: &[f64],
    sigma: &[f64],
    docs: &[Vec<u32>],
) -> Array2<f64> {
    let k = mu.len() + 1;
    let km1 = mu.len();
    let siginv = crate::linalg::spd_inverse(sigma, km1).unwrap_or_else(|| {
        let mut s = sigma.to_vec();
        crate::linalg::make_diagonally_dominant(&mut s, km1);
        crate::linalg::spd_inverse(&s, km1).unwrap()
    });
    let rows: Vec<Vec<f64>> = py.allow_threads(|| {
        docs.par_iter()
            .map(|doc| {
                let (words, counts) = ctm::doc_sparse(doc);
                ctm::infer_theta(beta, mu, &siginv, &words, &counts)
            })
            .collect()
    });
    let mut out = Array2::<f64>::zeros((rows.len(), k));
    for (d, row) in rows.iter().enumerate() {
        for (t, &v) in row.iter().enumerate() {
            out[[d, t]] = v;
        }
    }
    out
}

fn eta_posterior(model: &ctm::CtmModel) -> (Array2<f64>, Array3<f64>) {
    let d = model.lambda.len();
    let km1 = model.num_topics.saturating_sub(1);
    let mut mean = Array2::<f64>::zeros((d, km1));
    let mut cov = Array3::<f64>::zeros((d, km1, km1));
    for di in 0..d {
        for i in 0..km1 {
            mean[[di, i]] = model.lambda[di][i];
            for j in 0..km1 {
                cov[[di, i, j]] = model.nu[di][i * km1 + j];
            }
        }
    }
    (mean, cov)
}

/// Correlated Topic Model (Blei & Lafferty; the STM core). Topics are drawn
/// from a logistic-normal prior with a full covariance, so they can correlate —
/// unlike LDA's Dirichlet. Fit by variational EM (STM's Laplace E-step).
///
/// This is the engine STM builds on; prevalence/content covariates layer on top.
#[pyclass(module = "topica")]
pub struct CTM {
    num_topics: usize,
    sigma_shrink: f64,
    seed: u64,
    init_spectral: bool,

    fitted: bool,
    beta: Option<Array2<f64>>,  // (num_topics, num_words)
    theta: Option<Array2<f64>>, // (num_docs, num_topics)
    corr: Option<Array2<f64>>,  // (num_topics, num_topics)
    eta_mean: Option<Array2<f64>>, // (num_docs, num_topics-1) variational means λ
    eta_cov: Option<Array3<f64>>,  // (num_docs, K-1, K-1) variational covariances ν
    mu: Vec<f64>,                  // K-1 logistic-normal prior mean (for inference)
    sigma: Vec<f64>,               // (K-1)² logistic-normal prior covariance
    corpus: Option<corpus::Corpus>,
}

impl CTM {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl CTM {
    /// Create an unfitted model. `sigma_shrink` ∈ [0,1] shrinks the topic
    /// covariance toward its diagonal each M-step (stabilizes Σ). `init` is
    /// ``"spectral"`` (default; deterministic anchor-word init, matching STM's
    /// default — `seed` is then irrelevant) or ``"random"`` (seeded).
    #[new]
    #[pyo3(signature = (num_topics, *, sigma_shrink=0.0, seed=42, init="spectral"))]
    fn new(num_topics: usize, sigma_shrink: f64, seed: u64, init: &str) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if !(0.0..=1.0).contains(&sigma_shrink) {
            return Err(PyValueError::new_err("sigma_shrink must be in [0, 1]"));
        }
        let init_spectral = match init {
            "spectral" => true,
            "random" => false,
            _ => return Err(PyValueError::new_err("init must be 'spectral' or 'random'")),
        };
        Ok(CTM {
            num_topics,
            sigma_shrink,
            seed,
            init_spectral,
            fitted: false,
            beta: None,
            theta: None,
            corr: None,
            eta_mean: None,
            eta_cov: None,
            mu: Vec::new(),
            sigma: Vec::new(),
            corpus: None,
        })
    }

    /// Fit by variational EM for `em_iters` iterations. `data` is a
    /// :class:`Corpus` or `list[list[str]]`.
    #[pyo3(signature = (data, *, em_iters=50))]
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>, em_iters: usize) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }

        let k = self.num_topics;
        let num_types = corpus.num_types();
        let shrink = self.sigma_shrink;
        let spectral = self.init_spectral;
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let (model, corpus) = py.allow_threads(move || {
            let m = ctm::fit_ctm(
                &corpus.docs, k, num_types, em_iters, shrink, None, None, spectral, &mut rng,
            );
            (m, corpus)
        });

        let mut beta = Array2::<f64>::zeros((k, num_types));
        for t in 0..k {
            for v in 0..num_types {
                beta[[t, v]] = model.beta[t][v];
            }
        }
        let theta_v = model.doc_topics();
        let mut theta = Array2::<f64>::zeros((theta_v.len(), k));
        for (d, row) in theta_v.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[d, t]] = val;
            }
        }
        let corr_v = model.topic_correlation();
        let mut corr = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            for j in 0..k {
                corr[[i, j]] = corr_v[i][j];
            }
        }

        let (eta_mean, eta_cov) = eta_posterior(&model);

        self.beta = Some(beta);
        self.theta = Some(theta);
        self.corr = Some(corr);
        self.eta_mean = Some(eta_mean);
        self.eta_cov = Some(eta_cov);
        self.mu = model.mu.clone();
        self.sigma = model.sigma.clone();
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix β, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.beta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Topic-correlation matrix from the logistic-normal Σ, shape
    /// ``(num_topics, num_topics)``. Off-diagonal entries are genuine topic
    /// correlations (the whole point of CTM vs. LDA).
    #[getter]
    fn topic_correlation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.corr.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Per-document variational posterior means λ of the logistic-normal η,
    /// shape ``(num_docs, num_topics-1)``. Pairs with :attr:`eta_cov` to sample
    /// θ draws (method-of-composition uncertainty).
    #[getter]
    fn eta_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.eta_mean.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Per-document variational posterior covariances ν of η, shape
    /// ``(num_docs, num_topics-1, num_topics-1)``.
    #[getter]
    fn eta_cov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        self.require_fitted()?;
        Ok(self.eta_cov.as_ref().unwrap().to_pyarray_bound(py))
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// Top `n` words per topic (or one topic) as ``(word, probability)`` pairs.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let beta = self.beta.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tops = top_word_ids_phi(beta, self.num_topics, n);
        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err("topic out of range"));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| {
                    PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), beta[[t, w]].into_py(py)])
                })
                .collect();
            Ok(PyList::new_bound(py, items))
        };
        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.beta.as_ref().unwrap(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Infer topic proportions θ for *new* documents by the variational E-step
    /// against the fitted globals (β, logistic-normal prior μ, Σ). `data` is a
    /// :class:`Corpus` or `list[list[str]]`; tokens outside the training
    /// vocabulary are dropped. Returns a ``(num_docs, num_topics)`` array.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let docs = docs_to_ids(data, &self.corpus.as_ref().unwrap().id_to_word)?;
        let beta = self.beta.as_ref().unwrap();
        let beta_v: Vec<Vec<f64>> = beta.outer_iter().map(|r| r.to_vec()).collect();
        let theta = infer_theta_batch(py, &beta_v, &self.mu, &self.sigma, &docs);
        Ok(theta.to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `CTM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &CtmState {
            num_topics: self.num_topics, sigma_shrink: self.sigma_shrink, seed: self.seed,
            init_spectral: self.init_spectral, fitted: self.fitted,
            beta: arr2_opt(&self.beta), theta: arr2_opt(&self.theta), corr: arr2_opt(&self.corr),
            eta_mean: arr2_opt(&self.eta_mean), eta_cov: arr3_opt(&self.eta_cov),
            mu: self.mu.clone(), sigma: self.sigma.clone(),
            corpus: self.corpus.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: CtmState = read_state(path)?;
        Ok(CTM {
            num_topics: s.num_topics, sigma_shrink: s.sigma_shrink, seed: s.seed,
            init_spectral: s.init_spectral, fitted: s.fitted,
            beta: arr2_back(s.beta), theta: arr2_back(s.theta), corr: arr2_back(s.corr),
            eta_mean: arr2_back(s.eta_mean), eta_cov: arr3_back(s.eta_cov),
            mu: s.mu, sigma: s.sigma, corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!("CTM(num_topics={}, fitted={})", self.num_topics, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// STM: Structural Topic Model (CTM core + prevalence covariates)
// ---------------------------------------------------------------------------

/// Structural Topic Model (Roberts, Stewart & Tingley). The correlated-topic
/// core (:class:`CTM`) with **prevalence covariates**: a document's prior topic
/// mean is a regression on its covariates, `μ_d = X_d γ`, so covariates shift
/// which topics a document discusses. After fitting, `prevalence_effects` holds
/// the learned γ; pair it with `topica.stm.estimate_effect` for inference.
#[pyclass(module = "topica")]
pub struct STM {
    num_topics: usize,
    sigma_shrink: f64,
    seed: u64,
    init_spectral: bool,

    fitted: bool,
    beta: Option<Array2<f64>>,
    theta: Option<Array2<f64>>,
    corr: Option<Array2<f64>>,
    eta_mean: Option<Array2<f64>>, // (num_docs, num_topics-1) variational means λ
    eta_cov: Option<Array3<f64>>,  // (num_docs, K-1, K-1) variational covariances ν
    gamma: Option<Array2<f64>>, // (num_features, num_topics-1); None if no prevalence
    feature_names: Vec<String>,
    content_beta: Option<Vec<Vec<Vec<f64>>>>, // G×K×V; None if no content
    group_names: Vec<String>,
    mu: Vec<f64>,    // K-1 logistic-normal prior mean (covariate-free baseline)
    sigma: Vec<f64>, // (K-1)² logistic-normal prior covariance
    corpus: Option<corpus::Corpus>,
}

impl STM {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }

    fn resolve_group(&self, obj: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(i) = obj.extract::<usize>() {
            if i < self.group_names.len() {
                return Ok(i);
            }
            return Err(PyValueError::new_err("group index out of range"));
        }
        if let Ok(s) = obj.extract::<String>() {
            return self
                .group_names
                .iter()
                .position(|g| g == &s)
                .ok_or_else(|| PyValueError::new_err(format!("unknown group {:?}", s)));
        }
        Err(PyValueError::new_err("group must be a name (str) or index (int)"))
    }
}

#[pymethods]
impl STM {
    /// Create an unfitted model. `sigma_shrink` ∈ [0,1] shrinks Σ toward its
    /// diagonal each M-step. `init` is ``"spectral"`` (default; deterministic
    /// anchor-word init, matching STM's default — `seed` is then irrelevant for
    /// β init) or ``"random"`` (seeded). Spectral init applies to the
    /// topic-word β; with a content model the per-group β is always random.
    #[new]
    #[pyo3(signature = (num_topics, *, sigma_shrink=0.0, seed=42, init="spectral"))]
    fn new(num_topics: usize, sigma_shrink: f64, seed: u64, init: &str) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if !(0.0..=1.0).contains(&sigma_shrink) {
            return Err(PyValueError::new_err("sigma_shrink must be in [0, 1]"));
        }
        let init_spectral = match init {
            "spectral" => true,
            "random" => false,
            _ => return Err(PyValueError::new_err("init must be 'spectral' or 'random'")),
        };
        Ok(STM {
            num_topics,
            sigma_shrink,
            seed,
            init_spectral,
            fitted: false,
            beta: None,
            theta: None,
            corr: None,
            eta_mean: None,
            eta_cov: None,
            gamma: None,
            feature_names: Vec::new(),
            content_beta: None,
            group_names: Vec::new(),
            mu: Vec::new(),
            sigma: Vec::new(),
            corpus: None,
        })
    }

    /// Fit. `data` is a :class:`Corpus` or `list[list[str]]`. `prevalence`
    /// (optional, `(num_docs, F)` covariates) makes topic prevalence depend on
    /// covariates (`μ_d = X_d γ`); an intercept is prepended. `content`
    /// (optional, one group label per document) makes the topic-word
    /// distributions vary by group (the SAGE content model). At least one of
    /// `prevalence`/`content` should be given (else use :class:`CTM`).
    #[pyo3(signature = (data, prevalence=None, *, prevalence_names=None,
                        content=None, content_names=None, em_iters=50))]
    #[allow(clippy::too_many_arguments)]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        prevalence: Option<&Bound<'_, PyAny>>,
        prevalence_names: Option<Vec<String>>,
        content: Option<&Bound<'_, PyAny>>,
        content_names: Option<Vec<String>>,
        em_iters: usize,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?
        };
        let num_docs = corpus.num_docs();
        if num_docs == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        if prevalence.is_none() && content.is_none() {
            return Err(PyValueError::new_err(
                "STM needs prevalence and/or content covariates; use CTM for neither",
            ));
        }

        // --- Prevalence design (optional) ---
        let mut prevalence_x: Option<Vec<Vec<f64>>> = None;
        let mut feat_names: Vec<String> = Vec::new();
        if let Some(prev) = prevalence {
            let raw = parse_features(prev)?;
            if raw.len() != num_docs {
                return Err(PyValueError::new_err(format!(
                    "prevalence has {} rows but corpus has {} documents",
                    raw.len(),
                    num_docs
                )));
            }
            let f_in = raw.first().map(|r| r.len()).unwrap_or(0);
            if raw.iter().any(|r| r.len() != f_in) {
                return Err(PyValueError::new_err("all prevalence rows must have the same length"));
            }
            if let Some(names) = &prevalence_names {
                if names.len() != f_in {
                    return Err(PyValueError::new_err(
                        "prevalence_names length must match the number of covariate columns",
                    ));
                }
            }
            let nf = f_in + 1;
            prevalence_x = Some(
                raw.iter()
                    .map(|r| {
                        let mut v = Vec::with_capacity(nf);
                        v.push(1.0);
                        v.extend_from_slice(r);
                        v
                    })
                    .collect(),
            );
            feat_names.push("intercept".to_string());
            feat_names.extend(
                prevalence_names
                    .unwrap_or_else(|| (0..f_in).map(|i| format!("feature_{}", i)).collect()),
            );
        }

        // --- Content groups (optional) ---
        let mut content_groups: Option<(Vec<usize>, usize)> = None;
        let mut group_vocab: Vec<String> = Vec::new();
        if let Some(cont) = content {
            let groups_str = parse_groups(cont)?;
            if groups_str.len() != num_docs {
                return Err(PyValueError::new_err(format!(
                    "content has {} entries but corpus has {} documents",
                    groups_str.len(),
                    num_docs
                )));
            }
            group_vocab = match content_names {
                Some(n) => n,
                None => {
                    let mut set: HashSet<String> = groups_str.iter().cloned().collect();
                    let mut v: Vec<String> = set.drain().collect();
                    v.sort();
                    v
                }
            };
            let gindex: HashMap<&str, usize> = group_vocab
                .iter()
                .enumerate()
                .map(|(i, g)| (g.as_str(), i))
                .collect();
            let idx: Vec<usize> = groups_str
                .iter()
                .map(|g| {
                    gindex.get(g.as_str()).copied().ok_or_else(|| {
                        PyValueError::new_err(format!("content group {:?} not in content_names", g))
                    })
                })
                .collect::<PyResult<_>>()?;
            content_groups = Some((idx, group_vocab.len()));
        }

        let k = self.num_topics;
        let num_types = corpus.num_types();
        let shrink = self.sigma_shrink;
        let spectral = self.init_spectral;
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let (model, corpus) = py.allow_threads(move || {
            let prev_ref = prevalence_x.as_deref();
            let cont_ref = content_groups.as_ref().map(|(g, n)| (g.as_slice(), *n));
            let m = ctm::fit_ctm(
                &corpus.docs, k, num_types, em_iters, shrink, prev_ref, cont_ref, spectral,
                &mut rng,
            );
            (m, corpus)
        });

        let mut beta = Array2::<f64>::zeros((k, num_types));
        for t in 0..k {
            for v in 0..num_types {
                beta[[t, v]] = model.beta[t][v];
            }
        }
        let theta_v = model.doc_topics();
        let mut theta = Array2::<f64>::zeros((theta_v.len(), k));
        for (di, row) in theta_v.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[di, t]] = val;
            }
        }
        let corr_v = model.topic_correlation();
        let mut corr = Array2::<f64>::zeros((k, k));
        for i in 0..k {
            for j in 0..k {
                corr[[i, j]] = corr_v[i][j];
            }
        }
        self.gamma = model.gamma.as_ref().map(|g| {
            let nf = g.len();
            let mut arr = Array2::<f64>::zeros((nf, k - 1));
            for ff in 0..nf {
                for t in 0..(k - 1) {
                    arr[[ff, t]] = g[ff][t];
                }
            }
            arr
        });

        let (eta_mean, eta_cov) = eta_posterior(&model);

        self.beta = Some(beta);
        self.theta = Some(theta);
        self.corr = Some(corr);
        self.eta_mean = Some(eta_mean);
        self.eta_cov = Some(eta_cov);
        self.feature_names = feat_names;
        self.content_beta = model.content_beta;
        self.group_names = group_vocab;
        self.mu = model.mu.clone();
        self.sigma = model.sigma.clone();
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix β, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.beta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Topic-correlation matrix, shape ``(num_topics, num_topics)``.
    #[getter]
    fn topic_correlation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.corr.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Per-document variational posterior means λ of η, shape
    /// ``(num_docs, num_topics-1)``. With :attr:`eta_cov` this is the
    /// logistic-normal posterior used to draw θ samples for
    /// method-of-composition uncertainty in ``estimate_effect``.
    #[getter]
    fn eta_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.eta_mean.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Per-document variational posterior covariances ν of η, shape
    /// ``(num_docs, num_topics-1, num_topics-1)``.
    #[getter]
    fn eta_cov<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        self.require_fitted()?;
        Ok(self.eta_cov.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Prevalence coefficients γ, shape ``(num_features, num_topics-1)`` — how
    /// each covariate (row 0 is the intercept) shifts each topic's log-prior.
    /// The last topic is the softmax reference. For inference, prefer
    /// ``topica.stm.estimate_effect(model.doc_topic, X)``.
    #[getter]
    fn prevalence_effects<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let g = self
            .gamma
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("model was fit without prevalence covariates"))?;
        Ok(g.to_pyarray_bound(py))
    }

    /// Per-group topic-word distributions, shape ``(num_topics, num_groups,
    /// num_words)`` — only available when fit with `content` covariates.
    #[getter]
    fn topic_word_by_group<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f64>>> {
        self.require_fitted()?;
        let cb = self.content_beta.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("model was fit without content covariates")
        })?;
        let g = cb.len();
        let k = self.num_topics;
        let v = self.corpus.as_ref().unwrap().num_types();
        // cb is G×K×V; expose as (topics, groups, words).
        let mut arr = Array3::<f64>::zeros((k, g, v));
        for gg in 0..g {
            for t in 0..k {
                for w in 0..v {
                    arr[[t, gg, w]] = cb[gg][t][w];
                }
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// Content-covariate group names (axis-1 order of :attr:`topic_word_by_group`).
    #[getter]
    fn groups(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        if self.group_names.is_empty() {
            return Err(PyRuntimeError::new_err("model was fit without content covariates"));
        }
        Ok(self.group_names.clone())
    }

    /// Words that most distinguish how `topic` is worded in `group_a` vs
    /// `group_b` (log word-probability ratio; positive favours `group_a`).
    /// Requires content covariates.
    #[pyo3(signature = (topic, group_a, group_b, n=10))]
    fn word_contrast<'py>(
        &self,
        py: Python<'py>,
        topic: usize,
        group_a: &Bound<'py, PyAny>,
        group_b: &Bound<'py, PyAny>,
        n: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        self.require_fitted()?;
        let cb = self.content_beta.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("model was fit without content covariates")
        })?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        let ga = self.resolve_group(group_a)?;
        let gb = self.resolve_group(group_b)?;
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let a = &cb[ga][topic];
        let b = &cb[gb][topic];
        let ratio: Vec<f64> = (0..vocab.len())
            .map(|v| (a[v].max(1e-300) / b[v].max(1e-300)).ln())
            .collect();
        let mut idx: Vec<usize> = (0..vocab.len()).collect();
        idx.sort_by(|&x, &y| ratio[y].partial_cmp(&ratio[x]).unwrap());
        let items: Vec<Bound<'py, PyTuple>> = idx
            .iter()
            .take(n)
            .map(|&v| PyTuple::new_bound(py, &[vocab[v].clone().into_py(py), ratio[v].into_py(py)]))
            .collect();
        Ok(PyList::new_bound(py, items))
    }

    /// Covariate names aligned with the rows of :attr:`prevalence_effects`
    /// (``"intercept"`` first).
    #[getter]
    fn feature_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.feature_names.clone())
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// Top `n` words per topic (or one topic) as ``(word, probability)`` pairs.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let beta = self.beta.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tops = top_word_ids_phi(beta, self.num_topics, n);
        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err("topic out of range"));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| {
                    PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), beta[[t, w]].into_py(py)])
                })
                .collect();
            Ok(PyList::new_bound(py, items))
        };
        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.beta.as_ref().unwrap(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Infer topic proportions θ for *new* documents by the variational E-step
    /// against the fitted globals (β and the logistic-normal prior). `data` is a
    /// :class:`Corpus` or `list[list[str]]`; out-of-vocabulary tokens are
    /// dropped. Returns a ``(num_docs, num_topics)`` array.
    ///
    /// Note: the prior mean used is the covariate-free baseline μ learned at fit
    /// time (prevalence covariates for held-out docs are not applied here), and
    /// for a content model the marginal topic-word β is used. This is the same
    /// held-out inference stm's `fitNewDocuments` performs when no new covariate
    /// design is supplied.
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let docs = docs_to_ids(data, &self.corpus.as_ref().unwrap().id_to_word)?;
        let beta = self.beta.as_ref().unwrap();
        let beta_v: Vec<Vec<f64>> = beta.outer_iter().map(|r| r.to_vec()).collect();
        let theta = infer_theta_batch(py, &beta_v, &self.mu, &self.sigma, &docs);
        Ok(theta.to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `STM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &StmState {
            num_topics: self.num_topics, sigma_shrink: self.sigma_shrink, seed: self.seed,
            init_spectral: self.init_spectral, fitted: self.fitted,
            beta: arr2_opt(&self.beta), theta: arr2_opt(&self.theta), corr: arr2_opt(&self.corr),
            eta_mean: arr2_opt(&self.eta_mean), eta_cov: arr3_opt(&self.eta_cov),
            gamma: arr2_opt(&self.gamma), feature_names: self.feature_names.clone(),
            content_beta: self.content_beta.clone(),
            mu: self.mu.clone(), sigma: self.sigma.clone(),
            group_names: self.group_names.clone(),
            corpus: self.corpus.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: StmState = read_state(path)?;
        Ok(STM {
            num_topics: s.num_topics, sigma_shrink: s.sigma_shrink, seed: s.seed,
            init_spectral: s.init_spectral, fitted: s.fitted,
            beta: arr2_back(s.beta), theta: arr2_back(s.theta), corr: arr2_back(s.corr),
            eta_mean: arr2_back(s.eta_mean), eta_cov: arr3_back(s.eta_cov),
            gamma: arr2_back(s.gamma), feature_names: s.feature_names,
            content_beta: s.content_beta,
            mu: s.mu, sigma: s.sigma,
            group_names: s.group_names, corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!("STM(num_topics={}, fitted={})", self.num_topics, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

/// Window/document co-occurrence counts for coherence scoring.
///
/// `docs` holds relevant-word ids per token (`4294967295` marks a non-relevant
/// token). `pairs` are `(a, b)` with `a < b`. `window == 0` requests
/// document-level co-occurrence (one window per document, for UMass); a positive
/// width slides a window one token at a time. Returns
/// `(occ[num_relevant], co[len(pairs)], n_windows)`.
#[pyfunction]
fn window_cooccurrence(
    py: Python<'_>,
    docs: Vec<Vec<u32>>,
    num_relevant: usize,
    pairs: Vec<(u32, u32)>,
    window: u32,
) -> (Vec<f64>, Vec<f64>, f64) {
    py.allow_threads(move || coh::cooccurrence(&docs, num_relevant, &pairs, window))
}

/// Tokenize a string the way the corpus loader does: find regex tokens,
/// optionally lowercase, drop short tokens and stopwords. Handy for building
/// `list[list[str]]` input outside of `Corpus.from_text_file`.
#[pyfunction]
#[pyo3(signature = (text, *, lowercase=true, stopwords=None, token_regex=None, min_length=1))]
fn tokenize(
    text: &str,
    lowercase: bool,
    stopwords: Option<Vec<String>>,
    token_regex: Option<String>,
    min_length: usize,
) -> PyResult<Vec<String>> {
    let pattern = token_regex.unwrap_or_else(|| corpus::DEFAULT_TOKEN_REGEX.to_string());
    let re = Regex::new(&pattern).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let stop: HashSet<String> = stopwords.unwrap_or_default().into_iter().collect();

    let mut out = Vec::new();
    for m in re.find_iter(text) {
        let tok = if lowercase {
            m.as_str().to_lowercase()
        } else {
            m.as_str().to_string()
        };
        if tok.chars().count() < min_length {
            continue;
        }
        if stop.contains(&tok) {
            continue;
        }
        out.push(tok);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Module init
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// HDP: Hierarchical Dirichlet Process (nonparametric LDA — infers K)
// ---------------------------------------------------------------------------

/// Hierarchical Dirichlet Process topic model (Teh, Jordan, Beal & Blei 2006):
/// LDA that **infers the number of topics** rather than fixing it. Fit by the
/// direct-assignment Gibbs sampler (the Chinese Restaurant Franchise). The two
/// concentration parameters `alpha` (document level) and `gamma` (corpus level)
/// govern how readily new topics appear; by default both are resampled from the
/// data (a faithful port of blei-lab/hdp), so you typically don't tune them.
#[pyclass(module = "topica")]
pub struct HDP {
    alpha: f64,
    gamma: f64,
    eta: f64,
    seed: u64,
    resample_conc: bool,

    fitted: bool,
    num_topics: usize,
    learned_alpha: f64,
    learned_gamma: f64,
    beta: Option<Array2<f64>>,
    theta: Option<Array2<f64>>,
    corpus: Option<corpus::Corpus>,
}

impl HDP {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl HDP {
    /// Create an unfitted model. `alpha`/`gamma` are the initial document- and
    /// corpus-level DP concentrations; `eta` is the topic-word Dirichlet (base
    /// measure). With `resample_conc=True` (default) `alpha`/`gamma` are
    /// resampled each sweep and the given values are just starting points.
    #[new]
    #[pyo3(signature = (*, alpha=1.0, gamma=1.0, eta=0.01, seed=42, resample_conc=true))]
    fn new(alpha: f64, gamma: f64, eta: f64, seed: u64, resample_conc: bool) -> PyResult<Self> {
        if alpha <= 0.0 || gamma <= 0.0 {
            return Err(PyValueError::new_err("alpha and gamma must be > 0"));
        }
        if eta <= 0.0 {
            return Err(PyValueError::new_err("eta must be > 0"));
        }
        Ok(HDP {
            alpha,
            gamma,
            eta,
            seed,
            resample_conc,
            fitted: false,
            num_topics: 0,
            learned_alpha: alpha,
            learned_gamma: gamma,
            beta: None,
            theta: None,
            corpus: None,
        })
    }

    /// Fit by Gibbs sampling for `iters` sweeps. `data` is a :class:`Corpus` or
    /// `list[list[str]]`. The inferred topic count is available as `num_topics`.
    #[pyo3(signature = (data, *, iters=150))]
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>, iters: usize) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }

        let num_types = corpus.num_types();
        let (alpha, gamma, eta, conc) = (self.alpha, self.gamma, self.eta, self.resample_conc);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let (model, corpus) = py.allow_threads(move || {
            let m = hdp::fit_hdp(&corpus.docs, num_types, alpha, gamma, eta, iters, conc, &mut rng);
            (m, corpus)
        });

        let k = model.num_topics();
        let tw = model.topic_word();
        let mut beta = Array2::<f64>::zeros((k, num_types));
        for (t, row) in tw.iter().enumerate() {
            for (v, &val) in row.iter().enumerate() {
                beta[[t, v]] = val;
            }
        }
        let th = model.doc_topic();
        let mut theta = Array2::<f64>::zeros((th.len(), k));
        for (d, row) in th.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[d, t]] = val;
            }
        }

        self.num_topics = k;
        self.learned_alpha = model.alpha;
        self.learned_gamma = model.gamma;
        self.beta = Some(beta);
        self.theta = Some(theta);
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix β, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.beta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// The inferred number of topics K.
    #[getter]
    fn num_topics(&self) -> PyResult<usize> {
        self.require_fitted()?;
        Ok(self.num_topics)
    }

    /// The fitted document-level concentration α0 (resampled if enabled).
    #[getter]
    fn alpha(&self) -> f64 {
        self.learned_alpha
    }

    /// The fitted corpus-level concentration γ (resampled if enabled).
    #[getter]
    fn gamma(&self) -> f64 {
        self.learned_gamma
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    /// Top `n` words per topic (or one topic) as ``(word, probability)`` pairs.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let beta = self.beta.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tops = top_word_ids_phi(beta, self.num_topics, n);
        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err("topic out of range"));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| {
                    PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), beta[[t, w]].into_py(py)])
                })
                .collect();
            Ok(PyList::new_bound(py, items))
        };
        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.beta.as_ref().unwrap(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Infer topic proportions θ for *new* documents over the discovered topics,
    /// by collapsed Gibbs against the fixed topic-word matrix. `data` is a
    /// :class:`Corpus` or `list[list[str]]`; OOV tokens are dropped. The
    /// document-level prior is symmetric with total mass equal to the learned
    /// concentration α. Returns a ``(num_docs, num_topics)`` array.
    #[pyo3(signature = (data, *, iterations=100, burn_in=10, num_samples=10,
                        sample_interval=5, seed=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        iterations: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let k = self.num_topics;
        let alpha = vec![self.learned_alpha / k as f64; k];
        transform_gibbs(
            py, data, &self.corpus.as_ref().unwrap().id_to_word, self.beta.as_ref().unwrap(),
            &alpha, iterations, burn_in, num_samples, sample_interval,
            seed.unwrap_or(self.seed),
        )
    }

    /// Save the fitted model to `path`. Reload with `HDP.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &HdpState {
            alpha: self.alpha, gamma: self.gamma, eta: self.eta, seed: self.seed,
            resample_conc: self.resample_conc, fitted: self.fitted, num_topics: self.num_topics,
            learned_alpha: self.learned_alpha, learned_gamma: self.learned_gamma,
            beta: arr2_opt(&self.beta), theta: arr2_opt(&self.theta), corpus: self.corpus.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: HdpState = read_state(path)?;
        Ok(HDP {
            alpha: s.alpha, gamma: s.gamma, eta: s.eta, seed: s.seed,
            resample_conc: s.resample_conc, fitted: s.fitted, num_topics: s.num_topics,
            learned_alpha: s.learned_alpha, learned_gamma: s.learned_gamma,
            beta: arr2_back(s.beta), theta: arr2_back(s.theta), corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        if self.fitted {
            format!("HDP(num_topics={} [inferred], fitted=true)", self.num_topics)
        } else {
            format!("HDP(alpha={}, gamma={}, fitted=false)", self.alpha, self.gamma)
        }
    }
}

// ---------------------------------------------------------------------------
// DTM: Dynamic Topic Model (topics that evolve over time)
// ---------------------------------------------------------------------------

/// Dynamic Topic Model (Blei & Lafferty 2006): topics whose word distributions
/// **evolve across time slices**. Each topic-word chain follows a Gaussian
/// state-space model; inference is variational with Kalman smoothing, a faithful
/// port of Blei's C `dtm` / gensim's `LdaSeqModel`. After fitting, query a
/// topic's word distribution at any slice with `topic_word(time)` and trace a
/// word's trajectory with `word_evolution(topic, word)`.
#[pyclass(module = "topica")]
pub struct DTM {
    num_topics: usize,
    alpha: f64,
    chain_variance: f64,
    obs_variance: f64,
    seed: u64,

    fitted: bool,
    num_times: usize,
    bound: f64,
    // (num_times, num_topics, num_words): p(word | topic, time).
    topic_words: Option<Vec<Vec<Vec<f64>>>>,
    corpus: Option<corpus::Corpus>,
}

impl DTM {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl DTM {
    /// Create an unfitted model. `chain_variance` controls how much a topic may
    /// drift between adjacent slices (larger = freer to change; gensim's default
    /// is 0.005). `obs_variance` is the observation noise; `alpha` the Dirichlet
    /// concentration on document-topic proportions.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha=0.01, chain_variance=0.005, obs_variance=0.5, seed=42))]
    fn new(
        num_topics: usize,
        alpha: f64,
        chain_variance: f64,
        obs_variance: f64,
        seed: u64,
    ) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if alpha <= 0.0 || chain_variance <= 0.0 || obs_variance <= 0.0 {
            return Err(PyValueError::new_err(
                "alpha, chain_variance, obs_variance must be > 0",
            ));
        }
        Ok(DTM {
            num_topics,
            alpha,
            chain_variance,
            obs_variance,
            seed,
            fitted: false,
            num_times: 0,
            bound: 0.0,
            topic_words: None,
            corpus: None,
        })
    }

    /// Fit by variational EM. `data` is a :class:`Corpus` or `list[list[str]]`;
    /// `times` gives each document's integer time-slice index (0-based,
    /// contiguous). The number of slices is inferred as ``max(times) + 1``.
    #[pyo3(signature = (data, times, *, em_iters=20))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        times: Vec<i64>,
        em_iters: usize,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        if times.len() != corpus.num_docs() {
            return Err(PyValueError::new_err(format!(
                "times has length {} but there are {} documents",
                times.len(),
                corpus.num_docs()
            )));
        }
        if times.iter().any(|&t| t < 0) {
            return Err(PyValueError::new_err("time-slice indices must be >= 0"));
        }
        let times_u: Vec<usize> = times.iter().map(|&t| t as usize).collect();
        let num_times = times_u.iter().copied().max().unwrap() + 1;
        // Require every slice to be populated (contiguous 0..num_times).
        let mut seen = vec![false; num_times];
        for &t in &times_u {
            seen[t] = true;
        }
        if seen.iter().any(|&s| !s) {
            return Err(PyValueError::new_err(
                "time slices must be contiguous 0..max; some slice has no documents",
            ));
        }

        let num_types = corpus.num_types();
        let k = self.num_topics;
        let (alpha, cv, ov) = (self.alpha, self.chain_variance, self.obs_variance);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let (model, corpus) = py.allow_threads(move || {
            let m = dtm::fit_dtm(
                &corpus.docs, &times_u, num_types, k, num_times, alpha, cv, ov, em_iters, &mut rng,
            );
            (m, corpus)
        });

        // Precompute p(word | topic, time) for every slice.
        let tw: Vec<Vec<Vec<f64>>> =
            (0..num_times).map(|t| model.topic_word_matrix(t)).collect();

        self.num_times = num_times;
        self.bound = model.bound;
        self.topic_words = Some(tw);
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix at time slice `time`, shape ``(num_topics, num_words)``;
    /// rows sum to 1.
    fn topic_word<'py>(&self, py: Python<'py>, time: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        if time >= self.num_times {
            return Err(PyValueError::new_err("time out of range"));
        }
        let tw = &self.topic_words.as_ref().unwrap()[time];
        let mut arr = Array2::<f64>::zeros((self.num_topics, tw[0].len()));
        for (k, row) in tw.iter().enumerate() {
            for (w, &val) in row.iter().enumerate() {
                arr[[k, w]] = val;
            }
        }
        Ok(arr.to_pyarray_bound(py))
    }

    /// Trajectory of a word's probability in a topic across slices, shape
    /// ``(num_times,)``. `word` is a vocabulary string or its integer id.
    fn word_evolution<'py>(
        &self,
        py: Python<'py>,
        topic: usize,
        word: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let wid = if let Ok(i) = word.extract::<usize>() {
            i
        } else {
            let s = word.extract::<String>()?;
            vocab.iter().position(|w| w == &s).ok_or_else(|| {
                PyValueError::new_err(format!("word {:?} not in vocabulary", s))
            })?
        };
        if wid >= vocab.len() {
            return Err(PyValueError::new_err("word id out of range"));
        }
        let tw = self.topic_words.as_ref().unwrap();
        let traj: Vec<f64> = (0..self.num_times).map(|t| tw[t][topic][wid]).collect();
        Ok(Array1::from(traj).to_pyarray_bound(py))
    }

    /// Top `n` words for a topic at one time slice as ``(word, probability)``.
    #[pyo3(signature = (topic, time, n=10))]
    fn top_words(
        &self,
        topic: usize,
        time: usize,
        n: usize,
    ) -> PyResult<Vec<(String, f64)>> {
        self.require_fitted()?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        if time >= self.num_times {
            return Err(PyValueError::new_err("time out of range"));
        }
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let row = &self.topic_words.as_ref().unwrap()[time][topic];
        let mut idx: Vec<usize> = (0..row.len()).collect();
        idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        Ok(idx.into_iter().take(n).map(|w| (vocab[w].clone(), row[w])).collect())
    }

    /// Which words inside `topic` drift most between two time slices.
    ///
    /// For each word, the change in its probability within the topic from
    /// `from_time` to `to_time` (defaults: the first and last slices) is
    /// computed. Returns a dict with two keys, ``"rising"`` and ``"falling"``,
    /// each a list of ``(word, delta)`` pairs (largest gain first; largest drop
    /// first). This is how you see *what* makes a topic's vocabulary evolve, not
    /// just that it does.
    #[pyo3(signature = (topic, *, n=10, from_time=0, to_time=None))]
    fn word_drift<'py>(
        &self,
        py: Python<'py>,
        topic: usize,
        n: usize,
        from_time: usize,
        to_time: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        self.require_fitted()?;
        if topic >= self.num_topics {
            return Err(PyValueError::new_err("topic out of range"));
        }
        let to = to_time.unwrap_or(self.num_times - 1);
        if from_time >= self.num_times || to >= self.num_times {
            return Err(PyValueError::new_err("time slice out of range"));
        }
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tw = self.topic_words.as_ref().unwrap();
        let a = &tw[from_time][topic];
        let b = &tw[to][topic];
        let mut deltas: Vec<(usize, f64)> = (0..a.len()).map(|w| (w, b[w] - a[w])).collect();
        deltas.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap()); // descending by delta

        let to_pairs = |items: Vec<(usize, f64)>| -> Vec<(String, f64)> {
            items.into_iter().map(|(w, d)| (vocab[w].clone(), d)).collect()
        };
        let rising = to_pairs(
            deltas.iter().filter(|&&(_, d)| d > 0.0).take(n).copied().collect(),
        );
        let falling = to_pairs(
            deltas.iter().rev().filter(|&&(_, d)| d < 0.0).take(n).copied().collect(),
        );

        let out = PyDict::new_bound(py);
        out.set_item("rising", rising)?;
        out.set_item("falling", falling)?;
        Ok(out)
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    /// The number of time slices (available after fit).
    #[getter]
    fn num_times(&self) -> PyResult<usize> {
        self.require_fitted()?;
        Ok(self.num_times)
    }

    /// The final variational bound (ELBO) reached during fitting.
    #[getter]
    fn bound(&self) -> PyResult<f64> {
        self.require_fitted()?;
        Ok(self.bound)
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    /// Save the fitted model to `path`. Reload with `DTM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &DtmState {
            num_topics: self.num_topics, alpha: self.alpha, chain_variance: self.chain_variance,
            obs_variance: self.obs_variance, seed: self.seed, fitted: self.fitted,
            num_times: self.num_times, bound: self.bound,
            topic_words: self.topic_words.clone(), corpus: self.corpus.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: DtmState = read_state(path)?;
        Ok(DTM {
            num_topics: s.num_topics, alpha: s.alpha, chain_variance: s.chain_variance,
            obs_variance: s.obs_variance, seed: s.seed, fitted: s.fitted,
            num_times: s.num_times, bound: s.bound, topic_words: s.topic_words, corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        if self.fitted {
            format!(
                "DTM(num_topics={}, num_times={}, fitted=true)",
                self.num_topics, self.num_times
            )
        } else {
            format!("DTM(num_topics={}, fitted=false)", self.num_topics)
        }
    }
}

// ---------------------------------------------------------------------------
// SupervisedLDA: sLDA (topics shaped to predict a per-document response)
// ---------------------------------------------------------------------------

/// Supervised LDA (Blei & McAuliffe 2007): LDA in which each document carries a
/// real-valued response `y_d ~ N(ηᵀ z̄_d, σ²)` regressed on its topic usage.
/// Fitting is supervised by the response, so topics are shaped to be predictive
/// and the coefficients `η` report how each topic moves `y`. Fit by variational
/// EM; `predict` returns ŷ for new documents.
#[pyclass(module = "topica")]
pub struct SupervisedLDA {
    num_topics: usize,
    alpha: f64,
    seed: u64,

    fitted: bool,
    sigma2: f64,
    eta: Option<Array1<f64>>,
    beta: Option<Array2<f64>>,  // K × V
    theta: Option<Array2<f64>>, // D × K
    log_beta: Option<Vec<Vec<f64>>>,
    corpus: Option<corpus::Corpus>,
}

impl SupervisedLDA {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl SupervisedLDA {
    /// Create an unfitted model. `alpha` is the symmetric Dirichlet
    /// concentration on document-topic proportions.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha=0.1, seed=42))]
    fn new(num_topics: usize, alpha: f64, seed: u64) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if alpha <= 0.0 {
            return Err(PyValueError::new_err("alpha must be > 0"));
        }
        Ok(SupervisedLDA {
            num_topics,
            alpha,
            seed,
            fitted: false,
            sigma2: 0.0,
            eta: None,
            beta: None,
            theta: None,
            log_beta: None,
            corpus: None,
        })
    }

    /// Fit by variational EM. `data` is a :class:`Corpus` or `list[list[str]]`;
    /// `y` is the per-document real-valued response (length = number of docs).
    #[pyo3(signature = (data, y, *, em_iters=25, var_iters=15))]
    fn fit(
        &mut self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        y: Vec<f64>,
        em_iters: usize,
        var_iters: usize,
    ) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        if y.len() != corpus.num_docs() {
            return Err(PyValueError::new_err(format!(
                "y has length {} but there are {} documents",
                y.len(),
                corpus.num_docs()
            )));
        }

        let num_types = corpus.num_types();
        let (k, alpha) = (self.num_topics, self.alpha);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);

        let (model, corpus) = py.allow_threads(move || {
            let m = slda::fit_slda(&corpus.docs, &y, num_types, k, alpha, em_iters, var_iters, &mut rng);
            (m, corpus)
        });

        let mut beta = Array2::<f64>::zeros((k, num_types));
        let tw = model.topic_word();
        for (t, row) in tw.iter().enumerate() {
            for (w, &val) in row.iter().enumerate() {
                beta[[t, w]] = val;
            }
        }
        let th = model.doc_topic();
        let mut theta = Array2::<f64>::zeros((th.len(), k));
        for (di, row) in th.iter().enumerate() {
            for (t, &val) in row.iter().enumerate() {
                theta[[di, t]] = val;
            }
        }

        self.sigma2 = model.sigma2;
        self.eta = Some(Array1::from(model.eta.clone()));
        self.beta = Some(beta);
        self.theta = Some(theta);
        self.log_beta = Some(model.log_beta.clone());
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// Predict the response ŷ for new documents (`list[list[str]]` or a
    /// :class:`Corpus`). Out-of-vocabulary words are ignored. Returns a 1-D array
    /// of length = number of documents.
    #[pyo3(signature = (data, *, var_iters=20))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'_, PyAny>,
        var_iters: usize,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let word_id: std::collections::HashMap<&str, u32> =
            vocab.iter().enumerate().map(|(i, w)| (w.as_str(), i as u32)).collect();

        let docs: Vec<Vec<String>> = if let Ok(c) = data.extract::<Corpus>() {
            c.inner.docs.iter().map(|d| d.iter().map(|&w| c.inner.id_to_word[w as usize].clone()).collect()).collect()
        } else {
            data.extract().map_err(|_| {
                PyValueError::new_err("predict() expects a Corpus or a list of token lists")
            })?
        };

        let log_beta = self.log_beta.as_ref().unwrap();
        let model = slda::SldaModel {
            num_topics: self.num_topics,
            num_types: vocab.len(),
            alpha: self.alpha,
            log_beta: log_beta.clone(),
            eta: self.eta.as_ref().unwrap().to_vec(),
            sigma2: self.sigma2,
            gamma: Vec::new(),
        };

        let preds: Vec<f64> = docs
            .iter()
            .map(|doc| {
                let ids: Vec<u32> = doc.iter().filter_map(|w| word_id.get(w.as_str()).copied()).collect();
                slda::predict_one(&model, &ids, var_iters)
            })
            .collect();
        Ok(Array1::from(preds).to_pyarray_bound(py))
    }

    /// Topic-word matrix β, shape ``(num_topics, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.beta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// Regression coefficients η, shape ``(num_topics,)`` — how each topic moves
    /// the response (in the response's units, per unit of topic frequency).
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        Ok(self.eta.as_ref().unwrap().to_pyarray_bound(py))
    }

    /// The fitted response variance σ².
    #[getter]
    fn sigma2(&self) -> PyResult<f64> {
        self.require_fitted()?;
        Ok(self.sigma2)
    }

    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }

    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    /// Top `n` words per topic (or one topic) as ``(word, probability)`` pairs.
    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(
        &self,
        py: Python<'py>,
        n: usize,
        topic: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        let beta = self.beta.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let tops = top_word_ids_phi(beta, self.num_topics, n);
        let one = |t: usize| -> PyResult<Bound<'py, PyList>> {
            if t >= self.num_topics {
                return Err(PyValueError::new_err("topic out of range"));
            }
            let items: Vec<Bound<'py, PyTuple>> = tops[t]
                .iter()
                .map(|&w| {
                    PyTuple::new_bound(py, &[vocab[w].clone().into_py(py), beta[[t, w]].into_py(py)])
                })
                .collect();
            Ok(PyList::new_bound(py, items))
        };
        match topic {
            Some(t) => Ok(one(t)?.into_any()),
            None => {
                let all: Vec<Bound<'py, PyList>> =
                    (0..self.num_topics).map(one).collect::<PyResult<_>>()?;
                Ok(PyList::new_bound(py, all).into_any())
            }
        }
    }

    /// UMass topic coherence per topic, shape ``(num_topics,)``.
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.beta.as_ref().unwrap(), self.num_topics, n);
        let scores = umass_coherence(self.corpus.as_ref().unwrap(), &tops);
        Ok(Array1::from(scores).to_pyarray_bound(py))
    }

    /// Infer topic proportions θ for *new* documents by collapsed Gibbs against
    /// the fitted topic-word matrix (the response is not used — this is the
    /// unsupervised E-step). `data` is a :class:`Corpus` or `list[list[str]]`;
    /// OOV tokens are dropped. Returns ``(num_docs, num_topics)``. To predict the
    /// response for new documents, take ``transform(data) @ eta``.
    #[pyo3(signature = (data, *, iterations=100, burn_in=10, num_samples=10,
                        sample_interval=5, seed=None))]
    fn transform<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        iterations: usize,
        burn_in: usize,
        num_samples: usize,
        sample_interval: usize,
        seed: Option<u64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        let alpha = vec![self.alpha; self.num_topics];
        transform_gibbs(
            py, data, &self.corpus.as_ref().unwrap().id_to_word, self.beta.as_ref().unwrap(),
            &alpha, iterations, burn_in, num_samples, sample_interval,
            seed.unwrap_or(self.seed),
        )
    }

    /// Save the fitted model to `path`. Reload with `SupervisedLDA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &SldaState {
            num_topics: self.num_topics, alpha: self.alpha, seed: self.seed, fitted: self.fitted,
            sigma2: self.sigma2, eta: arr1_opt(&self.eta), beta: arr2_opt(&self.beta),
            theta: arr2_opt(&self.theta), log_beta: self.log_beta.clone(),
            corpus: self.corpus.clone(),
        })
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: SldaState = read_state(path)?;
        Ok(SupervisedLDA {
            num_topics: s.num_topics, alpha: s.alpha, seed: s.seed, fitted: s.fitted,
            sigma2: s.sigma2, eta: arr1_back(s.eta), beta: arr2_back(s.beta),
            theta: arr2_back(s.theta), log_beta: s.log_beta, corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!("SupervisedLDA(num_topics={}, fitted={})", self.num_topics, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// PT: Pseudo-document Topic Model (short texts)
// ---------------------------------------------------------------------------

/// Pseudo-document Topic Model (Zuo et al. 2016) for **short texts**. Documents
/// are aggregated into `num_pseudo` pseudo-documents that carry the topic
/// distributions, so the topic structure is estimated from richer aggregated
/// statistics than individual short documents would provide. Collapsed Gibbs.
#[pyclass(module = "topica")]
pub struct PT {
    num_topics: usize,
    num_pseudo: usize,
    alpha: f64,
    beta: f64,
    seed: u64,
    fitted: bool,
    phi: Option<Array2<f64>>,
    theta: Option<Array2<f64>>,
    corpus: Option<corpus::Corpus>,
}

impl PT {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl PT {
    /// Create an unfitted model. `num_pseudo` is the number of pseudo-documents
    /// short texts are aggregated into (more = finer, fewer = more aggregation).
    #[new]
    #[pyo3(signature = (num_topics, *, num_pseudo=100, alpha=0.1, beta=0.01, seed=42))]
    fn new(num_topics: usize, num_pseudo: usize, alpha: f64, beta: f64, seed: u64) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics must be >= 2"));
        }
        if num_pseudo < 1 {
            return Err(PyValueError::new_err("num_pseudo must be >= 1"));
        }
        if alpha <= 0.0 || beta <= 0.0 {
            return Err(PyValueError::new_err("alpha and beta must be > 0"));
        }
        Ok(PT {
            num_topics, num_pseudo, alpha, beta, seed,
            fitted: false, phi: None, theta: None, corpus: None,
        })
    }

    /// Fit by collapsed Gibbs sampling for `iters` sweeps.
    #[pyo3(signature = (data, *, iters=1000))]
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>, iters: usize) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_types = corpus.num_types();
        let (k, p, a, b) = (self.num_topics, self.num_pseudo, self.alpha, self.beta);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let (model, corpus) = py.allow_threads(move || {
            let m = pt::fit_ptm(&corpus.docs, num_types, k, p, a, b, iters, &mut rng);
            (m, corpus)
        });
        self.phi = Some(vecs_to_arr2(&model.topic_word()));
        self.theta = Some(vecs_to_arr2(&model.doc_topic()));
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(&self, py: Python<'py>, n: usize, topic: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        topic_words_helper(py, self.phi.as_ref().unwrap(), &self.corpus.as_ref().unwrap().id_to_word, self.num_topics, n, topic)
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_topics, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `PT.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &PtState {
            num_topics: self.num_topics, num_pseudo: self.num_pseudo, alpha: self.alpha,
            beta: self.beta, seed: self.seed, fitted: self.fitted,
            phi: arr2_opt(&self.phi), theta: arr2_opt(&self.theta), corpus: self.corpus.clone(),
        })
    }
    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: PtState = read_state(path)?;
        Ok(PT {
            num_topics: s.num_topics, num_pseudo: s.num_pseudo, alpha: s.alpha, beta: s.beta,
            seed: s.seed, fitted: s.fitted, phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!("PT(num_topics={}, num_pseudo={}, fitted={})", self.num_topics, self.num_pseudo, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// GSDMM: Gibbs Sampling Dirichlet Multinomial Mixture (short-text clustering)
// ---------------------------------------------------------------------------

/// GSDMM — the "Movie Group Process" (Yin & Wang 2014). A mixture model for
/// **short texts** (tweets, survey answers, headlines) where each document
/// belongs to exactly *one* topic, not a mixture. You set an upper bound `K` on
/// the number of clusters; empty clusters die out during sampling, so the
/// effective `num_topics` is inferred from the data (≤ K). Handles the sparsity
/// of short documents far better than LDA.
#[pyclass(module = "topica")]
pub struct GSDMM {
    k_max: usize,
    alpha: f64,
    beta: f64,
    seed: u64,
    fitted: bool,
    num_used: usize,
    phi: Option<Array2<f64>>,        // num_used × V (used clusters only)
    theta: Option<Array2<f64>>,      // num_docs × num_used (soft assignment)
    doc_cluster: Vec<usize>,         // hard assignment per doc, remapped to 0..num_used
    corpus: Option<corpus::Corpus>,
}

impl GSDMM {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl GSDMM {
    /// Create an unfitted model. `num_topics` is the *maximum* number of clusters
    /// `K`; the number actually used (non-empty after fitting) is reported by the
    /// `num_topics` getter and is usually smaller. `alpha` controls the pull
    /// toward populous clusters; `beta` is the word-Dirichlet smoothing.
    #[new]
    #[pyo3(signature = (num_topics, *, alpha=0.1, beta=0.1, seed=42))]
    fn new(num_topics: usize, alpha: f64, beta: f64, seed: u64) -> PyResult<Self> {
        if num_topics < 2 {
            return Err(PyValueError::new_err("num_topics (max clusters) must be >= 2"));
        }
        if alpha <= 0.0 || beta <= 0.0 {
            return Err(PyValueError::new_err("alpha and beta must be > 0"));
        }
        Ok(GSDMM {
            k_max: num_topics, alpha, beta, seed,
            fitted: false, num_used: 0, phi: None, theta: None,
            doc_cluster: Vec::new(), corpus: None,
        })
    }

    /// Fit by the Movie Group Process (collapsed Gibbs) for `iters` sweeps.
    #[pyo3(signature = (data, *, iters=30))]
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>, iters: usize) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_types = corpus.num_types();
        let (k, a, b) = (self.k_max, self.alpha, self.beta);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let (model, corpus) = py.allow_threads(move || {
            let m = gsdmm::fit_gsdmm(&corpus.docs, num_types, k, a, b, iters, &mut rng);
            (m, corpus)
        });

        // Keep only non-empty clusters; remap their ids to a dense 0..num_used.
        let used = model.used_clusters();
        let mut remap = vec![usize::MAX; self.k_max];
        for (new_i, &old) in used.iter().enumerate() {
            remap[old] = new_i;
        }
        let num_used = used.len();

        let phi_rows: Vec<Vec<f64>> = used.iter().map(|&k| model.cluster_word(k)).collect();
        self.phi = Some(vecs_to_arr2(&phi_rows));

        // Soft per-doc distribution restricted to the used clusters, renormalized.
        let dist = model.doc_cluster_dist(&corpus.docs);
        let d = dist.len();
        let mut theta = Array2::<f64>::zeros((d, num_used));
        for (di, row) in dist.iter().enumerate() {
            let mut s = 0.0;
            for &old in &used {
                s += row[old];
            }
            let s = if s > 0.0 { s } else { 1.0 };
            for (&old, ni) in used.iter().zip(0..) {
                theta[[di, ni]] = row[old] / s;
            }
        }
        self.theta = Some(theta);
        self.doc_cluster = model.doc_cluster().iter().map(|&c| remap[c]).collect();
        self.num_used = num_used;
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// Topic-word matrix β, shape ``(num_topics, num_words)`` (used clusters only).
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }
    /// Document-topic matrix θ, shape ``(num_docs, num_topics)``; rows sum to 1.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }
    /// Hard cluster assignment of each document, shape ``(num_docs,)``; values in
    /// ``0..num_topics``. GSDMM gives each document a single cluster.
    #[getter]
    fn doc_cluster<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        self.require_fitted()?;
        let v: Vec<i64> = self.doc_cluster.iter().map(|&c| c as i64).collect();
        Ok(Array1::from(v).to_pyarray_bound(py))
    }
    /// The number of *non-empty* clusters discovered (≤ the `K` you set).
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_used
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(&self, py: Python<'py>, n: usize, topic: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        topic_words_helper(py, self.phi.as_ref().unwrap(), &self.corpus.as_ref().unwrap().id_to_word, self.num_used, n, topic)
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_used, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `GSDMM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &GsdmmState {
            k_max: self.k_max, alpha: self.alpha, beta: self.beta, seed: self.seed,
            fitted: self.fitted, num_used: self.num_used,
            phi: arr2_opt(&self.phi), theta: arr2_opt(&self.theta),
            doc_cluster: self.doc_cluster.clone(), corpus: self.corpus.clone(),
        })
    }
    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: GsdmmState = read_state(path)?;
        Ok(GSDMM {
            k_max: s.k_max, alpha: s.alpha, beta: s.beta, seed: s.seed, fitted: s.fitted,
            num_used: s.num_used, phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            doc_cluster: s.doc_cluster, corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!("GSDMM(num_topics={}, k_max={}, fitted={})", self.num_used, self.k_max, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// SeededLDA: guided topics via seed-word priors
// ---------------------------------------------------------------------------

/// Parse a ``{topic_name: [words]}`` dict into ordered (names, word-lists),
/// preserving insertion order.
fn parse_seed_dict(d: &Bound<'_, PyDict>) -> PyResult<(Vec<String>, Vec<Vec<String>>)> {
    let mut names = Vec::new();
    let mut words = Vec::new();
    for (k, v) in d.iter() {
        names.push(k.extract::<String>().map_err(|_| {
            PyValueError::new_err("seed/keyword dict keys must be strings (topic names)")
        })?);
        words.push(v.extract::<Vec<String>>().map_err(|_| {
            PyValueError::new_err("seed/keyword dict values must be lists of strings")
        })?);
    }
    if names.is_empty() {
        return Err(PyValueError::new_err("provide at least one seeded/keyword topic"));
    }
    Ok((names, words))
}

/// Map per-topic seed/keyword *words* to vocabulary ids (dropping out-of-vocab),
/// padding with empty lists up to `num_topics` total topics.
fn seed_word_ids(
    word_strings: &[Vec<String>],
    id_to_word: &[String],
    num_topics: usize,
) -> Vec<Vec<usize>> {
    let index: HashMap<&str, usize> =
        id_to_word.iter().enumerate().map(|(i, w)| (w.as_str(), i)).collect();
    let mut out: Vec<Vec<usize>> = word_strings
        .iter()
        .map(|ws| ws.iter().filter_map(|w| index.get(w.as_str()).copied()).collect())
        .collect();
    out.resize(num_topics, Vec::new());
    out
}

/// Seeded LDA (guided topic modeling): you supply a few **seed words** per topic
/// and the model is steered so those topics form around them, while the rest of
/// each topic's vocabulary (and any `residual` unseeded topics) is still learned.
/// Useful when theory tells you which themes to expect (Jagarlamudi et al. 2012;
/// the seeding follows koheiw/seededlda — seed words get a `weight × 100`
/// prior pseudocount in their topic).
#[pyclass(module = "topica")]
pub struct SeededLDA {
    seed_names: Vec<String>,
    seed_words: Vec<Vec<String>>,
    residual: usize,
    alpha: f64,
    beta: f64,
    weight: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    phi: Option<Array2<f64>>,
    theta: Option<Array2<f64>>,
    corpus: Option<corpus::Corpus>,
}

impl SeededLDA {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
    fn num_topics_val(&self) -> usize {
        self.seed_names.len() + self.residual
    }
}

#[pymethods]
impl SeededLDA {
    /// Create an unfitted model. `seed_words` is ``{topic_name: [words]}``;
    /// `residual` adds that many extra unseeded topics. `weight` (default 0.01,
    /// matching the seededlda package) scales the seed prior. `alpha` is the
    /// per-topic Dirichlet, `beta` the base topic-word smoothing.
    #[new]
    #[pyo3(signature = (seed_words, *, residual=0, alpha=0.1, beta=0.01, weight=0.01, seed=42))]
    fn new(
        seed_words: &Bound<'_, PyDict>,
        residual: usize,
        alpha: f64,
        beta: f64,
        weight: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let (names, words) = parse_seed_dict(seed_words)?;
        if alpha <= 0.0 || beta <= 0.0 {
            return Err(PyValueError::new_err("alpha and beta must be > 0"));
        }
        if names.len() + residual < 2 {
            return Err(PyValueError::new_err("need at least 2 topics (seeded + residual)"));
        }
        Ok(SeededLDA {
            seed_names: names, seed_words: words, residual, alpha, beta, weight, seed,
            fitted: false, topic_names: Vec::new(), phi: None, theta: None, corpus: None,
        })
    }

    /// Fit by collapsed Gibbs for `iters` sweeps. Seeded topics come first (in
    /// the order given), then the residual topics.
    #[pyo3(signature = (data, *, iters=2000))]
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>, iters: usize) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_topics = self.num_topics_val();
        let num_types = corpus.num_types();
        let seeds = seed_word_ids(&self.seed_words, &corpus.id_to_word, num_topics);
        let (alpha, beta, seed_weight) = (self.alpha, self.beta, self.weight * 100.0);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let (model, corpus) = py.allow_threads(move || {
            let m = seeded::fit_seeded_lda(
                &corpus.docs, num_types, num_topics, &seeds, alpha, beta, seed_weight, iters,
                &mut rng,
            );
            (m, corpus)
        });
        self.phi = Some(vecs_to_arr2(&model.topic_word_all()));
        self.theta = Some(vecs_to_arr2(&model.doc_topic()));
        let mut names = self.seed_names.clone();
        for i in 0..self.residual {
            names.push(format!("residual_{}", i + 1));
        }
        self.topic_names = names;
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics_val()
    }
    /// The topic labels: the seed names you gave, then ``residual_1`` … for any
    /// unseeded topics.
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(&self, py: Python<'py>, n: usize, topic: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        topic_words_helper(py, self.phi.as_ref().unwrap(), &self.corpus.as_ref().unwrap().id_to_word, self.num_topics_val(), n, topic)
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_topics_val(), n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `SeededLDA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &SeededState {
            num_topics: self.num_topics_val(), alpha: self.alpha, beta: self.beta,
            weight: self.weight, seed: self.seed, fitted: self.fitted,
            topic_names: self.topic_names.clone(), phi: arr2_opt(&self.phi),
            theta: arr2_opt(&self.theta), corpus: self.corpus.clone(),
        })
    }
    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: SeededState = read_state(path)?;
        Ok(SeededLDA {
            seed_names: Vec::new(), seed_words: Vec::new(),
            residual: s.num_topics.saturating_sub(s.topic_names.iter().filter(|n| !n.starts_with("residual_")).count()),
            alpha: s.alpha, beta: s.beta, weight: s.weight, seed: s.seed, fitted: s.fitted,
            topic_names: s.topic_names, phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!("SeededLDA(seeded={}, residual={}, fitted={})", self.seed_names.len(), self.residual, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// KeyATM: keyword-assisted topic model (Eshima, Imai & Sasaki 2024)
// ---------------------------------------------------------------------------

/// Keyword-Assisted Topic Model (keyATM Base). Like LDA, but some topics carry a
/// researcher-supplied **keyword** list; a token in a keyword topic comes either
/// from a distribution over only that topic's keywords or from the topic's full
/// distribution. This anchors keyword topics to their keywords while still
/// learning the rest of the vocabulary. Faithful to keyATM/keyATM.
#[pyclass(module = "topica")]
pub struct KeyATM {
    key_names: Vec<String>,
    keywords: Vec<Vec<String>>,
    num_topics: usize,
    alpha: f64,
    beta: f64,
    beta_keyword: f64,
    gamma1: f64,
    gamma2: f64,
    seed: u64,
    fitted: bool,
    topic_names: Vec<String>,
    keyword_rate: Vec<f64>,
    phi: Option<Array2<f64>>,
    theta: Option<Array2<f64>>,
    corpus: Option<corpus::Corpus>,
}

impl KeyATM {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl KeyATM {
    /// Create an unfitted model. `keywords` is ``{topic_name: [words]}`` (the
    /// keyword topics, in order). `num_topics` (default = number of keyword
    /// topics) may be larger to add regular, no-keyword topics. `alpha` is the
    /// per-topic Dirichlet, `beta`/`beta_keyword` the regular and keyword
    /// topic-word smoothing, and `gamma1`/`gamma2` the Beta prior on the
    /// keyword-vs-regular switch.
    #[new]
    #[pyo3(signature = (keywords, *, num_topics=None, alpha=0.1, beta=0.01, beta_keyword=0.1, gamma1=1.0, gamma2=1.0, seed=42))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        keywords: &Bound<'_, PyDict>,
        num_topics: Option<usize>,
        alpha: f64,
        beta: f64,
        beta_keyword: f64,
        gamma1: f64,
        gamma2: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let (names, words) = parse_seed_dict(keywords)?;
        let k = num_topics.unwrap_or(names.len());
        if k < names.len() {
            return Err(PyValueError::new_err(
                "num_topics must be >= the number of keyword topics",
            ));
        }
        if k < 2 {
            return Err(PyValueError::new_err("need at least 2 topics"));
        }
        if alpha <= 0.0 || beta <= 0.0 || beta_keyword <= 0.0 || gamma1 <= 0.0 || gamma2 <= 0.0 {
            return Err(PyValueError::new_err("alpha, beta, beta_keyword, gamma1, gamma2 must be > 0"));
        }
        Ok(KeyATM {
            key_names: names, keywords: words, num_topics: k, alpha, beta, beta_keyword,
            gamma1, gamma2, seed, fitted: false, topic_names: Vec::new(),
            keyword_rate: Vec::new(), phi: None, theta: None, corpus: None,
        })
    }

    /// Fit by collapsed Gibbs for `iters` sweeps. Keyword topics come first (in
    /// the order given), then any regular topics.
    #[pyo3(signature = (data, *, iters=1500))]
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>, iters: usize) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_topics = self.num_topics;
        let num_types = corpus.num_types();
        let keys = seed_word_ids(&self.keywords, &corpus.id_to_word, num_topics);
        let (alpha, beta, beta_key, g1, g2) =
            (self.alpha, self.beta, self.beta_keyword, self.gamma1, self.gamma2);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let (model, corpus) = py.allow_threads(move || {
            let m = keyatm::fit_keyatm(
                &corpus.docs, num_types, num_topics, &keys, alpha, beta, beta_key, g1, g2, iters,
                &mut rng,
            );
            (m, corpus)
        });
        self.phi = Some(vecs_to_arr2(&model.topic_word_all()));
        self.theta = Some(vecs_to_arr2(&model.doc_topic()));
        self.keyword_rate = model.keyword_rate();
        let mut names = self.key_names.clone();
        for i in self.key_names.len()..num_topics {
            names.push(format!("topic_{}", i));
        }
        self.topic_names = names;
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }
    /// Per-topic keyword switch rate ``π_k`` (the share of a keyword topic's mass
    /// drawn from its keyword distribution); 0 for regular topics.
    #[getter]
    fn keyword_rate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        Ok(Array1::from(self.keyword_rate.clone()).to_pyarray_bound(py))
    }
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_topics
    }
    #[getter]
    fn topic_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.topic_names.clone())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(&self, py: Python<'py>, n: usize, topic: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        topic_words_helper(py, self.phi.as_ref().unwrap(), &self.corpus.as_ref().unwrap().id_to_word, self.num_topics, n, topic)
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_topics, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `KeyATM.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &KeyAtmState {
            num_topics: self.num_topics, alpha: self.alpha, beta: self.beta,
            beta_keyword: self.beta_keyword, gamma1: self.gamma1, gamma2: self.gamma2,
            seed: self.seed, fitted: self.fitted, topic_names: self.topic_names.clone(),
            keyword_rate: self.keyword_rate.clone(), phi: arr2_opt(&self.phi),
            theta: arr2_opt(&self.theta), corpus: self.corpus.clone(),
        })
    }
    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: KeyAtmState = read_state(path)?;
        Ok(KeyATM {
            key_names: Vec::new(), keywords: Vec::new(), num_topics: s.num_topics,
            alpha: s.alpha, beta: s.beta, beta_keyword: s.beta_keyword, gamma1: s.gamma1,
            gamma2: s.gamma2, seed: s.seed, fitted: s.fitted, topic_names: s.topic_names,
            keyword_rate: s.keyword_rate, phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!("KeyATM(keyword_topics={}, num_topics={}, fitted={})", self.key_names.len(), self.num_topics, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// PA: Pachinko Allocation Model (super-/sub-topic hierarchy)
// ---------------------------------------------------------------------------

/// Pachinko Allocation Model (Li & McCallum 2006): a DAG of `num_super`
/// super-topics over `num_sub` shared sub-topics over words, capturing topic
/// *correlations* — `super_sub` reports which sub-topics each super-topic groups
/// together. Collapsed Gibbs over (super, sub) pairs.
#[pyclass(module = "topica")]
pub struct PA {
    num_super: usize,
    num_sub: usize,
    alpha: f64,
    beta: f64,
    seed: u64,
    fitted: bool,
    phi: Option<Array2<f64>>,       // num_sub × V
    theta: Option<Array2<f64>>,     // num_docs × num_sub
    super_sub: Option<Array2<f64>>, // num_super × num_sub
    corpus: Option<corpus::Corpus>,
}

impl PA {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl PA {
    /// Create an unfitted model with `num_super` super-topics and `num_sub`
    /// sub-topics (the sub-topics are the word-level topics).
    #[new]
    #[pyo3(signature = (num_super, num_sub, *, alpha=0.1, beta=0.01, seed=42))]
    fn new(num_super: usize, num_sub: usize, alpha: f64, beta: f64, seed: u64) -> PyResult<Self> {
        if num_super < 1 || num_sub < 2 {
            return Err(PyValueError::new_err("num_super must be >= 1 and num_sub >= 2"));
        }
        if alpha <= 0.0 || beta <= 0.0 {
            return Err(PyValueError::new_err("alpha and beta must be > 0"));
        }
        Ok(PA {
            num_super, num_sub, alpha, beta, seed,
            fitted: false, phi: None, theta: None, super_sub: None, corpus: None,
        })
    }

    /// Fit by collapsed Gibbs sampling for `iters` sweeps.
    #[pyo3(signature = (data, *, iters=1000))]
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>, iters: usize) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_types = corpus.num_types();
        let (s, k, a, b) = (self.num_super, self.num_sub, self.alpha, self.beta);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let (model, corpus) = py.allow_threads(move || {
            let m = pa::fit_pam(&corpus.docs, num_types, s, k, a, b, iters, &mut rng);
            (m, corpus)
        });
        self.phi = Some(vecs_to_arr2(&model.topic_word()));
        self.theta = Some(vecs_to_arr2(&model.doc_topic()));
        self.super_sub = Some(vecs_to_arr2(&model.super_sub()));
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// Sub-topic word distributions, shape ``(num_sub, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.phi.as_ref().unwrap().to_pyarray_bound(py))
    }
    /// Document × sub-topic proportions, shape ``(num_docs, num_sub)``.
    #[getter]
    fn doc_topic<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.theta.as_ref().unwrap().to_pyarray_bound(py))
    }
    /// Super-topic → sub-topic association, shape ``(num_super, num_sub)``; row s
    /// shows which sub-topics super-topic s groups together (the correlations).
    #[getter]
    fn super_sub<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.super_sub.as_ref().unwrap().to_pyarray_bound(py))
    }
    #[getter]
    fn num_super(&self) -> usize {
        self.num_super
    }
    #[getter]
    fn num_sub(&self) -> usize {
        self.num_sub
    }
    /// Alias for `num_sub` (the word-level topics).
    #[getter]
    fn num_topics(&self) -> usize {
        self.num_sub
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }
    #[getter]
    fn doc_names(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().doc_names.clone())
    }

    #[pyo3(signature = (n=10, *, topic=None))]
    fn top_words<'py>(&self, py: Python<'py>, n: usize, topic: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        self.require_fitted()?;
        topic_words_helper(py, self.phi.as_ref().unwrap(), &self.corpus.as_ref().unwrap().id_to_word, self.num_sub, n, topic)
    }
    #[pyo3(signature = (n=10))]
    fn coherence<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.require_fitted()?;
        let tops = top_word_ids_phi(self.phi.as_ref().unwrap(), self.num_sub, n);
        Ok(Array1::from(umass_coherence(self.corpus.as_ref().unwrap(), &tops)).to_pyarray_bound(py))
    }

    /// Save the fitted model to `path`. Reload with `PA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &PaState {
            num_super: self.num_super, num_sub: self.num_sub, alpha: self.alpha, beta: self.beta,
            seed: self.seed, fitted: self.fitted, phi: arr2_opt(&self.phi),
            theta: arr2_opt(&self.theta), super_sub: arr2_opt(&self.super_sub),
            corpus: self.corpus.clone(),
        })
    }
    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: PaState = read_state(path)?;
        Ok(PA {
            num_super: s.num_super, num_sub: s.num_sub, alpha: s.alpha, beta: s.beta,
            seed: s.seed, fitted: s.fitted, phi: arr2_back(s.phi), theta: arr2_back(s.theta),
            super_sub: arr2_back(s.super_sub), corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        format!("PA(num_super={}, num_sub={}, fitted={})", self.num_super, self.num_sub, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// HLDA: Hierarchical LDA (nested CRP topic tree)
// ---------------------------------------------------------------------------

/// Hierarchical LDA (Blei, Griffiths & Jordan): topics organized in a tree of
/// fixed `depth`, inferred by the nested Chinese Restaurant Process. The root is
/// the shared (general) topic; deeper nodes are progressively more specific.
/// Each document follows a root-to-leaf path. Inspect the tree with
/// `topic_word`/`node_levels`/`node_parents`/`doc_paths`.
#[pyclass(module = "topica")]
pub struct HLDA {
    depth: usize,
    gamma: f64,
    eta: f64,
    alpha: f64,
    seed: u64,
    fitted: bool,
    num_nodes: usize,
    node_topic_word: Option<Array2<f64>>, // num_nodes × V
    node_levels: Vec<usize>,
    node_parents: Vec<i64>, // -1 for the root
    doc_paths: Vec<Vec<usize>>,
    corpus: Option<corpus::Corpus>,
}

impl HLDA {
    fn require_fitted(&self) -> PyResult<()> {
        if self.fitted {
            Ok(())
        } else {
            Err(PyRuntimeError::new_err("model is not fitted yet; call fit() first"))
        }
    }
}

#[pymethods]
impl HLDA {
    /// Create an unfitted model. `depth` is the (fixed) tree depth; `gamma` is
    /// the nested-CRP concentration (larger ⇒ more child topics); `eta` the
    /// topic-word Dirichlet; `alpha` the per-document level distribution.
    #[new]
    #[pyo3(signature = (*, depth=3, gamma=1.0, eta=0.01, alpha=0.1, seed=42))]
    fn new(depth: usize, gamma: f64, eta: f64, alpha: f64, seed: u64) -> PyResult<Self> {
        if depth < 2 {
            return Err(PyValueError::new_err("depth must be >= 2"));
        }
        if gamma <= 0.0 || eta <= 0.0 || alpha <= 0.0 {
            return Err(PyValueError::new_err("gamma, eta, alpha must be > 0"));
        }
        Ok(HLDA {
            depth, gamma, eta, alpha, seed,
            fitted: false, num_nodes: 0, node_topic_word: None,
            node_levels: Vec::new(), node_parents: Vec::new(), doc_paths: Vec::new(), corpus: None,
        })
    }

    /// Fit by nested-CRP collapsed Gibbs sampling for `iters` sweeps.
    #[pyo3(signature = (data, *, iters=500))]
    fn fit(&mut self, py: Python<'_>, data: &Bound<'_, PyAny>, iters: usize) -> PyResult<()> {
        let corpus: corpus::Corpus = if let Ok(c) = data.extract::<Corpus>() {
            c.inner
        } else {
            let docs: Vec<Vec<String>> = data.extract().map_err(|_| {
                PyValueError::new_err("fit() expects a Corpus or a list of token lists")
            })?;
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0, 0, 0)?
        };
        if corpus.num_docs() == 0 {
            return Err(PyValueError::new_err("corpus contains no documents"));
        }
        let num_types = corpus.num_types();
        let (depth, gamma, eta, alpha) = (self.depth, self.gamma, self.eta, self.alpha);
        let mut rng = ChaCha8Rng::seed_from_u64(self.seed);
        let (model, corpus) = py.allow_threads(move || {
            let m = hlda::fit_hlda(&corpus.docs, num_types, depth, gamma, eta, alpha, iters, &mut rng);
            (m, corpus)
        });

        let nn = model.num_nodes();
        let mut tw = Array2::<f64>::zeros((nn, num_types));
        for i in 0..nn {
            for (w, &val) in model.topic_word(i).iter().enumerate() {
                tw[[i, w]] = val;
            }
        }
        self.num_nodes = nn;
        self.node_topic_word = Some(tw);
        self.node_levels = (0..nn).map(|i| model.node_level(i)).collect();
        self.node_parents = (0..nn).map(|i| model.node_parent(i).map(|p| p as i64).unwrap_or(-1)).collect();
        self.doc_paths = (0..corpus.num_docs()).map(|d| model.doc_path(d)).collect();
        self.corpus = Some(corpus);
        self.fitted = true;
        Ok(())
    }

    /// The number of topic nodes in the inferred tree.
    #[getter]
    fn num_nodes(&self) -> PyResult<usize> {
        self.require_fitted()?;
        Ok(self.num_nodes)
    }
    /// Per-node word distributions, shape ``(num_nodes, num_words)``.
    #[getter]
    fn topic_word<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.require_fitted()?;
        Ok(self.node_topic_word.as_ref().unwrap().to_pyarray_bound(py))
    }
    /// The tree level (0 = root) of each node, length ``num_nodes``.
    #[getter]
    fn node_levels(&self) -> PyResult<Vec<usize>> {
        self.require_fitted()?;
        Ok(self.node_levels.clone())
    }
    /// The parent node id of each node (``-1`` for the root), length ``num_nodes``.
    #[getter]
    fn node_parents(&self) -> PyResult<Vec<i64>> {
        self.require_fitted()?;
        Ok(self.node_parents.clone())
    }
    /// Each document's root-to-leaf path (a list of node ids), length ``num_docs``.
    #[getter]
    fn doc_paths(&self) -> PyResult<Vec<Vec<usize>>> {
        self.require_fitted()?;
        Ok(self.doc_paths.clone())
    }
    /// The leaf node ids (nodes that are no node's parent).
    #[getter]
    fn leaves(&self) -> PyResult<Vec<usize>> {
        self.require_fitted()?;
        let parents: HashSet<i64> = self.node_parents.iter().copied().collect();
        Ok((0..self.num_nodes).filter(|&i| !parents.contains(&(i as i64))).collect())
    }
    #[getter]
    fn vocabulary(&self) -> PyResult<Vec<String>> {
        self.require_fitted()?;
        Ok(self.corpus.as_ref().unwrap().id_to_word.clone())
    }

    /// Top `n` words for one topic node as ``(word, probability)`` pairs.
    #[pyo3(signature = (node, n=10))]
    fn top_words(&self, node: usize, n: usize) -> PyResult<Vec<(String, f64)>> {
        self.require_fitted()?;
        if node >= self.num_nodes {
            return Err(PyValueError::new_err("node out of range"));
        }
        let tw = self.node_topic_word.as_ref().unwrap();
        let vocab = &self.corpus.as_ref().unwrap().id_to_word;
        let v = tw.shape()[1];
        let mut idx: Vec<usize> = (0..v).collect();
        idx.sort_by(|&a, &b| tw[[node, b]].partial_cmp(&tw[[node, a]]).unwrap());
        Ok(idx.into_iter().take(n).map(|w| (vocab[w].clone(), tw[[node, w]])).collect())
    }

    /// Save the fitted model to `path`. Reload with `HLDA.load`.
    fn save(&self, path: &str) -> PyResult<()> {
        self.require_fitted()?;
        write_state(path, &HldaState {
            depth: self.depth, gamma: self.gamma, eta: self.eta, alpha: self.alpha,
            seed: self.seed, fitted: self.fitted, num_nodes: self.num_nodes,
            node_topic_word: arr2_opt(&self.node_topic_word), node_levels: self.node_levels.clone(),
            node_parents: self.node_parents.clone(), doc_paths: self.doc_paths.clone(),
            corpus: self.corpus.clone(),
        })
    }
    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let s: HldaState = read_state(path)?;
        Ok(HLDA {
            depth: s.depth, gamma: s.gamma, eta: s.eta, alpha: s.alpha, seed: s.seed,
            fitted: s.fitted, num_nodes: s.num_nodes, node_topic_word: arr2_back(s.node_topic_word),
            node_levels: s.node_levels, node_parents: s.node_parents, doc_paths: s.doc_paths,
            corpus: s.corpus,
        })
    }

    fn __repr__(&self) -> String {
        if self.fitted {
            format!("HLDA(depth={}, num_nodes={}, fitted=true)", self.depth, self.num_nodes)
        } else {
            format!("HLDA(depth={}, fitted=false)", self.depth)
        }
    }
}

#[pymodule]
fn _topica(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LDA>()?;
    m.add_class::<DMR>()?;
    m.add_class::<LabeledLDA>()?;
    m.add_class::<SAGE>()?;
    m.add_class::<CTM>()?;
    m.add_class::<STM>()?;
    m.add_class::<HDP>()?;
    m.add_class::<DTM>()?;
    m.add_class::<SupervisedLDA>()?;
    m.add_class::<PT>()?;
    m.add_class::<GSDMM>()?;
    m.add_class::<SeededLDA>()?;
    m.add_class::<KeyATM>()?;
    m.add_class::<PA>()?;
    m.add_class::<HLDA>()?;
    m.add_class::<Corpus>()?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add_function(wrap_pyfunction!(window_cooccurrence, m)?)?;
    m.add("DEFAULT_TOKEN_REGEX", corpus::DEFAULT_TOKEN_REGEX)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
