//! PyO3 bindings: a Pythonic `LDA` + `Corpus` surface over the SparseLDA core.
//!
//! The compiled module is exposed to Python as `turbotopics._turbotopics`
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
use crate::hdp;
use crate::slda;
use crate::labeled;
use crate::sage;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use regex::Regex;

use crate::corpus::{self, InputFormat, LoadOptions};
use crate::model::TopicModel;
use crate::{ctm, optimize, output, sampler};

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn io_err(e: std::io::Error) -> PyErr {
    PyIOError::new_err(e.to_string())
}

// ---------------------------------------------------------------------------
// Corpus building from in-memory tokenised documents
// ---------------------------------------------------------------------------

/// Build a `corpus::Corpus` from already-tokenised documents.
///
/// Mirrors the vocab-construction and frequency-filtering logic of
/// `corpus::load_text_file`, minus the regex tokenisation/lowercasing — the
/// caller owns tokenisation here.
fn build_corpus_from_docs(
    docs_in: Vec<Vec<String>>,
    doc_names_in: Option<Vec<String>>,
    doc_labels_in: Option<Vec<String>>,
    stopwords: HashSet<String>,
    min_doc_freq: u32,
    max_doc_fraction: f64,
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

    // Frequency filtering (same policy as corpus::load_text_file).
    let max_df = (num_docs as f64 * max_doc_fraction).ceil() as u32;
    let keep: Vec<bool> = (0..num_types)
        .map(|id| doc_freqs[id] >= min_doc_freq && doc_freqs[id] <= max_df)
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
#[pyclass(module = "turbotopics")]
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
    /// label to every document. `stopwords` are dropped; `min_doc_freq` and
    /// `max_doc_fraction` prune words by document frequency.
    #[staticmethod]
    #[pyo3(signature = (documents, *, doc_names=None, doc_labels=None,
                        stopwords=None, min_doc_freq=1, max_doc_fraction=1.0))]
    fn from_documents(
        documents: Vec<Vec<String>>,
        doc_names: Option<Vec<String>>,
        doc_labels: Option<Vec<String>>,
        stopwords: Option<Vec<String>>,
        min_doc_freq: u32,
        max_doc_fraction: f64,
    ) -> PyResult<Self> {
        let stop: HashSet<String> = stopwords.unwrap_or_default().into_iter().collect();
        let inner = build_corpus_from_docs(
            documents,
            doc_names,
            doc_labels,
            stop,
            min_doc_freq,
            max_doc_fraction,
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
#[pyclass(module = "turbotopics")]
pub struct LDA {
    num_topics: usize,
    alpha_sum: Option<f64>,
    beta: f64,
    optimize_interval: usize,
    burn_in: usize,
    seed: u64,
    num_threads: usize,

    // Populated after fit().
    fitted: bool,
    phi: Option<Array2<f64>>,   // (num_topics, num_words)
    theta: Option<Array2<f64>>, // (num_docs, num_topics)
    model: Option<TopicModel>,
    corpus: Option<corpus::Corpus>,
}

impl LDA {
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
                        optimize_interval=50, burn_in=200, seed=42, num_threads=1))]
    fn new(
        num_topics: usize,
        alpha_sum: Option<f64>,
        beta: f64,
        optimize_interval: usize,
        burn_in: usize,
        seed: u64,
        num_threads: usize,
    ) -> PyResult<Self> {
        if num_topics == 0 {
            return Err(PyValueError::new_err("num_topics must be >= 1"));
        }
        if beta <= 0.0 {
            return Err(PyValueError::new_err("beta must be > 0"));
        }
        Ok(LDA {
            num_topics,
            alpha_sum,
            beta,
            optimize_interval,
            burn_in,
            seed,
            num_threads: num_threads.max(1),
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
            build_corpus_from_docs(docs, None, None, HashSet::new(), 1, 1.0)?
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

        // phi: transpose (word, topic) -> (topic, word) for the conventional
        // (num_topics, num_words) orientation.
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
#[pyclass(module = "turbotopics")]
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
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0)?
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
#[pyclass(module = "turbotopics")]
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
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0)?
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
#[pyclass(module = "turbotopics")]
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
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0)?
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

/// Correlated Topic Model (Blei & Lafferty; the STM core). Topics are drawn
/// from a logistic-normal prior with a full covariance, so they can correlate —
/// unlike LDA's Dirichlet. Fit by variational EM (STM's Laplace E-step).
///
/// This is the engine STM builds on; prevalence/content covariates layer on top.
#[pyclass(module = "turbotopics")]
pub struct CTM {
    num_topics: usize,
    sigma_shrink: f64,
    seed: u64,
    init_spectral: bool,

    fitted: bool,
    beta: Option<Array2<f64>>,  // (num_topics, num_words)
    theta: Option<Array2<f64>>, // (num_docs, num_topics)
    corr: Option<Array2<f64>>,  // (num_topics, num_topics)
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
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0)?
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

        self.beta = Some(beta);
        self.theta = Some(theta);
        self.corr = Some(corr);
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
/// the learned γ; pair it with `turbotopics.stm.estimate_effect` for inference.
#[pyclass(module = "turbotopics")]
pub struct STM {
    num_topics: usize,
    sigma_shrink: f64,
    seed: u64,
    init_spectral: bool,

    fitted: bool,
    beta: Option<Array2<f64>>,
    theta: Option<Array2<f64>>,
    corr: Option<Array2<f64>>,
    gamma: Option<Array2<f64>>, // (num_features, num_topics-1); None if no prevalence
    feature_names: Vec<String>,
    content_beta: Option<Vec<Vec<Vec<f64>>>>, // G×K×V; None if no content
    group_names: Vec<String>,
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
            gamma: None,
            feature_names: Vec::new(),
            content_beta: None,
            group_names: Vec::new(),
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
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0)?
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

        self.beta = Some(beta);
        self.theta = Some(theta);
        self.corr = Some(corr);
        self.feature_names = feat_names;
        self.content_beta = model.content_beta;
        self.group_names = group_vocab;
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

    /// Prevalence coefficients γ, shape ``(num_features, num_topics-1)`` — how
    /// each covariate (row 0 is the intercept) shifts each topic's log-prior.
    /// The last topic is the softmax reference. For inference, prefer
    /// ``turbotopics.stm.estimate_effect(model.doc_topic, X)``.
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

    fn __repr__(&self) -> String {
        format!("STM(num_topics={}, fitted={})", self.num_topics, self.fitted)
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

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
#[pyclass(module = "turbotopics")]
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
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0)?
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
#[pyclass(module = "turbotopics")]
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
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0)?
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
#[pyclass(module = "turbotopics")]
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
            build_corpus_from_docs(docs, None, None, std::collections::HashSet::new(), 1, 1.0)?
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

    fn __repr__(&self) -> String {
        format!("SupervisedLDA(num_topics={}, fitted={})", self.num_topics, self.fitted)
    }
}

#[pymodule]
fn _turbotopics(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LDA>()?;
    m.add_class::<DMR>()?;
    m.add_class::<LabeledLDA>()?;
    m.add_class::<SAGE>()?;
    m.add_class::<CTM>()?;
    m.add_class::<STM>()?;
    m.add_class::<HDP>()?;
    m.add_class::<DTM>()?;
    m.add_class::<SupervisedLDA>()?;
    m.add_class::<Corpus>()?;
    m.add_function(wrap_pyfunction!(tokenize, m)?)?;
    m.add("DEFAULT_TOKEN_REGEX", corpus::DEFAULT_TOKEN_REGEX)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
