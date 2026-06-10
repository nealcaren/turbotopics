from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Union, overload
import numpy
import numpy.typing

DEFAULT_TOKEN_REGEX: str
__version__: str


def tokenize(
    text: str,
    *,
    lowercase: bool = True,
    stopwords: Iterable[str] | None = None,
    token_regex: str | None = None,
    min_length: int = 1,
) -> list[str]:
    """Tokenize a string with the corpus loader's regex; lowercase, drop short
    tokens and stopwords. `stopwords` is any iterable of strings (list, set, or
    `topica.ENGLISH_STOPWORDS`). Convenience for building list[list[str]] input."""
    ...


def project(
    data: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
    n_components: int = 2,
    *,
    method: str = "pca",
    n_neighbors: int = 15,
    perplexity: float = 30.0,
    seed: int = 0,
) -> numpy.typing.NDArray[numpy.float64]:
    """Project a high-dimensional array to `n_components` for plotting or clustering.

    `method` is "pca" (default, deterministic, distance-faithful), "umap", or
    "tsne". UMAP and t-SNE preserve local neighborhoods but distort global geometry
    (between-cluster distances and cluster sizes are not meaningful) and are not
    reproducible across runs (a warning is issued); PCA is the honest default.
    `data` is a 2D float array or a list of float lists. Returns an
    `(n_rows, n_components)` array.
    """
    ...


def window_cooccurrence(
    docs: list[list[int]],
    num_relevant: int,
    pairs: list[tuple[int, int]],
    window: int,
) -> tuple[list[float], list[float], float]:
    """Window/document co-occurrence counts for coherence scoring (internal).

    docs holds relevant-word ids per token, 4294967295 marks a non-relevant
    token; pairs are (a, b) with a < b; window=0 requests document-level
    co-occurrence. Returns (occ, co, n_windows). Used by topica.coherence.
    """
    ...


class Corpus:
    """A preprocessed token corpus for LDA training."""

    @staticmethod
    def from_documents(
        documents: list[list[str]],
        *,
        doc_names: list[str] | None = None,
        doc_labels: list[str] | None = None,
        stopwords: list[str] | None = None,
        min_doc_freq: int = 1,
        max_doc_fraction: float = 1.0,
    ) -> Corpus:
        """Build a Corpus from a list of token lists.

        A document left with no tokens by pruning is dropped, so ``num_docs`` can
        be smaller than ``len(documents)``; the surviving original indices are in
        ``kept_indices`` (realign external covariates with ``X[corpus.kept_indices]``).
        """
        ...

    @staticmethod
    def from_text_file(
        path: str,
        *,
        format: str = "plain",
        id_field: bool = False,
        id_column: int = 0,
        label_column: int | None = 1,
        text_column: int = 2,
        token_regex: str | None = None,
        stopwords: list[str] | None = None,
        min_doc_freq: int = 1,
        max_doc_fraction: float = 1.0,
    ) -> Corpus:
        """Build a Corpus by reading and tokenizing a text file."""
        ...

    @staticmethod
    def load(path: str) -> Corpus:
        """Load a binary corpus previously saved by .save() or the preprocess CLI."""
        ...

    def save(self, path: str) -> None:
        """Serialize the corpus to a binary file."""
        ...

    @property
    def num_docs(self) -> int:
        """Number of documents in the corpus."""
        ...

    @property
    def num_words(self) -> int:
        """Vocabulary size (number of unique word types)."""
        ...

    @property
    def total_tokens(self) -> int:
        """Total number of tokens across all documents."""
        ...

    @property
    def doc_lengths(self) -> list[int]:
        """Tokens per document in the pruned vocabulary, parallel to a model's
        ``doc_topic`` rows. The N_d that ``dirichlet_theta_samples`` needs."""
        ...

    @property
    def vocabulary(self) -> list[str]:
        """Ordered list of vocabulary terms."""
        ...
    def documents(self) -> list[list[str]]:
        """The corpus as token lists (one per document), the inverse of
        from_documents."""
        ...

    @property
    def kept_indices(self) -> list[int]:
        """Original document indices that survived pruning, parallel to the rows
        of this corpus. Use to realign an external covariate array/DataFrame:
        ``X = X[corpus.kept_indices]`` (see :func:`topica.align`)."""
        ...

    metadata: object | None
    """Optional per-document metadata aligned to the surviving rows (a pandas
    DataFrame, set by :func:`topica.from_dataframe`, or assigned directly)."""

    @property
    def doc_names(self) -> list[str]:
        """Document identifiers, one per document."""
        ...

    @property
    def doc_labels(self) -> list[str]:
        """Document labels, one per document."""
        ...

    def __repr__(self) -> str: ...


class DMR:
    """Dirichlet-Multinomial Regression topic model (Mimno & McCallum 2008).

    Like LDA, but the per-document topic prior is log-linear in document
    features: alpha_{d,t} = exp(lambda_t . x_d). After fitting, the learned
    weights are in `feature_effects`.
    """

    def __init__(
        self,
        num_topics: int,
        *,
        beta: float = 0.01,
        optimize_interval: int = 50,
        burn_in: int = 200,
        seed: int = 42,
        prior_variance: float = 1.0,
        lbfgs_iters: int = 20,
    ) -> None:
        """Create an unfitted DMR model. prior_variance is the Gaussian prior
        variance on the feature weights; lbfgs_iters caps L-BFGS steps per round."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        features: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        *,
        feature_names: list[str] | None = None,
        iters: int = 1000,
        num_samples: int = 5,
        sample_interval: int = 25,
        progress: object | None = None,
        progress_interval: int = 50,
    ) -> None:
        """Fit by collapsed Gibbs with the per-document Dirichlet prior
        alpha_{d,t} = exp(lambda_t . x_d). `features` is required: an (num_docs, F)
        covariate matrix (no intercept column — one is prepended), with
        feature_names naming the F columns. The L-BFGS optimization of lambda runs
        every optimize_interval sweeps after burn_in; topic-word phi is averaged
        over num_samples samples taken every sample_interval sweeps."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """phi matrix of shape (num_topics, num_words)."""
        ...

    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """theta matrix of shape (num_docs, num_topics); rows sum to 1."""
        ...

    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The baseline document-topic Dirichlet prior alpha, shape (num_topics,):
        exp(lambda_intercept), the per-topic prior at covariates = 0."""
        ...

    @property
    def feature_effects(self) -> numpy.typing.NDArray[numpy.float64]:
        """Learned feature weights lambda, shape (num_topics, num_features). Column
        0 is the intercept; positive entries raise that topic's prevalence as the
        feature increases."""
        ...

    @property
    def feature_names(self) -> list[str]:
        """Feature names aligned with feature_effects columns ('intercept' first)."""
        ...

    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def num_topics(self) -> int: ...

    @overload
    def top_words(self, n: int = ..., *, topic: int) -> list[tuple[str, float]]: ...
    @overload
    def top_words(self, n: int = ..., *, topic: None = ...) -> list[list[tuple[str, float]]]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        """Top n (word, probability) pairs for one or all topics."""
        ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]:
        """UMass topic coherence per topic, shape (num_topics,)."""
        ...

    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        features: numpy.typing.NDArray[numpy.float64] | None = None,
        *,
        iterations: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic theta for new documents by collapsed Gibbs
        against the fitted topic-word matrix. `features` (optional, no intercept)
        sets each document's Dirichlet prior alpha_d = exp(Xgamma); if omitted
        the intercept-only baseline is used. Shape (num_new_docs, num_topics)."""
        ...

    def __repr__(self) -> str: ...


class CTM:
    """Correlated Topic Model (Blei & Lafferty; STM's logistic-normal core).
    Topics drawn from a logistic-normal prior with full covariance, so they can
    correlate (unlike LDA's Dirichlet). Fit by variational EM (STM's Laplace
    E-step)."""

    def __init__(
        self,
        num_topics: int,
        *,
        sigma_shrink: float = 0.0,
        seed: int = 42,
        init: str = "spectral",
    ) -> None:
        """num_topics >= 2. sigma_shrink in [0,1] shrinks topic covariance toward
        diagonal. init is "spectral" (default; deterministic anchor-word init,
        matching STM's default — seed is then irrelevant) or "random" (seeded)."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 500,
        em_tol: float = 1e-5,
    ) -> None:
        """EM stops once the relative change in the variational bound falls below
        em_tol or after iters iterations, whichever comes first. Pass em_tol=0
        to always run iters steps. Check converged and bound afterward."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def bound(self) -> float:
        """Final variational bound (approximate ELBO) at convergence."""
        ...
    @property
    def bound_history(self) -> list[float]:
        """Variational bound after each EM iteration (length = iterations run)."""
        ...
    @property
    def converged(self) -> bool:
        """True if EM met em_tol; False if it hit the iters cap."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def topic_correlation(self) -> numpy.typing.NDArray[numpy.float64]:
        """Topic-correlation matrix (num_topics, num_topics) from theta across docs."""
        ...
    @property
    def eta_mean(self) -> numpy.typing.NDArray[numpy.float64]:
        """Variational posterior means lambda, shape (num_docs, num_topics-1)."""
        ...
    @property
    def eta_cov(self) -> numpy.typing.NDArray[numpy.float64]:
        """Variational posterior covariances nu, shape (num_docs, K-1, K-1)."""
        ...
    @property
    def topic_covariance(self) -> numpy.typing.NDArray[numpy.float64]:
        """The fitted logistic-normal prior covariance Sigma over eta, shape
        (K-1, K-1); the last topic is the softmax reference. The model's own topic
        covariance (cf. topic_correlation, an across-document theta correlation)."""
        ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def num_topics(self) -> int: ...

    @overload
    def top_words(self, n: int = ..., *, topic: int) -> list[tuple[str, float]]: ...
    @overload
    def top_words(self, n: int = ..., *, topic: None = ...) -> list[list[tuple[str, float]]]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic theta for new documents by the variational
        E-step against the fitted globals. Shape (num_new_docs, num_topics)."""
        ...
    def __repr__(self) -> str: ...


class STM:
    """Structural Topic Model (Roberts, Stewart & Tingley): the correlated-topic
    core (CTM) plus prevalence covariates — the prior topic mean is a regression
    on document covariates (mu_d = X_d gamma)."""

    def __init__(
        self,
        num_topics: int,
        *,
        sigma_shrink: float = 0.0,
        seed: int = 42,
        init: str = "spectral",
    ) -> None:
        """init is "spectral" (default; deterministic anchor-word init matching
        STM's default) or "random" (seeded). With a content model the per-group
        beta is always random."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        prevalence: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        *,
        prevalence_names: list[str] | None = None,
        content: Sequence[str] | Sequence[int] | None = None,
        content_names: list[str] | None = None,
        iters: int = 500,
        em_tol: float = 1e-5,
    ) -> None:
        """Fit. prevalence is (num_docs, F) covariates driving topic proportions
        (mu_d = X_d gamma; intercept prepended). content is one group label per
        document, making topic-word distributions vary by group (SAGE). At least
        one of prevalence/content must be given.

        EM stops once the relative change in the variational bound falls below
        em_tol (R stm's emtol) or after iters iterations, whichever comes
        first. Pass em_tol=0 to always run iters steps. Check converged and
        bound afterward."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def bound(self) -> float:
        """Final variational bound (approximate ELBO) at convergence — R stm's
        convergence$bound."""
        ...
    @property
    def bound_history(self) -> list[float]:
        """Variational bound after each EM iteration (length = iterations run)."""
        ...
    @property
    def converged(self) -> bool:
        """True if EM met em_tol; False if it hit the iters cap."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def topic_correlation(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def eta_mean(self) -> numpy.typing.NDArray[numpy.float64]:
        """Variational posterior means lambda, shape (num_docs, num_topics-1).
        With eta_cov, the logistic-normal posterior for method-of-composition."""
        ...
    @property
    def eta_cov(self) -> numpy.typing.NDArray[numpy.float64]:
        """Variational posterior covariances nu, shape (num_docs, K-1, K-1)."""
        ...
    @property
    def topic_covariance(self) -> numpy.typing.NDArray[numpy.float64]:
        """The fitted logistic-normal prior covariance Sigma over eta, shape
        (K-1, K-1); the last topic is the softmax reference. The model's own topic
        covariance (cf. topic_correlation, an across-document theta correlation)."""
        ...
    @property
    def prevalence_effects(self) -> numpy.typing.NDArray[numpy.float64]:
        """gamma, shape (num_features, num_topics-1). RuntimeError if no
        prevalence. Prefer topica.stm.estimate_effect for inference."""
        ...
    @property
    def feature_names(self) -> list[str]: ...
    @property
    def topic_word_by_group(self) -> numpy.typing.NDArray[numpy.float64]:
        """Per-group topic-word, shape (num_topics, num_groups, num_words).
        RuntimeError if fit without content covariates."""
        ...
    @property
    def groups(self) -> list[str]:
        """Content group names (axis-1 of topic_word_by_group). RuntimeError if
        fit without content covariates."""
        ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def num_topics(self) -> int: ...

    @overload
    def top_words(self, n: int = ..., *, topic: int) -> list[tuple[str, float]]: ...
    @overload
    def top_words(self, n: int = ..., *, topic: None = ...) -> list[list[tuple[str, float]]]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...

    def word_contrast(
        self, topic: int, group_a: str | int, group_b: str | int, n: int = 10
    ) -> list[tuple[str, float]]:
        """Words most distinguishing how `topic` is worded in group_a vs group_b
        (log word-prob ratio; positive favours group_a). Requires content."""
        ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic theta for new documents by the variational
        E-step against the fitted globals (covariate-free baseline prior).
        Shape (num_new_docs, num_topics)."""
        ...
    def __repr__(self) -> str: ...


class HDP:
    """Hierarchical Dirichlet Process topic model (Teh, Jordan, Beal & Blei
    2006): the nonparametric LDA that *infers* the number of topics rather than
    fixing K. Fit by the direct-assignment Gibbs sampler (Chinese Restaurant
    Franchise). The inferred topic count is read from `num_topics` after fit."""

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        gamma: float = 1.0,
        eta: float = 0.01,
        seed: int = 42,
        resample_conc: bool = True,
    ) -> None:
        """alpha/gamma are the document- and corpus-level DP concentrations
        (initial values; resampled from the data when resample_conc=True, the
        default). eta is the topic-word Dirichlet (base measure). alpha, gamma,
        eta must be > 0."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 150,
        report_interval: int = 0,
    ) -> None:
        """Fit by `iters` Gibbs sweeps. The inferred K is then `num_topics`.

        report_interval controls the discovery/convergence trace
        (topic_count_history / log_likelihood_history / concentration_history):
        0 (default) records ~50 evenly spaced points; a positive value records
        every that-many sweeps."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """Topic-word matrix, shape (num_topics, num_words); rows sum to 1."""
        ...
    @property
    def topic_count_history(self) -> list[tuple[int, int]]:
        """Topic-discovery trajectory: (iteration, num_topics) pairs over the
        fit. Watching K stabilize is HDP's headline convergence check."""
        ...
    @property
    def log_likelihood_history(self) -> list[tuple[int, float]]:
        """Convergence trace: (iteration, per-token log-likelihood) pairs."""
        ...
    @property
    def concentration_history(self) -> list[tuple[int, float, float]]:
        """Learned-concentration trace: (iteration, alpha, gamma) triples
        (informative when resample_conc=True)."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """Document-topic matrix, shape (num_docs, num_topics); rows sum to 1."""
        ...
    @property
    def num_topics(self) -> int:
        """The inferred number of topics K (RuntimeError before fit)."""
        ...
    @property
    def alpha(self) -> float:
        """The fitted document-level concentration alpha0."""
        ...
    @property
    def gamma(self) -> float:
        """The fitted corpus-level concentration gamma."""
        ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...

    @overload
    def top_words(self, n: int = ..., *, topic: int) -> list[tuple[str, float]]: ...
    @overload
    def top_words(self, n: int = ..., *, topic: None = ...) -> list[list[tuple[str, float]]]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iterations: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer theta over the discovered topics for new documents by collapsed
        Gibbs against the fixed topic-word matrix. Shape (num_new_docs,
        num_topics)."""
        ...
    def __repr__(self) -> str: ...


class DTM:
    """Dynamic Topic Model (Blei & Lafferty 2006): topics whose word
    distributions evolve across time slices via a Gaussian state-space model.
    Fit variationally with Kalman smoothing (a port of Blei's C dtm /
    gensim's LdaSeqModel). Query a topic's distribution at a slice with
    topic_word(time) and a word's trajectory with word_evolution(topic, word)."""

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float = 0.01,
        chain_variance: float = 0.005,
        obs_variance: float = 0.5,
        seed: int = 42,
    ) -> None:
        """num_topics >= 2. chain_variance controls how much a topic may drift
        between adjacent slices (larger = freer). alpha, chain_variance,
        obs_variance must be > 0."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        times: Sequence[int],
        *,
        iters: int = 20,
    ) -> None:
        """Fit by variational EM. `times` is each document's integer time-slice
        index (0-based, contiguous); the slice count is max(times)+1."""
        ...

    def topic_word(self, time: int) -> numpy.typing.NDArray[numpy.float64]:
        """Topic-word matrix at `time`, shape (num_topics, num_words); rows sum to 1."""
        ...

    def word_evolution(
        self, topic: int, word: str | int
    ) -> numpy.typing.NDArray[numpy.float64]:
        """A word's probability in `topic` across slices, shape (num_times,)."""
        ...

    def top_words(self, topic: int, time: int, n: int = 10) -> list[tuple[str, float]]:
        """Top n words for `topic` at slice `time` as (word, probability) pairs."""
        ...

    def word_drift(
        self, topic: int, *, n: int = 10, from_time: int = 0, to_time: int | None = None
    ) -> dict[str, list[tuple[str, float]]]:
        """Words inside `topic` whose probability changed most between two slices
        (default first and last). Returns {"rising": [(word, delta)], "falling":
        [(word, delta)]} — what makes the topic's vocabulary evolve."""
        ...

    @property
    def num_topics(self) -> int: ...
    @property
    def num_times(self) -> int: ...
    @property
    def bound(self) -> float:
        """The final variational bound (ELBO) reached during fitting."""
        ...
    @property
    def vocabulary(self) -> list[str]: ...

    def __repr__(self) -> str: ...


class SupervisedLDA:
    """Supervised LDA (Blei & McAuliffe 2007): LDA where each document has a
    real-valued response y_d ~ N(eta^T zbar_d, sigma^2) regressed on its topic
    usage. Topics are shaped to predict the response; `coefficients` (eta) report
    how each topic moves y. Fit by variational EM; `predict` scores new docs."""

    def __init__(self, num_topics: int, *, alpha: float = 0.1, seed: int = 42) -> None:
        """num_topics >= 2. alpha is the Dirichlet concentration on doc-topic
        proportions; both must be > 0."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        y: Sequence[float],
        *,
        iters: int = 25,
        var_iters: int = 15,
    ) -> None:
        """Fit by variational EM. `y` is the per-document response (length =
        number of documents)."""
        ...

    def predict(
        self, data: Corpus | Sequence[Sequence[str]], *, var_iters: int = 20
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Predict y-hat for new documents. Out-of-vocabulary words are ignored.
        Returns a 1-D array of length = number of documents."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The symmetric document-topic Dirichlet prior alpha, shape (num_topics,)."""
        ...
    @property
    def coefficients(self) -> numpy.typing.NDArray[numpy.float64]:
        """Regression coefficients eta, shape (num_topics,) — how each topic
        moves the response per unit of topic frequency."""
        ...
    @property
    def sigma2(self) -> float:
        """The fitted response variance sigma^2."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...

    @overload
    def top_words(self, n: int = ..., *, topic: int) -> list[tuple[str, float]]: ...
    @overload
    def top_words(self, n: int = ..., *, topic: None = ...) -> list[list[tuple[str, float]]]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iterations: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic theta for new documents by collapsed Gibbs
        against the fitted topic-word matrix (the response is not used). Shape
        (num_new_docs, num_topics). Predict the response with transform @ eta."""
        ...
    def __repr__(self) -> str: ...


class SAGE:
    """Content-covariate topic model (SAGE / the STM content model). Topics are
    shared but each topic's word distribution varies by a document-level group
    covariate, so you can read how a topic is worded differently across groups."""

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float = 0.1,
        prior_variance: float = 1.0,
        optimize_interval: int = 50,
        burn_in: int = 100,
        seed: int = 42,
        lbfgs_iters: int = 20,
    ) -> None: ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        groups: Sequence[str] | Sequence[int],
        *,
        group_names: list[str] | None = None,
        iters: int = 1000,
        num_samples: int = 5,
        sample_interval: int = 25,
        progress: Optional[object] = None,
        progress_interval: int = 50,
    ) -> None:
        """Fit. groups is one group label per document (strings or ints);
        group_names fixes group order (default: sorted union)."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """Per-group topic-word, shape (num_topics, num_groups, num_words)."""
        ...

    @property
    def topic_word_marginal(self) -> numpy.typing.NDArray[numpy.float64]:
        """Group-averaged topic-word, shape (num_topics, num_words)."""
        ...

    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """theta, shape (num_docs, num_topics); rows sum to 1."""
        ...

    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The symmetric document-topic Dirichlet prior alpha, shape (num_topics,).
        SAGE's sparse additive parameterization is on the word side; the document
        side is an ordinary Dirichlet."""
        ...

    @property
    def groups(self) -> list[str]:
        """Group names, in the index order of topic_word's second axis."""
        ...

    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def num_topics(self) -> int: ...
    @property
    def num_groups(self) -> int: ...

    def top_words(
        self, topic: int, *, group: str | int | None = None, n: int = 10
    ) -> list[tuple[str, float]]:
        """Top n (word, prob) for a topic; for a given group (name/index) or the
        group-averaged distribution when group is None."""
        ...

    def word_contrast(
        self, topic: int, group_a: str | int, group_b: str | int, n: int = 10
    ) -> list[tuple[str, float]]:
        """Words most distinguishing how `topic` is worded in group_a vs group_b,
        by log word-probability ratio (positive favours group_a)."""
        ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def __repr__(self) -> str: ...


class LabeledLDA:
    """Labeled LDA (Ramage et al. 2009): supervised topics constrained to each
    document's label set. The number of topics equals the number of labels."""

    def __init__(self, *, alpha: float = 0.1, beta: float = 0.01, seed: int = 42) -> None:
        """Create an unfitted model. alpha is the symmetric per-topic prior."""
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        labels: Sequence[Sequence[str]],
        *,
        label_names: list[str] | None = None,
        iters: int = 1000,
        num_samples: int = 5,
        sample_interval: int = 25,
        progress: Optional[object] = None,
        progress_interval: int = 50,
    ) -> None:
        """Fit the model. labels is one label-list per document; the topic set is
        the union of all labels (or label_names, which fixes topic order). An
        empty label list leaves that document unconstrained (all topics)."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """phi matrix of shape (num_topics, num_words)."""
        ...

    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """theta matrix (num_docs, num_topics); only a document's label topics
        are non-zero, rows sum to 1."""
        ...

    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The symmetric document-topic Dirichlet prior alpha, shape (num_topics,)."""
        ...

    @property
    def labels(self) -> list[str]:
        """Label name for each topic, in topic (column) order."""
        ...

    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    @property
    def num_topics(self) -> int: ...

    @overload
    def top_words(self, n: int = ..., *, topic: int) -> list[tuple[str, float]]: ...
    @overload
    def top_words(self, n: int = ..., *, topic: None = ...) -> list[list[tuple[str, float]]]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        """Top n (word, probability) pairs for one or all topics."""
        ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]:
        """UMass topic coherence per topic, shape (num_topics,)."""
        ...

    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iterations: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer label (topic) proportions theta for new documents by collapsed
        Gibbs against the fitted topic-word matrix, treating every label as
        available. Shape (num_new_docs, num_topics); columns align with labels."""
        ...

    def __repr__(self) -> str: ...


class LDA:
    """Sparse LDA topic model (MALLET's algorithm) implemented in Rust."""

    def __init__(
        self,
        num_topics: int,
        *,
        alpha_sum: float | None = None,
        beta: float = 0.01,
        optimize_interval: int = 50,
        burn_in: int = 200,
        seed: int = 42,
        num_threads: int = 1,
        sampler: str = "sparse",
        mh_steps: int = 2,
        use_symmetric_alpha: bool = False,
    ) -> None:
        """Create an LDA model. alpha_sum defaults to num_topics if None.

        use_symmetric_alpha mirrors MALLET's --use-symmetric-alpha: when True,
        hyperparameter optimization learns only the alpha concentration and
        keeps every per-topic alpha equal, instead of learning an asymmetric
        per-topic prior (the default, MALLET's Wallach optimization).

        num_threads > 1 enables MALLET-style approximate parallel Gibbs
        sampling in fit() (faster on multicore; results differ from the exact
        single-threaded path but remain deterministic for a fixed
        num_threads + seed). num_threads=1 is the exact, CLI-identical path.

        sampler selects the inference backend: "sparse" (default) is MALLET's
        SparseLDA collapsed Gibbs sampler; "lightlda" is the alias-table
        Metropolis-Hastings sampler of Yuan et al. (2015), an O(1)-per-token
        cycle-proposal sampler for the same model. The alias sampler is built
        for the very-large-K / long-document regime; "sparse" is faster at the
        topic counts typical of social-science work. mh_steps is the number of
        MH proposals per token (alias sampler only).
        """
        ...

    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 1000,
        num_samples: int = 5,
        sample_interval: int = 25,
        progress: Optional[object] = None,
        progress_interval: int = 50,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
    ) -> None:
        """Run Gibbs sampling to fit the model on data.

        With ``keep_theta_draws`` (default on), the last ``num_theta_draws``
        thinned MCMC theta snapshots are retained as :attr:`theta_draws` for
        ``composition_theta`` standard errors. Set ``keep_theta_draws=False`` to
        save memory (``num_theta_draws x num_docs x num_topics`` f32)."""
        ...

    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """phi matrix of shape (num_topics, num_words)."""
        ...

    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """theta matrix of shape (num_docs, num_topics); rows sum to 1."""
        ...

    @property
    def theta_draws(self) -> Optional[numpy.typing.NDArray[numpy.float32]]:
        """Thinned MCMC theta draws, shape (num_draws, num_docs, num_topics), or
        None when fit with keep_theta_draws=False. Real cross-sweep posterior
        samples that composition_theta prefers over the Dirichlet approximation."""
        ...

    @property
    def doc_lengths(self) -> list[int]:
        """Per-document token counts (length num_docs), in doc_topic row order.
        Lets composition_theta recover N_d without re-threading the Corpus."""
        ...

    @property
    def vocabulary(self) -> list[str]:
        """Vocabulary list; column order matches topic_word."""
        ...

    @property
    def doc_names(self) -> list[str]:
        """Document names; row order matches doc_topic."""
        ...

    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """Per-topic alpha (Dirichlet prior), shape (num_topics,)."""
        ...

    @property
    def beta(self) -> float:
        """Scalar beta hyperparameter."""
        ...

    @property
    def num_topics(self) -> int:
        """Number of topics (available before fit)."""
        ...

    @overload
    def top_words(
        self, n: int = ..., *, topic: int
    ) -> list[tuple[str, float]]: ...

    @overload
    def top_words(
        self, n: int = ..., *, topic: None = ...
    ) -> list[list[tuple[str, float]]]: ...

    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]:
        """Return top n (word, probability) pairs for one or all topics."""
        ...

    def log_likelihood(self) -> float:
        """MALLET-formula model log-likelihood of the final sampler state (in-sample)."""
        ...

    def evaluate(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        num_particles: int = 10,
        seed: int | None = None,
    ) -> dict[str, Union[float, int]]:
        """Held-out evaluation via the Wallach (2009) left-to-right estimator.

        Returns a dict with `log_likelihood`, `perplexity`, `num_tokens`, `num_oov`.
        Out-of-vocabulary tokens (not seen in training) are dropped and counted.
        """
        ...

    def perplexity(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        num_particles: int = 10,
        seed: int | None = None,
    ) -> float:
        """Held-out perplexity (lower is better); convenience wrapper over evaluate()."""
        ...

    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]:
        """UMass topic coherence per topic, shape (num_topics,). Higher (nearer 0) is better."""
        ...

    def diagnostics(self, n: int = 10) -> list[dict[str, Any]]:
        """Per-topic diagnostics (MALLET-style), one dict per topic.

        Keys: topic, tokens, coherence, exclusivity, effective_words,
        document_entropy, uniform_dist, corpus_dist, rank1_docs, alpha,
        top_words. Suitable for pandas.DataFrame(...).
        """
        ...

    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iterations: int = 100,
        burn_in: int = 10,
        num_samples: int = 10,
        sample_interval: int = 5,
        seed: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]:
        """Infer document-topic distributions for new, unseen documents under
        the fitted model. Returns shape (num_new_docs, num_topics); rows sum to 1."""
        ...

    def top_documents(self, topic: int, n: int = 10) -> list[tuple[str, float]]:
        """The n training documents most associated with `topic`, as
        (doc_name, weight) pairs sorted by descending theta."""
        ...

    @property
    def topic_divergence(self) -> numpy.typing.NDArray[numpy.float64]:
        """Pairwise Jensen-Shannon divergence between topic-word distributions,
        shape (num_topics, num_topics), base 2 in [0, 1]; 0 on the diagonal."""
        ...

    def similar_documents(self, doc: int, n: int = 10) -> list[tuple[str, float]]:
        """The n training documents most similar to document `doc` (by index),
        as (doc_name, divergence) pairs sorted by ascending JS divergence."""
        ...

    def save_topic_word(self, path: str) -> None:
        """Write topic-word matrix to a TSV file (topic, word, probability)."""
        ...

    def save_doc_topic(self, path: str) -> None:
        """Write doc-topic matrix to a TSV file (doc[, label], topic_0, ...)."""
        ...

    def save_state(self, path: str) -> None:
        """Write the token-level Gibbs state to a gzipped file in MALLET's
        --output-state format: a header, #alpha/#beta lines, then one row per
        token (doc source pos typeindex type topic) giving the final topic
        assignment of every token in the training corpus. Use it to feed custom
        visualizations (e.g. pyLDAvis) or corpus metrics."""
        ...

    @staticmethod
    def load_state(path: str) -> "LDA":
        """Reconstruct a fitted LDA from a MALLET-format Gibbs state file (the
        inverse of save_state; MALLET --input-state). The file may be gzipped or
        plain text. Vocabulary, documents, per-token topic assignments, and the
        #alpha/#beta hyperparameters are restored, so the model supports the
        read-only surface (topic_word, doc_topic, top_words, ...) and transform
        on new documents."""
        ...

    def save(self, path: str) -> None:
        """Persist the fitted model (topic-word state, hyperparameters, and the
        training corpus) to `path`. Reload with `LDA.load` to run `transform`
        inference later without retraining (MALLET --output-model)."""
        ...

    @staticmethod
    def load(path: str) -> "LDA":
        """Load a model written by `save`, ready for `transform` inference on new
        documents (MALLET --input-model / --inferencer-filename)."""
        ...

    def __repr__(self) -> str: ...


class PT:
    """Pseudo-document Topic model (Zuo et al. 2016) for short texts: aggregates
    documents into `num_pseudo` pseudo-documents so LDA-style mixed membership is
    estimable on short, sparse texts. Fit by collapsed Gibbs."""

    def __init__(
        self,
        num_topics: int,
        *,
        num_pseudo: int = 100,
        alpha: float = 0.1,
        beta: float = 0.01,
        seed: int = 42,
    ) -> None: ...
    def fit(self, data: Corpus | Sequence[Sequence[str]], *, iters: int = 1000) -> None: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The symmetric document-topic Dirichlet prior alpha, shape (num_topics,)."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "PT": ...
    def __repr__(self) -> str: ...


class GSDMM:
    """Gibbs Sampling Dirichlet Multinomial Mixture, the "Movie Group Process"
    (Yin & Wang 2014): a one-topic-per-document mixture for short texts. You set
    an upper bound K (`num_topics`); empty clusters die out, so the effective
    number of topics is read from `num_topics` after fit."""

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float = 0.1,
        beta: float = 0.1,
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 30,
        report_interval: int = 0,
    ) -> None:
        """Fit by the Movie Group Process. report_interval controls the
        cluster-discovery trace (0 = auto ~50 points)."""
        ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def cluster_count_history(self) -> list[tuple[int, int]]:
        """Cluster-discovery trajectory: (iteration, num_clusters) pairs.
        Watching the count collapse to a stable value is GSDMM's headline
        convergence check."""
        ...
    @property
    def log_likelihood_history(self) -> list[tuple[int, float]]:
        """Convergence trace: (iteration, per-token log-likelihood) pairs."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_cluster(self) -> numpy.typing.NDArray[numpy.int64]:
        """Hard cluster assignment per document, shape (num_docs,)."""
        ...
    @property
    def num_topics(self) -> int:
        """The number of non-empty clusters after fitting (the effective K)."""
        ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "GSDMM": ...
    def __repr__(self) -> str: ...


class PA:
    """Pachinko Allocation Model (Li & McCallum 2006): a DAG of `num_super`
    super-topics over `num_sub` shared sub-topics over words, capturing topic
    correlations. `super_sub` reports which sub-topics each super-topic groups."""

    def __init__(
        self,
        num_super: int,
        num_sub: int,
        *,
        alpha: float = 0.1,
        beta: float = 0.01,
        seed: int = 42,
    ) -> None: ...
    def fit(self, data: Corpus | Sequence[Sequence[str]], *, iters: int = 1000) -> None: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """Sub-topic word matrix, shape (num_sub, num_words); rows sum to 1."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]:
        """Document sub-topic matrix, shape (num_docs, num_sub)."""
        ...
    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The symmetric sub-topic Dirichlet prior alpha, shape (num_sub,)."""
        ...
    @property
    def super_sub(self) -> numpy.typing.NDArray[numpy.float64]:
        """Super-topic to sub-topic association, shape (num_super, num_sub)."""
        ...
    @property
    def num_super(self) -> int: ...
    @property
    def num_sub(self) -> int: ...
    @property
    def vocabulary(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "PA": ...
    def __repr__(self) -> str: ...


class HLDA:
    """Hierarchical LDA (Blei et al.): topics arranged in a `depth`-level tree via
    the nested Chinese Restaurant Process. Each document follows a root-to-leaf
    path; general words sit near the root, specific words near the leaves."""

    def __init__(
        self,
        *,
        depth: int = 3,
        gamma: float = 1.0,
        eta: float = 0.01,
        alpha: float = 0.1,
        seed: int = 42,
    ) -> None: ...
    def fit(self, data: Corpus | Sequence[Sequence[str]], *, iters: int = 1000) -> None: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]:
        """Node-word matrix, shape (num_nodes, num_words); rows sum to 1."""
        ...
    @property
    def num_nodes(self) -> int: ...
    @property
    def node_levels(self) -> list[int]:
        """Tree level of each node (0 = root)."""
        ...
    @property
    def node_parents(self) -> list[int]:
        """Parent node index of each node (-1 for the root)."""
        ...
    @property
    def doc_paths(self) -> list[list[int]]:
        """Each document's root-to-leaf path as a list of node indices."""
        ...
    @property
    def leaves(self) -> list[int]: ...
    @property
    def vocabulary(self) -> list[str]: ...
    def top_words(self, node: int, n: int = 10) -> list[tuple[str, float]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "HLDA": ...
    def __repr__(self) -> str: ...


class SeededLDA:
    """Seeded (guided) LDA: supply a few seed words per topic and the model is
    steered so those topics form around them, while the rest of each topic's
    vocabulary and any `residual` unseeded topics are still learned. Seeding
    follows the seededlda package (seed words get a `weight * 100` prior
    pseudocount in their topic, plus seeded initialization)."""

    def __init__(
        self,
        seed_words: dict[str, Sequence[str]],
        *,
        residual: int = 0,
        alpha: float = 0.1,
        beta: float = 0.01,
        weight: float = 0.01,
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 2000,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
    ) -> None: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def theta_draws(self) -> Optional[numpy.typing.NDArray[numpy.float32]]:
        """Thinned MCMC theta draws, shape (num_draws, num_docs, num_topics), or
        None when fit with keep_theta_draws=False."""
        ...
    @property
    def doc_lengths(self) -> list[int]:
        """Per-document token counts (length num_docs), in doc_topic row order."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The symmetric document-topic Dirichlet prior alpha, shape (num_topics,)."""
        ...
    @property
    def topic_names(self) -> list[str]:
        """The seed names you gave, then 'residual_1' ... for unseeded topics."""
        ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "SeededLDA": ...
    def __repr__(self) -> str: ...


class Top2Vec:
    """Top2Vec (Angelov 2020): topics by clustering document embeddings. The
    embeddings are reduced (randomized PCA), density-clustered (HDBSCAN), and
    each topic is read off its cluster: the topic vector is the mean of its
    documents' embeddings and its words are the nearest vocabulary terms. You
    bring the embeddings; the topic count is discovered, not set. No embedder of
    your own? ``topica.llm_embed(texts, model=...)`` builds the matrix."""

    def __init__(
        self,
        *,
        n_components: int = 5,
        min_cluster_size: int = 15,
        min_samples: int | None = None,
        reducer: str = "pca",
        n_neighbors: int = 15,
        clusterer: str = "hdbscan",
        num_clusters: int | None = None,
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        *,
        word_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        vocabulary: Sequence[str] | None = None,
    ) -> None:
        """Fit on token documents plus one `doc_embeddings` row per document.
        Pass `word_embeddings` with the aligned `vocabulary` (same space) to
        enable `topic_neighbors`; they are realigned to topica's vocabulary."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def topic_vectors(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def labels(self) -> list[int]: ...
    @property
    def topic_names(self) -> list[str]: ...
    @property
    def vocabulary(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None, representation: str | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def topic_neighbors(self, topic: int, *, n: int = 10) -> list[tuple[str, float]]: ...
    def transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def fit_transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        *,
        word_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        vocabulary: Sequence[str] | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def merge_topics(self, groups: Sequence[Sequence[int]]) -> None: ...
    def reduce_outliers(self) -> int: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "Top2Vec": ...
    def __repr__(self) -> str: ...


class BERTopic:
    """BERTopic (Grootendorst 2022): the same reduce/cluster pipeline as Top2Vec,
    but topics are defined by class-based TF-IDF over their documents' words, so
    no word embeddings are needed. `nr_topics` merges the most similar topics down
    to a target; `doc_topic` is the approximate distribution. You bring the
    document embeddings; the topic count is discovered (before any reduction).
    No embedder of your own? ``topica.llm_embed(texts, model=...)`` builds it."""

    def __init__(
        self,
        *,
        n_components: int = 5,
        min_cluster_size: int = 15,
        min_samples: int | None = None,
        nr_topics: int | None = None,
        window: int = 4,
        stride: int = 1,
        reducer: str = "pca",
        n_neighbors: int = 15,
        bm25: bool = False,
        reduce_frequent: bool = False,
        clusterer: str = "hdbscan",
        num_clusters: int | None = None,
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
    ) -> None:
        """Fit on token documents plus one `doc_embeddings` row per document."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def labels(self) -> list[int]: ...
    @property
    def topic_names(self) -> list[str]: ...
    @property
    def vocabulary(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def approximate_distribution(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        window: int | None = None,
        stride: int | None = None,
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def fit_transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def merge_topics(self, groups: Sequence[Sequence[int]]) -> None: ...
    def reduce_outliers(self) -> int: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "BERTopic": ...
    def __repr__(self) -> str: ...


class ETM:
    """Embedded Topic Model (Dieng, Ruiz & Blei 2020): LDA with the topic-word
    matrix factored through embeddings, beta_{k,v} = softmax_v(rho_v . alpha_k),
    and a logistic-normal document prior. You bring the word embeddings rho;
    topica fits the topic embeddings alpha. `inference="em"` (default) uses
    per-document variational EM; `inference="vae"` uses the reference's amortized
    autoencoder, which scales to large corpora and maps new documents with a single
    encoder pass. Neither uses PyTorch. No embedder of your own?
    ``topica.llm_embed(vocabulary, model=...)`` builds the word embeddings rho."""

    def __init__(
        self,
        num_topics: int,
        *,
        inference: str = "em",
        em_iters: int = 100,
        em_tol: float = 1e-4,
        sigma_shrink: float = 0.0,
        prior_variance: float = 1e6,
        max_inner: int = 25,
        hidden_size: int = 800,
        epochs: int = 150,
        batch_size: int = 1000,
        lr: float = 0.005,
        wdecay: float = 1.2e-6,
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        word_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        vocabulary: Sequence[str],
    ) -> None:
        """Fit on token documents plus word embeddings (len(vocabulary) x E) and
        the aligned vocabulary, which defines the word ids."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def inference(self) -> str: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def topic_embeddings(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def bound(self) -> float: ...
    @property
    def converged(self) -> bool: ...
    @property
    def topic_names(self) -> list[str]: ...
    @property
    def vocabulary(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def transform(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def fit_transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        word_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
        vocabulary: Sequence[str],
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def __repr__(self) -> str: ...


class ProdLDA:
    """ProdLDA (Srivastava & Sutton 2017), the AVITM autoencoding-variational topic
    model. LDA with the word-level mixture replaced by a product of experts:
    the word distribution is softmax(beta . theta) with an unnormalized beta,
    yielding more coherent topics. Inference is an amortized VAE trained by
    minibatch Adam on the ELBO; batch normalization and high-momentum Adam guard
    against component collapse. Unlike ETM you bring no embeddings: beta is learned
    directly. New documents transform with a single encoder forward pass."""

    def __init__(
        self,
        num_topics: int,
        *,
        alpha: float = 1.0,
        hidden_size: int = 100,
        dropout: float = 0.2,
        epochs: int = 200,
        batch_size: int = 200,
        lr: float = 0.002,
        em_tol: float = 0.0,
        seed: int = 42,
    ) -> None: ...
    def fit(self, data: Corpus | Sequence[Sequence[str]]) -> None:
        """Fit on a Corpus or a list of token lists."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def bound(self) -> float: ...
    @property
    def bound_history(self) -> list[float]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def epochs_run(self) -> int: ...
    @property
    def topic_names(self) -> list[str]: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def transform(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def fit_transform(
        self, data: Corpus | Sequence[Sequence[str]]
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def __repr__(self) -> str: ...


class FASTopic:
    """FASTopic (Wu et al. 2024): a topic model with no encoder or neural network.
    The topic proportions theta and topic-word matrix beta are read off two
    entropic optimal-transport plans between embedding sets. You bring the document
    embeddings; topica learns the topic embeddings, word embeddings (same space),
    and transport marginals, minimizing a bag-of-words reconstruction plus the two
    transport costs. Held-out documents are mapped by a distance-softmax over the
    fitted topic embeddings, so `transform` needs only their embeddings. No
    embedder of your own? ``topica.llm_embed(texts, model=...)`` builds it."""

    def __init__(
        self,
        num_topics: int,
        *,
        epochs: int = 200,
        lr: float = 0.002,
        dt_alpha: float = 3.0,
        tw_alpha: float = 2.0,
        theta_temp: float = 1.0,
        em_tol: float = 1e-6,
        sinkhorn_iters: int = 50,
        sinkhorn_tol: float = 1e-4,
        seed: int = 42,
    ) -> None: ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
    ) -> None:
        """Fit on token documents plus frozen document embeddings (num_docs x E).
        The vocabulary is taken from the corpus; the word embeddings are learned."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def topic_embeddings(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def word_embeddings(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def loss_history(self) -> list[float]: ...
    @property
    def converged(self) -> bool: ...
    @property
    def topic_names(self) -> list[str]: ...
    @property
    def vocabulary(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def transform(
        self, doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]]
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def fit_transform(
        self,
        data: Corpus | Sequence[Sequence[str]],
        doc_embeddings: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]],
    ) -> numpy.typing.NDArray[numpy.float64]: ...
    def __repr__(self) -> str: ...


class KeyATM:
    """Keyword-Assisted Topic Model (keyATM Base; Eshima, Imai & Sasaki 2024).
    Some topics carry a keyword list; a token in a keyword topic comes either from
    a distribution over only that topic's keywords or from its full distribution,
    anchoring keyword topics to their keywords. `num_topics` may exceed the number
    of keyword topics to add regular, no-keyword topics."""

    def __init__(
        self,
        keywords: dict[str, Sequence[str]],
        *,
        num_topics: int | None = None,
        alpha: float | None = None,
        beta: float = 0.01,
        beta_keyword: float = 0.1,
        gamma1: float = 1.0,
        gamma2: float = 1.0,
        seed: int = 42,
        estimate_alpha: bool = True,
    ) -> None:
        """estimate_alpha (default True, matching R keyATM) slice-samples an
        asymmetric document-topic prior alpha each sweep. Set it False for a
        fixed symmetric alpha: a faster fit (it skips the dominant non-sweep
        cost) at the price of the R-matching asymmetric prior. The base model
        only; the covariate and dynamic models always learn their priors."""
        ...
    @staticmethod
    def weighted_lda(
        num_topics: int,
        *,
        alpha: float = 0.1,
        beta: float = 0.01,
        seed: int = 42,
    ) -> "KeyATM":
        """keyATM's weightedLDA: a keyword-free model (no keyword topics) — plain
        LDA fit with keyATM's token weighting and estimated asymmetric alpha. Fit
        it like a KeyATM; keyword outputs (keyword_rate, pi_history) are empty."""
        ...
    def fit(
        self,
        data: Corpus | Sequence[Sequence[str]],
        *,
        iters: int = 1500,
        covariates: numpy.typing.NDArray[numpy.float64] | Sequence[Sequence[float]] | None = None,
        feature_names: list[str] | None = None,
        timestamps: Sequence[float] | Sequence[str] | None = None,
        num_states: int = 5,
        weights: str = "information-theory",
        num_threads: int = 1,
        optimize_interval: int = 50,
        burn_in: int = 200,
        prior_variance: float = 1.0,
        lbfgs_iters: int = 20,
        report_interval: int = 0,
        keep_theta_draws: bool = True,
        num_theta_draws: int = 25,
    ) -> None:
        """Fit by collapsed Gibbs. Pass `covariates` (num_docs x F) for the
        covariate keyATM: the document-topic prior becomes a DMR,
        alpha_{d,k} = exp(x_d . lambda_k) (an intercept is prepended), and the
        learned lambda is exposed as `feature_effects`.

        Pass `timestamps` (one per document) for the dynamic keyATM: a Chib (1998)
        change-point HMM lets topic prevalence shift over `num_states` regimes.
        The smoothed path is exposed as `time_prevalence` (aligned with
        `time_labels`) and the per-segment regime as `time_state`. `timestamps`
        and `covariates` are mutually exclusive.

        `weights` is keyATM's token weighting: 'information-theory' (default,
        each token counts by its word's surprisal in bits), 'inv-freq', or
        'none' (unweighted). Weighting downweights frequent words and applies to
        every variant (base, covariate, dynamic).

        `report_interval` sets how often model_fit is recorded for
        `log_likelihood_history` (keyATM's model_fit / plot_modelfit): 0
        (default) records ~50 evenly spaced points across the run; a positive
        value records every that-many sweeps."""
        ...
    @property
    def topic_word(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def log_likelihood_history(self) -> list[tuple[int, float, float]]:
        """Convergence trace: (iteration, log_likelihood, perplexity) triples —
        the three columns of keyATM's model_fit / plot_modelfit. The
        log-likelihood is the collapsed marginal and perplexity is
        exp(-loglik / total_weighted_tokens), both on R keyATM's scale. Empty if
        tracing was disabled."""
        ...
    @property
    def alpha_history(self) -> list[tuple[int, list[float]]]:
        """Trace of the estimated document-topic prior alpha: (iteration, alpha)
        pairs (alpha length K) — keyATM's plot_alpha / values_iter$alpha_iter.
        Base model only; empty for covariate (traces lambda) and dynamic."""
        ...
    @property
    def pi_history(self) -> list[tuple[int, list[float]]]:
        """Trace of the per-topic keyword switch rate pi: (iteration, pi) pairs
        (pi length K, 0 for regular topics) — keyATM's plot_pi /
        values_iter$pi_iter. Empty for a keyword-free model."""
        ...
    @property
    def doc_topic(self) -> numpy.typing.NDArray[numpy.float64]: ...
    @property
    def theta_draws(self) -> Optional[numpy.typing.NDArray[numpy.float32]]:
        """Thinned MCMC theta draws, shape (num_draws, num_docs, num_topics), or
        None when fit with keep_theta_draws=False. Real cross-sweep posterior
        samples that composition_theta prefers over the Dirichlet approximation.
        Collected for the base, covariate, and dynamic variants."""
        ...
    @property
    def doc_lengths(self) -> list[int]:
        """Per-document token counts (length num_docs), in doc_topic row order."""
        ...
    @property
    def feature_effects(self) -> numpy.typing.NDArray[numpy.float64]:
        """Covariate model: learned lambda, shape (num_topics, F+1); column 0 is
        the intercept. Raises if fit without covariates."""
        ...
    @property
    def feature_names(self) -> list[str]:
        """Covariate model: names for feature_effects columns ('intercept' first)."""
        ...
    @property
    def keyword_rate(self) -> numpy.typing.NDArray[numpy.float64]:
        """Per-topic keyword switch rate (0 for regular topics)."""
        ...
    @property
    def alpha(self) -> numpy.typing.NDArray[numpy.float64]:
        """The document-topic Dirichlet prior alpha, shape (num_topics,). Base model:
        the estimated asymmetric prior; covariate/dynamic models fall back to the
        symmetric base value."""
        ...
    @property
    def time_prevalence(self) -> numpy.typing.NDArray[numpy.float64]:
        """Dynamic model: smoothed topic prevalence per time segment, shape
        (T, num_topics), aligned with `time_labels`. Raises if fit without
        `timestamps`."""
        ...
    @property
    def time_state(self) -> list[int]:
        """Dynamic model: latent HMM regime of each time segment (length T).
        Empty for non-dynamic models."""
        ...
    @property
    def time_labels(self) -> list[str]:
        """Dynamic model: sorted distinct timestamp labels, one per time segment.
        Empty for non-dynamic models."""
        ...
    @property
    def transition_matrix(self) -> numpy.typing.NDArray[numpy.float64]:
        """Dynamic model: left-to-right state transition matrix, shape
        (num_states, num_states). Raises if fit without `timestamps`."""
        ...
    @property
    def num_topics(self) -> int: ...
    @property
    def topic_names(self) -> list[str]: ...
    @property
    def vocabulary(self) -> list[str]: ...
    @property
    def doc_names(self) -> list[str]: ...
    def top_words(
        self, n: int = 10, *, topic: int | None = None
    ) -> list[tuple[str, float]] | list[list[tuple[str, float]]]: ...
    def coherence(self, n: int = 10) -> numpy.typing.NDArray[numpy.float64]: ...
    def save(self, path: str) -> None: ...
    @staticmethod
    def load(path: str) -> "KeyATM": ...
    def __repr__(self) -> str: ...
