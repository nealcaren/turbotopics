# 2. Pick the right model

Before `K`, choose the *model*. This is a substantive decision — it encodes what
you think generates the text and what you want to learn — and reviewers will ask
you to justify it. Pick the simplest model that matches your question and your
data structure.

## First: is a topic model even the right tool?

!!! note "Good fit"
    - Exploratory: *what themes exist in this corpus?*
    - Large collections where manual reading is infeasible.
    - You want to **discover** structure, not impose categories.
    - You need to track theme prevalence over time or across groups.

!!! warning "Poor fit"
    - **Confirmatory** questions with known categories → use a dictionary or a
      supervised classifier.
    - Very short documents (< ~50 words) → use a [short-text model](../guides/short-text.md),
      not standard LDA.
    - Highly technical or formulaic text, or a need for crisp category boundaries.

## Match the model to your question

| Your question / data | Model | Why |
|----------------------|-------|-----|
| "What themes are here?" (baseline) | **`LDA`** | Standard, fast, well understood |
| "Does theme prevalence depend on metadata?" | **`STM`** | Prevalence covariates with valid effect estimation — the social-science default |
| "Is the same theme worded differently by group?" | **`STM`** (content) / **`SAGE`** | Content covariates on the topic-word distribution |
| "Do themes co-occur?" | **`CTM`** / `STM` | Logistic-normal allows topic correlation |
| "How many themes are there?" | **`HDP`** | Infers `K` nonparametrically (a check on your `K`) |
| "How does theme *vocabulary* drift over time?" | **`DTM`** | Topics evolve across ordered time slices |
| "Do themes predict an outcome?" | **`SupervisedLDA`** / `DMR` | Response- or covariate-conditioned topics |
| Tweets / survey answers / headlines | **`GSDMM`**, **`PT`** | Built for short, sparse documents |

## Why STM is the social-science default

For most published social-science work the answer is the **Structural Topic
Model**. It is LDA's correlated cousin *plus* a regression layer, so a single fit
gives you topics, their correlations, and how covariates move them — with
[proper uncertainty](effects.md). The R `stm` package made it the field standard;
turbotopics gives you the same model (spectral init, prevalence and content
covariates, `estimateEffect`, `searchK`, FREX) in Python, faster, with
[clustered standard errors](effects.md) and GLM links on top.

## Justify it in the paper

State the model and *why it fits your design* in one or two sentences:

> Because our research question concerns how immigration framing differs by party
> and shifts across the campaign, we use the Structural Topic Model (STM), which
> lets topic prevalence depend on covariates (party, week) and provides valid
> uncertainty for those effects via the method of composition.

A nonparametric (`HDP`) or alternative-model robustness check — "the substantive
topics were stable when we instead used LDA / a different `K`" — pre-empts the
"why this model?" objection.

→ Next: [Choose and justify K](choosing-k.md).
