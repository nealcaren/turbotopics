# Distinguishing words

Sometimes the question isn't "what are the topics" but "which words separate
these two groups." **Fighting Words** (Monroe, Colaresi & Quinn 2008) answers
that with statistical significance, and, unlike a raw log-odds ratio, it doesn't
let rare words dominate.

```python
import turbotopics as tt

conservative = [tokenize(t) for t in con_texts]
liberal      = [tokenize(t) for t in lib_texts]

scored = tt.fighting_words(conservative, liberal, prior=0.05)
# sorted by z-score: corpus-A markers at the top, corpus-B at the bottom
for word, z in scored[:10]:
    print(f"{word:20s} {z:+.1f}")     # |z| > 1.96 ~ significant at 95%
```

A large positive `z` marks a word distinctive of the first corpus; a large
negative `z`, the second. Because the estimator's variance grows for rare words,
the z-score already accounts for how much evidence each word carries.

## Top words per side

```python
top = tt.top_fighting_words(conservative, liberal, n=15)
print("conservative:", [w for w, _ in top["a"]])
print("liberal:     ", [w for w, _ in top["b"]])
```

## Informative prior

By default the prior is a symmetric pseudocount. Pass `informative=True` to scale
the prior by each word's overall frequency: Monroe et al.'s informative
Dirichlet prior, which pulls extreme estimates toward the corpus background:

```python
tt.fighting_words(conservative, liberal, prior=0.01, informative=True)
```

This pairs naturally with [SAGE / content STM](covariates.md), which find
group-distinguishing wording *within a fitted topic model*, where Fighting Words
works directly on two raw corpora with no model at all.
