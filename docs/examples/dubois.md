# W.E.B. Du Bois in *The Crisis*

A full end-to-end analysis on a real corpus: **704 articles from *The Crisis***,
the NAACP magazine Du Bois edited from 1910 to 1934. It walks the whole library:
preprocessing, phrase detection, LDA, STM with decade prevalence, dynamic topics,
HDP, and held-out inference.

!!! info "Focus of this example"
    Corpus building & cleaning · **temporal / dynamic topics** · using `HDP` as a
    `K` sanity check. For topic validation and clustered errors see
    [Poliblog](poliblog.md); for the experimental effect see [Gadarian](gadarian.md).

> :material-notebook: **[Run the full notebook](https://github.com/nealcaren/turbotopics/blob/main/examples/dubois_tutorial.ipynb)**
> ([script version](https://github.com/nealcaren/turbotopics/blob/main/examples/dubois_tutorial.py)) ·
> data: [`examples/dubois_crisis.csv`](https://github.com/nealcaren/turbotopics/blob/main/examples/dubois_crisis.csv)

## What it covers

1. **Preprocess** — tokenize with a stoplist, prune the vocabulary
   (`min_doc_freq=10`, `rm_top=20`).
2. **Phrases** — detect 118 collocations (*jim crow*, *booker washington*) before
   modeling.
3. **LDA (K=15)** — clean topics: WWI Black regiments, lynching, voting and
   disfranchisement, education, the NAACP, labor, Africa and the international
   scene. Scored with `c_v` coherence and topic diversity.
4. **STM** — topic prevalence on decade, with `estimate_effect` by the method of
   composition. The significant movers are historically exact: **labor /
   communism rises** (Du Bois's economic turn) and **women's suffrage falls**
   (the 19th Amendment passed in 1920).
5. **DTM** — word trajectories across decades. **War** recedes after the WWI
   1910s, **labor** climbs toward the 1930s, and **Africa** peaks in the
   Pan-African 1920s.
6. **HDP** — independently infers ≈ 17 topics, close to the `K=15` chosen for
   LDA.
7. **Utilities** — `summary`, save/load, pyLDAvis, and held-out `transform`.

## A taste

```python
import csv, turbotopics as tt
from turbotopics import Corpus, tokenize

rows = list(csv.DictReader(open("examples/dubois_crisis.csv")))
docs = [tokenize(r["text"], stopwords=stop, min_length=3) for r in rows]
corpus = Corpus.from_documents(docs, min_doc_freq=10, rm_top=20)

lda = tt.LDA(num_topics=15, seed=1)
lda.fit(corpus, iterations=400)
for t in range(15):
    print(f"T{t}:", ", ".join(w for w, _ in lda.top_words(8, topic=t)))
```

```
T 3: lynching, southern, state, law, murder, crime, georgia, justice
T 4: schools, education, children, public, teachers, college, training
T 6: africa, england, war, india, europe, government, peace, african
T 8: labor, industry, industrial, capital, economic, workers, wages
...
```

The notebook applies the full [publishing workflow](../publishing/index.md).
