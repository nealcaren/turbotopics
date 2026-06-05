"""LLM topic labeling as plumbing: prompt assembly, the BYO-callable path, and
the optional `llm` backend. No network or real model is used; the model is a
deterministic stub callable."""

import importlib.util

import numpy as np
import pytest

import topica


@pytest.fixture(scope="module")
def model_and_texts():
    pets = ["cat dog pet cat dog fur"]
    sky = ["star moon sky star moon night"]
    docs = [d.split() for d in (pets * 20 + sky * 20)]
    texts = pets * 20 + sky * 20
    m = topica.LDA(num_topics=2, seed=1)
    m.fit(docs, iterations=300)
    return m, texts


def test_prompts_carry_words_and_docs(model_and_texts):
    m, texts = model_and_texts
    prompts = topica.topic_label_prompts(m, texts, n_words=5, n_docs=2)
    assert len(prompts) == 2
    # Every topic's prompt names its top words and shows representative documents.
    joined = " ".join(prompts).lower()
    assert "top words:" in joined
    assert "representative documents:" in joined
    # The two themes' words appear somewhere across the prompts.
    assert "cat" in joined and "moon" in joined


def test_prompts_without_texts_have_no_doc_block(model_and_texts):
    m, _ = model_and_texts
    prompts = topica.topic_label_prompts(m, n_words=5)
    assert all("Representative documents" not in p for p in prompts)
    assert all("Top words:" in p for p in prompts)


def test_byo_callable_labels_each_topic(model_and_texts):
    m, texts = model_and_texts
    calls = []

    def fake(prompt):
        calls.append(prompt)
        # A deterministic "label": the first top word after the "Top words:" line.
        line = next(ln for ln in prompt.splitlines() if ln.startswith("Top words:"))
        return line.split(":", 1)[1].split(",")[0].strip()

    labels = topica.llm_topic_labels(m, texts, call=fake, n_words=4)
    assert len(labels) == 2 == len(calls)
    assert all(isinstance(x, str) and x for x in labels)


def test_set_labels_flows_into_topic_info(model_and_texts):
    m, texts = model_and_texts
    labels = topica.llm_topic_labels(
        m, texts, call=lambda p: "THEME", set_labels=True
    )
    assert labels == ["THEME", "THEME"]
    info = topica.topic_info(m)
    assert all(row["label"] == "THEME" for row in info if row["topic"] >= 0)
    assert topica.topic_labels(m) == ["THEME", "THEME"]
    # reset so the module-scoped model does not leak custom labels to other tests
    topica.set_topic_labels(m, {})


def test_instructions_override(model_and_texts):
    m, _ = model_and_texts
    prompts = topica.topic_label_prompts(m, instructions="LABEL THIS THING")
    assert all(p.startswith("LABEL THIS THING") for p in prompts)


def test_llm_backend_requires_llm():
    # When the optional `llm` package is absent, llm_backend raises a clear error;
    # when present we cannot call a real model in tests, so just check it builds.
    if importlib.util.find_spec("llm") is None:
        with pytest.raises(ImportError, match="llm"):
            topica.llm_backend("gpt-4o-mini")
    else:
        assert callable(topica.llm_backend("gpt-4o-mini"))
