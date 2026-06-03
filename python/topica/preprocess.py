"""Preprocessing helpers for the document-as-data workflow.

Topic models assume documents are roughly comparable bags of words, but social
science corpora are full of *long* documents — full speeches, legislative
transcripts, interviews, book chapters — that violate that assumption. The usual
fix is to segment them into shorter, comparable chunks; the chore everyone
hand-writes is keeping each chunk tied to its source document's metadata. This
module does both.
"""

from __future__ import annotations

import re

_SENT = re.compile(r"[^.!?]+[.!?]+|\S.*$")


def _sentences(text):
    """Split text into sentences on ./!/? boundaries (lightweight, no deps)."""
    return [s.strip() for s in _SENT.findall(text) if s.strip()]


def split_documents(
    texts,
    metadata=None,
    *,
    max_words=200,
    min_words=50,
    sentence_aware=True,
):
    """Segment long documents into shorter chunks, **propagating metadata**.

    Each source document is split into chunks of roughly ``max_words`` words; the
    metadata row for the source document is copied onto every chunk it produces,
    with two bookkeeping keys added: ``parent`` (the source document's index) and
    ``chunk`` (the chunk's position within that document). The chunked texts and
    chunked metadata stay aligned, so you can feed the chunks to a model and still
    aggregate or condition on the original document-level covariates.

    Parameters
    ----------
    texts : sequence of ``str`` (raw text) or ``list[str]`` (pre-tokenized).
        Pre-tokenized input is chunked by token count and returned as token lists;
        raw strings are chunked and returned as strings.
    metadata : sequence aligned with ``texts``, optional.
        One entry per source document — typically a ``dict`` of covariates. A
        mapping is shallow-copied onto each chunk; a non-mapping value is stored
        under a ``"metadata"`` key. If omitted, chunk metadata carries just
        ``parent`` / ``chunk``.
    max_words : int, default 200
        Target chunk length in words/tokens.
    min_words : int, default 50
        A trailing chunk shorter than this is merged back into the previous chunk
        (so no text is dropped and no runt chunks are produced). A whole document
        shorter than ``min_words`` is still emitted as a single chunk.
    sentence_aware : bool, default True
        For raw-string input, pack whole sentences up to ``max_words`` rather than
        cutting mid-sentence (a sentence longer than ``max_words`` is hard-split).
        Ignored for pre-tokenized input.

    Returns
    -------
    (chunks, chunk_metadata) : the chunked documents (same element type as the
    input) and a list of metadata dicts, aligned and the same length.
    """
    if metadata is not None and len(metadata) != len(texts):
        raise ValueError("metadata must be the same length as texts")

    chunks = []
    chunk_meta = []
    for i, doc in enumerate(texts):
        tokenized = not isinstance(doc, str)
        if tokenized:
            pieces = _chunk_tokens(list(doc), max_words, min_words)
            emit = lambda p: list(p)
        elif sentence_aware:
            pieces = _chunk_sentences(_sentences(doc), max_words, min_words)
            emit = lambda p: " ".join(p)
        else:
            pieces = _chunk_tokens(doc.split(), max_words, min_words)
            emit = lambda p: " ".join(p)

        base = metadata[i] if metadata is not None else None
        for j, piece in enumerate(pieces):
            chunks.append(emit(piece))
            chunk_meta.append(_meta_for(base, parent=i, chunk=j))
    return chunks, chunk_meta


def _meta_for(base, *, parent, chunk):
    if hasattr(base, "keys"):           # mapping → shallow copy
        row = dict(base)
    elif base is None:
        row = {}
    else:
        row = {"metadata": base}
    row["parent"] = parent
    row["chunk"] = chunk
    return row


def _chunk_tokens(tokens, max_words, min_words):
    if not tokens:
        return [[]]
    out = [tokens[k:k + max_words] for k in range(0, len(tokens), max_words)]
    if len(out) > 1 and len(out[-1]) < min_words:
        out[-2].extend(out.pop())       # merge runt tail into previous
    return out


def _chunk_sentences(sentences, max_words, min_words):
    if not sentences:
        return [""]
    chunks, cur, cur_n = [], [], 0
    for sent in sentences:
        words = sent.split()
        if len(words) > max_words:      # a single over-long sentence: hard-split
            if cur:
                chunks.append(cur); cur, cur_n = [], 0
            for k in range(0, len(words), max_words):
                chunks.append([" ".join(words[k:k + max_words])])
            continue
        if cur_n + len(words) > max_words and cur:
            chunks.append(cur); cur, cur_n = [], 0
        cur.append(sent); cur_n += len(words)
    if cur:
        chunks.append(cur)
    # Flatten each chunk's sentence list into a word list for length checks.
    flat = [" ".join(c).split() for c in chunks]
    if len(flat) > 1 and len(flat[-1]) < min_words:
        flat[-2].extend(flat.pop())
    return flat


__all__ = ["split_documents"]
