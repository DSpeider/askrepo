"""Rank index chunks for a question.

``Retriever`` is the only seam ``ask`` uses. ``LexicalRetriever`` scores path
tokens, extracted symbol names, import hints, and IDF-weighted keyword overlap.
Nothing in this module embeds text or walks a call graph.

Reserved backend names (see ``load_retriever``):

- ``embedding`` — same ``retrieve`` signature. Store vectors outside the chunk
  text, keyed by ``Chunk.id``, and rank by similarity. Lexical scores can stay
  as a hybrid term; do not replace the index schema just to try a model.
- ``graph`` — still return ``list[Hit]``. Expand or rerank the lexical (or
  embedding) hits with edges built from ``Chunk.imports`` / ``Chunk.symbols``
  and, later, real call edges. ``answer`` should not learn about edges.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from askrepo.types import Chunk, Hit, Index


# English function words only. Identifiers like "class" or "from" stay, because
# people do ask "class Invoice" and IDF already downweights tokens that are
# everywhere in the tree.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "how",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "what",
        "when",
        "where",
        "who",
        "why",
        "with",
        "you",
    }
)

_TOKEN = re.compile(r"[A-Za-z0-9_]+")
_CAMEL = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")

_PATH_WEIGHT = 1.5
_SYMBOL_WEIGHT = 2.0
_IMPORT_WEIGHT = 0.5
_PER_FILE = 2

_RESERVED = {
    "embedding": (
        "retriever 'embedding' is reserved and not implemented. "
        "Implement Retriever.retrieve with vectors keyed by chunk id, "
        "then set ASKREPO_RETRIEVER=embedding."
    ),
    "graph": (
        "retriever 'graph' is reserved and not implemented. "
        "A graph backend should rerank or expand hits using import and call "
        "edges and still return list[Hit]."
    ),
}


class Retriever(Protocol):
    """Rank ``index.chunks`` for a natural-language query.

    Implementations may read chunks and, later, sidecar vectors or edges.
    They must not mutate the index and must not call the chat model.
    """

    def retrieve(self, query: str, index: Index, k: int) -> list[Hit]:
        """Return at most ``k`` hits, best first. Empty when nothing scores."""


class LexicalRetriever:
    """Path match + keyword overlap, with a bonus when a symbol name hits.

    There is no inverted index. Scoring scans the chunk list, which is fine
    for a small repository and the wrong shape once the tree is large.
    """

    def retrieve(self, query: str, index: Index, k: int) -> list[Hit]:
        if k < 1:
            raise ValueError("k must be at least 1")
        tokens = query_tokens(query)
        if not tokens or not index.chunks:
            return []
        idf = _idf(index.chunks, tokens)
        ranked: list[Hit] = []
        for chunk in index.chunks:
            score, reasons = _score_chunk(tokens, chunk, idf)
            if score <= 0:
                continue
            ranked.append(Hit(chunk=chunk, score=score, reasons=reasons))
        ranked.sort(key=lambda hit: (-hit.score, hit.chunk.path, hit.chunk.start_line))
        return _diversify(ranked, k)


def load_retriever(name: str = "lexical") -> Retriever:
    """Return the named backend.

    Only ``lexical`` exists. ``embedding`` and ``graph`` raise
    ``NotImplementedError`` with the contract a replacement should meet.
    """

    key = name.strip().lower()
    if key == "lexical":
        return LexicalRetriever()
    if key in _RESERVED:
        raise NotImplementedError(_RESERVED[key])
    known = ", ".join(["lexical", *_RESERVED])
    raise ValueError(f"unknown retriever {name!r}. Known names: {known}.")


def retrieve(query: str, index: Index, k: int = 5, backend: str = "lexical") -> list[Hit]:
    return load_retriever(backend).retrieve(query, index, k)


def tokenize(text: str) -> list[str]:
    """Split on punctuation, then on ``_`` and camel case. Tokens shorter than 2 are dropped."""

    out: list[str] = []
    for raw in _TOKEN.findall(text):
        pieces = raw.split("_") if "_" in raw else [raw]
        for piece in pieces:
            if not piece:
                continue
            parts = _CAMEL.findall(piece)
            # ``_CAMEL`` drops a single leading lowercase run mixed oddly; fall back.
            if not parts:
                parts = [piece]
            for part in parts:
                token = part.lower()
                if len(token) >= 2:
                    out.append(token)
    return out


def query_tokens(query: str) -> list[str]:
    tokens = tokenize(query)
    content = [token for token in tokens if token not in _STOPWORDS]
    chosen = content or tokens
    seen: set[str] = set()
    unique: list[str] = []
    for token in chosen:
        if token in seen:
            continue
        seen.add(token)
        unique.append(token)
    return unique


def _idf(chunks: tuple[Chunk, ...], tokens: list[str]) -> dict[str, float]:
    sets = [set(tokenize(chunk.text)) | set(tokenize(chunk.path)) for chunk in chunks]
    total = len(chunks)
    weights: dict[str, float] = {}
    for token in tokens:
        document_frequency = sum(1 for token_set in sets if token in token_set)
        weights[token] = math.log((total + 1) / (document_frequency + 1)) + 1.0
    return weights


def _score_chunk(
    tokens: list[str],
    chunk: Chunk,
    idf: dict[str, float],
) -> tuple[float, tuple[str, ...]]:
    query_set = set(tokens)
    body = set(tokenize(chunk.text))
    overlap = [token for token in tokens if token in body]
    content = sum(idf[token] for token in overlap) / len(tokens)

    path_tokens = set(tokenize(chunk.path))
    path_hits = [token for token in tokens if token in path_tokens]

    symbol_tokens: set[str] = set()
    matched_symbols: list[str] = []
    for name in chunk.symbols:
        parts = set(tokenize(name))
        parts.add(name.lower())
        symbol_tokens |= parts
        if parts & query_set:
            matched_symbols.append(name)

    import_tokens: set[str] = set()
    for name in chunk.imports:
        import_tokens |= set(tokenize(name))
    import_hits = [token for token in tokens if token in import_tokens]

    symbol_frac = len(query_set & symbol_tokens) / len(tokens)
    path_frac = len(path_hits) / len(tokens)
    import_frac = len(import_hits) / len(tokens)
    score = content + (_SYMBOL_WEIGHT * symbol_frac) + (_PATH_WEIGHT * path_frac) + (_IMPORT_WEIGHT * import_frac)

    reasons: list[str] = []
    if matched_symbols:
        reasons.append("symbol:" + ",".join(matched_symbols[:3]))
    if path_hits:
        reasons.append("path:" + ",".join(path_hits))
    if import_hits:
        reasons.append("import:" + ",".join(import_hits))
    if overlap:
        reasons.append("term:" + ",".join(overlap[:6]))
    return score, tuple(reasons)


def _diversify(ranked: list[Hit], k: int, per_file: int = _PER_FILE) -> list[Hit]:
    """Keep one verbose file from filling the window the model will see."""

    chosen: list[Hit] = []
    overflow: list[Hit] = []
    counts: Counter[str] = Counter()
    for hit in ranked:
        if counts[hit.chunk.path] >= per_file:
            overflow.append(hit)
            continue
        chosen.append(hit)
        counts[hit.chunk.path] += 1
        if len(chosen) == k:
            return chosen
    for hit in overflow:
        chosen.append(hit)
        if len(chosen) == k:
            break
    return chosen
