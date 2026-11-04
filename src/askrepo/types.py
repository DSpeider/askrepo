"""Shared records for the index, retrieval hits, and answers.

Chunk identity is ``path`` plus the line range. Later index versions can hang
embeddings or graph edges off that id; this module does not store either.
"""

from dataclasses import dataclass


INDEX_VERSION = 1
INDEX_DIR = ".askrepo"
INDEX_FILENAME = "index.json"


@dataclass(frozen=True)
class Chunk:
    """One retrievable window of a text file.

    ``symbols`` and ``imports`` are regex hints, not a symbol table. They are
    stored so a later code-graph pass has something to consume, and so lexical
    ranking can prefer a chunk that defines the name in the question.
    """

    id: str
    path: str
    start_line: int
    end_line: int
    text: str
    language: str
    symbols: tuple[str, ...]
    imports: tuple[str, ...]


@dataclass(frozen=True)
class Index:
    """In-memory view of ``.askrepo/index.json`` (schema version 1)."""

    version: int
    created_at: str
    root: str
    files_indexed: int
    files_skipped: int
    dirs_pruned: int
    chunks: tuple[Chunk, ...]

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)


@dataclass(frozen=True)
class Hit:
    """A chunk plus the score that put it in the result list.

    ``reasons`` are short strings (``path:…``, ``symbol:…``, ``term:…``) so a
    later graph reranker can add ``edge:calls`` without changing ``answer``.
    """

    chunk: Chunk
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Answer:
    """What ``ask`` prints after retrieval.

    ``mode`` is ``offline`` when no model was called, ``model`` otherwise.
    """

    text: str
    mode: str
    model: str | None
