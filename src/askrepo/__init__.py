"""Ask natural-language questions about a local codebase.

The package is split so each stage can change on its own:

- ``ingest`` walks a tree and writes ``.askrepo/index.json``
- ``retrieve`` ranks chunks; only the lexical backend exists today
- ``answer`` turns hits into an offline note or a chat completion
- ``eval`` scores a gold set of questions against a retriever
"""

__version__ = "0.1.0"
