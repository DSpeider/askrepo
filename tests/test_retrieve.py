from askrepo.ingest import build_index
from askrepo.retrieve import LexicalRetriever, load_retriever, query_tokens, retrieve, tokenize
from askrepo.types import Chunk, Index


def test_tokenize_splits_snake_and_camel() -> None:
    assert tokenize("calculate_invoice_total") == ["calculate", "invoice", "total"]
    assert "invoice" in tokenize("src/billing/invoice.py")
    assert tokenize("parseHTTPResponse") == ["parse", "http", "response"]
    assert query_tokens("how is the invoice total calculated") == ["invoice", "total", "calculated"]


def test_invoice_query_prefers_defining_file(tiny_repo) -> None:
    index = build_index(tiny_repo)
    hits = LexicalRetriever().retrieve("calculate invoice total", index, k=3)

    assert hits
    assert hits[0].chunk.path == "src/billing/invoice.py"
    assert any(reason.startswith("symbol:calculate_invoice_total") for reason in hits[0].reasons)
    assert all("node_modules" not in hit.chunk.path for hit in hits)


def test_tax_query_prefers_tax_helper(tiny_repo) -> None:
    index = build_index(tiny_repo)
    hits = retrieve("apply tax", index, k=3)

    assert hits[0].chunk.path == "src/billing/tax.py"
    assert "apply_tax" in hits[0].chunk.symbols


def test_unknown_query_returns_nothing(tiny_repo) -> None:
    index = build_index(tiny_repo)
    assert retrieve("zzzz_quantum_flux", index, k=3) == []


def test_embedding_and_graph_backends_are_reserved() -> None:
    for name in ("embedding", "graph"):
        try:
            load_retriever(name)
        except NotImplementedError as exc:
            message = str(exc)
            assert name in message
        else:
            raise AssertionError(f"{name} should be reserved")


def test_one_file_does_not_fill_the_window() -> None:
    chunks = [
        _chunk("a.py", start, "alpha beta gamma", ("alpha_beta",))
        for start in range(1, 5)
    ]
    chunks.append(_chunk("b.py", 1, "alpha beta gamma"))
    hits = LexicalRetriever().retrieve("alpha beta", _index(chunks), k=3)

    assert [hit.chunk.path for hit in hits] == ["a.py", "a.py", "b.py"]


def _chunk(path: str, start: int, text: str, symbols: tuple[str, ...] = ()) -> Chunk:
    return Chunk(
        id=f"{path}:{start}",
        path=path,
        start_line=start,
        end_line=start + 1,
        text=text,
        language="python",
        symbols=symbols,
        imports=(),
    )


def _index(chunks: list[Chunk]) -> Index:
    return Index(
        version=1,
        created_at="2026-01-01T00:00:00+00:00",
        root="/tmp/example",
        files_indexed=len({chunk.path for chunk in chunks}),
        files_skipped=0,
        dirs_pruned=0,
        chunks=tuple(chunks),
    )
