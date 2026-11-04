import json
import shutil
from pathlib import Path

import pytest

from askrepo.ingest import IngestError, build_index, discover_index, index_tree, read_index, write_index


def test_tiny_repo_skips_junk_and_records_hints(tiny_repo: Path, tmp_path: Path) -> None:
    # Copy so a local .askrepo/ from a previous `askrepo index` does not change the counts.
    repo = tmp_path / "tiny"
    shutil.copytree(tiny_repo, repo, ignore=shutil.ignore_patterns(".askrepo"))
    index = build_index(repo)
    paths = {chunk.path for chunk in index.chunks}

    assert "src/billing/invoice.py" in paths
    assert "src/billing/tax.py" in paths
    assert "docs/overview.md" in paths
    assert "README.md" in paths
    assert not any("node_modules" in path or path.startswith("build/") or path == ".env" for path in paths)
    assert index.dirs_pruned == 2
    assert index.files_skipped == 1

    invoice = next(chunk for chunk in index.chunks if chunk.path == "src/billing/invoice.py")
    assert invoice.language == "python"
    assert "calculate_invoice_total" in invoice.symbols
    assert "billing.tax" in invoice.imports
    assert "API_TOKEN" not in "\n".join(chunk.text for chunk in index.chunks)


def test_long_file_is_split_on_line_windows(tmp_path: Path) -> None:
    body = "".join(f"def func_{i}():\n    return {i}\n" for i in range(90))
    (tmp_path / "big.py").write_text(body, encoding="utf-8")

    index = build_index(tmp_path)

    assert index.files_indexed == 1
    assert len(index.chunks) >= 2
    assert index.chunks[0].start_line == 1
    assert index.chunks[0].symbols[0] == "func_0"
    assert any("func_89" in chunk.symbols for chunk in index.chunks)
    assert index.chunks[0].path == "big.py"


def test_skips_binary_minified_git_and_egg_info(tmp_path: Path) -> None:
    (tmp_path / "app.min.js").write_text("function nope() { return 1; }\n", encoding="utf-8")
    (tmp_path / "wide.py").write_text("x = '" + ("a" * 2500) + "'\n", encoding="utf-8")
    (tmp_path / "bin.py").write_bytes(b"def hidden():\n    return 1\n\0")
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("calculate_invoice_total\n", encoding="utf-8")
    (tmp_path / "real.py").write_text("def real_func():\n    return 1\n", encoding="utf-8")
    egg = tmp_path / "demo.egg-info"
    egg.mkdir()
    (egg / "PKG-INFO").write_text("Name: demo\n", encoding="utf-8")

    index = build_index(tmp_path)

    assert [chunk.path for chunk in index.chunks] == ["real.py"]
    assert index.dirs_pruned >= 1
    assert "real_func" in index.chunks[0].symbols


def test_roundtrip_preserves_chunks(tmp_path: Path, tiny_repo: Path) -> None:
    index = build_index(tiny_repo)
    destination = tmp_path / "index.json"
    write_index(index, destination)
    loaded = read_index(destination)

    assert loaded.chunks == index.chunks
    assert loaded.files_indexed == index.files_indexed
    assert loaded.root == index.root


def test_rejects_unsupported_version(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"version": 2, "chunks": []}), encoding="utf-8")
    with pytest.raises(IngestError, match="unsupported index version"):
        read_index(path)


def test_discover_walks_upward(tmp_path: Path) -> None:
    root = tmp_path / "proj"
    root.mkdir()
    (root / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    _, written = index_tree(root)
    nested = root / "sub" / "deep"
    nested.mkdir(parents=True)

    found = discover_index(nested)

    assert found == written.resolve()
