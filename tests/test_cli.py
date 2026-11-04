import json
import shutil
from pathlib import Path

from typer.testing import CliRunner

from askrepo.cli import app


runner = CliRunner()


def test_index_and_ask_run_offline(tmp_path: Path, tiny_repo: Path, monkeypatch) -> None:
    monkeypatch.delenv("ASKREPO_API_KEY", raising=False)
    monkeypatch.delenv("ASKREPO_RETRIEVER", raising=False)
    repo = tmp_path / "repo"
    shutil.copytree(tiny_repo, repo)

    indexed = runner.invoke(app, ["index", str(repo)])
    assert indexed.exit_code == 0, indexed.output
    assert "Wrote" in indexed.output
    assert "chunks" in indexed.output

    written = json.loads((repo / ".askrepo" / "index.json").read_text(encoding="utf-8"))
    assert written["version"] == 1
    assert written["tool"] == "askrepo"
    paths = {chunk["path"] for chunk in written["chunks"]}
    assert "src/billing/invoice.py" in paths
    assert not any("node_modules" in path for path in paths)

    monkeypatch.chdir(repo)
    asked = runner.invoke(app, ["ask", "calculate invoice total"])
    assert asked.exit_code == 0, asked.output
    assert "1. src/billing/invoice.py:" in asked.output
    assert "Answer · offline" in asked.output
    assert "ASKREPO_API_KEY" in asked.output
    assert "node_modules/should_skip.js" not in asked.output


def test_ask_without_an_index_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ASKREPO_API_KEY", raising=False)
    result = runner.invoke(app, ["ask", "where is main"])
    assert result.exit_code != 0
    assert "index" in result.output.lower()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "askrepo 0.1.0" in result.output
