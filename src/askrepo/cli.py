"""Command line entry points for ``index`` and ``ask``."""

from dataclasses import replace
from pathlib import Path
from typing import Optional

import typer

from askrepo import __version__
from askrepo.answer import AnswerError, answer_question
from askrepo.config import Settings
from askrepo.ingest import IngestError, discover_index, index_tree, read_index, resolve_index_file
from askrepo.retrieve import load_retriever
from askrepo.types import Answer, Hit, Index


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Answer natural-language questions about a local codebase.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"askrepo {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Answer natural-language questions about a local codebase."""


@app.command()
def index(
    path: Optional[Path] = typer.Argument(
        None,
        help="Directory to index. Defaults to the current directory.",
    ),
) -> None:
    """Walk a repository and write .askrepo/index.json."""

    target = path if path is not None else Path(".")
    try:
        built, destination = index_tree(target)
    except IngestError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        "Indexed "
        f"{_counted(built.files_indexed, 'file')} → {_counted(built.chunk_count, 'chunk')} "
        f"(skipped {_counted(built.files_skipped, 'file')}, "
        f"pruned {_counted(built.dirs_pruned, 'directory', 'directories')})"
    )
    typer.echo(f"Wrote {destination}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural-language question about the indexed tree."),
    k: int = typer.Option(5, "--k", min=1, max=50, help="How many chunks to retrieve."),
    index_path: Optional[Path] = typer.Option(
        None,
        "--index",
        help="Index file, project directory, or .askrepo directory. Default: search upward from cwd.",
    ),
    retriever_name: Optional[str] = typer.Option(
        None,
        "--retriever",
        help="Retrieval backend. Only 'lexical' is implemented.",
    ),
) -> None:
    """Retrieve chunks and answer. Offline when ASKREPO_API_KEY is unset."""

    settings = Settings.from_env()
    if retriever_name:
        settings = replace(settings, retriever=retriever_name)
    try:
        location = discover_index() if index_path is None else resolve_index_file(index_path)
        loaded = read_index(location)
        retriever = load_retriever(settings.retriever)
        hits = retriever.retrieve(question, loaded, k)
        result = answer_question(question, hits, settings)
    except (IngestError, AnswerError, NotImplementedError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(render_ask(location, loaded, settings.retriever, hits, result), nl=False)


def render_ask(
    index_path: Path,
    index: Index,
    retriever_name: str,
    hits: list[Hit],
    result: Answer,
) -> str:
    lines = [
        f"Index: {index_path} ({index.chunk_count} chunks, retriever={retriever_name})",
        "",
    ]
    if not hits:
        lines.append("No chunks matched.")
        lines.append("")
    else:
        lines.append(f"{_counted(len(hits), 'chunk')}:")
        lines.append("")
        for number, hit in enumerate(hits, start=1):
            chunk = hit.chunk
            reason = " · ".join(hit.reasons) if hit.reasons else "match"
            lines.append(
                f"{number}. {chunk.path}:{chunk.start_line}-{chunk.end_line}  score={hit.score:.3f}"
            )
            lines.append(f"   {reason}")
            for line in _preview(chunk.text).splitlines():
                lines.append(f"   {line}")
            lines.append("")
    heading = "Answer · offline" if result.mode == "offline" else f"Answer · {result.model}"
    lines.append(heading)
    lines.append(result.text.rstrip())
    return "\n".join(lines).rstrip() + "\n"


def _counted(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {word}"


def _preview(text: str, max_lines: int = 24) -> str:
    lines = text.splitlines() or [""]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    hidden = len(lines) - max_lines
    return "\n".join(lines[:max_lines]) + f"\n… ({hidden} more lines)"
