"""Walk a repository, chunk text files, and read or write the local index.

The walker is deliberately boring: a fixed junk list, line windows, and
regex-level symbol and import hints. It does not parse with tree-sitter, does
not follow ``.gitignore``, and does not build edges between chunks. Those are
later passes that should read this index, not grow inside the walker.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from askrepo import __version__
from askrepo.types import INDEX_DIR, INDEX_FILENAME, INDEX_VERSION, Chunk, Index


MAX_FILE_BYTES = 400_000
MAX_CHUNK_LINES = 80
CHUNK_OVERLAP_LINES = 10
MAX_LINE_CHARS = 2_000
MAX_HINTS = 50

# Directory names pruned anywhere in the tree. ``build`` and ``env`` can also
# be real source folders; the list is fixed on purpose until it is configurable.
SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".askrepo",
        ".idea",
        ".vscode",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "node_modules",
        "bower_components",
        "site-packages",
        "venv",
        ".venv",
        "virtualenv",
        "env",
        "dist",
        "build",
        "target",
        "vendor",
        "htmlcov",
        ".eggs",
    }
)

LOCKFILES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "cargo.lock",
        "poetry.lock",
        "pipfile.lock",
        "composer.lock",
        "gemfile.lock",
    }
)

# Suffixes checked with ``str.endswith``. Multi-part names (``.min.js``) are
# included explicitly because ``Path.suffix`` would only see ``.js``.
SKIP_SUFFIXES = (
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".bmp",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".7z",
    ".xz",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
    ".mp3",
    ".mp4",
    ".mov",
    ".wav",
    ".ogg",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".a",
    ".o",
    ".class",
    ".jar",
    ".war",
    ".pyc",
    ".pyo",
    ".whl",
    ".sqlite",
    ".db",
    ".bin",
    ".wasm",
    ".min.js",
    ".min.css",
    ".map",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".keystore",
)

_EXT_LANGUAGE = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".sql": "sql",
    ".md": "markdown",
    ".rst": "rst",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".xml": "xml",
    ".txt": "text",
}

_NAME_LANGUAGE = {
    "makefile": "make",
    "dockerfile": "docker",
    "jenkinsfile": "groovy",
    "gemfile": "ruby",
    "rakefile": "ruby",
}

_PY_SYMBOL = re.compile(r"^(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)
_JS_SYMBOL = re.compile(
    r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)"
    r"|^(?:export\s+)?class\s+([A-Za-z_$][\w$]*)"
    r"|^(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][\w$]*)\s*=>",
    re.MULTILINE,
)
_GO_SYMBOL = re.compile(
    r"^func\s+(?:\([^)\n]*\)\s*)?([A-Za-z_][A-Za-z0-9_]*)"
    r"|^type\s+([A-Za-z_][A-Za-z0-9_]*)\s+(?:struct|interface)",
    re.MULTILINE,
)
_RUST_SYMBOL = re.compile(
    r"^(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)"
    r"|^(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
_JAVA_SYMBOL = re.compile(
    r"^(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?(?:class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
_JS_IMPORT = re.compile(
    r"""import\s+(?:[^'"\n]+?\s+from\s+)?['"]([^'"]+)['"]|require\(\s*['"]([^'"]+)['"]\s*\)"""
)
_GO_IMPORT = re.compile(r'^\s*(?:import\s+)?"([A-Za-z0-9_./\-]+)"', re.MULTILINE)
_RUST_USE = re.compile(r"^use\s+([A-Za-z0-9_:]+)", re.MULTILINE)
_JAVA_IMPORT = re.compile(r"^import\s+(?:static\s+)?([A-Za-z_][\w.]*)\s*;", re.MULTILINE)


class IngestError(ValueError):
    """The path cannot be indexed, or an index file is not version 1."""


def index_tree(root: Path) -> tuple[Index, Path]:
    """Build an index for ``root`` and write ``<root>/.askrepo/index.json``."""

    index = build_index(root)
    destination = index_file(root)
    write_index(index, destination)
    return index, destination


def index_file(root: Path) -> Path:
    return root.resolve() / INDEX_DIR / INDEX_FILENAME


def build_index(root: Path) -> Index:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise IngestError(f"index path is not a directory: {root}")

    chunks: list[Chunk] = []
    files_indexed = 0
    files_skipped = 0
    dirs_pruned = 0

    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        kept: list[str] = []
        for name in sorted(dirnames):
            if _skip_dir(name):
                dirs_pruned += 1
                continue
            kept.append(name)
        dirnames[:] = kept

        for name in sorted(filenames):
            path = Path(dirpath) / name
            if path.is_symlink():
                files_skipped += 1
                continue
            if _skip_filename(name):
                files_skipped += 1
                continue
            try:
                text = _read_text(path)
            except OSError:
                files_skipped += 1
                continue
            if text is None:
                files_skipped += 1
                continue
            if not text.strip():
                continue

            relative = path.relative_to(root).as_posix()
            language = language_for(path, text)
            file_imports = extract_imports(language, text)
            file_chunks = chunk_text(
                relative=relative,
                text=text,
                language=language,
                file_imports=file_imports,
            )
            if not file_chunks:
                files_skipped += 1
                continue
            chunks.extend(file_chunks)
            files_indexed += 1

    return Index(
        version=INDEX_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        root=str(root),
        files_indexed=files_indexed,
        files_skipped=files_skipped,
        dirs_pruned=dirs_pruned,
        chunks=tuple(chunks),
    )


def write_index(index: Index, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = index_to_dict(index)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def read_index(path: Path) -> Index:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IngestError(f"cannot read index {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IngestError(f"index is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise IngestError(f"index is not a JSON object: {path}")
    version = data.get("version")
    if version != INDEX_VERSION:
        raise IngestError(
            f"unsupported index version {version!r} in {path}. "
            f"This build reads version {INDEX_VERSION}. Re-run `askrepo index`."
        )
    chunks_raw = data.get("chunks")
    if not isinstance(chunks_raw, list):
        raise IngestError(f"index is missing a chunks array: {path}")
    chunks = tuple(_chunk_from_dict(item, path) for item in chunks_raw)
    return Index(
        version=INDEX_VERSION,
        created_at=str(data.get("created_at", "")),
        root=str(data.get("root", "")),
        files_indexed=int(data.get("files_indexed", 0)),
        files_skipped=int(data.get("files_skipped", 0)),
        dirs_pruned=int(data.get("dirs_pruned", 0)),
        chunks=chunks,
    )


def resolve_index_file(explicit: Path) -> Path:
    """Return an index file from a file path, a project root, or a ``.askrepo`` dir."""

    candidate = explicit.expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.exists():
        raise IngestError(f"index path does not exist: {candidate}")
    if not candidate.is_dir():
        raise IngestError(f"index path is not a file or directory: {candidate}")

    if candidate.name == INDEX_DIR:
        nested = candidate / INDEX_FILENAME
        if nested.is_file():
            return nested.resolve()
    direct = candidate / INDEX_DIR / INDEX_FILENAME
    if direct.is_file():
        return direct.resolve()
    raise IngestError(
        f"no {INDEX_DIR}/{INDEX_FILENAME} under {candidate.resolve()}. "
        f"Run `askrepo index {candidate}`."
    )


def discover_index(start: Path | None = None) -> Path:
    """Walk from ``start`` (default: cwd) toward the filesystem root."""

    current = (start or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for directory in (current, *current.parents):
        candidate = directory / INDEX_DIR / INDEX_FILENAME
        if candidate.is_file():
            return candidate
    raise IngestError(
        "no .askrepo/index.json found from the current directory upward. "
        "Run `askrepo index <path>` first, or pass --index."
    )


def index_to_dict(index: Index) -> dict:
    return {
        "version": index.version,
        "tool": "askrepo",
        "tool_version": __version__,
        "created_at": index.created_at,
        "root": index.root,
        "files_indexed": index.files_indexed,
        "files_skipped": index.files_skipped,
        "dirs_pruned": index.dirs_pruned,
        "chunks": [
            {
                "id": chunk.id,
                "path": chunk.path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "language": chunk.language,
                "symbols": list(chunk.symbols),
                "imports": list(chunk.imports),
                "text": chunk.text,
            }
            for chunk in index.chunks
        ],
    }


def language_for(path: Path, text: str) -> str:
    named = _NAME_LANGUAGE.get(path.name.lower())
    if named:
        return named
    by_ext = _EXT_LANGUAGE.get(path.suffix.lower())
    if by_ext:
        return by_ext
    if text.startswith("#!"):
        first = text.splitlines()[0].lower()
        if "python" in first:
            return "python"
        if "node" in first:
            return "javascript"
        if "bash" in first or first.endswith(" sh") or "/sh" in first:
            return "shell"
    return "text"


def extract_imports(language: str, text: str) -> tuple[str, ...]:
    """File-level import hints. Attached to every chunk of the file."""

    found: list[str] = []
    if language == "python":
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("from ") and " import " in stripped:
                parts = stripped.split()
                if len(parts) >= 2:
                    found.append(parts[1])
            elif stripped.startswith("import "):
                rest = stripped[len("import ") :].split("#", 1)[0]
                for part in rest.split(","):
                    name = part.strip().split(" as ", 1)[0].strip()
                    if name:
                        found.append(name)
    elif language in {"javascript", "typescript"}:
        for match in _JS_IMPORT.finditer(text):
            found.append(next(group for group in match.groups() if group))
    elif language == "go":
        found.extend(_GO_IMPORT.findall(text))
    elif language == "rust":
        found.extend(_RUST_USE.findall(text))
    elif language == "java":
        found.extend(_JAVA_IMPORT.findall(text))
    return _unique(found)


def extract_symbols(language: str, text: str) -> tuple[str, ...]:
    """Names defined in this window. Regexes, so decorators and macros slip through."""

    patterns = {
        "python": _PY_SYMBOL,
        "javascript": _JS_SYMBOL,
        "typescript": _JS_SYMBOL,
        "go": _GO_SYMBOL,
        "rust": _RUST_SYMBOL,
        "java": _JAVA_SYMBOL,
    }
    pattern = patterns.get(language)
    if pattern is None:
        return ()
    names = [group for match in pattern.finditer(text) for group in match.groups() if group]
    return _unique(names)


def chunk_text(
    *,
    relative: str,
    text: str,
    language: str,
    file_imports: tuple[str, ...],
) -> list[Chunk]:
    lines = text.splitlines(keepends=True)
    if any(len(line) > MAX_LINE_CHARS for line in lines):
        return []
    windows = _line_windows(lines, MAX_CHUNK_LINES, CHUNK_OVERLAP_LINES)
    chunks: list[Chunk] = []
    for start, end, body in windows:
        symbols = extract_symbols(language, body)
        chunks.append(
            Chunk(
                id=f"{relative}:{start}-{end}",
                path=relative,
                start_line=start,
                end_line=end,
                text=body,
                language=language,
                symbols=symbols,
                imports=file_imports,
            )
        )
    return chunks


def _line_windows(lines: list[str], max_lines: int, overlap: int) -> list[tuple[int, int, str]]:
    if not lines:
        return []
    if overlap >= max_lines:
        raise IngestError("chunk overlap must be smaller than the chunk line limit")
    count = len(lines)
    if count <= max_lines:
        return [(1, count, "".join(lines))]
    step = max_lines - overlap
    windows: list[tuple[int, int, str]] = []
    start = 0
    while start < count:
        end = min(start + max_lines, count)
        windows.append((start + 1, end, "".join(lines[start:end])))
        if end == count:
            break
        start += step
    return windows


def _skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.endswith(".egg-info")


def _skip_filename(name: str) -> bool:
    lower = name.lower()
    if lower in LOCKFILES:
        return True
    if lower in {".env", ".npmrc", ".pypirc", ".netrc", "id_rsa", "id_dsa"}:
        return True
    if lower.startswith(".env."):
        return True
    return lower.endswith(SKIP_SUFFIXES)


def _read_text(path: Path) -> str | None:
    try:
        size = path.stat().st_size
    except OSError:
        return None
    if size > MAX_FILE_BYTES:
        return None
    data = path.read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _chunk_from_dict(item: object, path: Path) -> Chunk:
    if not isinstance(item, dict):
        raise IngestError(f"chunk is not an object in {path}")
    try:
        symbols = item.get("symbols", [])
        imports = item.get("imports", [])
        return Chunk(
            id=str(item["id"]),
            path=str(item["path"]),
            start_line=int(item["start_line"]),
            end_line=int(item["end_line"]),
            text=str(item["text"]),
            language=str(item.get("language", "text")),
            symbols=tuple(str(name) for name in symbols),
            imports=tuple(str(name) for name in imports),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IngestError(f"chunk in {path} is missing required fields") from exc


def _unique(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        cleaned = item.strip().strip(",")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        ordered.append(cleaned)
        if len(ordered) >= MAX_HINTS:
            break
    return tuple(ordered)
