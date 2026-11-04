"""Score a retriever against a gold set of question → file paths.

The set in ``evals/tiny_repo.json`` is the baseline for retrieval experiments.
Required cases must land in the top k (or at the rank named on the case).
Challenge cases are expected lexical misses: a later graph or embedding
retriever should raise that hit rate without dropping the required cases.

Nothing here calls a chat model or downloads an embedding model. Scoring uses
``Retriever.retrieve`` and the in-memory index from ``build_index``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from askrepo.ingest import build_index
from askrepo.retrieve import load_retriever
from askrepo.types import Hit


_TIERS = frozenset({"required", "challenge"})
_CASE_KEYS = frozenset({"id", "question", "expect", "tier", "at", "note"})
_FILE_KEYS = frozenset({"name", "description", "repo", "k", "retriever", "cases"})
_DEFAULT_K = 5
_DEFAULT_RETRIEVER = "lexical"


class EvalError(ValueError):
    """The gold file, the repository path, or a case record is not usable."""


@dataclass(frozen=True)
class GoldCase:
    """One question and the repo-relative paths that answer it.

    ``at`` is the worst acceptable rank for a required case. Rank 1 is the
    first retrieved chunk. Challenge cases are reported and not gated on ``at``.
    """

    id: str
    question: str
    expect: tuple[str, ...]
    tier: str
    at: int
    note: str


@dataclass(frozen=True)
class GoldSet:
    """A gold file plus the repository it is scored against."""

    name: str
    description: str
    repo: str
    k: int
    retriever: str
    cases: tuple[GoldCase, ...]
    path: Path


@dataclass(frozen=True)
class CaseResult:
    """How one case ranked inside the retrieved window."""

    case: GoldCase
    rank: int | None
    top_paths: tuple[str, ...]

    def hit_at(self, k: int) -> bool:
        return self.rank is not None and self.rank <= k

    def passed(self, k: int) -> bool:
        """Required cases pass at ``min(case.at, k)``. Challenge cases are not gated."""

        if self.case.tier != "required":
            return True
        return self.hit_at(min(self.case.at, k))


@dataclass(frozen=True)
class EvalReport:
    """Hit rates and per-case ranks for one run."""

    repo: Path
    gold: GoldSet
    retriever: str
    k: int
    files_indexed: int
    chunk_count: int
    results: tuple[CaseResult, ...]
    check_baseline: bool = False

    def hits(self, tier: str, at: int) -> tuple[int, int]:
        subset = [result for result in self.results if result.case.tier == tier]
        got = sum(1 for result in subset if result.hit_at(at))
        return got, len(subset)

    def problems(self) -> tuple[CaseResult, ...]:
        """Required misses, plus challenge hits when the lexical baseline is locked."""

        found: list[CaseResult] = []
        for result in self.results:
            if not result.passed(self.k):
                found.append(result)
            elif self.check_baseline and result.case.tier == "challenge" and result.hit_at(self.k):
                found.append(result)
        return tuple(found)


def load_gold(path: Path) -> GoldSet:
    """Read a gold file. Paths are checked for shape, not for presence in a tree."""

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalError(f"cannot read gold file {path}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvalError(f"gold file is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise EvalError(f"gold file is not a JSON object: {path}")
    unknown = set(data) - _FILE_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise EvalError(f"gold file {path} has unknown fields: {names}")

    name = _text_field(data, "name", path)
    description = str(data.get("description", ""))
    repo = _text_field(data, "repo", path)
    k = _positive_int(data.get("k", _DEFAULT_K), f"{path} k")
    retriever = str(data.get("retriever", _DEFAULT_RETRIEVER)).strip() or _DEFAULT_RETRIEVER
    cases_raw = data.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise EvalError(f"gold file {path} needs a non-empty cases array")

    cases: list[GoldCase] = []
    seen: set[str] = set()
    for index, item in enumerate(cases_raw, start=1):
        case = _case_from_dict(item, path, index, default_at=k)
        if case.id in seen:
            raise EvalError(f"duplicate case id {case.id!r} in {path}")
        seen.add(case.id)
        cases.append(case)
    return GoldSet(
        name=name,
        description=description,
        repo=repo,
        k=k,
        retriever=retriever,
        cases=tuple(cases),
        path=path.resolve(),
    )


def resolve_cases_path(explicit: Path | None = None) -> Path:
    """Find the gold file from ``--cases``, the cwd, or this checkout."""

    if explicit is not None:
        candidate = explicit.expanduser()
        if not candidate.is_file():
            raise EvalError(f"gold file does not exist: {candidate}")
        return candidate.resolve()
    cwd_candidate = Path.cwd() / "evals" / "tiny_repo.json"
    if cwd_candidate.is_file():
        return cwd_candidate.resolve()
    checkout = Path(__file__).resolve().parents[2] / "evals" / "tiny_repo.json"
    if checkout.is_file():
        return checkout.resolve()
    raise EvalError(
        "could not find evals/tiny_repo.json. Pass --cases, or run from the askrepo checkout."
    )


def resolve_repo(gold: GoldSet, explicit: Path | None = None) -> Path:
    """Return the tree to index. An argument wins; otherwise use the gold file's ``repo``."""

    if explicit is not None:
        candidate = explicit.expanduser()
        if not candidate.is_dir():
            raise EvalError(f"eval path is not a directory: {candidate}")
        return candidate.resolve()

    raw = Path(gold.repo)
    if raw.is_absolute():
        if not raw.is_dir():
            raise EvalError(f"gold repo does not exist: {raw}")
        return raw
    # evals/tiny_repo.json lives one directory under the checkout root.
    beside_gold = gold.path.parent.parent / raw
    if beside_gold.is_dir():
        return beside_gold.resolve()
    from_cwd = Path.cwd() / raw
    if from_cwd.is_dir():
        return from_cwd.resolve()
    raise EvalError(
        f"gold repo does not exist: {gold.repo}. Pass the repository path to `askrepo eval`."
    )


def evaluate(
    repo: Path,
    gold: GoldSet,
    *,
    k: int | None = None,
    retriever_name: str | None = None,
    check_baseline: bool = False,
) -> EvalReport:
    """Build an in-memory index and rank every case. Does not write ``.askrepo/``."""

    cutoff = gold.k if k is None else k
    if cutoff < 1:
        raise EvalError("k must be at least 1")
    backend = retriever_name or gold.retriever
    index = build_index(repo)
    indexed = {chunk.path for chunk in index.chunks}
    missing: list[str] = []
    for case in gold.cases:
        for path in case.expect:
            if path not in indexed and path not in missing:
                missing.append(path)
    if missing:
        listed = ", ".join(missing)
        raise EvalError(f"gold expect paths are not in the index: {listed}")

    retriever = load_retriever(backend)
    results = tuple(
        _score_case(case, retriever.retrieve(case.question, index, cutoff)) for case in gold.cases
    )
    return EvalReport(
        repo=repo,
        gold=gold,
        retriever=backend,
        k=cutoff,
        files_indexed=index.files_indexed,
        chunk_count=index.chunk_count,
        results=results,
        check_baseline=check_baseline,
    )


def format_report(report: EvalReport) -> str:
    """Plain-text hit rates and a pass/fail line per case."""

    lines = [
        f"Repo: {report.repo}",
        (
            f"Index: {_counted(report.files_indexed, 'file')}, "
            f"{_counted(report.chunk_count, 'chunk')}, "
            f"retriever={report.retriever}, k={report.k}"
        ),
        f"Cases: {report.gold.path} ({len(report.results)})",
        "",
    ]
    cutoffs = (1,) if report.k == 1 else (1, report.k)
    for cutoff in cutoffs:
        required = _ratio(report.hits("required", cutoff))
        challenge = _ratio(report.hits("challenge", cutoff))
        lines.append(f"hit@{cutoff}  required {required}  challenge {challenge}")
    lines.append("Gate: required rank cutoffs. Challenge rows are reported, not gated.")
    lines.append("")

    width = max((len(result.case.id) for result in report.results), default=0)
    previous: str | None = None
    for result in report.results:
        if previous is not None and result.case.tier != previous:
            lines.append("")
        previous = result.case.tier
        lines.append(_format_case(result, report.k, width))

    lines.append("")
    problems = report.problems()
    if not problems:
        required_total = report.hits("required", report.k)[1]
        lines.append(f"Required cases passed ({required_total}).")
        return "\n".join(lines).rstrip() + "\n"

    required_misses = [result for result in problems if result.case.tier == "required"]
    challenge_hits = [result for result in problems if result.case.tier == "challenge"]
    if required_misses:
        lines.append(
            f"{_counted(len(required_misses), 'required case')} missed the rank cutoff."
        )
    if challenge_hits:
        lines.append(
            f"{_counted(len(challenge_hits), 'challenge case')} hit; "
            "the lexical baseline expects a miss. Move it to required if that is intentional."
        )
    return "\n".join(lines).rstrip() + "\n"


def run_eval(
    *,
    repo: Path | None = None,
    cases: Path | None = None,
    k: int | None = None,
    retriever_name: str | None = None,
    check_baseline: bool = False,
) -> EvalReport:
    """Load the gold file, index ``repo``, and score it."""

    gold = load_gold(resolve_cases_path(cases))
    root = resolve_repo(gold, repo)
    return evaluate(
        root,
        gold,
        k=k,
        retriever_name=retriever_name,
        check_baseline=check_baseline,
    )


def _score_case(case: GoldCase, hits: list[Hit]) -> CaseResult:
    first_at: dict[str, int] = {}
    top: list[str] = []
    for offset, hit in enumerate(hits, start=1):
        path = hit.chunk.path
        if path in first_at:
            continue
        first_at[path] = offset
        top.append(path)
    if all(path in first_at for path in case.expect):
        rank = max(first_at[path] for path in case.expect)
    else:
        rank = None
    return CaseResult(case=case, rank=rank, top_paths=tuple(top))


def _format_case(result: CaseResult, k: int, width: int) -> str:
    case = result.case
    name = case.id.ljust(width)
    expect = ", ".join(case.expect)
    appeared = [path for path in case.expect if path in result.top_paths]
    if case.tier == "challenge" and result.hit_at(k):
        return f"BEAT  {name}  @{result.rank}  {expect}"
    if case.tier == "required" and result.passed(k):
        line = f"PASS  {name}  @{result.rank}  {expect}"
        if case.note and result.rank != 1:
            line += f"\n      {case.note}"
        return line

    top = ", ".join(result.top_paths) or "(none)"
    found = ", ".join(appeared) or "(none)"
    if case.tier == "challenge":
        mark = "MISS"
        detail = f"expected {expect}; appeared {found}; top: {top}"
    else:
        mark = "FAIL"
        detail = f"need @{min(case.at, k)}  expected {expect}; appeared {found}; top: {top}"
    line = f"{mark}  {name}  {detail}"
    if case.note:
        line += f"\n      {case.note}"
    return line


def _ratio(counts: tuple[int, int]) -> str:
    got, total = counts
    if total == 0:
        return "n/a"
    return f"{got}/{total} ({got / total:.3f})"


def _counted(count: int, singular: str) -> str:
    word = singular if count == 1 else f"{singular}s"
    return f"{count} {word}"


def _case_from_dict(item: object, path: Path, index: int, *, default_at: int) -> GoldCase:
    where = f"{path} case {index}"
    if not isinstance(item, dict):
        raise EvalError(f"{where} is not an object")
    unknown = set(item) - _CASE_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise EvalError(f"{where} has unknown fields: {names}")
    case_id = _text_field(item, "id", where)
    question = _text_field(item, "question", where)
    tier = _text_field(item, "tier", where)
    if tier not in _TIERS:
        raise EvalError(f"{where} tier must be required or challenge, not {tier!r}")
    expect_raw = item.get("expect")
    if not isinstance(expect_raw, list) or not expect_raw:
        raise EvalError(f"{where} needs a non-empty expect array")
    expect: list[str] = []
    for raw_path in expect_raw:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise EvalError(f"{where} expect entries must be path strings")
        cleaned = raw_path.strip()
        if cleaned.startswith("/") or ".." in Path(cleaned).parts:
            raise EvalError(f"{where} expect path must be repo-relative: {cleaned}")
        if cleaned in expect:
            raise EvalError(f"{where} lists {cleaned} more than once")
        expect.append(cleaned)
    if "at" in item and item["at"] is not None:
        at = _positive_int(item["at"], f"{where} at")
    else:
        at = default_at
    note = item.get("note", "")
    if not isinstance(note, str):
        raise EvalError(f"{where} note must be a string")
    return GoldCase(
        id=case_id,
        question=question,
        expect=tuple(expect),
        tier=tier,
        at=at,
        note=note.strip(),
    )


def _text_field(data: dict, key: str, where: Path | str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvalError(f"{where} needs a non-empty string field {key!r}")
    return value.strip()


def _positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise EvalError(f"{where} must be an integer >= 1")
    return value
