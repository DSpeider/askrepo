import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from askrepo.cli import app
from askrepo.eval import EvalError, format_report, load_gold, run_eval


runner = CliRunner()
ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "evals" / "tiny_repo.json"


def test_required_cases_hit_and_challenge_cases_miss(tiny_repo: Path) -> None:
    report = run_eval(repo=tiny_repo, cases=GOLD, check_baseline=True)

    assert report.k == 5
    assert report.retriever == "lexical"
    assert report.problems() == (), format_report(report)
    required_at_1, required_total = report.hits("required", 1)
    required_at_5, _ = report.hits("required", 5)
    challenge_at_5, challenge_total = report.hits("challenge", 5)
    assert required_total >= 10
    assert challenge_total >= 2
    assert required_at_5 == required_total
    assert challenge_at_5 == 0
    # The call site stays in the window, but the definition still ranks first.
    assert required_at_1 == required_total - 1
    caller = next(result for result in report.results if result.case.id == "caller-apply-tax")
    assert caller.rank is not None and caller.rank > 1
    for case_id in ("gateway-hand-off", "payment-message-callee", "gross-up-paraphrase"):
        result = next(item for item in report.results if item.case.id == case_id)
        assert result.rank is None
        assert result.case.expect[0] not in result.top_paths


def test_eval_command_prints_hit_rates_and_passes(tiny_repo: Path, monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    result = runner.invoke(app, ["eval", str(tiny_repo)])

    assert result.exit_code == 0, result.output
    assert "hit@1" in result.output
    assert "hit@5" in result.output
    assert "PASS  define-apply-tax" in result.output
    assert "PASS  caller-apply-tax" in result.output
    assert "MISS  gateway-hand-off" in result.output
    assert "MISS  gross-up-paraphrase" in result.output
    assert "Required cases passed" in result.output
    assert "FAIL" not in result.output
    assert "BEAT" not in result.output


def test_eval_command_from_checkout_defaults(monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    result = runner.invoke(app, ["eval"])

    assert result.exit_code == 0, result.output
    assert "retriever=lexical, k=5" in result.output


def test_eval_command_fails_when_a_required_case_misses(tiny_repo: Path, tmp_path: Path) -> None:
    cases = tmp_path / "gold.json"
    cases.write_text(
        json.dumps(
            {
                "name": "broken",
                "repo": str(tiny_repo),
                "k": 5,
                "cases": [
                    {
                        "id": "missing-name",
                        "question": "zzzz_quantum_flux",
                        "expect": ["src/billing/invoice.py"],
                        "tier": "required",
                        "at": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["eval", "--cases", str(cases)])

    assert result.exit_code == 1
    assert "FAIL  missing-name" in result.output
    assert "missed the rank cutoff" in result.output


def test_eval_rejects_an_expect_path_that_was_not_indexed(tiny_repo: Path, tmp_path: Path) -> None:
    cases = tmp_path / "gold.json"
    cases.write_text(
        json.dumps(
            {
                "name": "typo",
                "repo": str(tiny_repo),
                "k": 5,
                "cases": [
                    {
                        "id": "not-a-file",
                        "question": "where is apply_tax defined",
                        "expect": ["src/billing/nope.py"],
                        "tier": "challenge",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["eval", "--cases", str(cases)])

    assert result.exit_code == 1
    assert "not in the index" in result.output
    assert "src/billing/nope.py" in result.output


def test_load_gold_rejects_a_duplicate_id(tmp_path: Path) -> None:
    path = tmp_path / "gold.json"
    case = {
        "id": "same",
        "question": "where is apply_tax defined",
        "expect": ["src/billing/tax.py"],
        "tier": "required",
    }
    path.write_text(
        json.dumps({"name": "dup", "repo": "examples/tiny_repo", "k": 5, "cases": [case, case]}),
        encoding="utf-8",
    )

    with pytest.raises(EvalError, match="duplicate case id"):
        load_gold(path)


def test_shipped_gold_file_loads() -> None:
    gold = load_gold(GOLD)

    assert gold.k == 5
    assert gold.retriever == "lexical"
    assert len(gold.cases) >= 15
    assert {case.tier for case in gold.cases} == {"required", "challenge"}
