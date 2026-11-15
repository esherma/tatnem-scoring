"""``fbscore verify`` round-trips on the published fixtures and catches tampering."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from fbscore.export import write_parquet
from fbscore.verify import main, verify_document

from .fixtures_gen import FIXTURES, dump, fixtures

SCHEMA = json.loads((Path(__file__).parents[1] / "schemas" / "settlement-export.v0.schema.json").read_text())
NAMES = sorted(fixtures())


def load(name: str) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads((FIXTURES / name).read_text())
    return doc


@pytest.mark.parametrize("name", NAMES)
def test_committed_fixtures_are_golden(name: str) -> None:
    assert (FIXTURES / name).read_text() == dump(fixtures()[name]), (
        "rules output changed; regenerate only for a new rule"
    )


@pytest.mark.parametrize("name", NAMES)
def test_fixture_matches_schema(name: str) -> None:
    jsonschema.validate(load(name), SCHEMA)


@pytest.mark.parametrize("name", NAMES)
def test_verify_json_ok(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", str(FIXTURES / name)]) == 0
    assert capsys.readouterr().out.startswith("OK")


@pytest.mark.parametrize("name", NAMES)
def test_verify_parquet_round_trip(name: str, tmp_path: Path) -> None:
    path = tmp_path / "export.parquet"
    write_parquet(load(name), path)
    assert main(["verify", str(path)]) == 0


def _tamper(doc: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    d = copy.deepcopy(doc)
    node: Any = d
    keys = path.split(".")
    for k in keys[:-1]:
        node = node[int(k)] if isinstance(node, list) else node[k]
    last = keys[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value
    return d


@pytest.mark.parametrize(
    ("name", "path", "value", "needle"),
    [
        ("binary_cash_haircut.json", "settlement.lines.0.prize_minor", 28_132, "prize_minor"),
        ("binary_cash_haircut.json", "settlement.refund_minor", 0, "refund_minor"),
        ("binary_cash_haircut.json", "settlement.haircut", 1.0, "haircut"),
        ("binary_cash_haircut.json", "settlement.lines.1.credits_burned_minor", 1, "credits_burned_minor"),
        ("binary_cash_haircut.json", "settlement.lines.1.lock_minor", 1, "lock_minor"),
        ("binary_cash_haircut.json", "submissions.3.c", 2880.0, ".c"),
        ("binary_cash_haircut.json", "submissions.3.pmf", [0.2, 0.8], "points"),
        ("binary_cash_haircut.json", "question.c_max", 1.0, "c_max"),
        ("binary_cash_haircut.json", "question.scoring.rule", "crps", "rule"),
        ("binary_cash_haircut.json", "outcome.x", 0, "points"),
        ("binary_cash_haircut.json", "settlement.payout_mode", "sandbox", "payout_mode"),
        ("numeric_credit_prize.json", "settlement.lines.1.credits_earned_minor", 10**9, "credits_earned_minor"),
        ("numeric_sandbox.json", "settlement.scores.0.points", 1.0, "points"),
    ],
)
def test_verify_detects_tampering(name: str, path: str, value: Any, needle: str) -> None:
    result = verify_document(_tamper(load(name), path, value))
    assert not result.ok
    assert any(needle in m for m in result.mismatches), result.mismatches


def test_verify_detects_missing_and_extra_rows() -> None:
    doc = load("binary_cash_haircut.json")
    d = copy.deepcopy(doc)
    d["settlement"]["scores"].pop()
    d["settlement"]["lines"].pop()
    d["settlement"]["scores"].append({"seq": 99, "points": 0.0, "norm_points": 0.0})
    d["settlement"]["lines"].append({**doc["settlement"]["lines"][0], "principal_id": "ghost"})
    mismatches = verify_document(d).mismatches
    assert any("missing" in m for m in mismatches)
    assert any("ghost" in m for m in mismatches)
    assert any("99" in m for m in mismatches)


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(_tamper(load("binary_cash_haircut.json"), "settlement.refund_minor", 5)))
    assert main(["verify", str(bad)]) == 1
    assert "MISMATCH" in capsys.readouterr().out
    for content in ("not json", "[]", json.dumps({"format": "other"})):
        bad.write_text(content)
        assert main(["verify", str(bad)]) == 2
    assert main(["verify", str(tmp_path / "missing.json")]) == 2


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("question.outcome_space", {"type": "date"}),
        ("question.prior", [0.9, 0.9]),
        ("question.scoring.scoring_version", "9.9.9"),
        ("question.payout_mode", "lottery"),
        ("question.open_at", "yesterday"),
        ("question.open_at", "2026-01-01T00:00:00"),
        ("question.open_at", 5),
        ("question.prize_pool_minor", True),
        ("submissions.0.pmf", [2.0, -1.0]),
        ("submissions.0.submitted_at", "2027-01-01T00:00:00Z"),
        ("submissions.1.seq", 0),
        ("submissions", {}),
        ("submissions.0", []),
        ("outcome", {}),
    ],
)
def test_malformed_exports_exit_2(path: str, value: Any, tmp_path: Path) -> None:
    f = tmp_path / "x.json"
    f.write_text(json.dumps(_tamper(load("binary_cash_haircut.json"), path, value)))
    assert main(["verify", str(f)]) == 2


def test_parquet_without_metadata_is_rejected(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    f = tmp_path / "plain.parquet"
    pq.write_table(pa.table({"a": [1]}), f)
    assert main(["verify", str(f)]) == 2


def test_module_entry_point() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "fbscore", "verify", str(FIXTURES / NAMES[0])],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
