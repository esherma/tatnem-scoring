"""Build the settlement-export fixtures in ``tests/fixtures``.

Run ``python tests/fixtures_gen.py`` to regenerate. ``test_verify.py`` checks that the
committed files equal what this produces, so the fixtures double as golden outputs.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import fbscore as fb
from fbscore.export import build_export
from fbscore.rules import v1

FIXTURES = Path(__file__).parent / "fixtures"
OPEN = datetime(2026, 1, 1, tzinfo=UTC)
CLOSE = OPEN + timedelta(days=10)


def _doc(
    qid: str,
    space: fb.OutcomeSpace,
    prior: tuple[float, ...],
    mode: fb.PayoutMode,
    pool_minor: int,
    reports: list[tuple[str, str, tuple[float, ...], timedelta]],
    x: float,
    *,
    k: float = 1.0,
    decay: float = 0.5,
    caps: dict[str, int] | None = None,
) -> dict[str, Any]:
    cmax = v1.c_max(space, prior, k=k, reserve=0.1, prize_pool=pool_minor / 100)
    subs: list[fb.Submission] = []
    rows: list[dict[str, Any]] = []
    for i, (principal, forecaster, pmf, dt) in enumerate(reports, start=1):
        at = OPEN + dt
        c = v1.c_at(c_max=cmax, decay=decay, open_at=OPEN, close_at=CLOSE, at=at)
        subs.append(fb.Submission(i, principal, forecaster, pmf, c))
        rows.append(
            {"seq": i, "forecaster_id": forecaster, "principal_id": principal, "submitted_at": at, "pmf": pmf, "c": c}
        )
    if mode == "cash":
        st = v1.settle_cash(space, prior, subs, x, prize_pool_minor=pool_minor)
    elif mode == "credit_prize":
        assert caps is not None
        st = v1.settle_credit_prize(space, prior, subs, x, remaining_mint_cap_minor=caps)
    else:
        st = v1.settle_sandbox(space, prior, subs, x)
    question = {
        "id": qid,
        "outcome_space": space.to_dict(),
        "prior": list(prior),
        "scoring": {"rule": v1.rule_name(space), "k": k, "reserve": 0.1, "decay": decay, "scoring_version": "0.1.0"},
        "c_max": cmax,
        "open_at": OPEN,
        "close_at": CLOSE,
        "payout_mode": mode,
        "prize_pool_minor": pool_minor,
    }
    return build_export(question=question, submissions=rows, settlement=st)


def fixtures() -> dict[str, dict[str, Any]]:
    b = fb.BinarySpace()
    p = lambda v: fb.canonicalize(b, "binary_p", v)  # noqa: E731
    n = fb.NumericSpace(fb.Grid(0.0, 100.0, 20))
    q = lambda *pairs: fb.canonicalize(n, "quantiles", [list(x) for x in pairs])  # noqa: E731
    numeric_reports = [
        ("a", "alpha", q((0.1, 20.0), (0.5, 40.0), (0.9, 70.0)), timedelta(hours=3)),
        ("b", "bravo", q((0.05, 30.0), (0.5, 55.0), (0.95, 80.0)), timedelta(days=2)),
        ("a", "alpha-2", q((0.1, 35.0), (0.5, 50.0), (0.9, 62.0)), timedelta(days=4, hours=6)),
        ("c", "charlie", q((0.25, 45.0), (0.5, 52.0), (0.75, 58.0)), timedelta(days=9, hours=23)),
    ]
    return {
        "binary_cash_haircut.json": _doc(
            "q-binary-cash",
            b,
            fb.uniform(b),
            "cash",
            40_000,
            [
                ("a", "A", p(0.70), timedelta(0)),
                ("b", "B", p(0.65), timedelta(days=1)),
                ("c", "C", p(0.65), timedelta(days=2)),
                ("d", "D", p(0.90), timedelta(days=8)),
            ],
            1,
            k=2.0,
        ),
        "numeric_credit_prize.json": _doc(
            "q-numeric-credit",
            n,
            v1.prior_from_sponsor(n, fb.canonicalize(n, "quantiles", [[0.25, 30.0], [0.5, 50.0], [0.75, 70.0]])),
            "credit_prize",
            50_000,
            numeric_reports,
            53.7,
            caps={"a": 1_500, "b": 99_999, "c": 99_999},
        ),
        "numeric_sandbox.json": _doc("q-numeric-sandbox", n, fb.uniform(n), "sandbox", 10_000, numeric_reports, 101.0),
    }


def dump(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    FIXTURES.mkdir(exist_ok=True)
    for name, doc in fixtures().items():
        (FIXTURES / name).write_text(dump(doc), encoding="utf-8")
