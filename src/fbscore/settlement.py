"""Deterministic settlement at resolution (spec §7) for cash, credit-prize and sandbox questions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from ._types import Pmf
from .grid import OutcomeSpace
from .losses import max_loss
from .money import MINOR_PER_MAJOR, ceil_minor, floor_minor
from .msr import score_submission
from .stake import Step, lock_minor

__all__ = [
    "PayoutMode",
    "ScoreRow",
    "Settlement",
    "SettlementLine",
    "Submission",
    "score_submissions",
    "settle_cash",
    "settle_credit_prize",
    "settle_sandbox",
]

PayoutMode = Literal["cash", "credit_prize", "sandbox"]


@dataclass(frozen=True, slots=True)
class Submission:
    """An accepted, non-voided submission. ``c`` is the decayed scale stored at acceptance."""

    seq: int
    principal_id: str
    forecaster_id: str
    pmf: Pmf
    c: float


@dataclass(frozen=True, slots=True)
class ScoreRow:
    seq: int
    principal_id: str
    forecaster_id: str
    loss_before: float
    loss_report: float
    c: float
    points: float
    norm_points: float
    is_noop: bool


@dataclass(frozen=True, slots=True)
class SettlementLine:
    """Per-principal result. Amounts are integer minor units (cents or centi-credits)."""

    principal_id: str
    net_points: float
    prize_minor: int
    credits_burned_minor: int
    credits_earned_minor: int
    lock_minor: int
    lock_released_minor: int


@dataclass(frozen=True, slots=True)
class Settlement:
    payout_mode: PayoutMode
    outcome: float
    scores: tuple[ScoreRow, ...]
    lines: tuple[SettlementLine, ...]
    net_total: float
    gross_positive: float
    haircut: float
    prizes_total_minor: int
    refund_minor: int


def _validate(space: OutcomeSpace, prior: Pmf, submissions: Sequence[Submission]) -> None:
    if len(prior) != space.size:
        raise ValueError("prior length does not match the outcome space")
    prev_seq: int | None = None
    prev_c = math.inf
    for s in submissions:
        if len(s.pmf) != space.size:
            raise ValueError(f"submission {s.seq}: pmf length does not match the outcome space")
        if prev_seq is not None and s.seq <= prev_seq:
            raise ValueError("submissions must be in strictly increasing seq order")
        if not (0.0 <= s.c <= prev_c):
            raise ValueError(f"submission {s.seq}: c must be >= 0 and non-increasing in seq order")
        prev_seq, prev_c = s.seq, s.c


def score_submissions(space: OutcomeSpace, prior: Pmf, submissions: Sequence[Submission], x: float) -> list[ScoreRow]:
    """Step 1: score every submission against the MSR state just before it."""
    _validate(space, prior, submissions)
    lstar = max_loss(space, prior)
    rows: list[ScoreRow] = []
    prev = prior
    for s in submissions:
        sc = score_submission(space, prev=prev, report=s.pmf, c=s.c, x=x, max_prior_loss=lstar)
        rows.append(
            ScoreRow(
                s.seq,
                s.principal_id,
                s.forecaster_id,
                sc.loss_before,
                sc.loss_report,
                sc.c,
                sc.points,
                sc.norm_points,
                sc.is_noop,
            )
        )
        prev = s.pmf
    return rows


def _steps_by_principal(prior: Pmf, submissions: Sequence[Submission]) -> dict[str, list[Step]]:
    steps: dict[str, list[Step]] = {}
    prev = prior
    for s in submissions:
        steps.setdefault(s.principal_id, []).append(Step(prev, s.pmf, s.c))
        prev = s.pmf
    return steps


def _nets(rows: Sequence[ScoreRow]) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r.principal_id, []).append(r.points)
    return {p: math.fsum(v) for p, v in sorted(by.items())}


def _locks(
    space: OutcomeSpace, prior: Pmf, submissions: Sequence[Submission], min_lock_minor_per_submission: int
) -> dict[str, int]:
    return {
        p: lock_minor(space, st, min_lock_minor_per_submission=min_lock_minor_per_submission)
        for p, st in _steps_by_principal(prior, submissions).items()
    }


def _burned(net: float, lock: int, principal_id: str) -> int:
    burned = ceil_minor(max(-net, 0.0))
    if burned > lock:
        raise AssertionError(f"principal {principal_id}: burn {burned} exceeds lock {lock}")
    return burned


def settle_cash(
    space: OutcomeSpace,
    prior: Pmf,
    submissions: Sequence[Submission],
    x: float,
    *,
    prize_pool_minor: int,
    alpha: float,
    min_lock_minor_per_submission: int,
) -> Settlement:
    """Cash question: positive nets paid from the pool (haircut if G > P), negative nets burn credits."""
    if prize_pool_minor < 0:
        raise ValueError("prize_pool_minor must be >= 0")
    rows = score_submissions(space, prior, submissions, x)
    nets = _nets(rows)
    locks = _locks(space, prior, submissions, min_lock_minor_per_submission)
    pool = prize_pool_minor / MINOR_PER_MAJOR
    gross = math.fsum(max(n, 0.0) for n in nets.values())
    h = 1.0 if gross <= pool else pool / gross
    lines: list[SettlementLine] = []
    for p, net in nets.items():
        prize = floor_minor(h * net) if net > 0 else 0
        burned = _burned(net, locks[p], p)
        earned = floor_minor(alpha * max(net, 0.0))
        lines.append(SettlementLine(p, net, prize, burned, earned, locks[p], locks[p] - burned))
    paid = sum(line.prize_minor for line in lines)
    if paid > prize_pool_minor:
        raise AssertionError("prizes exceed the prize pool")
    return Settlement(
        "cash", x, tuple(rows), tuple(lines), math.fsum(nets.values()), gross, h, paid, prize_pool_minor - paid
    )


def settle_credit_prize(
    space: OutcomeSpace,
    prior: Pmf,
    submissions: Sequence[Submission],
    x: float,
    *,
    min_lock_minor_per_submission: int,
    remaining_mint_cap_minor: Mapping[str, int],
) -> Settlement:
    """Credit-prize question: positive nets mint credits 1:1 up to each principal's remaining cap."""
    rows = score_submissions(space, prior, submissions, x)
    nets = _nets(rows)
    locks = _locks(space, prior, submissions, min_lock_minor_per_submission)
    lines: list[SettlementLine] = []
    for p, net in nets.items():
        if p not in remaining_mint_cap_minor:
            raise ValueError(f"no remaining mint cap supplied for principal {p}")
        cap = remaining_mint_cap_minor[p]
        if cap < 0:
            raise ValueError(f"negative mint cap for principal {p}")
        minted = min(floor_minor(max(net, 0.0)), cap)
        burned = _burned(net, locks[p], p)
        lines.append(SettlementLine(p, net, 0, burned, minted, locks[p], locks[p] - burned))
    gross = math.fsum(max(n, 0.0) for n in nets.values())
    return Settlement("credit_prize", x, tuple(rows), tuple(lines), math.fsum(nets.values()), gross, 1.0, 0, 0)


def settle_sandbox(space: OutcomeSpace, prior: Pmf, submissions: Sequence[Submission], x: float) -> Settlement:
    """Sandbox question: scores for display only. No locks, no money, no credits."""
    rows = score_submissions(space, prior, submissions, x)
    nets = _nets(rows)
    lines = tuple(SettlementLine(p, net, 0, 0, 0, 0, 0) for p, net in nets.items())
    gross = math.fsum(max(n, 0.0) for n in nets.values())
    return Settlement("sandbox", x, tuple(rows), lines, math.fsum(nets.values()), gross, 1.0, 0, 0)
