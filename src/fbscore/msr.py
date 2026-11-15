"""Market scoring rule: budget scale, time decay and per-submission points (spec §4-§5)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from ._types import Pmf
from .grid import OutcomeSpace
from .losses import loss, max_loss

__all__ = ["Score", "c_max", "decayed_c", "elapsed_fraction", "norm_points", "score_chain", "score_submission"]


def c_max(*, k: float, reserve: float, prize_pool: float, max_prior_loss: float) -> float:
    """``c_max = k (1 - r) P / L*_prior``: pool units per unit of loss improvement."""
    if not (k > 0 and math.isfinite(k)):
        raise ValueError("k must be positive")
    if not 0.0 <= reserve < 1.0:
        raise ValueError("reserve must be in [0, 1)")
    if not (prize_pool >= 0 and math.isfinite(prize_pool)):
        raise ValueError("prize_pool must be >= 0")
    if not max_prior_loss > 0:
        raise ValueError("prior has zero max loss")
    return k * (1.0 - reserve) * prize_pool / max_prior_loss


def elapsed_fraction(*, open_at: datetime | float, close_at: datetime | float, at: datetime | float) -> float:
    """``(at - open_at) / (close_at - open_at)``, required to lie in [0, 1]."""
    if isinstance(open_at, datetime) and isinstance(close_at, datetime) and isinstance(at, datetime):
        if close_at <= open_at:
            raise ValueError("close_at must be after open_at")
        frac = (at - open_at) / (close_at - open_at)
    elif not isinstance(open_at, datetime) and not isinstance(close_at, datetime) and not isinstance(at, datetime):
        if not close_at > open_at:
            raise ValueError("close_at must be after open_at")
        frac = (at - open_at) / (close_at - open_at)
    else:
        raise TypeError("open_at, close_at and at must all be datetimes or all be numbers")
    if not 0.0 <= frac <= 1.0:
        raise ValueError("submission time is outside the open window")
    return frac


def decayed_c(
    *, c_max: float, decay: float, open_at: datetime | float, close_at: datetime | float, at: datetime | float
) -> float:
    """``c_k = c_max * (1 - decay * elapsed_fraction)``; linear from c_max to (1 - decay) c_max."""
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be in [0, 1]")
    return c_max * (1.0 - decay * elapsed_fraction(open_at=open_at, close_at=close_at, at=at))


def norm_points(loss_before: float, loss_report: float, max_prior_loss: float) -> float:
    """Scale-free information contribution ``(L_before - L_report) / L*_prior``."""
    return (loss_before - loss_report) / max_prior_loss


@dataclass(frozen=True, slots=True)
class Score:
    """One submission scored at outcome x."""

    loss_before: float
    loss_report: float
    c: float
    points: float
    norm_points: float
    is_noop: bool


def score_submission(
    space: OutcomeSpace, *, prev: Pmf, report: Pmf, c: float, x: float, max_prior_loss: float
) -> Score:
    """``points = c (L(prev, x) - L(report, x))``; a bitwise copy of ``prev`` scores exactly 0."""
    is_noop = report == prev
    lb = loss(space, prev, x)
    if is_noop:
        return Score(lb, lb, c, 0.0, 0.0, True)
    lr = loss(space, report, x)
    return Score(lb, lr, c, c * (lb - lr), norm_points(lb, lr, max_prior_loss), False)


def score_chain(space: OutcomeSpace, prior: Pmf, reports: Sequence[tuple[Pmf, float]], x: float) -> list[Score]:
    """Score ``[(pmf, c), ...]`` in seq order against outcome ``x``, starting from ``prior``."""
    lstar = max_loss(space, prior)
    out: list[Score] = []
    prev = prior
    for pmf, c in reports:
        out.append(score_submission(space, prev=prev, report=pmf, c=c, x=x, max_prior_loss=lstar))
        prev = pmf
    return out
