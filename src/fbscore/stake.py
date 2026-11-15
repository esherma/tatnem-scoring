"""Per-principal stake locks (spec §6)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ._types import Pmf
from .grid import OutcomeSpace
from .losses import loss_profile
from .money import ceil_minor

__all__ = ["Step", "lock_minor", "net_by_outcome", "worst_case_loss"]


@dataclass(frozen=True, slots=True)
class Step:
    """One of a principal's submissions: the MSR state before it, its report, and its c."""

    prev: Pmf
    report: Pmf
    c: float

    @property
    def is_noop(self) -> bool:
        return self.report == self.prev


def net_by_outcome(space: OutcomeSpace, steps: Sequence[Step]) -> tuple[float, ...]:
    """``net_i(x) = sum_k c_k (L(q_{k-1}, x) - L(p_k, x))`` at each representative outcome."""
    cache: dict[Pmf, tuple[float, ...]] = {}

    def profile(pmf: Pmf) -> tuple[float, ...]:
        if pmf not in cache:
            cache[pmf] = loss_profile(space, pmf)
        return cache[pmf]

    live = [(s.c, profile(s.prev), profile(s.report)) for s in steps if not s.is_noop]
    return tuple(
        math.fsum(c * (before[b] - after[b]) for c, before, after in live)
        for b in range(len(space.representative_outcomes))
    )


def worst_case_loss(space: OutcomeSpace, steps: Sequence[Step]) -> float:
    """``max(0, max_x -net_i(x))`` in pool units. Exact, because losses are constant per bin."""
    return max(0.0, *(-n for n in net_by_outcome(space, steps)))


def lock_minor(space: OutcomeSpace, steps: Sequence[Step], *, min_lock_minor_per_submission: int) -> int:
    """The principal's lock on one question, in minor units (c-cr).

    ``max(ceil_minor(worst_case_loss), min_lock_minor_per_submission * n_non_noop)``.
    It depends only on this principal's own steps.
    """
    if min_lock_minor_per_submission < 0:
        raise ValueError("min_lock_minor_per_submission must be >= 0")
    n_live = sum(1 for s in steps if not s.is_noop)
    return max(ceil_minor(worst_case_loss(space, steps)), min_lock_minor_per_submission * n_live)
