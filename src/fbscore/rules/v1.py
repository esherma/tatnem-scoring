"""Contest rules v1. FROZEN: never edit behavior in place; add ``v2`` instead.

Public contest constants for every question settled under v1, plus the functions that
bind them. Per-question parameters (grid, prior, k, reserve, decay, pool, times) come
from the question. Abuse-relevant parameters stay required arguments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final

from .. import losses as _losses
from .. import msr as _msr
from .. import settlement as _settlement
from .. import stake as _stake
from .._types import Pmf
from ..canonicalize import mix_prior
from ..grid import NumericSpace, OutcomeSpace

RULE_ID: Final = "v1"
BINARY_LOSS: Final = "brier"
NUMERIC_LOSS: Final = "crps"
ALPHA: Final = 0.5
"""Cash questions: credits earned per positive net point."""
MIN_LOCK_MINOR_PER_SUBMISSION: Final = 100
"""Every non-noop submission locks at least 1 credit (100 c-cr)."""
SPONSOR_PRIOR_WEIGHT: Final = 0.9
"""A sponsor prior is mixed 0.9 * sponsor + 0.1 * uniform."""


def rule_name(space: OutcomeSpace) -> str:
    return NUMERIC_LOSS if isinstance(space, NumericSpace) else BINARY_LOSS


def loss(space: OutcomeSpace, pmf: Pmf, x: float) -> float:
    return _losses.loss(space, pmf, x)


def prior_from_sponsor(space: OutcomeSpace, sponsor_pmf: Pmf) -> Pmf:
    return mix_prior(space, sponsor_pmf, sponsor_weight=SPONSOR_PRIOR_WEIGHT)


def c_max(space: OutcomeSpace, prior: Pmf, *, k: float, reserve: float, prize_pool: float) -> float:
    return _msr.c_max(k=k, reserve=reserve, prize_pool=prize_pool, max_prior_loss=_losses.max_loss(space, prior))


def c_at(
    *, c_max: float, decay: float, open_at: datetime | float, close_at: datetime | float, at: datetime | float
) -> float:
    return _msr.decayed_c(c_max=c_max, decay=decay, open_at=open_at, close_at=close_at, at=at)


def lock_minor(space: OutcomeSpace, steps: Sequence[_stake.Step]) -> int:
    return _stake.lock_minor(space, steps, min_lock_minor_per_submission=MIN_LOCK_MINOR_PER_SUBMISSION)


def settle_cash(
    space: OutcomeSpace, prior: Pmf, submissions: Sequence[_settlement.Submission], x: float, *, prize_pool_minor: int
) -> _settlement.Settlement:
    return _settlement.settle_cash(
        space,
        prior,
        submissions,
        x,
        prize_pool_minor=prize_pool_minor,
        alpha=ALPHA,
        min_lock_minor_per_submission=MIN_LOCK_MINOR_PER_SUBMISSION,
    )


def settle_credit_prize(
    space: OutcomeSpace,
    prior: Pmf,
    submissions: Sequence[_settlement.Submission],
    x: float,
    *,
    remaining_mint_cap_minor: Mapping[str, int],
) -> _settlement.Settlement:
    return _settlement.settle_credit_prize(
        space,
        prior,
        submissions,
        x,
        min_lock_minor_per_submission=MIN_LOCK_MINOR_PER_SUBMISSION,
        remaining_mint_cap_minor=remaining_mint_cap_minor,
    )


def settle_sandbox(
    space: OutcomeSpace, prior: Pmf, submissions: Sequence[_settlement.Submission], x: float
) -> _settlement.Settlement:
    return _settlement.settle_sandbox(space, prior, submissions, x)
