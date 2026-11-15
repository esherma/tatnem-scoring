"""fbscore: the Tatnem market-scoring-rule library (distribution ``tatnem-scoring``).

Pure, deterministic functions for loss computation, canonicalization, MSR payouts,
stake locks, settlement and the delivered-consensus aggregation primitive. See
``spec/scoring.md`` for the normative specification.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .abuse import ABUSE_PARAMETER_NAMES
from .aggregate import Consensus, Member, cap_principal_weights, delivered_consensus, weighted_median
from .canonicalize import CanonicalizationError, canonicalize, mix_prior, uniform
from .grid import BinarySpace, Grid, NumericSpace, OutcomeSpace, space_from_dict
from .losses import brier, crps, loss, max_loss
from .money import ceil_minor, floor_minor
from .msr import Score, c_max, decayed_c, norm_points, score_chain, score_submission
from .settlement import (
    PayoutMode,
    ScoreRow,
    Settlement,
    SettlementLine,
    Submission,
    score_submissions,
    settle_cash,
    settle_credit_prize,
    settle_sandbox,
)
from .stake import Step, lock_minor, net_by_outcome, worst_case_loss

__all__ = [
    "ABUSE_PARAMETER_NAMES",
    "BinarySpace",
    "CanonicalizationError",
    "Consensus",
    "Grid",
    "Member",
    "NumericSpace",
    "OutcomeSpace",
    "PayoutMode",
    "Score",
    "ScoreRow",
    "Settlement",
    "SettlementLine",
    "Step",
    "Submission",
    "__version__",
    "brier",
    "c_max",
    "canonicalize",
    "cap_principal_weights",
    "ceil_minor",
    "crps",
    "decayed_c",
    "delivered_consensus",
    "floor_minor",
    "lock_minor",
    "loss",
    "max_loss",
    "mix_prior",
    "net_by_outcome",
    "norm_points",
    "score_chain",
    "score_submission",
    "score_submissions",
    "settle_cash",
    "settle_credit_prize",
    "settle_sandbox",
    "space_from_dict",
    "uniform",
    "weighted_median",
    "worst_case_loss",
]
