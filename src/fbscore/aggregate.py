"""Delivered-consensus aggregation primitive (spec §11, method shape ``wmedian-v1``).

This module contains only the aggregation primitive. How member weights are computed
(track record, recency) and the values of the cap and participation thresholds are
platform configuration: every one of them is a required argument here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ._types import Pmf
from .grid import BinarySpace, OutcomeSpace
from .losses import cdf_at_interior_edges

__all__ = ["Consensus", "Member", "cap_principal_weights", "delivered_consensus", "weighted_median"]


@dataclass(frozen=True, slots=True)
class Member:
    """A forecaster's latest report and its (already computed) weight."""

    forecaster_id: str
    principal_id: str
    weight: float
    pmf: Pmf


@dataclass(frozen=True, slots=True)
class Consensus:
    pmf: Pmf
    low_participation: bool
    cap_applied: bool
    weights: tuple[tuple[str, float], ...]
    """Normalized effective weight per forecaster, sorted by forecaster id."""


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Lower weighted median: the smallest value whose cumulative weight reaches half the total.

    Independent of input order: pairs are sorted by (value, weight) before accumulating.
    """
    if len(values) != len(weights) or not values:
        raise ValueError("need equally many values and weights, at least one")
    if any(not (math.isfinite(w) and w > 0) for w in weights):
        raise ValueError("weights must be finite and > 0")
    pairs = sorted(zip(values, weights, strict=True))
    total = math.fsum(w for _, w in pairs)
    acc: list[float] = []
    for v, w in pairs:
        acc.append(w)
        if 2.0 * math.fsum(acc) >= total:
            return v
    return pairs[-1][0]  # pragma: no cover - unreachable: the last prefix equals the total


def cap_principal_weights(members: Sequence[Member], *, principal_cap: float) -> dict[str, float]:
    """Normalized weights such that no principal holds more than ``principal_cap`` of the total.

    Water-filling: principals over the cap are pinned at exactly the cap, the rest share the
    remainder in proportion to their raw weight, repeated until no principal exceeds the cap.
    Weights within a principal keep their proportions.
    """
    if not 0.0 < principal_cap <= 1.0:
        raise ValueError("principal_cap must be in (0, 1]")
    if any(not (math.isfinite(m.weight) and m.weight > 0) for m in members):
        raise ValueError("member weights must be finite and > 0")
    by_p: dict[str, list[Member]] = {}
    for m in sorted(members, key=lambda m: (m.principal_id, m.forecaster_id)):
        by_p.setdefault(m.principal_id, []).append(m)
    if len(by_p) * principal_cap < 1.0:
        raise ValueError("cap is infeasible: n_principals * principal_cap < 1")
    raw = {p: math.fsum(m.weight for m in ms) for p, ms in by_p.items()}
    order = sorted(raw, key=lambda p: (-raw[p], p))
    share: dict[str, float] = {}
    for n_capped in range(len(order) + 1):
        rest = order[n_capped:]
        remaining = 1.0 - n_capped * principal_cap
        u = math.fsum(raw[p] for p in rest)
        if not rest or raw[rest[0]] / u * remaining <= principal_cap:
            share = dict.fromkeys(order[:n_capped], principal_cap)
            share.update({p: raw[p] / u * remaining for p in rest})
            break
    out: dict[str, float] = {}
    for p, ms in by_p.items():
        for m in ms:
            out[m.forecaster_id] = share[p] * m.weight / raw[p]
    return out


def delivered_consensus(
    space: OutcomeSpace,
    members: Sequence[Member],
    *,
    msr_state: Pmf,
    principal_cap: float,
    cap_min_principals: int,
    low_participation_min_principals: int,
) -> Consensus:
    """Weighted median of members' reports (binary p, or CDF per interior edge).

    * Fewer than ``low_participation_min_principals`` distinct principals: return
      ``msr_state`` flagged ``low_participation``.
    * The per-principal cap applies only with at least ``cap_min_principals`` principals.
    """
    if len({m.forecaster_id for m in members}) != len(members):
        raise ValueError("one member per forecaster")
    if any(len(m.pmf) != space.size for m in members) or len(msr_state) != space.size:
        raise ValueError("pmf length does not match the outcome space")
    n_principals = len({m.principal_id for m in members})
    if n_principals < low_participation_min_principals or not members:
        return Consensus(tuple(msr_state), True, False, ())
    cap_applied = n_principals >= cap_min_principals
    if cap_applied:
        w = cap_principal_weights(members, principal_cap=principal_cap)
    else:
        if any(not (math.isfinite(m.weight) and m.weight > 0) for m in members):
            raise ValueError("member weights must be finite and > 0")
        total = math.fsum(m.weight for m in sorted(members, key=lambda m: m.forecaster_id))
        w = {m.forecaster_id: m.weight / total for m in members}
    ms = sorted(members, key=lambda m: m.forecaster_id)
    ws = [w[m.forecaster_id] for m in ms]
    if isinstance(space, BinarySpace):
        p = weighted_median([m.pmf[1] for m in ms], ws)
        pmf: Pmf = (1.0 - p, p)
    else:
        cdfs = [cdf_at_interior_edges(m.pmf) for m in ms]
        f = [weighted_median([c[j] for c in cdfs], ws) for j in range(space.size - 1)]
        for j in range(1, len(f)):  # medians of non-decreasing CDFs are non-decreasing; guard rounding
            f[j] = max(f[j], f[j - 1])
        f = [min(max(v, 0.0), 1.0) for v in f]
        full = [0.0, *f, 1.0]
        pmf = tuple(full[j + 1] - full[j] for j in range(space.size))
    return Consensus(pmf, False, cap_applied, tuple(sorted(w.items())))
