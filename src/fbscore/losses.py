"""Loss functions (spec §1). Lower is better; every loss is >= 0 and bounded."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import accumulate

from ._types import Pmf
from .grid import BinarySpace, Grid, OutcomeSpace

__all__ = ["brier", "cdf_at_interior_edges", "crps", "loss", "loss_profile", "max_loss"]


def brier(pmf: Sequence[float], x: float) -> float:
    """Quadratic (Brier) loss ``(p - x)^2`` for a binary pmf ``(1 - p, p)``, x in {0, 1}."""
    if len(pmf) != 2:
        raise ValueError("binary pmf must have length 2")
    if x not in (0, 1):
        raise ValueError("binary outcome must be 0 or 1")
    d = pmf[1] - float(x)
    return d * d


def cdf_at_interior_edges(pmf: Sequence[float]) -> tuple[float, ...]:
    """``F_j = sum_{i <= j} p_i`` for j = 1..N-1, summed left to right (fixed order)."""
    return tuple(accumulate(pmf[:-1]))


def _squared_terms(pmf: Sequence[float], grid: Grid) -> tuple[list[float], list[float]]:
    if len(pmf) != grid.n_bins:
        raise ValueError("pmf length does not match grid")
    cdf = cdf_at_interior_edges(pmf)
    return [f * f for f in cdf], [(f - 1.0) * (f - 1.0) for f in cdf]


def crps(pmf: Sequence[float], grid: Grid, x: float) -> float:
    """Discretized CRPS evaluated at interior edges.

    ``L(p, x) = w * sum_{j=1}^{N-1} (F_j - 1[x <= e_j])^2``. An outcome on an edge belongs
    to the lower bin, and outcomes outside ``[lo, hi]`` clamp automatically. With ``b`` the
    0-based bin of ``x``, the indicator is 0 for the first ``b`` interior edges and 1 after.
    """
    below, above = _squared_terms(pmf, grid)
    b = grid.bin_index(x)
    return grid.width * math.fsum(below[:b] + above[b:])


def loss_profile(space: OutcomeSpace, pmf: Sequence[float]) -> tuple[float, ...]:
    """``L(pmf, x)`` at every representative outcome, bit-identical to :func:`loss` at each."""
    if isinstance(space, BinarySpace):
        return (brier(pmf, 0), brier(pmf, 1))
    below, above = _squared_terms(pmf, space.grid)
    w = space.grid.width
    return tuple(w * math.fsum(below[:b] + above[b:]) for b in range(space.grid.n_bins))


def loss(space: OutcomeSpace, pmf: Sequence[float], x: float) -> float:
    """The rule's loss for ``space``: Brier for binary, CRPS for numeric."""
    if isinstance(space, BinarySpace):
        return brier(pmf, x)
    return crps(pmf, space.grid, x)


def max_loss(space: OutcomeSpace, pmf: Pmf) -> float:
    """``max_x L(pmf, x)``, exact: binary over {0, 1}; numeric over bin midpoints."""
    return max(loss_profile(space, pmf))
