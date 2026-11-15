"""Outcome spaces: binary questions and bounded numeric questions on a uniform grid."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, field

__all__ = ["BinarySpace", "Grid", "NumericSpace", "OutcomeSpace", "space_from_dict"]

MAX_BINS = 500
"""Largest supported grid (spec §1.2)."""


@dataclass(frozen=True, slots=True)
class Grid:
    """A uniform grid on ``[lo, hi]`` with ``n_bins`` equal-width bins.

    Edges are ``e_0 = lo < e_1 < ... < e_N = hi``. Bin ``j`` (0-based) is the half-open
    interval ``(e_j, e_{j+1}]``, except that bin 0 also contains everything ``<= e_1``:
    an outcome exactly on an edge belongs to the lower bin, and outcomes outside
    ``[lo, hi]`` are clamped to the end bins.
    """

    lo: float
    hi: float
    n_bins: int
    edges: tuple[float, ...] = field(init=False, repr=False, compare=False)
    width: float = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)) or not self.lo < self.hi:
            raise ValueError("grid requires finite lo < hi")
        if isinstance(self.n_bins, bool) or not isinstance(self.n_bins, int):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("n_bins must be an int")
        if not 2 <= self.n_bins <= MAX_BINS:
            raise ValueError(f"n_bins must be in [2, {MAX_BINS}]")
        lo, hi, n = float(self.lo), float(self.hi), self.n_bins
        edges = tuple([lo + (hi - lo) * j / n for j in range(n)] + [hi])
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "width", (hi - lo) / n)

    @property
    def interior_edges(self) -> tuple[float, ...]:
        """``e_1 ... e_{N-1}``."""
        return self.edges[1:-1]

    @property
    def midpoints(self) -> tuple[float, ...]:
        """Bin midpoints. Every loss is constant on a bin, so these represent all outcomes."""
        e = self.edges
        return tuple(0.5 * (e[j] + e[j + 1]) for j in range(self.n_bins))

    def bin_index(self, x: float) -> int:
        """0-based bin containing ``x`` (edges go to the lower bin; out of range clamps)."""
        if math.isnan(x):
            raise ValueError("outcome is NaN")
        i = bisect_left(self.edges, x)  # first edge >= x
        return min(max(i - 1, 0), self.n_bins - 1)


@dataclass(frozen=True, slots=True)
class BinarySpace:
    """Outcome x in {0, 1}; a pmf is ``(1 - p, p)`` with ``p = P(x = 1)``."""

    @property
    def size(self) -> int:
        return 2

    @property
    def representative_outcomes(self) -> tuple[float, ...]:
        return (0.0, 1.0)

    def to_dict(self) -> dict[str, object]:
        return {"type": "binary"}


@dataclass(frozen=True, slots=True)
class NumericSpace:
    """A bounded numeric quantity on a uniform grid, out-of-range outcomes clamped."""

    grid: Grid

    @property
    def size(self) -> int:
        return self.grid.n_bins

    @property
    def representative_outcomes(self) -> tuple[float, ...]:
        return self.grid.midpoints

    def to_dict(self) -> dict[str, object]:
        g = self.grid
        return {"type": "numeric", "lo": g.lo, "hi": g.hi, "n_bins": g.n_bins}


OutcomeSpace = BinarySpace | NumericSpace


def space_from_dict(d: dict[str, object]) -> OutcomeSpace:
    """Build a space from ``{"type": "binary"}`` or ``{"type": "numeric", "lo", "hi", "n_bins"}``."""
    kind = d.get("type")
    if kind == "binary":
        return BinarySpace()
    if kind == "numeric":
        lo, hi, n = d.get("lo"), d.get("hi"), d.get("n_bins")
        if not isinstance(lo, int | float) or not isinstance(hi, int | float) or not isinstance(n, int):
            raise ValueError("numeric space needs numeric lo, hi and integer n_bins")
        return NumericSpace(Grid(float(lo), float(hi), n))
    raise ValueError(f"unknown outcome space type: {kind!r}")
