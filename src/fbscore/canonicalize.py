"""Canonicalization of submissions into a pmf (spec §2).

Every rejection raises :class:`CanonicalizationError` with a stable ``code`` that API
clients can match on. Codes are part of the public contract and never change meaning.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from itertools import pairwise
from typing import Final, Literal

from ._types import Pmf
from .grid import BinarySpace, Grid, OutcomeSpace

__all__ = [
    "ERROR_CODES",
    "PMF_SUM_TOLERANCE",
    "CanonicalizationError",
    "InputFormat",
    "canonicalize",
    "check_canonical",
    "from_binary_p",
    "from_cdf",
    "from_pmf",
    "from_quantiles",
    "mix_prior",
    "uniform",
]

InputFormat = Literal["binary_p", "pmf", "cdf", "quantiles"]

PMF_SUM_TOLERANCE: Final = 1e-6
MIN_QUANTILES: Final = 3
MAX_QUANTILES: Final = 99

ERROR_CODES: Final[dict[str, str]] = {
    "unsupported_format": "input format is not one of binary_p, pmf, cdf, quantiles",
    "format_not_allowed": "input format is not allowed for this question type",
    "not_a_number": "a value is not a finite number",
    "p_out_of_range": "binary probability is outside [0, 1]",
    "wrong_length": "array length does not match the outcome space",
    "negative_mass": "a pmf entry is negative",
    "bad_sum": "pmf does not sum to 1 within 1e-6",
    "cdf_out_of_range": "a CDF value is outside [0, 1]",
    "cdf_not_monotone": "CDF values decrease",
    "quantile_count": "number of quantiles is outside [3, 99]",
    "quantile_level_range": "a quantile level is outside (0, 1)",
    "quantile_value_range": "a quantile value is outside [lo, hi]",
    "quantile_not_increasing": "quantile levels or values are not strictly increasing",
    "bad_shape": "payload has the wrong shape",
}


class CanonicalizationError(ValueError):
    """Invalid submission. ``code`` is one of :data:`ERROR_CODES`."""

    def __init__(self, code: str, detail: str = "") -> None:
        if code not in ERROR_CODES:
            raise AssertionError(f"unregistered error code {code!r}")
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail or ERROR_CODES[code]}")


def _finite(v: object) -> float:
    if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
        raise CanonicalizationError("not_a_number", repr(v))
    return float(v)


def uniform(space: OutcomeSpace) -> Pmf:
    """The uniform pmf on ``space`` (binary: p = 0.5)."""
    n = space.size
    return tuple([1.0 / n] * n)


def from_binary_p(p: object) -> Pmf:
    """``p`` in [0, 1] (endpoints allowed) becomes ``(1 - p, p)``."""
    fp = _finite(p)
    if not 0.0 <= fp <= 1.0:
        raise CanonicalizationError("p_out_of_range", repr(p))
    return (1.0 - fp, fp)


def from_pmf(space: OutcomeSpace, values: Sequence[object]) -> Pmf:
    """Length-N, non-negative, sums to 1 within 1e-6; renormalized."""
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise CanonicalizationError("bad_shape", "pmf must be an array")
    if len(values) != space.size:
        raise CanonicalizationError("wrong_length", f"expected {space.size}, got {len(values)}")
    vals = [_finite(v) for v in values]
    if any(v < 0.0 for v in vals):
        raise CanonicalizationError("negative_mass")
    total = math.fsum(vals)
    if abs(total - 1.0) > PMF_SUM_TOLERANCE:
        raise CanonicalizationError("bad_sum", f"sum = {total!r}")
    return tuple(v / total for v in vals)


CANONICAL_SUM_TOLERANCE: Final = 1e-9


def check_canonical(space: OutcomeSpace, values: Sequence[object]) -> Pmf:
    """Validate an already-canonical pmf (e.g. from an export) and return it unchanged.

    Unlike :func:`from_pmf` this does not renormalize, so stored values are scored bit for bit.
    """
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise CanonicalizationError("bad_shape", "pmf must be an array")
    if len(values) != space.size:
        raise CanonicalizationError("wrong_length", f"expected {space.size}, got {len(values)}")
    vals = tuple(_finite(v) for v in values)
    if any(v < 0.0 for v in vals):
        raise CanonicalizationError("negative_mass")
    if abs(math.fsum(vals) - 1.0) > CANONICAL_SUM_TOLERANCE:
        raise CanonicalizationError("bad_sum", "stored pmf must sum to 1 within 1e-9")
    return vals


def _pmf_from_interior_cdf(cdf: Sequence[float]) -> Pmf:
    full = [0.0, *cdf, 1.0]
    return tuple(full[j + 1] - full[j] for j in range(len(full) - 1))


def from_cdf(grid: Grid, values: Sequence[object]) -> Pmf:
    """CDF values at the interior edges ``e_1..e_{N-1}``, non-decreasing in [0, 1]."""
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise CanonicalizationError("bad_shape", "cdf must be an array")
    if len(values) != grid.n_bins - 1:
        raise CanonicalizationError("wrong_length", f"expected {grid.n_bins - 1}, got {len(values)}")
    vals = [_finite(v) for v in values]
    if any(not 0.0 <= v <= 1.0 for v in vals):
        raise CanonicalizationError("cdf_out_of_range")
    if any(b < a for a, b in pairwise(vals)):
        raise CanonicalizationError("cdf_not_monotone")
    return _pmf_from_interior_cdf(vals)


def from_quantiles(grid: Grid, pairs: Sequence[object]) -> Pmf:
    """3-99 ``(level, value)`` pairs, strictly increasing in both.

    The CDF is piecewise linear through ``(lo, 0), (v_1, t_1), ..., (v_m, t_m), (hi, 1)``,
    evaluated at the interior edges and differenced. ``v_1 = lo`` puts mass ``t_1`` in
    the first bin; ``v_m = hi`` puts mass ``1 - t_m`` in the last.
    """
    if isinstance(pairs, str | bytes) or not isinstance(pairs, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise CanonicalizationError("bad_shape", "quantiles must be an array of [level, value]")
    if not MIN_QUANTILES <= len(pairs) <= MAX_QUANTILES:
        raise CanonicalizationError("quantile_count", str(len(pairs)))
    levels: list[float] = []
    values: list[float] = []
    for pair in pairs:
        if isinstance(pair, str | bytes) or not isinstance(pair, Sequence) or len(pair) != 2:  # pyright: ignore[reportUnknownArgumentType]
            raise CanonicalizationError("bad_shape", "each quantile must be [level, value]")
        t, v = _finite(pair[0]), _finite(pair[1])  # pyright: ignore[reportUnknownArgumentType]
        if not 0.0 < t < 1.0:
            raise CanonicalizationError("quantile_level_range", repr(t))
        if not grid.lo <= v <= grid.hi:
            raise CanonicalizationError("quantile_value_range", repr(v))
        levels.append(t)
        values.append(v)
    for a, b in zip(range(len(levels) - 1), range(1, len(levels)), strict=True):
        if not (levels[b] > levels[a] and values[b] > values[a]):
            raise CanonicalizationError("quantile_not_increasing")
    xs = [grid.lo, *values, grid.hi]
    fs = [0.0, *levels, 1.0]
    cdf: list[float] = []
    prev = 0.0
    for e in grid.interior_edges:
        i = bisect_right(xs, e) - 1  # xs[i] <= e < xs[i + 1]; e is strictly inside (lo, hi)
        x0, x1, f0, f1 = xs[i], xs[i + 1], fs[i], fs[i + 1]
        f = f0 + (f1 - f0) * (e - x0) / (x1 - x0)
        f = min(max(f, prev), 1.0)  # guard against rounding: keep the CDF monotone
        cdf.append(f)
        prev = f
    return _pmf_from_interior_cdf(cdf)


def canonicalize(space: OutcomeSpace, input_format: str, payload: object) -> Pmf:
    """Turn a raw submission into the canonical pmf that gets scored."""
    if input_format not in ("binary_p", "pmf", "cdf", "quantiles"):
        raise CanonicalizationError("unsupported_format", repr(input_format))
    if isinstance(space, BinarySpace):
        if input_format == "binary_p":
            return from_binary_p(payload)
        if input_format == "pmf":
            return from_pmf(space, _seq(payload))
        raise CanonicalizationError("format_not_allowed", f"{input_format} on a binary question")
    if input_format == "binary_p":
        raise CanonicalizationError("format_not_allowed", "binary_p on a numeric question")
    if input_format == "pmf":
        return from_pmf(space, _seq(payload))
    if input_format == "cdf":
        return from_cdf(space.grid, _seq(payload))
    return from_quantiles(space.grid, _seq(payload))


def _seq(payload: object) -> Sequence[object]:
    if isinstance(payload, str | bytes) or not isinstance(payload, Sequence):
        raise CanonicalizationError("bad_shape", "expected an array")
    return payload  # pyright: ignore[reportUnknownVariableType]


def mix_prior(space: OutcomeSpace, sponsor_pmf: Pmf, *, sponsor_weight: float) -> Pmf:
    """``w * sponsor + (1 - w) * uniform``, so no region is effectively ruled out (spec §3)."""
    if not 0.0 <= sponsor_weight <= 1.0:
        raise ValueError("sponsor_weight must be in [0, 1]")
    base = from_pmf(space, sponsor_pmf)
    u = uniform(space)
    mixed = [sponsor_weight * a + (1.0 - sponsor_weight) * b for a, b in zip(base, u, strict=True)]
    total = math.fsum(mixed)
    return tuple(v / total for v in mixed)
