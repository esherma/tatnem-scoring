"""Conversion of point amounts to integer minor units (cents, centi-credits).

Points are floats. To make ``ceil``/``floor`` immune to binary floating-point noise
(``1440 * 0.24 * 100 == 34559.999999999996``), an amount is first rounded to 1e-6
minor units, then ceiled or floored. This is part of the scoring rules.
"""

from __future__ import annotations

import math
from typing import Final

__all__ = ["MINOR_PER_MAJOR", "ceil_minor", "floor_minor"]

MINOR_PER_MAJOR: Final = 100
_ROUND_DIGITS: Final = 6


def _scaled(amount: float) -> float:
    if not math.isfinite(amount):
        raise ValueError("amount is not finite")
    return round(amount * MINOR_PER_MAJOR, _ROUND_DIGITS)


def ceil_minor(amount: float) -> int:
    """Smallest whole number of minor units >= ``amount`` (after 1e-6 rounding)."""
    return math.ceil(_scaled(amount))


def floor_minor(amount: float) -> int:
    """Largest whole number of minor units <= ``amount`` (after 1e-6 rounding)."""
    return math.floor(_scaled(amount))
