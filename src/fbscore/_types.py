"""Shared value types."""

from __future__ import annotations

type Pmf = tuple[float, ...]
"""A canonical probability mass function: finite, non-negative floats summing to 1."""
