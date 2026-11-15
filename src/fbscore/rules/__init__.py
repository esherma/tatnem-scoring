"""Frozen contest-rule modules and dispatch by scoring version.

A question freezes ``scoring_version`` (the ``tatnem-scoring`` package version) when it
opens. Settlement dispatches to the rule module that was current at that version. A
released rule module is never changed in place: a behavior change ships as a new module
(``v2``, ...) registered below with the first package version that uses it.
"""

from __future__ import annotations

from types import ModuleType

from .. import __version__
from . import v1

__all__ = ["RULE_MODULES", "for_scoring_version", "parse_version"]

RULE_MODULES: tuple[tuple[tuple[int, int, int], ModuleType], ...] = (((0, 1, 0), v1),)
"""``(first package version, rule module)``, in increasing version order."""


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse ``MAJOR.MINOR.PATCH`` (pre-release and build suffixes are rejected)."""
    parts = version.split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"scoring_version must be MAJOR.MINOR.PATCH, got {version!r}")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def for_scoring_version(version: str) -> ModuleType:
    """The rule module that settles questions frozen at ``version``."""
    v = parse_version(version)
    if v > parse_version(__version__):
        raise ValueError(f"scoring_version {version} is newer than installed tatnem-scoring {__version__}")
    chosen: ModuleType | None = None
    for first, module in RULE_MODULES:
        if v >= first:
            chosen = module
    if chosen is None:
        raise ValueError(f"no rule module for scoring_version {version}")
    return chosen
