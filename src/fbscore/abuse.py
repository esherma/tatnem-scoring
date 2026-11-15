"""Names of abuse-relevant parameters.

Every function in ``fbscore`` that takes one of these takes it as a **required** argument
with no default. Their values are platform configuration and are never published here.
Tests and examples use deliberately unrealistic ``EXAMPLE_*`` values. A test in this
repository fails if any function gives one of these names a default value.
"""

from __future__ import annotations

from typing import Final

__all__ = ["ABUSE_PARAMETER_NAMES"]

ABUSE_PARAMETER_NAMES: Final[frozenset[str]] = frozenset(
    {
        "principal_cap",
        "cap_min_principals",
        "low_participation_min_principals",
        "remaining_mint_cap_minor",
    }
)
