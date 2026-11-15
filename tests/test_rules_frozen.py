"""Released rule modules are never changed in place (TDD §2.3.3)."""

from __future__ import annotations

import hashlib
from pathlib import Path

from fbscore.rules import v1

# Update ONLY by adding a new rule module. Changing v1 behavior breaks every question frozen at v1.
FROZEN_SHA256 = {
    "v1.py": "9d0bd730f0184d3e2c4c82354559de2bb7dde527d3082aadf97e16c29c64c365",
}


def test_v1_source_is_frozen() -> None:
    src = Path(v1.__file__).read_bytes()
    assert hashlib.sha256(src).hexdigest() == FROZEN_SHA256["v1.py"], (
        "fbscore/rules/v1.py changed. Released rules are frozen: add rules/v2.py and register it instead."
    )


def test_v1_public_constants() -> None:
    assert (v1.RULE_ID, v1.BINARY_LOSS, v1.NUMERIC_LOSS) == ("v1", "brier", "crps")
    assert (v1.ALPHA, v1.MIN_LOCK_MINOR_PER_SUBMISSION, v1.SPONSOR_PRIOR_WEIGHT) == (0.5, 100, 0.9)
