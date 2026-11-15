"""Aggregation primitive (spec §11). Fixture values are deliberately unrealistic."""

from __future__ import annotations

import pytest

import fbscore as fb

EXAMPLE_PRINCIPAL_CAP = 0.4
EXAMPLE_CAP_MIN_PRINCIPALS = 3
EXAMPLE_LOW_PARTICIPATION_MIN_PRINCIPALS = 2

BIN = fb.BinarySpace()
NUM = fb.NumericSpace(fb.Grid(0.0, 10.0, 5))


def consensus(space: fb.OutcomeSpace, members: list[fb.Member]) -> fb.Consensus:
    return fb.delivered_consensus(
        space,
        members,
        msr_state=fb.uniform(space),
        principal_cap=EXAMPLE_PRINCIPAL_CAP,
        cap_min_principals=EXAMPLE_CAP_MIN_PRINCIPALS,
        low_participation_min_principals=EXAMPLE_LOW_PARTICIPATION_MIN_PRINCIPALS,
    )


def test_weighted_median() -> None:
    assert fb.weighted_median([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) == 2.0
    assert fb.weighted_median([1.0, 2.0], [1.0, 1.0]) == 1.0  # lower median
    assert fb.weighted_median([1.0, 2.0, 3.0], [1.0, 1.0, 5.0]) == 3.0
    with pytest.raises(ValueError):
        fb.weighted_median([], [])
    with pytest.raises(ValueError):
        fb.weighted_median([1.0], [0.0])


def test_cap_water_filling() -> None:
    members = [
        fb.Member("f1", "big", 10.0, fb.uniform(BIN)),
        fb.Member("f2", "big", 10.0, fb.uniform(BIN)),
        fb.Member("f3", "s1", 1.0, fb.uniform(BIN)),
        fb.Member("f4", "s2", 1.0, fb.uniform(BIN)),
        fb.Member("f5", "s3", 2.0, fb.uniform(BIN)),
    ]
    w = fb.cap_principal_weights(members, principal_cap=EXAMPLE_PRINCIPAL_CAP)
    assert w["f1"] + w["f2"] == pytest.approx(0.4)
    assert w["f1"] == pytest.approx(w["f2"])
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["f5"] == pytest.approx(2 * w["f3"])
    with pytest.raises(ValueError, match="infeasible"):
        fb.cap_principal_weights(members[:3], principal_cap=0.3)
    with pytest.raises(ValueError, match="principal_cap"):
        fb.cap_principal_weights(members, principal_cap=0.0)
    with pytest.raises(ValueError, match="weights"):
        fb.cap_principal_weights([fb.Member("x", "p", -1.0, fb.uniform(BIN))], principal_cap=1.0)


def test_binary_consensus_is_weighted_median() -> None:
    members = [
        fb.Member("f1", "a", 1.0, fb.canonicalize(BIN, "binary_p", 0.2)),
        fb.Member("f2", "b", 1.0, fb.canonicalize(BIN, "binary_p", 0.6)),
        fb.Member("f3", "c", 1.0, fb.canonicalize(BIN, "binary_p", 0.9)),
    ]
    out = consensus(BIN, members)
    assert out.pmf[1] == 0.6
    assert not out.low_participation and out.cap_applied


def test_numeric_consensus_is_a_valid_distribution() -> None:
    members = [
        fb.Member("f1", "a", 1.0, (0.1, 0.2, 0.4, 0.2, 0.1)),
        fb.Member("f2", "b", 2.0, (0.0, 0.1, 0.3, 0.4, 0.2)),
        fb.Member("f3", "b", 1.0, (0.5, 0.5, 0.0, 0.0, 0.0)),
    ]
    out = consensus(NUM, members)
    assert not out.cap_applied  # only 2 principals
    assert all(v >= 0 for v in out.pmf)
    assert sum(out.pmf) == pytest.approx(1.0)


def test_low_participation_falls_back_to_msr_state() -> None:
    state = fb.canonicalize(BIN, "binary_p", 0.8)
    out = fb.delivered_consensus(
        BIN,
        [fb.Member("f1", "a", 1.0, fb.uniform(BIN))],
        msr_state=state,
        principal_cap=EXAMPLE_PRINCIPAL_CAP,
        cap_min_principals=EXAMPLE_CAP_MIN_PRINCIPALS,
        low_participation_min_principals=EXAMPLE_LOW_PARTICIPATION_MIN_PRINCIPALS,
    )
    assert out.pmf == state and out.low_participation


def test_consensus_validation() -> None:
    m = fb.Member("f1", "a", 1.0, fb.uniform(BIN))
    with pytest.raises(ValueError, match="one member per forecaster"):
        consensus(BIN, [m, m])
    with pytest.raises(ValueError, match="length"):
        consensus(NUM, [m, fb.Member("f2", "b", 1.0, fb.uniform(BIN))])
    with pytest.raises(ValueError, match="weights"):
        consensus(BIN, [m, fb.Member("f2", "b", 0.0, fb.uniform(BIN))])
