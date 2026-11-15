"""Unit tests (spec §12.1 of the TDD): losses, clamping, canonicalization, c_max, money."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

import fbscore as fb
from fbscore.canonicalize import ERROR_CODES, CanonicalizationError
from fbscore.rules import for_scoring_version, parse_version, v1

NUM = fb.NumericSpace(fb.Grid(0.0, 10.0, 5))
BIN = fb.BinarySpace()


# ---- losses -------------------------------------------------------------------------------


def test_brier_values() -> None:
    assert fb.brier((0.3, 0.7), 1) == pytest.approx(0.09)
    assert fb.brier((0.3, 0.7), 0) == pytest.approx(0.49)
    assert fb.brier((1.0, 0.0), 0) == 0.0
    with pytest.raises(ValueError, match="0 or 1"):
        fb.brier((0.5, 0.5), 0.5)
    with pytest.raises(ValueError, match="length 2"):
        fb.brier((1.0,), 1)


@pytest.mark.parametrize("j", range(5))
def test_point_mass_in_correct_bin_scores_zero(j: int) -> None:
    pmf = tuple(1.0 if i == j else 0.0 for i in range(5))
    x = NUM.grid.midpoints[j]
    assert fb.crps(pmf, NUM.grid, x) == 0.0


def test_edge_belongs_to_lower_bin() -> None:
    lower = (0.0, 1.0, 0.0, 0.0, 0.0)  # bin (2, 4]
    upper = (0.0, 0.0, 1.0, 0.0, 0.0)  # bin (4, 6]
    assert fb.crps(lower, NUM.grid, 4.0) == 0.0
    assert fb.crps(upper, NUM.grid, 4.0) > 0.0
    assert NUM.grid.bin_index(4.0) == 1
    assert NUM.grid.bin_index(4.0000001) == 2


def test_clamping() -> None:
    pmf = (0.1, 0.2, 0.4, 0.2, 0.1)
    assert fb.crps(pmf, NUM.grid, -50.0) == fb.crps(pmf, NUM.grid, 0.0)
    assert fb.crps(pmf, NUM.grid, 1e9) == fb.crps(pmf, NUM.grid, 10.0)
    assert NUM.grid.bin_index(-1.0) == 0
    assert NUM.grid.bin_index(11.0) == 4


def test_crps_validation() -> None:
    with pytest.raises(ValueError, match="length"):
        fb.crps((0.5, 0.5), NUM.grid, 1.0)
    with pytest.raises(ValueError, match="NaN"):
        fb.crps(fb.uniform(NUM), NUM.grid, math.nan)
    with pytest.raises(ValueError, match="NaN"):
        NUM.grid.bin_index(math.nan)


def test_crps_of_binary_grid_matches_brier() -> None:
    # CRPS on a 2-bin grid over [0, 1] with unit spacing equals Brier: (F_1 - 1[x <= 1/2])^2.
    g = fb.Grid(-0.5, 1.5, 2)
    for p in (0.0, 0.2, 0.5, 0.9, 1.0):
        pmf = (1 - p, p)
        for x in (0, 1):
            assert fb.crps(pmf, g, x) == pytest.approx(fb.brier(pmf, x))


def test_max_prior_loss_midpoints_equal_brute_force() -> None:
    for space, prior in (
        (NUM, fb.uniform(NUM)),
        (NUM, fb.canonicalize(NUM, "pmf", [0.05, 0.1, 0.5, 0.3, 0.05])),
        (fb.NumericSpace(fb.Grid(-3.0, 7.0, 40)), fb.uniform(fb.NumericSpace(fb.Grid(-3.0, 7.0, 40)))),
    ):
        assert isinstance(space, fb.NumericSpace)
        g = space.grid
        fine = [g.lo - 1 + (g.hi - g.lo + 2) * i / 20_000 for i in range(20_001)]
        brute = max(fb.crps(prior, g, x) for x in fine)
        assert fb.max_loss(space, prior) == pytest.approx(brute, abs=1e-12)
    assert fb.max_loss(BIN, (0.3, 0.7)) == pytest.approx(0.49)


def test_c_max_uniform_and_custom_prior() -> None:
    assert v1.c_max(NUM, fb.uniform(NUM), k=1, reserve=0.1, prize_pool=400) == pytest.approx(150)
    sharp = fb.canonicalize(NUM, "pmf", [0.0, 0.0, 1.0, 0.0, 0.0])
    # A sharper prior has a larger max loss, so c_max goes down (spec §3).
    assert v1.c_max(NUM, sharp, k=1, reserve=0.1, prize_pool=400) < 150
    assert v1.c_max(BIN, (0.2, 0.8), k=1, reserve=0.0, prize_pool=100) == pytest.approx(100 / 0.64)
    for kw in ({"k": 0.0}, {"reserve": 1.0}, {"prize_pool": -1.0}):
        args: dict[str, float] = {"k": 1.0, "reserve": 0.1, "prize_pool": 1.0, "max_prior_loss": 1.0} | kw
        with pytest.raises(ValueError):
            fb.c_max(**args)
    with pytest.raises(ValueError, match="zero max loss"):
        fb.c_max(k=1, reserve=0, prize_pool=1, max_prior_loss=0.0)


def test_decay_with_datetimes() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    t1 = t0 + timedelta(days=10)
    assert v1.c_at(c_max=100.0, decay=0.5, open_at=t0, close_at=t1, at=t0) == 100.0
    assert v1.c_at(c_max=100.0, decay=0.5, open_at=t0, close_at=t1, at=t1) == 50.0
    assert v1.c_at(c_max=100.0, decay=0.5, open_at=t0, close_at=t1, at=t0 + timedelta(days=8)) == pytest.approx(60.0)
    with pytest.raises(ValueError, match="outside"):
        v1.c_at(c_max=1.0, decay=0.5, open_at=t0, close_at=t1, at=t1 + timedelta(seconds=1))
    with pytest.raises(ValueError, match="after"):
        v1.c_at(c_max=1.0, decay=0.5, open_at=t1, close_at=t0, at=t1)
    with pytest.raises(ValueError, match="after"):
        v1.c_at(c_max=1.0, decay=0.5, open_at=1.0, close_at=1.0, at=1.0)
    with pytest.raises(TypeError):
        v1.c_at(c_max=1.0, decay=0.5, open_at=t0, close_at=1.0, at=0.5)
    with pytest.raises(ValueError, match="decay"):
        v1.c_at(c_max=1.0, decay=1.5, open_at=0.0, close_at=1.0, at=0.5)


def test_money_rounding_absorbs_float_noise() -> None:
    assert 0.1 * 3 * 100 != 30  # the noise we guard against
    assert fb.ceil_minor(0.1 * 3) == 30
    assert fb.floor_minor(0.29 * 3) == 87
    assert fb.ceil_minor(0.001) == 1
    assert fb.floor_minor(-0.001) == -1
    with pytest.raises(ValueError):
        fb.ceil_minor(math.inf)


# ---- canonicalization ---------------------------------------------------------------------


def test_binary_p() -> None:
    assert fb.canonicalize(BIN, "binary_p", 0.7) == (pytest.approx(0.3), 0.7)
    assert fb.canonicalize(BIN, "binary_p", 0) == (1.0, 0.0)
    assert fb.canonicalize(BIN, "binary_p", 1) == (0.0, 1.0)
    assert fb.canonicalize(BIN, "pmf", [0.25, 0.75]) == (0.25, 0.75)


def test_pmf_renormalized() -> None:
    pmf = fb.canonicalize(NUM, "pmf", [0.2, 0.2, 0.2, 0.2, 0.2000005])
    assert math.fsum(pmf) == pytest.approx(1.0, abs=1e-12)


def test_cdf_input() -> None:
    assert fb.canonicalize(NUM, "cdf", [0.1, 0.3, 0.7, 0.9]) == pytest.approx((0.1, 0.2, 0.4, 0.2, 0.1))


def test_quantiles_round_trip_cdf() -> None:
    # Quantiles placed exactly on interior edges reproduce that CDF.
    q = [[0.1, 2.0], [0.3, 4.0], [0.7, 6.0], [0.9, 8.0]]
    from_q = fb.canonicalize(NUM, "quantiles", q)
    from_cdf = fb.canonicalize(NUM, "cdf", [0.1, 0.3, 0.7, 0.9])
    assert from_q == pytest.approx(from_cdf)


def test_quantiles_monotone_and_sum_to_one() -> None:
    g = fb.NumericSpace(fb.Grid(0.0, 100.0, 200))
    pmf = fb.canonicalize(g, "quantiles", [[0.05, 10.0], [0.5, 42.0], [0.95, 77.7]])
    assert all(v >= 0 for v in pmf)
    assert math.fsum(pmf) == pytest.approx(1.0, abs=1e-12)


def test_quantiles_mass_at_bounds() -> None:
    pmf = fb.canonicalize(NUM, "quantiles", [[0.2, 0.0], [0.5, 5.0], [0.9, 10.0]])
    assert pmf[0] >= 0.2 - 1e-12  # v1 = lo puts mass t1 in bin 1
    assert pmf[-1] >= 0.1 - 1e-12  # v_m = hi puts 1 - t_m in the last bin


@pytest.mark.parametrize(
    ("space", "fmt", "payload", "code"),
    [
        (BIN, "binary_p", 1.2, "p_out_of_range"),
        (BIN, "binary_p", math.nan, "not_a_number"),
        (BIN, "binary_p", True, "not_a_number"),
        (BIN, "binary_p", "0.5", "not_a_number"),
        (BIN, "cdf", [0.5], "format_not_allowed"),
        (NUM, "binary_p", 0.5, "format_not_allowed"),
        (NUM, "logits", [0.0], "unsupported_format"),
        (NUM, "pmf", [0.5, 0.5], "wrong_length"),
        (NUM, "pmf", [0.6, -0.1, 0.2, 0.2, 0.1], "negative_mass"),
        (NUM, "pmf", [0.2, 0.2, 0.2, 0.2, 0.3], "bad_sum"),
        (NUM, "pmf", "abc", "bad_shape"),
        (NUM, "pmf", 3, "bad_shape"),
        (NUM, "cdf", [0.1, 0.3, 0.7], "wrong_length"),
        (NUM, "cdf", [0.1, 0.3, 0.7, 1.2], "cdf_out_of_range"),
        (NUM, "cdf", [0.1, 0.5, 0.4, 0.9], "cdf_not_monotone"),
        (NUM, "quantiles", [[0.1, 1.0], [0.5, 5.0]], "quantile_count"),
        (NUM, "quantiles", [[0.0, 1.0], [0.5, 5.0], [0.9, 8.0]], "quantile_level_range"),
        (NUM, "quantiles", [[0.1, -1.0], [0.5, 5.0], [0.9, 8.0]], "quantile_value_range"),
        (NUM, "quantiles", [[0.1, 1.0], [0.5, 5.0], [0.5, 8.0]], "quantile_not_increasing"),
        (NUM, "quantiles", [[0.1, 1.0], [0.5, 5.0], [0.6, 5.0]], "quantile_not_increasing"),
        (NUM, "quantiles", [[0.1, 1.0], [0.5], [0.9, 8.0]], "bad_shape"),
        (NUM, "quantiles", [[0.1, 1.0]] * 100, "quantile_count"),
    ],
)
def test_invalid_inputs_have_stable_codes(space: fb.OutcomeSpace, fmt: str, payload: object, code: str) -> None:
    with pytest.raises(CanonicalizationError) as ei:
        fb.canonicalize(space, fmt, payload)
    assert ei.value.code == code
    assert code in ERROR_CODES


def test_error_code_registry_is_closed() -> None:
    with pytest.raises(AssertionError):
        CanonicalizationError("made_up")


def test_direct_helpers_reject_non_arrays() -> None:
    from fbscore.canonicalize import check_canonical, from_cdf, from_pmf, from_quantiles

    with pytest.raises(CanonicalizationError, match="bad_shape"):
        from_pmf(NUM, "x")
    with pytest.raises(CanonicalizationError, match="bad_shape"):
        from_cdf(NUM.grid, "x")
    with pytest.raises(CanonicalizationError, match="bad_shape"):
        from_quantiles(NUM.grid, "x")
    with pytest.raises(CanonicalizationError, match="bad_shape"):
        check_canonical(NUM, "x")


def test_check_canonical() -> None:
    from fbscore.canonicalize import check_canonical

    u = fb.uniform(NUM)
    assert check_canonical(NUM, list(u)) == u
    for bad, code in (
        ([0.2] * 4, "wrong_length"),
        ([0.5, 0.5, 0.1, -0.1, 0.0], "negative_mass"),
        ([0.3] * 5, "bad_sum"),
    ):
        with pytest.raises(CanonicalizationError) as ei:
            check_canonical(NUM, bad)
        assert ei.value.code == code


def test_sponsor_prior_mix() -> None:
    sponsor = (1.0, 0.0, 0.0, 0.0, 0.0)
    mixed = v1.prior_from_sponsor(NUM, sponsor)
    assert mixed == pytest.approx((0.92, 0.02, 0.02, 0.02, 0.02))
    assert min(mixed) > 0
    with pytest.raises(ValueError, match="sponsor_weight"):
        fb.mix_prior(NUM, sponsor, sponsor_weight=1.5)


# ---- grid and spaces ----------------------------------------------------------------------


def test_grid_validation_and_spaces() -> None:
    g = fb.Grid(0, 1, 4)
    assert g.edges == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert g.width == 0.25
    for args in ((1.0, 0.0, 4), (0.0, math.inf, 4), (0.0, 1.0, 1), (0.0, 1.0, 501)):
        with pytest.raises(ValueError):
            fb.Grid(*args)
    with pytest.raises(TypeError):
        fb.Grid(0.0, 1.0, 4.0)  # type: ignore[arg-type]
    assert fb.space_from_dict({"type": "binary"}) == BIN
    assert fb.space_from_dict(NUM.to_dict()) == NUM
    assert BIN.to_dict() == {"type": "binary"}
    for bad in ({"type": "date"}, {"type": "numeric", "lo": 0, "hi": 1}):
        with pytest.raises(ValueError):
            fb.space_from_dict(bad)


# ---- rule dispatch ------------------------------------------------------------------------


def test_rule_dispatch() -> None:
    assert for_scoring_version("0.1.0") is v1
    assert for_scoring_version(fb.__version__) is v1
    with pytest.raises(ValueError, match="newer"):
        for_scoring_version("99.0.0")
    with pytest.raises(ValueError, match="no rule module"):
        for_scoring_version("0.0.9")
    for bad in ("1.0", "1.0.0rc1", "a.b.c"):
        with pytest.raises(ValueError, match="MAJOR"):
            parse_version(bad)
    assert v1.rule_name(BIN) == "brier"
    assert v1.rule_name(NUM) == "crps"
    assert v1.loss(BIN, (0.5, 0.5), 1) == 0.25


def test_loss_profile_is_bit_identical_to_loss() -> None:
    from fbscore.losses import loss_profile

    for space in (BIN, NUM, fb.NumericSpace(fb.Grid(-7.0, 3.3, 37))):
        raw = [float((i * 7919) % 13 + 1) for i in range(space.size)]
        pmf = fb.canonicalize(space, "pmf", [v / sum(raw) for v in raw])
        prof = loss_profile(space, pmf)
        assert prof == tuple(fb.loss(space, pmf, x) for x in space.representative_outcomes)
