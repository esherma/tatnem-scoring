"""Spec §13 worked examples, reproduced exactly in cents and centi-credits."""

from __future__ import annotations

import math

import pytest

import fbscore as fb
from fbscore.rules import v1

BIN = fb.BinarySpace()
PRIOR = fb.uniform(BIN)
POOL = 40_000  # $400.00


def p(x: float) -> tuple[float, float]:
    return fb.canonicalize(BIN, "binary_p", x)


def chain(cmax: float, *, d_c: float | None = None, same_ad: bool = False) -> list[fb.Submission]:
    b = p(0.65)
    return [
        fb.Submission(1, "a", "A", p(0.70), cmax),
        fb.Submission(2, "b", "B", b, cmax),
        fb.Submission(3, "c", "C", b, cmax),  # bitwise copy of the state
        fb.Submission(4, "a" if same_ad else "d", "D", p(0.90), cmax if d_c is None else d_c),
    ]


def lines(st: fb.Settlement) -> dict[str, fb.SettlementLine]:
    return {ln.principal_id: ln for ln in st.lines}


def test_binary_c_max() -> None:
    assert v1.c_max(BIN, PRIOR, k=1, reserve=0.1, prize_pool=400) == pytest.approx(1440, abs=1e-9)


def test_binary_outcome_1() -> None:
    st = v1.settle_cash(BIN, PRIOR, chain(1440.0), 1, prize_pool_minor=POOL)
    pts = [r.points for r in st.scores]
    assert pts == pytest.approx([230.4, -46.8, 0.0, 162.0], abs=1e-9)
    assert st.scores[2].points == 0.0 and st.scores[2].is_noop
    assert st.net_total == pytest.approx(1440 * (0.25 - 0.01), abs=1e-9)  # telescopes
    assert st.gross_positive == pytest.approx(392.4, abs=1e-9)
    assert st.haircut == 1.0
    ln = lines(st)
    assert (ln["a"].prize_minor, ln["d"].prize_minor, ln["b"].prize_minor) == (23_040, 16_200, 0)
    assert ln["b"].credits_burned_minor == 4_680
    assert (ln["a"].credits_earned_minor, ln["d"].credits_earned_minor) == (11_520, 8_100)
    assert (ln["a"].lock_minor, ln["b"].lock_minor, ln["c"].lock_minor, ln["d"].lock_minor) == (
        34_560,
        4_680,
        0,  # 1 cr floor waived for no-ops
        55_800,
    )
    assert st.refund_minor == 760


def test_binary_outcome_0() -> None:
    st = v1.settle_cash(BIN, PRIOR, chain(1440.0), 0, prize_pool_minor=POOL)
    assert st.net_total == pytest.approx(-806.4, abs=1e-9)
    assert st.gross_positive == pytest.approx(97.2, abs=1e-9)
    ln = lines(st)
    assert ln["b"].prize_minor == 9_720
    assert (ln["a"].credits_burned_minor, ln["d"].credits_burned_minor) == (34_560, 55_800)
    assert (ln["a"].lock_released_minor, ln["d"].lock_released_minor) == (0, 0)
    assert st.refund_minor == 30_280


def test_binary_same_principal_netting() -> None:
    st = v1.settle_cash(BIN, PRIOR, chain(1440.0, same_ad=True), 1, prize_pool_minor=POOL)
    ln = lines(st)
    assert ln["a"].net_points == pytest.approx(392.4, abs=1e-9)
    assert ln["a"].prize_minor == 39_240
    assert ln["a"].lock_minor == 90_360  # max(-(230.4 + 162), 345.6 + 558) = 903.6 cr


def test_binary_decay() -> None:
    c_d = v1.c_at(c_max=1440.0, decay=0.5, open_at=0.0, close_at=10.0, at=8.0)
    assert c_d == pytest.approx(864.0, abs=1e-9)
    st = v1.settle_cash(BIN, PRIOR, chain(1440.0, d_c=c_d), 1, prize_pool_minor=POOL)
    assert st.scores[3].points == pytest.approx(97.2, abs=1e-9)
    assert lines(st)["d"].lock_minor == 33_480


def test_binary_k2_haircut() -> None:
    cmax = v1.c_max(BIN, PRIOR, k=2, reserve=0.1, prize_pool=400)
    assert cmax == pytest.approx(2880, abs=1e-9)
    st = v1.settle_cash(BIN, PRIOR, chain(cmax), 1, prize_pool_minor=POOL)
    assert [r.points for r in st.scores] == pytest.approx([460.8, -93.6, 0.0, 324.0], abs=1e-9)
    assert st.gross_positive == pytest.approx(784.8, abs=1e-9)
    assert st.haircut == pytest.approx(400 / 784.8)
    ln = lines(st)
    # h * net, floored to the cent: 234.8623... -> $234.86 and 165.1376... -> $165.13.
    # (TDD v0.2 §6.13 printed $165.14; the floor rule gives $165.13 and a 1-cent refund.)
    assert (ln["a"].prize_minor, ln["d"].prize_minor) == (23_486, 16_513)
    assert math.floor(400 / 784.8 * 324.0 * 100) == 16_513
    assert st.refund_minor == 1
    assert st.prizes_total_minor + st.refund_minor == POOL


NUM = fb.NumericSpace(fb.Grid(0.0, 10.0, 5))
NPRIOR = fb.uniform(NUM)


def test_numeric_example() -> None:
    assert fb.max_loss(NUM, NPRIOR) == pytest.approx(2.4)
    cmax = v1.c_max(NUM, NPRIOR, k=1, reserve=0.1, prize_pool=400)
    assert cmax == pytest.approx(150)
    a = fb.canonicalize(NUM, "pmf", [0.1, 0.2, 0.4, 0.2, 0.1])
    b = fb.canonicalize(NUM, "pmf", [0.0, 0.1, 0.3, 0.4, 0.2])
    assert fb.crps(NPRIOR, NUM.grid, 5.4) == pytest.approx(0.80)
    assert fb.crps(a, NUM.grid, 5.4) == pytest.approx(0.40)
    assert fb.crps(b, NUM.grid, 5.4) == pytest.approx(0.82)
    subs = [
        fb.Submission(1, "a", "A", a, cmax),
        fb.Submission(2, "b", "B", b, cmax),
        fb.Submission(3, "c", "C", b, cmax),
    ]
    st = v1.settle_cash(NUM, NPRIOR, subs, 5.4, prize_pool_minor=POOL)
    assert [r.points for r in st.scores] == pytest.approx([60.0, -63.0, 0.0], abs=1e-9)
    assert st.net_total == pytest.approx(-3.0, abs=1e-9)
    ln = lines(st)
    assert (ln["a"].lock_minor, ln["b"].lock_minor, ln["c"].lock_minor) == (6_000, 24_300, 0)
    assert ln["a"].prize_minor == 6_000
    assert (ln["b"].credits_burned_minor, ln["b"].lock_released_minor) == (6_300, 18_000)
    assert st.refund_minor == 34_000
