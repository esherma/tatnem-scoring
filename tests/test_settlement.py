"""Settlement modes and edge cases (spec §7-§8)."""

from __future__ import annotations

import pytest

import fbscore as fb
from fbscore.rules import v1

BIN = fb.BinarySpace()
PRIOR = fb.uniform(BIN)


def p(x: float) -> tuple[float, float]:
    return fb.canonicalize(BIN, "binary_p", x)


SUBS = [
    fb.Submission(1, "a", "A", p(0.7), 1440.0),
    fb.Submission(2, "b", "B", p(0.65), 1440.0),
    fb.Submission(3, "d", "D", p(0.9), 1440.0),
]


def test_credit_prize_mints_up_to_cap_without_haircut() -> None:
    st = v1.settle_credit_prize(BIN, PRIOR, SUBS, 1, remaining_mint_cap_minor={"a": 10_000, "b": 0, "d": 1_000_000})
    ln = {x.principal_id: x for x in st.lines}
    assert ln["a"].credits_earned_minor == 10_000  # capped (net 230.40 cr)
    assert ln["d"].credits_earned_minor == 16_200  # under cap: net minted 1:1
    assert ln["b"].credits_burned_minor == 4_680
    assert all(x.prize_minor == 0 for x in st.lines)
    assert (st.haircut, st.prizes_total_minor, st.refund_minor) == (1.0, 0, 0)


def test_credit_prize_requires_cap_for_every_principal() -> None:
    with pytest.raises(ValueError, match="no remaining mint cap"):
        v1.settle_credit_prize(BIN, PRIOR, SUBS, 1, remaining_mint_cap_minor={"a": 1})
    with pytest.raises(ValueError, match="negative"):
        v1.settle_credit_prize(BIN, PRIOR, SUBS, 1, remaining_mint_cap_minor={"a": -1, "b": 0, "d": 0})


def test_sandbox_has_scores_and_no_money() -> None:
    st = v1.settle_sandbox(BIN, PRIOR, SUBS, 1)
    assert [r.points for r in st.scores] == pytest.approx([230.4, -46.8, 162.0])
    assert all(
        (x.prize_minor, x.credits_burned_minor, x.credits_earned_minor, x.lock_minor) == (0, 0, 0, 0) for x in st.lines
    )


def test_no_submissions_refunds_everything() -> None:
    st = v1.settle_cash(BIN, PRIOR, [], 1, prize_pool_minor=40_000)
    assert (st.lines, st.refund_minor, st.gross_positive, st.haircut) == ((), 40_000, 0.0, 1.0)


def test_only_noops_refund_everything() -> None:
    subs = [fb.Submission(1, "a", "A", PRIOR, 1440.0), fb.Submission(2, "b", "B", PRIOR, 1000.0)]
    st = v1.settle_cash(BIN, PRIOR, subs, 1, prize_pool_minor=40_000)
    assert st.refund_minor == 40_000
    assert all(x.lock_minor == 0 for x in st.lines)


def test_g_zero_refunds_pool() -> None:
    st = v1.settle_cash(BIN, PRIOR, [fb.Submission(1, "a", "A", p(0.9), 100.0)], 0, prize_pool_minor=5_000)
    assert st.gross_positive == 0.0
    assert st.refund_minor == 5_000
    assert st.lines[0].credits_burned_minor == fb.ceil_minor(100 * (0.81 - 0.25))


def test_min_lock_floor_per_non_noop_submission() -> None:
    tiny = [fb.Submission(1, "a", "A", p(0.5000001), 1.0), fb.Submission(2, "a", "A", p(0.5000002), 1.0)]
    st = v1.settle_cash(BIN, PRIOR, tiny, 1, prize_pool_minor=100)
    assert st.lines[0].lock_minor == 2 * v1.MIN_LOCK_MINOR_PER_SUBMISSION
    with pytest.raises(ValueError, match=">= 0"):
        fb.lock_minor(BIN, [], min_lock_minor_per_submission=-1)


@pytest.mark.parametrize(
    ("subs", "match"),
    [
        ([fb.Submission(2, "a", "A", p(0.6), 1.0), fb.Submission(1, "b", "B", p(0.7), 1.0)], "seq"),
        ([fb.Submission(1, "a", "A", p(0.6), 1.0), fb.Submission(2, "b", "B", p(0.7), 2.0)], "non-increasing"),
        ([fb.Submission(1, "a", "A", (0.2, 0.3, 0.5), 1.0)], "length"),
        ([fb.Submission(1, "a", "A", p(0.6), -1.0)], "non-increasing"),
    ],
)
def test_input_validation(subs: list[fb.Submission], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        v1.settle_cash(BIN, PRIOR, subs, 1, prize_pool_minor=100)


def test_prior_length_and_pool_validation() -> None:
    with pytest.raises(ValueError, match="prior length"):
        v1.settle_cash(BIN, (0.2, 0.3, 0.5), [], 1, prize_pool_minor=100)
    with pytest.raises(ValueError, match="prize_pool_minor"):
        v1.settle_cash(BIN, PRIOR, [], 1, prize_pool_minor=-1)


def test_score_chain_helper_matches_settlement() -> None:
    chain = fb.score_chain(BIN, PRIOR, [(s.pmf, s.c) for s in SUBS], 1)
    st = v1.settle_sandbox(BIN, PRIOR, SUBS, 1)
    assert [c.points for c in chain] == [r.points for r in st.scores]
