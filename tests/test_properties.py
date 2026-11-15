"""Property-based tests (TDD §12.2, public-repo rows). Profiles: ci (default), nightly, dev."""

from __future__ import annotations

import math
import random
from itertools import pairwise

from hypothesis import assume, given
from hypothesis import strategies as st

import fbscore as fb
from fbscore.rules import v1

EXAMPLE_PRINCIPAL_CAP = 0.3
EXAMPLE_CAP_MIN_PRINCIPALS = 4
EXAMPLE_LOW_PARTICIPATION_MIN_PRINCIPALS = 1

PRINCIPALS = ("p1", "p2", "p3", "p4", "p5")


@st.composite
def spaces(draw: st.DrawFn) -> fb.OutcomeSpace:
    if draw(st.booleans()):
        return fb.BinarySpace()
    lo = draw(st.floats(-1e3, 1e3))
    span = draw(st.floats(0.5, 1e3))
    return fb.NumericSpace(fb.Grid(lo, lo + span, draw(st.integers(2, 12))))


@st.composite
def pmfs(draw: st.DrawFn, space: fb.OutcomeSpace) -> tuple[float, ...]:
    if isinstance(space, fb.BinarySpace):
        return fb.canonicalize(space, "binary_p", draw(st.floats(0.0, 1.0)))
    raw = draw(st.lists(st.floats(0.0, 1.0), min_size=space.size, max_size=space.size))
    assume(math.fsum(raw) > 1e-3)
    total = math.fsum(raw)
    return fb.canonicalize(space, "pmf", [v / total for v in raw])


@st.composite
def chains(draw: st.DrawFn, *, decay: bool) -> tuple[fb.OutcomeSpace, tuple[float, ...], list[fb.Submission], float]:
    space = draw(spaces())
    prior = draw(pmfs(space)) if draw(st.booleans()) else fb.uniform(space)
    assume(fb.max_loss(space, prior) > 1e-9)
    cmax = v1.c_max(space, prior, k=draw(st.floats(0.1, 3.0)), reserve=0.1, prize_pool=draw(st.floats(1.0, 1e4)))
    n = draw(st.integers(1, 30))
    times = sorted(draw(st.lists(st.floats(0.0, 1.0), min_size=n, max_size=n)))
    delta = draw(st.floats(0.0, 1.0)) if decay else 0.0
    subs: list[fb.Submission] = []
    state = prior
    for k, t in enumerate(times):
        pmf = state if draw(st.integers(0, 5)) == 0 else draw(pmfs(space))  # sometimes copy
        c = v1.c_at(c_max=cmax, decay=delta, open_at=0.0, close_at=1.0, at=t)
        subs.append(fb.Submission(k + 1, draw(st.sampled_from(PRINCIPALS)), f"f{k}", pmf, c))
        state = pmf
    x = draw(st.sampled_from(space.representative_outcomes))
    return space, prior, subs, x


def _expected(space: fb.OutcomeSpace, belief: tuple[float, ...], report: tuple[float, ...]) -> float:
    xs = space.representative_outcomes
    return math.fsum(b * fb.loss(space, report, x) for b, x in zip(belief, xs, strict=True))


@given(st.data())
def test_strict_properness(data: st.DataObject) -> None:
    space = data.draw(spaces())
    b = data.draw(pmfs(space))
    r = data.draw(pmfs(space))
    fb_, fr = fb.losses.cdf_at_interior_edges(b), fb.losses.cdf_at_interior_edges(r)
    assume(max((abs(u - v) for u, v in zip(fb_, fr, strict=True)), default=abs(b[1] - r[1])) > 1e-4)
    if isinstance(space, fb.BinarySpace):
        assume(abs(b[1] - r[1]) > 1e-4)
    assert _expected(space, b, r) > _expected(space, b, b)


@given(chains(decay=True))
def test_copy_scores_exactly_zero(case: tuple[fb.OutcomeSpace, tuple[float, ...], list[fb.Submission], float]) -> None:
    space, prior, subs, x = case
    st_ = v1.settle_sandbox(space, prior, subs, x)
    prev = prior
    for s, row in zip(subs, st_.scores, strict=True):
        if s.pmf == prev:
            assert row.points == 0.0 and row.is_noop
        prev = s.pmf


@given(chains(decay=False))
def test_telescoping_without_decay(case: tuple[fb.OutcomeSpace, tuple[float, ...], list[fb.Submission], float]) -> None:
    space, prior, subs, x = case
    st_ = v1.settle_sandbox(space, prior, subs, x)
    c = subs[0].c
    expected = c * (fb.loss(space, prior, x) - fb.loss(space, subs[-1].pmf, x))
    assert math.isclose(math.fsum(r.points for r in st_.scores), expected, abs_tol=1e-9 * max(1.0, c) * len(subs))


@given(chains(decay=True))
def test_budget_bound_under_decay(case: tuple[fb.OutcomeSpace, tuple[float, ...], list[fb.Submission], float]) -> None:
    space, prior, subs, _ = case
    bound = subs[0].c * fb.max_loss(space, prior)  # c_1 L*_prior <= c_max L*_prior
    for x in space.representative_outcomes:
        total = math.fsum(r.points for r in v1.settle_sandbox(space, prior, subs, x).scores)
        assert total <= bound + 1e-9 * max(1.0, bound) * len(subs)


@given(chains(decay=True))
def test_lock_covers_loss(case: tuple[fb.OutcomeSpace, tuple[float, ...], list[fb.Submission], float]) -> None:
    space, prior, subs, _ = case
    for x in space.representative_outcomes:
        st_ = v1.settle_cash(space, prior, subs, x, prize_pool_minor=10**9)
        for line in st_.lines:
            assert -line.net_points <= line.lock_minor / 100 + 1e-9
            assert line.credits_burned_minor <= line.lock_minor


@given(chains(decay=True), st.data())
def test_lock_independent_of_later_other_principals(
    case: tuple[fb.OutcomeSpace, tuple[float, ...], list[fb.Submission], float], data: st.DataObject
) -> None:
    space, prior, subs, x = case
    before = {ln.principal_id: ln.lock_minor for ln in v1.settle_cash(space, prior, subs, x, prize_pool_minor=1).lines}
    extra = fb.Submission(subs[-1].seq + 1, "zz", "fz", data.draw(pmfs(space)), subs[-1].c)
    after = {
        ln.principal_id: ln.lock_minor
        for ln in v1.settle_cash(space, prior, [*subs, extra], x, prize_pool_minor=1).lines
    }
    assert all(after[p] == lock for p, lock in before.items())


@given(chains(decay=True), st.integers(1, 10**7))
def test_cash_cap(case: tuple[fb.OutcomeSpace, tuple[float, ...], list[fb.Submission], float], pool: int) -> None:
    space, prior, subs, x = case
    st_ = v1.settle_cash(space, prior, subs, x, prize_pool_minor=pool)
    assert st_.prizes_total_minor <= pool
    assert 0.0 < st_.haircut <= 1.0
    assert st_.refund_minor >= 0
    assert st_.prizes_total_minor + st_.refund_minor == pool


@st.composite
def member_sets(draw: st.DrawFn) -> tuple[fb.OutcomeSpace, list[fb.Member]]:
    space = draw(spaces())
    n = draw(st.integers(1, 15))
    members = [
        fb.Member(f"f{i}", draw(st.sampled_from(PRINCIPALS)), draw(st.floats(1e-3, 1e3)), draw(pmfs(space)))
        for i in range(n)
    ]
    return space, members


def _consensus(space: fb.OutcomeSpace, members: list[fb.Member]) -> fb.Consensus:
    return fb.delivered_consensus(
        space,
        members,
        msr_state=fb.uniform(space),
        principal_cap=EXAMPLE_PRINCIPAL_CAP,
        cap_min_principals=EXAMPLE_CAP_MIN_PRINCIPALS,
        low_participation_min_principals=EXAMPLE_LOW_PARTICIPATION_MIN_PRINCIPALS,
    )


@given(member_sets())
def test_delivered_consensus_validity(case: tuple[fb.OutcomeSpace, list[fb.Member]]) -> None:
    space, members = case
    out = _consensus(space, members)
    assert len(out.pmf) == space.size
    assert all(v >= 0.0 for v in out.pmf)
    cdf = fb.losses.cdf_at_interior_edges(out.pmf)
    assert all(0.0 <= f <= 1.0 + 1e-12 for f in cdf)
    assert all(b >= a for a, b in pairwise(cdf))
    assert math.isclose(math.fsum(out.pmf), 1.0, abs_tol=1e-12)
    if out.cap_applied:
        by_p: dict[str, float] = {}
        weights = dict(out.weights)
        for m in members:
            by_p[m.principal_id] = by_p.get(m.principal_id, 0.0) + weights[m.forecaster_id]
        assert all(v <= EXAMPLE_PRINCIPAL_CAP + 1e-12 for v in by_p.values())
        assert math.isclose(math.fsum(weights.values()), 1.0, abs_tol=1e-12)


@given(member_sets(), st.randoms(use_true_random=False))
def test_consensus_is_order_independent(case: tuple[fb.OutcomeSpace, list[fb.Member]], rnd: random.Random) -> None:
    space, members = case
    shuffled = members[:]
    rnd.shuffle(shuffled)
    assert _consensus(space, members) == _consensus(space, shuffled)


@given(chains(decay=True), st.randoms(use_true_random=False))
def test_settlement_is_deterministic(
    case: tuple[fb.OutcomeSpace, tuple[float, ...], list[fb.Submission], float], rnd: random.Random
) -> None:
    space, prior, subs, x = case
    a = v1.settle_cash(space, prior, subs, x, prize_pool_minor=123_456)
    b = v1.settle_cash(space, prior, list(subs), x, prize_pool_minor=123_456)
    assert a == b
    # Principal-level results do not depend on the order lines are produced in.
    assert [ln.principal_id for ln in a.lines] == sorted(ln.principal_id for ln in a.lines)
    _ = rnd


@given(st.data())
def test_quantiles_canonicalize_to_valid_pmf(data: st.DataObject) -> None:
    g = fb.Grid(0.0, 100.0, data.draw(st.integers(2, 200)))
    m = data.draw(st.integers(3, 20))
    levels = sorted(set(data.draw(st.lists(st.floats(0.001, 0.999), min_size=m, max_size=m))))
    values = sorted(set(data.draw(st.lists(st.floats(0.0, 100.0), min_size=m, max_size=m))))
    k = min(len(levels), len(values))
    assume(k >= 3)
    pmf = fb.canonicalize(fb.NumericSpace(g), "quantiles", [[levels[i], values[i]] for i in range(k)])
    assert all(v >= 0.0 for v in pmf)
    assert math.isclose(math.fsum(pmf), 1.0, abs_tol=1e-12)
