"""Port of ``forecast_bounty_sim.py`` / ``settlement_variants.py`` onto ``fbscore``.

Same population and information model as the reference sim (X ~ N(50, 15^2) on [0, 100],
100 bins, Bayesian honest forecasters, an extremizer, 10 copier sybils run by one
operator, optional spammers), but every loss, scale factor, decayed c, settlement and the
delivered consensus come from ``fbscore``. Each question is settled two ways:

* reference settlement: delta = 0, reserve = 0, k = 1, per-account net including negatives
  (the original sim's MSR payout);
* production settlement: rules v1 cash settlement with per-principal netting, haircut,
  credits burned, decay and reserve (TDD defaults k = 1, r = 0.1, delta = 0.5).

The naive inverse-distance scheme of the original proposal is also computed, for S8.
Aggregation parameters are deliberately unrealistic ``EXAMPLE_*`` fixture values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import norm

import fbscore as fb
from fbscore.rules import v1

LO, HI, NB = 0.0, 100.0, 100
SPACE = fb.NumericSpace(fb.Grid(LO, HI, NB))
EDGES = np.array(SPACE.grid.edges)
CENTERS = np.array(SPACE.grid.midpoints)
PRIOR_MU, PRIOR_SD = 50.0, 15.0
POOL_MINOR = 100_000  # $1,000

EXAMPLE_PRINCIPAL_CAP = 0.45
EXAMPLE_CAP_MIN_PRINCIPALS = 3
EXAMPLE_LOW_PARTICIPATION_MIN_PRINCIPALS = 2

TYPES = ("honest_sharp", "honest_mid", "honest_weak", "extremizer", "copier", "spam")


def agents(n_spam: int) -> list[tuple[str, str, float | None]]:
    return (
        [("honest_sharp", "honest", 5.0)] * 3
        + [("honest_mid", "honest", 10.0)] * 4
        + [("honest_weak", "honest", 20.0)] * 3
        + [("extremizer", "extremizer", 10.0)] * 3
        + [("copier", "copier", None)] * 10
        + [("spam", "spam", None)] * n_spam
    )


def gauss(mu: float, sd: float) -> tuple[float, ...]:
    p = np.clip(np.diff(norm.cdf(EDGES, mu, sd)), 1e-12, None)
    return fb.canonicalize(SPACE, "pmf", (p / p.sum()).tolist())


def posterior(signals: list[tuple[float, float]]) -> tuple[float, float]:
    prec, num = 1 / PRIOR_SD**2, PRIOR_MU / PRIOR_SD**2
    for s, sd in signals:
        prec += 1 / sd**2
        num += s / sd**2
    return num / prec, float(np.sqrt(1 / prec))


def pmf_median(p: tuple[float, ...]) -> float:
    return float(CENTERS[np.searchsorted(np.cumsum(p), 0.5)])


@dataclass
class Results:
    """Per-question, per-type totals (group sums over accounts of that type)."""

    n_accounts: dict[str, int]
    reference: dict[str, list[float]] = field(default_factory=lambda: {t: [] for t in TYPES})
    production_cash: dict[str, list[float]] = field(default_factory=lambda: {t: [] for t in TYPES})
    naive: dict[str, list[float]] = field(default_factory=lambda: {t: [] for t in TYPES})
    reference_cost: list[float] = field(default_factory=lambda: [])
    production_net_cost: list[float] = field(default_factory=lambda: [])
    production_cash_paid: list[float] = field(default_factory=lambda: [])
    production_bound: float = 0.0
    reference_bound: float = 0.0
    haircut: list[bool] = field(default_factory=lambda: [])
    crps_prior: list[float] = field(default_factory=lambda: [])
    crps_msr_final: list[float] = field(default_factory=lambda: [])
    crps_delivered: list[float] = field(default_factory=lambda: [])

    def per_account(self, scheme: dict[str, list[float]], t: str) -> np.ndarray:
        return np.asarray(scheme[t]) / self.n_accounts[t]


def run(n_questions: int, *, n_spam: int, seed: int = 0) -> Results:
    rng = np.random.default_rng(seed)
    population = agents(n_spam)
    counts = {t: sum(1 for a in population if a[0] == t) for t in TYPES}
    res = Results(n_accounts={t: max(c, 1) for t, c in counts.items()})
    prior = gauss(PRIOR_MU, PRIOR_SD)
    ref_c = v1.c_max(SPACE, prior, k=1.0, reserve=0.0, prize_pool=POOL_MINOR / 100)
    prod_c = v1.c_max(SPACE, prior, k=1.0, reserve=0.1, prize_pool=POOL_MINOR / 100)
    lstar = fb.max_loss(SPACE, prior)
    res.reference_bound, res.production_bound = ref_c * lstar, prod_c * lstar
    principal_of = ["copier-operator" if a[0] == "copier" else f"acct-{i}" for i, a in enumerate(population)]

    for _ in range(n_questions):
        x = float(np.clip(rng.normal(PRIOR_MU, PRIOR_SD), LO + 0.5, HI - 0.5))
        order = rng.permutation(len(population))
        T = len(order)
        revealed: list[tuple[float, float]] = []
        msr_reports: list[tuple[int, tuple[float, ...]]] = []
        naive_points: list[tuple[int, float, int]] = []
        naive_dists: list[tuple[float, ...]] = []
        state = prior
        for t, idx in enumerate(order):
            label, kind, sd = population[idx]
            if kind in ("honest", "extremizer"):
                assert sd is not None
                s = x + rng.normal(0, sd)
                mu, psd = posterior([*revealed, (s, sd)])
                dist = gauss(mu, psd * (0.4 if kind == "extremizer" else 1.0))
                revealed.append((s, sd))
                msr_dist = naive_dist = dist
            elif kind == "copier":
                msr_dist = state  # bitwise copy of the MSR state
                naive_dist = tuple(np.mean(naive_dists, axis=0).tolist()) if naive_dists else prior
            else:
                msr_dist = naive_dist = gauss(float(rng.uniform(LO, HI)), 4.0)
            msr_reports.append((int(idx), msr_dist))
            naive_dists.append(naive_dist)
            naive_points.append((int(idx), pmf_median(naive_dist), t))
            state = msr_dist

        # (a) reference settlement: no decay, no reserve, per-account net incl. negatives.
        ref_subs = [fb.Submission(k + 1, f"acct-{i}", f"acct-{i}", p, ref_c) for k, (i, p) in enumerate(msr_reports)]
        ref = v1.settle_sandbox(SPACE, prior, ref_subs, x)
        ref_by_acct = {line.principal_id: line.net_points for line in ref.lines}
        res.reference_cost.append(ref.net_total)

        # (b) production settlement under rules v1 (cash), per-principal netting.
        prod_subs = [
            fb.Submission(
                k + 1,
                principal_of[i],
                f"acct-{i}",
                p,
                v1.c_at(c_max=prod_c, decay=0.5, open_at=0.0, close_at=1.0, at=k / max(T - 1, 1)),
            )
            for k, (i, p) in enumerate(msr_reports)
        ]
        prod = v1.settle_cash(SPACE, prior, prod_subs, x, prize_pool_minor=POOL_MINOR)
        prize_by_principal = {line.principal_id: line.prize_minor / 100 for line in prod.lines}
        res.production_net_cost.append(prod.net_total)
        res.production_cash_paid.append(prod.prizes_total_minor / 100)
        res.haircut.append(prod.haircut < 1.0)

        # Naive scheme (original proposal): inverse distance, time weighted.
        w = np.array([(1 - 0.5 * t / T) / (abs(m - x) + 1) for _, m, t in naive_points])
        share = (POOL_MINOR / 100) * w / w.sum()

        for t in TYPES:
            res.reference[t].append(0.0)
            res.production_cash[t].append(0.0)
            res.naive[t].append(0.0)
        n_copiers = counts["copier"]
        for i, (label, _, _) in enumerate(population):
            res.reference[label][-1] += ref_by_acct.get(f"acct-{i}", 0.0)
            pid = principal_of[i]
            res.production_cash[label][-1] += prize_by_principal.get(pid, 0.0) / (n_copiers if label == "copier" else 1)
        for (i, _, _), sh in zip(naive_points, share, strict=True):
            res.naive[population[i][0]][-1] += float(sh)

        # Accuracy: prior vs MSR final state vs delivered consensus (wmedian primitive).
        members = [fb.Member(f"acct-{i}", principal_of[i], 1.0, p) for i, p in msr_reports]
        delivered = fb.delivered_consensus(
            SPACE,
            members,
            msr_state=state,
            principal_cap=EXAMPLE_PRINCIPAL_CAP,
            cap_min_principals=EXAMPLE_CAP_MIN_PRINCIPALS,
            low_participation_min_principals=EXAMPLE_LOW_PARTICIPATION_MIN_PRINCIPALS,
        )
        res.crps_prior.append(fb.crps(prior, SPACE.grid, x))
        res.crps_msr_final.append(fb.crps(state, SPACE.grid, x))
        res.crps_delivered.append(fb.crps(delivered.pmf, SPACE.grid, x))
    return res


def summary(res: Results) -> str:
    lines = [f"{'type':<14}{'reference/acct':>16}{'production cash/acct':>22}{'naive/acct':>12}"]
    for t in TYPES:
        if res.n_accounts[t] and res.reference[t]:
            lines.append(
                f"{t:<14}{res.per_account(res.reference, t).mean():>16.1f}"
                f"{res.per_account(res.production_cash, t).mean():>22.1f}{res.per_account(res.naive, t).mean():>12.1f}"
            )
    lines.append(
        f"CRPS prior {np.mean(res.crps_prior):.2f}  MSR final {np.mean(res.crps_msr_final):.2f}  "
        f"delivered {np.mean(res.crps_delivered):.2f}  haircut freq {np.mean(res.haircut):.2f}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    for spam in (3, 0):
        print(f"--- spam accounts: {spam}, questions: {n}")
        print(summary(run(n, n_spam=spam)))
