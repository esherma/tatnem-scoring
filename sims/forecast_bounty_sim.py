"""
Forecast bounty payout simulation.

Compares three payout mechanisms for a sponsor-funded forecasting bounty
on a continuous quantity X, under a mix of forecaster behaviors.

Mechanisms
----------
1. naive      : point estimates; bounty split in proportion to
                time_weight / (|estimate - X| + 1)   (the original proposal)
2. tournament : distributional forecasts scored by CRPS; top 3 split 50/30/20
3. msr        : market scoring rule with CRPS; each forecaster is paid
                c * (CRPS(previous consensus) - CRPS(their report)).
                Payments telescope, so sponsor cost <= bounty by construction.
                Payments can be negative -> forecasters post a (credit) stake.

Forecaster types
----------------
honest      : Bayesian; reports posterior given base prior + own signal + the
              signals of earlier informative forecasters (assumed recoverable
              from their reported means, e.g. via track records).
extremizer  : same information as an honest sd=10 forecaster, but reports an
              overconfident posterior (sd shrunk to 40%).
copier      : no signal; one operator runs many sybil accounts, each
              submitting the current consensus.
spam        : no signal; random sharp distribution.

Model: X ~ N(50, 15^2) on [0, 100] discretized into 100 bins.
Signal_i = X + N(0, sd_i^2).
"""

import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(0)

# ---------------- setup ----------------
LO, HI, NB = 0.0, 100.0, 100
edges = np.linspace(LO, HI, NB + 1)
centers = 0.5 * (edges[:-1] + edges[1:])
BW = edges[1] - edges[0]
PRIOR_MU, PRIOR_SD = 50.0, 15.0
BOUNTY = 1000.0
N_QUESTIONS = 3000

AGENTS = (
    [("honest_sharp", "honest", 5.0)] * 3
    + [("honest_mid", "honest", 10.0)] * 4
    + [("honest_weak", "honest", 20.0)] * 3
    + [("extremizer", "extremizer", 10.0)] * 3
    + [("copier", "copier", None)] * 10       # one operator, 10 sybils
    + [("spam", "spam", None)] * 3
)
TYPES = ["honest_sharp", "honest_mid", "honest_weak", "extremizer", "copier", "spam"]


def gauss_dist(mu, sd):
    cdf = norm.cdf(edges, mu, sd)
    p = np.diff(cdf)
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def crps(p, x):
    """Discretized CRPS (ranked probability score * bin width). Lower = better."""
    F = np.cumsum(p)
    H = (centers >= x).astype(float)
    return np.sum((F - H) ** 2) * BW


def median(p):
    return centers[np.searchsorted(np.cumsum(p), 0.5)]


prior = gauss_dist(PRIOR_MU, PRIOR_SD)
# MSR scale: worst-case sponsor cost = c * max_x CRPS(prior, x) must equal BOUNTY
worst = max(crps(prior, x) for x in np.linspace(LO, HI, 201))
C_MSR = BOUNTY / worst


def posterior(signals_sds):
    prec = 1 / PRIOR_SD**2
    num = PRIOR_MU / PRIOR_SD**2
    for s, sd in signals_sds:
        prec += 1 / sd**2
        num += s / sd**2
    return num / prec, np.sqrt(1 / prec)


def run_question():
    x = np.clip(rng.normal(PRIOR_MU, PRIOR_SD), LO + 0.5, HI - 0.5)
    order = rng.permutation(len(AGENTS))
    T = len(order)

    revealed = []           # (signal, sd) of informative forecasters so far
    reports = []            # (label, dist, entry_index)
    msr_pay = {}
    consensus_msr = prior.copy()

    for t, idx in enumerate(order):
        label, kind, sd = AGENTS[idx]
        if kind in ("honest", "extremizer"):
            s = x + rng.normal(0, sd)
            mu, psd = posterior(revealed + [(s, sd)])
            if kind == "extremizer":
                psd *= 0.4
            dist = gauss_dist(mu, psd)
            revealed.append((s, sd))
        elif kind == "copier":
            # copies the current consensus (latest report under MSR;
            # linear pool of submissions so far otherwise)
            if reports:
                pool = np.mean([r[1] for r in reports], axis=0)
            else:
                pool = prior
            dist = pool
            dist_msr = consensus_msr.copy()
        else:  # spam
            dist = gauss_dist(rng.uniform(LO, HI), 4.0)

        d_for_msr = dist_msr if kind == "copier" else dist
        pay = C_MSR * (crps(consensus_msr, x) - crps(d_for_msr, x))
        msr_pay[t] = (label, pay)
        consensus_msr = d_for_msr
        reports.append((label, dist, t))

    out = {}

    # ---- 1. naive: inverse distance, time weighted ----
    w = np.array([(1 - 0.5 * r[2] / T) / (abs(median(r[1]) - x) + 1) for r in reports])
    share = BOUNTY * w / w.sum()
    naive = {k: 0.0 for k in TYPES}
    for r, sh in zip(reports, share):
        naive[r[0]] += sh
    agg_point = np.mean([median(r[1]) for r in reports])
    out["naive"] = (naive, BOUNTY, abs(agg_point - x))  # point forecast CRPS = |err|

    # ---- 2. tournament: top-3 by CRPS ----
    scores = np.array([crps(r[1], x) for r in reports])
    top = np.argsort(scores)[:3]
    tour = {k: 0.0 for k in TYPES}
    for rank, i in enumerate(top):
        tour[reports[i][0]] += BOUNTY * [0.5, 0.3, 0.2][rank]
    pool = np.mean([r[1] for r in reports], axis=0)
    out["tournament"] = (tour, BOUNTY, crps(pool, x))

    # ---- 3. MSR ----
    msr = {k: 0.0 for k in TYPES}
    for label, pay in msr_pay.values():
        msr[label] += pay
    cost = sum(p for _, p in msr_pay.values())
    out["msr"] = (msr, cost, crps(consensus_msr, x))
    out["_prior_crps"] = crps(prior, x)
    return out


# ---------------- run ----------------
counts = {k: sum(1 for a in AGENTS if a[0] == k) for k in TYPES}
mechs = ["naive", "tournament", "msr"]
tot = {m: {k: [] for k in TYPES} for m in mechs}
cost = {m: [] for m in mechs}
acc = {m: [] for m in mechs}
prior_acc = []

for _ in range(N_QUESTIONS):
    o = run_question()
    prior_acc.append(o["_prior_crps"])
    for m in mechs:
        pays, c, a = o[m]
        for k in TYPES:
            tot[m][k].append(pays[k])
        cost[m].append(c)
        acc[m].append(a)

print(f"Bounty per question: ${BOUNTY:.0f}   Questions simulated: {N_QUESTIONS}")
print(f"Accounts: " + ", ".join(f"{k}={counts[k]}" for k in TYPES))
print()
hdr = f"{'type':<14}" + "".join(f"{m+' (group)':>18}{m+' (per acct)':>18}" for m in mechs)
print(hdr)
for k in TYPES:
    row = f"{k:<14}"
    for m in mechs:
        g = np.mean(tot[m][k])
        row += f"{g:>18.1f}{g / counts[k]:>18.1f}"
    print(row)
print()
print(f"{'':<14}" + "".join(f"{m:>18}" for m in mechs))
print(f"{'sponsor cost':<14}" + "".join(f"{np.mean(cost[m]):>18.1f}" for m in mechs))
print(f"{'deliv. CRPS':<14}" + "".join(f"{np.mean(acc[m]):>18.2f}" for m in mechs)
      + f"   (prior alone: {np.mean(prior_acc):.2f}; lower is better)")
print(f"{'spam P(loss)':<14}" + "".join(
    f"{np.mean(np.array(tot[m]['spam']) < 0):>18.2f}" for m in mechs))

# ---------------- chart ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 5))
xpos = np.arange(len(TYPES))
wbar = 0.27
labels = {"naive": "Original (inverse distance)", "tournament": "Top-3 tournament",
          "msr": "Market scoring rule"}
for i, m in enumerate(mechs):
    vals = [np.mean(tot[m][k]) / counts[k] for k in TYPES]
    ax.bar(xpos + (i - 1) * wbar, vals, wbar, label=labels[m])
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(xpos)
ax.set_xticklabels(["Honest\n(sharp)", "Honest\n(mid)", "Honest\n(weak)",
                    "Extremizer", "Copier\n(per sybil)", "Spam"])
ax.set_ylabel("Avg payout per account ($)")
ax.set_title(f"Payout per account by forecaster type (${BOUNTY:.0f} bounty, {N_QUESTIONS} questions)")
ax.legend()
plt.tight_layout()
plt.savefig("payouts_by_type.png", dpi=150)
print("\nSaved payouts_by_type.png")
