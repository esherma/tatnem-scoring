"""Variants of the reference sim to size design parameters.

Settlement as proposed in the TDD:
  - per-submission MSR points with c(t) decaying linearly from c_max to (1-delta)*c_max
  - net per account per question
  - cash = positive nets, haircut pro rata if sum > prize pool P
  - negative nets burn credits (never cash)
CRPS uses the TDD's edge convention: sum_{j=1}^{N-1} (F(e_j) - 1[x <= e_j])^2 * BW
"""
import sys
import numpy as np
from scipy.stats import norm

LO, HI, NB = 0.0, 100.0, 100
edges = np.linspace(LO, HI, NB + 1)
BW = edges[1] - edges[0]
PRIOR_MU, PRIOR_SD = 50.0, 15.0
P = 1000.0


def gauss(mu, sd):
    p = np.diff(norm.cdf(edges, mu, sd))
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def crps(p, x):
    F = np.cumsum(p)[:-1]
    H = (x <= edges[1:-1]).astype(float)
    return float(np.sum((F - H) ** 2) * BW)


def post(sig):
    prec = 1 / PRIOR_SD**2
    num = PRIOR_MU / PRIOR_SD**2
    for s, sd in sig:
        prec += 1 / sd**2
        num += s / sd**2
    return num / prec, np.sqrt(1 / prec)


def run(n_q, spam, delta, reserve, prior_kind, seed=0, k=1.0, full=False):
    rng = np.random.default_rng(seed)
    agents = ([("sharp", "h", 5.0)] * 3 + [("mid", "h", 10.0)] * 4 + [("weak", "h", 20.0)] * 3
              + [("extrem", "e", 10.0)] * 3 + [("copier", "c", None)] * 10
              + ([("spam", "s", None)] * spam))
    types = sorted(set(a[0] for a in agents))
    prior = gauss(PRIOR_MU, PRIOR_SD) if prior_kind == "gauss" else np.full(NB, 1 / NB)
    worst = max(crps(prior, x) for x in edges)
    cmax = k * (1 - reserve) * P / worst
    cash = {t: [] for t in types}
    burn = {t: [] for t in types}
    G_over_P, net_over_P, haircut = [], [], 0
    for _ in range(n_q):
        x = float(np.clip(rng.normal(PRIOR_MU, PRIOR_SD), LO + 0.5, HI - 0.5))
        order = rng.permutation(len(agents))
        T = len(order)
        q = prior.copy()
        rev = []
        net = np.zeros(len(agents))
        for t, idx in enumerate(order):
            _, kind, sd = agents[idx]
            if kind in "he":
                s = x + rng.normal(0, sd)
                mu, psd = post(rev + [(s, sd)])
                if kind == "e":
                    psd *= 0.4
                d = gauss(mu, psd)
                rev.append((s, sd))
            elif kind == "c":
                d = q.copy()
            else:
                d = gauss(rng.uniform(LO, HI), 4.0)
            c = cmax * (1 - delta * t / max(T - 1, 1))
            net[idx] += c * (crps(q, x) - crps(d, x))
            q = d
        pos = np.clip(net, 0, None)
        G = pos.sum()
        scale = 1.0
        if G > P or (full and G > 0):
            scale = P / G
            haircut += 1
        G_over_P.append(G / P)
        net_over_P.append(net.sum() / P)
        for t in types:
            m = [i for i, a in enumerate(agents) if a[0] == t]
            cash[t].append(np.mean(pos[m] * scale))
            burn[t].append(np.mean(np.clip(-net[m], 0, None)))
    print(f"k={k} full={full} spam={spam} delta={delta} reserve={reserve} prior={prior_kind} cmax={cmax:.1f}")
    for t in ["sharp", "mid", "weak", "extrem", "copier", "spam"]:
        if t in cash:
            print(f"  {t:<7} cash/acct {np.mean(cash[t]):7.1f}   credits burned/acct {np.mean(burn[t]):7.1f}")
    g = np.array(G_over_P)
    print(f"  cash paid/P mean {np.mean(np.minimum(g,1)):.2f}  G/P p50 {np.median(g):.2f} p95 {np.quantile(g,.95):.2f}"
          f"  haircut freq {haircut/n_q:.2f}  net/P mean {np.mean(net_over_P):.2f}")


if __name__ == "__main__":
    # Configurations behind the tables in docs/TDD.md §6.4, §6.5, §12.3.
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    run(n, 3, 0.0, 0.0, "gauss")               # spam, no decay, no reserve
    run(n, 0, 0.0, 0.0, "gauss")               # no spam
    run(n, 0, 0.0, 0.1, "gauss")               # + reserve
    run(n, 0, 0.5, 0.1, "gauss")               # TDD defaults
    run(n, 3, 0.5, 0.1, "gauss")               # defaults with spam
    run(n, 0, 0.5, 0.1, "uniform")             # uniform prior
    run(n, 0, 0.5, 0.1, "gauss", k=2.0)
    run(n, 0, 0.5, 0.1, "gauss", k=3.0)
    run(n, 0, 0.5, 0.1, "gauss", full=True)    # pro-rata (cash/P line is pre-scaling)
