# Tatnem scoring specification

| | |
|---|---|
| Spec version | 1 (rule module `fbscore.rules.v1`) |
| First package version | `tatnem-scoring` 0.1.0 |
| Status | Normative. `fbscore` is the reference implementation. |

This document defines how every submission on a Tatnem question is scored, how much each forecaster must lock as a stake, and how a question is settled. It is complete: with a question's published parameters and its submissions, anyone can recompute every payout (§10). Conventions: **MUST / SHOULD / MAY** as in RFC 2119. A *principal* is the legal identity (person or organization) that holds credits and receives prizes. A *forecaster* is a human or agent that submits on a principal's behalf. Amounts marked `cr` are credits, which are non-cash.

## 1. Outcome spaces and losses

Each question has an outcome space Ω, a canonical distribution p (a pmf), a realized outcome x and a loss L(p, x) ≥ 0. Lower loss is better.

### 1.1 Binary questions: Brier loss

p = P(x = 1), and the pmf is (1 − p, p).

  L(p, x) = (p − x)²,  x ∈ {0, 1},  range [0, 1].

Brier is bounded (so stakes are finite), strictly proper on all of [0, 1], and equal to CRPS on a binary outcome, so one family covers both question types.

### 1.2 Numeric questions: discretized CRPS

A numeric question has a range [lo, hi] and N equal-width bins (2 ≤ N ≤ 500). Edges are e₀ = lo < e₁ < … < e_N = hi, with width w = (hi − lo)/N and e_j = lo + (hi − lo)·j/N. For a pmf p, F_j = Σ_{i≤j} p_i is the CDF at edge e_j.

  L(p, x) = w · Σ_{j=1}^{N−1} (F_j − 𝟙[x ≤ e_j])²

- L = 0 exactly when all mass is in the bin containing x.
- L is constant in x on each bin (e_{j−1}, e_j]. **An outcome exactly on an edge belongs to the lower bin.**
- **Out-of-range outcomes clamp:** x < lo scores as x = lo, and x > hi scores as x = hi. The formula does this automatically, because the indicator vector is then all ones or all zeros.
- Because L is constant per bin, max_x L(p, x) is evaluated exactly over the N bin midpoints.

### 1.3 Numerics

- F_j is accumulated left to right in floating point. The sum over j uses `math.fsum` over the terms in increasing j, with (F_j)² for edges below the outcome's bin and (F_j − 1)² from it upward.
- Point amounts are IEEE-754 double precision. Results are bit-for-bit reproducible across conforming platforms.
- **Conversion to minor units** (cents, or centi-credits where 100 c-cr = 1 cr): an amount a is first rounded to 1e−6 minor units, `round(100·a, 6)`, then ceiled or floored as each rule below says. This makes `ceil`/`floor` immune to binary floating-point noise.

## 2. Canonicalization

A submission arrives in one of four formats and is converted to the canonical pmf that gets scored. The raw input is stored as well. Invalid input is rejected with a stable error code (§2.1).

| Format | Question type | Rule |
|---|---|---|
| `binary_p` | binary | p ∈ [0, 1] (0 and 1 are allowed). pmf = (1 − p, p). |
| `pmf` | both | Length = number of outcomes (2 or N), all entries ≥ 0, sum within 1e−6 of 1. Renormalized by dividing each entry by the `fsum` of the entries. |
| `cdf` | numeric | N − 1 values at e₁…e_{N−1}, non-decreasing, in [0, 1]. pmf = diff([0, F, 1]). |
| `quantiles` | numeric | 3–99 pairs (τ_i, v_i), strictly increasing in both, τ ∈ (0, 1), v ∈ [lo, hi]. The CDF is piecewise linear through (lo, 0), (v₁, τ₁), …, (v_m, τ_m), (hi, 1), evaluated at the interior edges and then differenced. If v₁ = lo, mass τ₁ sits in the first bin; if v_m = hi, mass 1 − τ_m sits in the last. |

### 2.1 Error codes

`unsupported_format`, `format_not_allowed`, `not_a_number`, `p_out_of_range`, `wrong_length`, `negative_mass`, `bad_sum`, `cdf_out_of_range`, `cdf_not_monotone`, `quantile_count`, `quantile_level_range`, `quantile_value_range`, `quantile_not_increasing`, `bad_shape`. A code never changes meaning once published.

## 3. Prior

The prior q₀ is the MSR state before the first submission. The default is **uniform**: p = 0.5 for binary, 1/N per bin for numeric. A sponsor MAY supply a prior, subject to review. It is mixed as **0.9 · sponsor + 0.1 · uniform**, so that no region is effectively ruled out.

Under an MSR the sponsor pays only for improvement over the prior. A sharper prior does not reduce worst-case cost, though. A point-mass prior has a larger maximum loss than a uniform one, so c_max goes down and prizes shrink.

## 4. The scale factor and the budget cap

Losses are dimensionless. The scale factor c converts a loss improvement into pool units: dollars of prize on a cash question, or credits on a credit-prize question.

Let P be the prize pool (in major units) and L*_prior = max_x L(q₀, x). Then

  c_max = k · (1 − r) · P / L*_prior

where k (budget multiplier) and r (reserve) are per-question parameters, frozen at open and published with the question. Platform defaults are k = 1 and r = 0.10.

## 5. Per-submission points, with time decay

Submission k arrives at time t_k (server time when its question lock was taken). With q_{k−1} the MSR state just before it (q₀ = prior) and p_k its canonical report:

  c_k = c_max · (1 − δ · (t_k − open_at) / (close_at − open_at))

  points_k = c_k · (L(q_{k−1}, x) − L(p_k, x))

  the MSR state becomes q_k = p_k.

δ is the per-question decay, frozen at open (platform default 0.5). c_k is stored on the submission. A submission whose pmf is bitwise equal to q_{k−1} is a **no-op**: its points are exactly 0.0.

Properties (each is a test in the reference implementation):

1. **Strict properness.** For a forecaster with belief b, E_{x∼b}[points_k] = c_k · (E_b L(q_{k−1}) − E_b L(p_k)), uniquely maximized at p_k = b.
2. **Copying earns exactly zero.**
3. **Budget bound under decay.** Because c is non-increasing in seq order, Σ_k c_k (L_{k−1} − L_k) ≤ c₁ L₀ ≤ c_max · L*_prior = k(1 − r)P for every outcome and every submission sequence. With δ = 0 the sum telescopes to c · (L(prior, x) − L(final, x)).
4. **Why decay.** It makes early information worth more than late information (the sponsor is buying a *live* forecast), and it makes "push wrong early, correct later" strictly unprofitable in points.

`norm_points_k = (L(q_{k−1}, x) − L(p_k, x)) / L*_prior` is the scale-free information contribution used for reputation and leaderboards.

## 6. Stakes

Settlement nets points per **principal** per question, so the stake is computed per principal. For principal i:

  net_i(x) = Σ_{k ∈ subs(i)} c_k (L(q_{k−1}, x) − L(p_k, x))

  lock_i = max( ceil_minor( max(0, max_x −net_i(x)) ),  100 c-cr × (number of i's non-no-op submissions) )

The max is over x ∈ {0, 1} (binary) or the N bin midpoints (numeric), which is exact. One credit stands for one unit of points. **Every non-no-op submission locks at least 1 cr**, so zero-risk spam is impossible. No-ops lock nothing.

On each new submission the platform recomputes lock_i and moves the difference between the principal's available and locked credits. It rejects the submission if the principal has too few credits available. Each term of net_i depends only on i's report and the state just before it. So a later submission by another principal never changes lock_i. Exposure caps may apply to new principals. Each principal can see its own current cap.

## 7. Settlement

Settlement runs once, at resolution, over the accepted and non-voided submissions in seq order. Voiding is always a time suffix (submissions after an effective close), so the chain of the remaining submissions is unchanged. A mid-chain submission is never removed from the chain.

```
1. for each submission k: L_before, L_report, points_k, norm_points_k
2. for each principal i: net_i = fsum of i's points in seq order
3. lock_i as in §6;  burned_i = ceil_minor(max(−net_i, 0))   (≤ lock_i; asserted)
```

### 7.1 Cash questions

```
4. G = fsum over principals of max(net_i, 0);  h = 1 if G ≤ P else P / G     (the haircut)
5. prize_i = floor_minor(h · net_i) for net_i > 0, else 0                    (cents)
6. credits_earned_i = floor_minor(α · max(net_i, 0)),  α = 0.5               (raw net, not haircut-adjusted)
7. refund = P − Σ prize_i  (≥ 0) → back to the sponsor
```

Losses are burned credits and never reach the sponsor, so cash out is G (capped at P), not the telescoped net. With the default k = 1, a haircut is rare. Where one applies, strict properness holds only up to the haircut.

### 7.2 Credit-prize questions

No haircut and no cash. Positive nets **mint credits 1:1**, up to each principal's remaining monthly mint allowance. The rest of a net still counts for reputation:

```
4'. minted_i = min(floor_minor(max(net_i, 0)), remaining allowance of i)
```

Net credit supply from one question is the telescoped total, which is at most k(1 − r)·P_cr by §5.3. Because gains and losses are in the same unit, the rule is strictly proper for anyone who values credits roughly linearly.

### 7.3 Sandbox questions

Scores are computed for display. There are no locks, no credits and no money.

### 7.4 Edge cases

| Case | Rule |
|---|---|
| No submissions, or only no-ops | Nothing is paid. A cash pool is refunded in full. |
| G = 0 | Every net ≤ 0, so there is no cash. The pool is refunded. |
| Outcome outside [lo, hi] | Clamp (§1.2). |
| Outcome exactly on a bin edge | Lower bin. |
| Two submissions "at the same time" | Impossible: seq is assigned under a per-question lock. |
| Submission after the effective close | Voided: unscored, lock released. |

## 8. Delivered consensus (method `wmedian-v1`)

The delivered consensus is what the sponsor receives. It affects **no** payout or stake. Shape:

1. Members: for each forecaster, its latest non-voided report.
2. Each member has a weight derived from its track record and the recency of its report. The platform computes the weights.
3. A per-principal cap limits the total weight that any one principal's forecasters can hold. It is applied by water-filling: capped principals are pinned at the cap, and the rest share the remainder in proportion.
4. Binary: the weighted median of the members' p. Numeric: at each interior edge, the weighted median of the members' CDF values. A pointwise median of non-decreasing functions is non-decreasing, so the result is a valid CDF. The median is the *lower* weighted median.
5. With too few distinct principals, the MSR state is delivered instead, flagged `low_participation`.

The tuning values (weights, cap, thresholds) are platform configuration and are not part of this spec. In `fbscore` they are required arguments with no defaults.

## 9. Visibility of the rules

Everything that decides a forecaster's points, stake or prize is in this document or in the question's published parameters: grid, prior, rule, k, r, δ, c_max, P, open and close times, and scoring version. Submissions are monitored for collusion and abuse. Detection never changes the rules above. A principal found abusing them may have prizes withheld or locks burned under the platform terms.

## 10. Versioning and verification

- **The scoring version is the `tatnem-scoring` package version** (semver). A question freezes its `scoring_version` when it opens.
- Each scoring version maps to one frozen rule module. This spec is rule module `v1`, used from package version 0.1.0. **A released rule module is never changed in place.** A behavior change ships as a new module (`v2`, …) in a new release, and questions frozen at an older version keep settling under their own module.
- **Anyone can verify a settlement.** `fbscore verify <export.json|export.parquet>` recomputes c_max, every c_k (from the submission times), every submission's points, each principal's net, lock, burn and prize or mint, the haircut and the refund, and compares them with the published settlement. The export format is `schemas/settlement-export.v0.schema.json`. For credit-prize questions the remaining mint allowance is not exported, so verify checks that each published mint is between 0 and floor_minor(max(net, 0)).

## 11. Worked examples

All are tests in `tests/test_worked_examples.py`.

**Binary.** P = $400, uniform prior (L*_prior = 0.25), k = 1, r = 0.1, so c_max = 0.9 · 400 / 0.25 = **1,440**. δ = 0.

| Seq | Principal | Report | Lock | Points if x = 1 | Points if x = 0 |
|---|---|---|---|---|---|
| 1 | a | 0.70 | 345.60 cr | +230.40 | −345.60 |
| 2 | b | 0.65 | 46.80 cr | −46.80 | +97.20 |
| 3 | c (copies the state) | 0.65 | 0 (no-op) | 0 | 0 |
| 4 | d | 0.90 | 558.00 cr | +162.00 | −558.00 |

x = 1: net = 345.6 = 1440 · (0.25 − 0.01), which telescopes. G = 392.40 ≤ 400, so h = 1. Prizes: a $230.40, d $162.00. b burns 46.80 cr. Refund $7.60. Credits earned (α = 0.5): a 115.20 cr, d 81.00 cr.

x = 0: G = 97.20. b gets $97.20. a burns 345.60 cr and d burns 558.00 cr. Refund $302.80.

If a and d were one principal, the lock is max(−(230.4 + 162), 345.6 + 558) = 903.60 cr.

With δ = 0.5 and d submitting 80% of the way through the window: c_d = 864, d's points at x = 1 are 97.20, and d's lock is 334.80 cr.

With k = 2 (c_max = 2,880) at x = 1: a = +460.80, b = −93.60, d = +324.00, G = 784.80 > 400, so h = 400/784.8 = 0.50968… Prizes are floored to the cent: a $234.86, d $165.13. The refund is $0.01.

**Numeric.** Range [0, 10], N = 5 (w = 2), uniform prior, P = $400, k = 1, r = 0.1, δ = 0. L*_prior = 2.4, so c_max = **150**. x = 5.4.

| | pmf | L(·, 5.4) |
|---|---|---|
| Prior | [.2, .2, .2, .2, .2] | 0.80 |
| a | [.1, .2, .4, .2, .1] | 0.40 |
| b | [0, .1, .3, .4, .2] | 0.82 |
| c = copy of b | | 0.82 |

Points: a +60, b −63, c 0. Net −3 = 150 · (0.80 − 0.82). Locks: a 60 cr, b 243 cr, c 0. Settlement: a gets $60.00. b burns 63 cr (180 cr released). Refund $340.00.
