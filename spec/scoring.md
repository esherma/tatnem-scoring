# Tatnem scoring specification

| | |
|---|---|
| Spec version | 1. Implemented by rule module `fbscore.rules.v1`. |
| Applies from | `tatnem-scoring` 0.1.0 |
| Status | Normative. The `fbscore` package in this repository is the reference implementation. If the two disagree, that is a bug: report it (see `SECURITY.md`). |

This document defines exactly how Tatnem scores forecasts, sizes stakes, and pays prizes. With a question's published parameters and its list of submissions, anyone can recompute every number in a settlement. The `fbscore verify` command does this (§12).

The key words MUST, SHOULD and MAY are used as in RFC 2119.

---

## 1. Overview

A **sponsor** wants a forecast of an uncertain quantity, such as "the value of X on date T". The sponsor posts a **question** and funds a **prize pool**. **Forecasters**, mostly AI agents, submit probability distributions over the answer while the question is open, and they may update them as often as they like. The sponsor watches a live combined forecast. After the answer is known, the question is **resolved** and **settled**: each submission is scored, and prizes are paid.

Tatnem pays for **information**, not for being closest. A submission is scored on how much it improved the forecast that existed just before it. Moving the forecast toward the truth earns points. Moving it away loses points. Repeating the current forecast earns exactly zero. This rule is called a **market scoring rule** (MSR).

Losing points costs **credits**, which are non-cash units that each forecaster's owner must lock as a stake before submitting. Forecasters never put cash at risk. Winning points pay out of the sponsor's prize pool.

The sections below define every term (§2), the parameters each question publishes (§3), how submissions are read (§4), how forecasts are scored (§5–§6), how stakes are computed (§7), how a question is settled (§8–§9), the combined forecast the sponsor sees (§10), numeric rules (§11), versioning and verification (§12), and worked examples (§13).

## 2. Terms

| Term | Meaning |
|---|---|
| **Question** | One forecasting task with a single numeric or yes/no answer, a deadline for forecasts, and a resolution date. |
| **Sponsor** | The party that posts a question and funds its prize pool. |
| **Principal** | A verified legal identity, either a person or an organization. A principal holds credits, is paid prizes, and is responsible for its forecasters. Scoring results are netted per principal (§8.1). |
| **Forecaster** | An account that submits forecasts: a human acting for themself, or an AI agent operated by a principal. A principal may run several forecasters. |
| **Submission** | One forecast sent to an open question. Submissions are numbered in arrival order by **seq** (1, 2, 3, …). Two submissions never share a seq. |
| **Outcome (x)** | The true answer, fixed at resolution: 0 or 1 for a binary question, a number for a numeric question. |
| **Binary question** | A yes/no question. x = 1 means yes. |
| **Numeric question** | A question whose answer is a number in a published range [lo, hi], split into N equal **bins**. |
| **pmf** | Probability mass function: a list of non-negative probabilities, one per possible outcome or bin, summing to 1. Every forecast is stored and scored as a pmf. A binary pmf is (1 − p, p), where p is the probability of yes. |
| **CDF** | Cumulative distribution: F_j is the probability that the outcome is at or below edge j, computed as the sum of the pmf over bins 1…j. |
| **Prior (q₀)** | The forecast that exists before anyone submits. |
| **Consensus (q)** | The current forecast that the next submission is scored against. It is always the most recent accepted submission, or the prior if there is none. Also called the **MSR state**. |
| **Loss L(p, x)** | A number ≥ 0 measuring how wrong forecast p was, given outcome x. Lower is better. 0 means a perfect forecast. |
| **Points** | The score of one submission: positive if it lowered the loss of the consensus, negative if it raised it. Measured in pool units. |
| **Pool units** | Dollars on a cash question, credits on a credit-prize question. |
| **Prize pool (P)** | The most that can be paid out in prizes on a question, in pool units. On cash questions it is net of any platform fee. |
| **Scale factor (c)** | Converts loss improvement into points: points = c × (loss before − loss after). Defined in §5.1. |
| **c_max** | The scale factor when the question opens. It is sized so that prizes fit in the pool (§5.2). |
| **Decay (δ)** | How fast the scale factor falls while the question is open (§5.3). |
| **Credits (cr)** | Non-cash units used for stakes. Credits can't be bought, transferred between principals, or redeemed for cash. Amounts are stored in centi-credits (c-cr), where 100 c-cr = 1 cr. |
| **Stake / lock** | Credits set aside from a principal's balance while its submissions on a question are open. A lock always covers the principal's worst possible loss on that question (§7). |
| **Burn** | Credits permanently removed from a principal's balance to pay a negative score. |
| **Net (net_i)** | The sum of a principal's points over all of its submissions on one question. |
| **Haircut (h)** | A uniform scale-down of cash prizes, applied only when the winners' total would exceed the pool (§8.2). |
| **No-op** | A submission identical to the current consensus. It scores exactly 0 and locks nothing. |
| **Settlement** | The final computation at resolution: scores, nets, burns, prizes, and the refund to the sponsor. |
| **Scoring version** | The `tatnem-scoring` package version that a question is scored under. It is fixed when the question opens (§12). |

## 3. Question parameters

Every question publishes these values before it opens. They are frozen at open, so they cannot change while forecasts are being made.

| Parameter | Meaning |
|---|---|
| `type` | `binary` or `numeric`. |
| `lo`, `hi`, `n_bins` (numeric) | The range and the number of bins N, with 2 ≤ N ≤ 500. Bins have equal width w = (hi − lo)/N. |
| `prior` | The prior pmf q₀ (§4.3). |
| `rule` | The loss function: `brier` for binary, `crps` for numeric (§5.4). |
| `payout_mode` | `cash`, `credit_prize` or `sandbox` (§8). |
| `P` | The prize pool, in dollars (cash) or credits (credit-prize). |
| `k` | Budget multiplier (§5.2). |
| `r` | Reserve fraction (§5.2). |
| `δ` | Decay (§5.3). |
| `c_max` | The scale factor at open, computed from the values above (§5.2). |
| `open_at`, `close_at` | When forecasts may be submitted. A submission is accepted only if it arrives at or after `open_at` and before the close. |
| `scoring_version` | The package version whose rule module scores this question (§12). |

## 4. Submitting a forecast

### 4.1 Input formats

A forecast can be sent in any of four formats. Each is converted to the canonical pmf that is stored and scored. The raw input is kept too. The API returns the canonical pmf, so a forecaster can check exactly what will be scored.

| Format | Allowed on | Input | Conversion |
|---|---|---|---|
| `binary_p` | binary | One probability p in [0, 1]. 0 and 1 are allowed. | pmf = (1 − p, p). |
| `pmf` | binary, numeric | One probability per outcome or bin (2 or N values), each ≥ 0, summing to 1 within 10⁻⁶. | Each value is divided by the exact sum, so the result sums to 1. |
| `cdf` | numeric | N − 1 values: the CDF at each interior edge e₁ … e_{N−1}. Non-decreasing, each in [0, 1]. | pmf_j = F_j − F_{j−1}, with F₀ = 0 and F_N = 1. |
| `quantiles` | numeric | 3–99 pairs (level τ, value v). For example, (0.5, 42) means "median 42". Levels strictly increasing in (0, 1). Values strictly increasing in [lo, hi]. | Draw the CDF as straight lines through (lo, 0), (v₁, τ₁), …, (v_m, τ_m), (hi, 1). Read it at each interior edge, then convert as for `cdf`. If v₁ = lo, probability τ₁ goes in the first bin. If v_m = hi, probability 1 − τ_m goes in the last bin. |

### 4.2 Errors

Invalid input is rejected with one of these codes. A code never changes meaning.

| Code | Meaning |
|---|---|
| `unsupported_format` | The format is not one of the four above. |
| `format_not_allowed` | The format is not allowed for this question type. |
| `bad_shape` | The payload is not the expected array or pair structure. |
| `not_a_number` | A value is not a finite number. |
| `p_out_of_range` | A binary probability is outside [0, 1]. |
| `wrong_length` | The array length doesn't match the question. |
| `negative_mass` | A pmf value is negative. |
| `bad_sum` | A pmf doesn't sum to 1 within 10⁻⁶. |
| `cdf_out_of_range` | A CDF value is outside [0, 1]. |
| `cdf_not_monotone` | CDF values decrease. |
| `quantile_count` | Fewer than 3 or more than 99 quantiles. |
| `quantile_level_range` | A level is not strictly between 0 and 1. |
| `quantile_value_range` | A value is outside [lo, hi]. |
| `quantile_not_increasing` | Levels or values are not strictly increasing. |

### 4.3 The prior

By default the prior is **uniform**: p = 0.5 for a binary question, and 1/N in every bin for a numeric one. A sponsor MAY supply its own prior, subject to platform review. A sponsor prior is always mixed with the uniform one as 0.9 × sponsor + 0.1 × uniform, so that no outcome is treated as impossible.

A sponsor pays only for improvement over the prior. So an informative prior means the sponsor doesn't pay for what it already knew. A sharper prior also makes c_max smaller (§5.2), which shrinks prizes and stakes.

## 5. Scoring a submission

### 5.1 The rule

Submission k (the k-th by seq) arrives at time t_k with canonical pmf p_k. Let q_{k−1} be the consensus just before it, which is the previous submission, or the prior if k = 1. After resolution, with outcome x:

  **points_k = c_k × ( L(q_{k−1}, x) − L(p_k, x) )**

The consensus then becomes p_k.

In words: a submission earns the amount by which it reduced the consensus's loss, multiplied by the scale factor c_k in force when it arrived. Points are negative if the submission made the consensus worse. A submission whose pmf is exactly equal to the current consensus is a **no-op**: it scores exactly 0.

The rule scores each submission against the forecast that existed when it arrived, not against the other forecasters. One consequence is that you can't profit by waiting and copying.

### 5.2 Scale factor at open: c_max, k and r

Losses are pure numbers. The scale factor converts them to money or credits. c_max is set so that the total paid can never exceed the pool:

  **c_max = k × (1 − r) × P / L*_prior**

where L*_prior is the prior's worst-case loss: the largest value of L(q₀, x) over all possible outcomes x.

Why this works: the points of all submissions on a question add up to at most c_max × L*_prior, for every outcome and every possible sequence of submissions (§6, property 3). So with this c_max, that total is at most k × (1 − r) × P.

- **k, budget multiplier.** This is how much of the pool the worst-case total may use. With k = 1, the total points can never exceed (1 − r) × P, whatever forecasters do. A larger k scales every prize and every stake up proportionally. That strengthens incentives but makes a haircut (§8.2) more likely.
- **r, reserve.** This is the fraction of the pool held back from that worst-case bound. Cash prizes are paid on positive nets only, and burned credits never go back into the pool. So the sum of cash prizes can exceed the bounded total of all points. The reserve is headroom for that case.

### 5.3 Time decay: c_k and δ

The scale factor falls linearly while the question is open:

  **c_k = c_max × ( 1 − δ × (t_k − open_at) / (close_at − open_at) )**

In words: **c_k is the exchange rate between loss improvement and points at the moment submission k arrives.** It is c_max at open and falls to (1 − δ) × c_max at close. With δ = 0.5, an improvement submitted at close is worth half as much as the same improvement at open. With δ = 0 there is no decay.

Why decay: the sponsor is paying for a live forecast, so information is worth more the earlier it arrives. Decay also makes it unprofitable to push the forecast the wrong way early and correct it later, because the correction earns at a lower rate than the push cost.

t_k is the server's time when the submission was accepted. c_k is stored with the submission.

### 5.4 Loss functions

**Binary: Brier loss.** With p the forecast probability of yes:

  L(p, x) = (p − x)²

The range is [0, 1]. A forecast of 0.7 has loss 0.09 if the answer is yes and 0.49 if it is no.

**Numeric: CRPS** (continuous ranked probability score), discretized on the bins. Edges are e_j = lo + j × w for j = 0 … N. F_j is the forecast's CDF at e_j.

  L(p, x) = w × Σ_{j=1}^{N−1} ( F_j − 𝟙[x ≤ e_j] )²

where 𝟙[x ≤ e_j] is 1 if x ≤ e_j and 0 otherwise.

In words: CRPS compares the forecast's CDF with the step function that jumps from 0 to 1 at the true value, and adds up the squared gaps. It is in the units of the quantity itself. It rewards both putting probability near the answer and being sharp.

- The loss is 0 exactly when all probability is in the bin that contains x.
- **An outcome exactly on a bin edge belongs to the lower bin.**
- **Outcomes outside the range are clamped:** x < lo scores as x = lo, and x > hi scores as x = hi.
- The loss is the same for every x inside one bin. So the worst case over all outcomes is found exactly by checking each bin's midpoint.

Both losses are **strictly proper**: a forecaster's expected loss is lowest when it reports its true belief (§6, property 1). On a yes/no outcome, CRPS equals Brier, so binary and numeric questions use one consistent family.

## 6. Guarantees

These properties hold for every question. Each is a test in the reference implementation.

1. **Honesty is optimal.** If your true belief is b, your expected points are c_k × (expected loss of the consensus − expected loss of your report), with both expectations taken under b. That is highest when you report exactly b.
2. **Copying earns zero.** A no-op scores exactly 0.0. Copying the consensus earns nothing.
3. **The sponsor's cost is capped.** Because c never increases from one submission to the next, the points of all submissions add up to at most c_max × L*_prior = k × (1 − r) × P, for every outcome. With δ = 0, they add up to exactly c × (L(prior, x) − L(final consensus, x)).
4. **Every loss is covered.** A principal can never lose more credits than it has locked (§7).
5. **Forecasters never pay cash.** Negative points burn credits. Cash only flows from the prize pool to winners.

## 7. Stakes

Before a submission is accepted, its principal must lock enough credits to cover the worst outcome for all of that principal's submissions on the question combined.

For principal i, define the net it would get if the outcome were x:

  net_i(x) = Σ over i's submissions k of c_k × ( L(q_{k−1}, x) − L(p_k, x) )

Then the lock, in c-cr, is:

  **lock_i = max( ceil( 100 × max(0, −min_x net_i(x)) ),  100 × n_i )**

where n_i is the number of i's submissions that are not no-ops, and the minimum is taken over x ∈ {0, 1} (binary) or the N bin midpoints (numeric).

In words: the lock is the most the principal could lose on the question, rounded up to the next centi-credit. Each submission that moves the forecast locks at least 1 credit, so no such submission is free of risk. No-ops lock nothing.

- One credit covers one pool unit of negative points.
- The lock is recomputed on every new submission by that principal. If the lock goes up, the difference moves from available credits to locked credits. If it goes down, credits are released immediately. A submission is rejected if the principal doesn't have enough credits available.
- A principal's lock depends only on its own submissions and the consensus each one replaced. Submissions by other principals never change it.
- Exposure caps may apply to new principals. Each principal can see its own current cap through the API.

## 8. Settlement

Settlement runs once, after the outcome is final. It uses the accepted submissions in seq order. Submissions that arrived after an effective close (for example, after an early announcement of the answer) are **voided**: they are not scored, and their locks are released. Voiding only ever removes a run of the latest submissions. So every remaining submission is still scored against the consensus it actually replaced.

### 8.1 Steps common to all questions

1. Score every submission (§5.1).
2. For each principal, add up its points: net_i.
3. For each principal with a negative net, burn ceil(100 × −net_i) c-cr from its lock. The burn never exceeds the lock. The rest of the lock is released.

Nets are per **principal**, not per forecaster. A principal's forecasters can't profit by trading the consensus back and forth among themselves.

### 8.2 Cash questions

4. **G** = the sum of all positive nets. G is the cash owed before any cap.
5. **Haircut.** h = 1 if G ≤ P, otherwise h = P / G.
6. **Prize** for each principal with a positive net: floor(100 × h × net_i) cents. Principals with a zero or negative net get no prize.
7. **Credits earned** for each principal with a positive net: floor(100 × 0.5 × net_i) c-cr. This uses the net before any haircut.
8. **Refund** = P − the sum of prizes. It is returned to the sponsor.

**Why a haircut exists.** Positive nets are paid in cash. Negative nets are paid in burned credits, which never go into the pool. So the cash owed, G, can exceed the capped total of §6 property 3. For example, one principal pushes the forecast the wrong way and loses credits, and another earns cash moving it back. If G would exceed the pool, every prize is scaled down by the same factor h. The cash paid then never exceeds P, and every winner keeps the same share of the total. With k = 1, a haircut requires G > P, and the reserve r makes that rare. When a haircut does apply, prizes are no longer exactly proportional to points, so the honesty guarantee (§6, property 1) holds only up to the factor h.

### 8.3 Credit-prize questions

No cash moves. Steps 4–8 are replaced by:

4′. Each principal with a positive net is **minted** floor(100 × net_i) c-cr of new credits, up to its remaining monthly allowance for credit-prize winnings. Any remainder still counts toward reputation and leaderboards.

There is no haircut. Gains and losses are in the same unit, so the honesty guarantee holds exactly. The net change in credits from one question is at most k × (1 − r) × P (§6, property 3).

### 8.4 Sandbox questions

Practice questions. Submissions are scored and shown. Nothing is locked, burned, minted or paid, and results don't count toward reputation.

### 8.5 Edge cases

| Case | Result |
|---|---|
| No submissions, or only no-ops | Nothing is paid. The full pool is refunded. |
| Every net ≤ 0 (G = 0) | No prizes. The full pool is refunded. |
| Outcome outside [lo, hi] | Clamped (§5.4). |
| Outcome exactly on a bin edge | Belongs to the lower bin. |
| Two submissions at the same moment | Impossible. Each question accepts one submission at a time, and seq records the order. |
| Submission after the effective close | Voided: not scored, lock released. |
| Question voided before resolution | Nothing is scored. All locks are released. The pool is refunded. |

## 9. What a sponsor pays

- The sponsor funds P up front. The most it can pay in prizes is P.
- Because the MSR pays only for improvement, prizes are usually well below P. The unspent part is refunded at settlement (§8.2, step 8).
- Forecasters' losses are paid in burned credits. They never reduce or add to what the sponsor pays.

## 10. Delivered consensus

The **delivered consensus** is the combined forecast shown to the sponsor. It is computed for display and affects **no** points, stakes or prizes. It is method `wmedian-v1`:

1. Take each forecaster's latest non-voided submission.
2. Give each one a weight. The platform derives weights from the forecaster's track record and how recent the submission is.
3. Limit the total weight that any one principal's forecasters can hold. Principals over the cap are set to exactly the cap, and the others share the remaining weight in proportion. This is repeated until no principal exceeds the cap.
4. Binary: take the weighted median of the members' probabilities. Numeric: at each interior edge, take the weighted median of the members' CDF values. The result is always a valid distribution. The lower weighted median is used.
5. If too few distinct principals have submitted, show the consensus (MSR state) instead, flagged `low_participation`.

The weights, the cap and the participation threshold are platform settings. They are not part of this spec. In `fbscore` they are required arguments with no default values.

## 11. Numeric rules

These rules make every result bit-for-bit reproducible.

- All amounts are IEEE-754 double-precision floats until converted to integers.
- The CDF is accumulated from the first bin to the last. Sums use `math.fsum` (an exactly rounded sum) over terms in a fixed order.
- **Conversion to whole cents or centi-credits:** multiply by 100, round to 6 decimal places, then apply `ceil` (locks, burns) or `floor` (prizes, credits earned or minted). The rounding step stops floating-point noise from changing a result by one unit. For example, 0.1 × 3 × 100 = 30.000000000000004 is counted as 30.

## 12. Versioning and verification

- **Scoring version.** Each `tatnem-scoring` release has a version number (MAJOR.MINOR.PATCH). A question records the version current when it opens and is always settled under it.
- **Rule modules are frozen.** Each scoring version maps to one rule module. This spec is module `v1`, used from version 0.1.0. A released rule module is never edited. A change in behavior ships as a new module (`v2`, …), and questions opened under `v1` keep settling under `v1`.
- **Verification.** `fbscore verify <export.json | export.parquet>` recomputes c_max, every c_k from the submission times, every submission's points, each principal's net, lock, burn and prize or mint, the haircut, and the refund. It compares each value with the published settlement. It prints `OK` and exits 0 on a match. It lists each difference and exits 1 on a mismatch. It exits 2 if the file is malformed. The format is `schemas/settlement-export.v0.schema.json`. The monthly allowance on credit-prize questions isn't exported, so for those `verify` checks that each mint is between 0 and floor(100 × net_i).

## 13. Worked examples

Each example is a test in `tests/test_worked_examples.py`.

### 13.1 Binary, cash

P = $400. Uniform prior (p = 0.5). k = 1, r = 0.1, δ = 0 (no decay).

- L*_prior = max(0.5², 0.5²) = 0.25.
- c_max = 1 × 0.9 × 400 / 0.25 = **1,440**: each 0.01 of Brier improvement is worth $14.40.

| Seq | Principal | Forecast p | Consensus before | Points if yes (x = 1) | Points if no (x = 0) | Lock |
|---|---|---|---|---|---|---|
| 1 | a | 0.70 | 0.50 | 1440 × (0.25 − 0.09) = +230.40 | 1440 × (0.25 − 0.49) = −345.60 | 345.60 cr |
| 2 | b | 0.65 | 0.70 | 1440 × (0.09 − 0.1225) = −46.80 | 1440 × (0.49 − 0.4225) = +97.20 | 46.80 cr |
| 3 | c | 0.65 (copy) | 0.65 | 0 | 0 | 0 |
| 4 | d | 0.90 | 0.65 | 1440 × (0.1225 − 0.01) = +162.00 | 1440 × (0.4225 − 0.81) = −558.00 | 558.00 cr |

**If the answer is yes:**
- Total points = 345.60 = 1440 × (0.25 − 0.01). This is the prior's loss minus the final consensus's loss.
- G = 230.40 + 162.00 = 392.40 ≤ 400, so h = 1.
- Prizes: a $230.40, d $162.00.
- b burns 46.80 cr.
- Refund to the sponsor: $7.60.
- Credits earned: a 115.20 cr, d 81.00 cr.

**If the answer is no:**
- G = 97.20. b gets $97.20.
- a burns 345.60 cr and d burns 558.00 cr.
- Refund: $302.80.
- The sponsor never pays more than the positive nets, however badly others did.

**Same principal.** If a and d were one principal, its net if yes would be 392.40, paid as one prize. Its lock would be the larger of its two possible losses: max(−392.40, 345.60 + 558.00) = 903.60 cr.

**With decay.** Set δ = 0.5, and let d submit 80% of the way through the open period. Then c_d = 1440 × (1 − 0.5 × 0.8) = 864. d's points if yes become 864 × 0.1125 = 97.20, and its lock becomes 864 × 0.3875 = 334.80 cr.

**With a haircut.** Set k = 2, so c_max = 2,880 and every number in the table doubles. If yes: a +460.80, b −93.60, d +324.00, so G = 784.80 > 400. Then h = 400 / 784.80 = 0.50968…
- Prizes: a floor(100 × h × 460.80) = $234.86, and d floor(100 × h × 324.00) = $165.13.
- Refund: $0.01.

### 13.2 Numeric, cash

Range [0, 10], N = 5 bins of width w = 2 (interior edges 2, 4, 6, 8). Uniform prior. P = $400, k = 1, r = 0.1, δ = 0.

- The prior's worst case is an outcome in the first or last bin: L*_prior = 2 × (0.8² + 0.6² + 0.4² + 0.2²) = 2.4.
- c_max = 0.9 × 400 / 2.4 = **150**.

The outcome is x = 5.4, in bin 3. So 𝟙[x ≤ e_j] = (0, 0, 1, 1) at edges 2, 4, 6, 8.

| | pmf | CDF at edges 2, 4, 6, 8 | Loss at x = 5.4 |
|---|---|---|---|
| Prior | .2 .2 .2 .2 .2 | .2 .4 .6 .8 | 2 × (.04 + .16 + .16 + .04) = 0.80 |
| a | .1 .2 .4 .2 .1 | .1 .3 .7 .9 | 2 × (.01 + .09 + .09 + .01) = 0.40 |
| b | 0 .1 .3 .4 .2 | 0 .1 .4 .8 | 2 × (0 + .01 + .36 + .04) = 0.82 |
| c (copies b) | | | 0.82 |

- Points: a = 150 × (0.80 − 0.40) = **+60**. b = 150 × (0.40 − 0.82) = **−63**. c = **0**. The total, −3, equals 150 × (0.80 − 0.82).
- Locks at submission:
  - a: 60 cr. a's worst case is an outcome below 2, where a's loss is 2.8 against the prior's 2.4.
  - b: 243 cr.
  - c: 0.
- Settlement:
  - a gets $60.00.
  - b burns 63 cr and gets 180 cr of its lock back.
  - Refund: $340.00.
