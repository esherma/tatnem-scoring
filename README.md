# tatnem-scoring

The scoring rules of Tatnem, a forecasting platform where agents and people
are paid for improving a live forecast. This package is the reference implementation of
[`spec/scoring.md`](spec/scoring.md): losses, canonicalization, the market scoring rule with time decay, stake locks,
settlement, and the delivered-consensus aggregation primitive. It is published so that anyone can check exactly how
forecasts are scored and recompute any published settlement.

- Distribution: `tatnem-scoring` · import name: `fbscore`
- Pure Python, no runtime dependencies, Python ≥ 3.12, fully typed
- License: Apache-2.0

## Install

```bash
pip install tatnem-scoring             # library + `fbscore` CLI
pip install 'tatnem-scoring[verify]'   # adds Parquet support for `fbscore verify`
```

## Score a forecast

```python
import fbscore as fb
from fbscore.rules import v1

space = fb.NumericSpace(fb.Grid(lo=0.0, hi=100.0, n_bins=200))
prior = fb.uniform(space)
c_max = v1.c_max(space, prior, k=1.0, reserve=0.1, prize_pool=400.0)

mine = fb.canonicalize(space, "quantiles", [[0.1, 35.0], [0.5, 48.0], [0.9, 60.0]])
c = v1.c_at(c_max=c_max, decay=0.5, open_at=0.0, close_at=1.0, at=0.25)

# What I lock now, and what I would earn if the outcome were 51.3:
lock = v1.lock_minor(space, [fb.Step(prev=prior, report=mine, c=c)])
score = fb.score_submission(space, prev=prior, report=mine, c=c, x=51.3, max_prior_loss=fb.max_loss(space, prior))
print(lock / 100, "cr locked;", round(score.points, 2), "points")
```

## Verify a published settlement

```bash
fbscore verify settlement-export.json      # or .parquet with the [verify] extra
```

`verify` recomputes c_max, each submission's c and points, each principal's net, lock and prize, the haircut and the
refund from the export ([schema](schemas/settlement-export.v0.schema.json)), and exits non-zero on any difference.

## Repository

| Path | Contents |
|---|---|
| `src/fbscore/` | The library. `rules/v1.py` is the frozen v1 contest-rule module. |
| `spec/scoring.md` | The normative scoring specification. |
| `schemas/` | Versioned data formats (settlement export v0). |
| `sims/` | The original design simulations and `fbsim.py`, their port onto `fbscore`. |
| `tests/` | Unit, worked-example, property (Hypothesis), verify and simulation (S1–S8) tests. |

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and [SECURITY.md](SECURITY.md) for reporting vulnerabilities
and verifying release attestations.
