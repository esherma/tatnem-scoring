"""Simulation tests S1-S8 (TDD §12.3). Marked ``slow``: nightly and on scoring changes.

Run with ``pytest -m slow``. Assertions use 95% paired bootstrap CIs over questions.
Set ``SIM_REPORT=path.json`` to write the measured values (including the spam regime,
which is recorded, not asserted).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SIMS = Path(__file__).resolve().parents[1] / "sims"
sys.path.insert(0, str(SIMS))
import fbsim  # noqa: E402

pytestmark = pytest.mark.slow

N_QUESTIONS = 2000
# In the no-spam production regime the extremizer trails honest_mid by ~1.4 per account
# (TDD baseline 14.6 vs 15.5). Resolving that gap at 95% needs about 8,000 questions.
N_QUESTIONS_S7 = 8000
REPORT: dict[str, Any] = {}


def ci(d: np.ndarray, seed: int = 12345, n_boot: int = 4000) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)


@pytest.fixture(scope="module")
def with_spam() -> fbsim.Results:
    return fbsim.run(N_QUESTIONS, n_spam=3, seed=0)


@pytest.fixture(scope="module")
def no_spam() -> fbsim.Results:
    return fbsim.run(N_QUESTIONS_S7, n_spam=0, seed=0)


def diff(res: fbsim.Results, scheme: dict[str, list[float]], a: str, b: str) -> tuple[float, float, float]:
    return ci(res.per_account(scheme, a) - res.per_account(scheme, b))


def test_s1_copier_earns_exactly_zero(with_spam: fbsim.Results, no_spam: fbsim.Results) -> None:
    for res in (with_spam, no_spam):
        assert all(v == 0.0 for v in res.reference["copier"])
        assert all(v == 0.0 for v in res.production_cash["copier"])


def test_s2_extremizer_below_honest_mid(no_spam: fbsim.Results) -> None:
    for name, scheme in (("reference", no_spam.reference), ("production", no_spam.production_cash)):
        m, _, hi = diff(no_spam, scheme, "extremizer", "honest_mid")
        REPORT[f"S2_{name}_extremizer_minus_mid"] = m
        assert hi < 0, (name, m, hi)


def test_s3_skill_ordering(no_spam: fbsim.Results) -> None:
    for name, scheme in (("reference", no_spam.reference), ("production", no_spam.production_cash)):
        for a, b in (("honest_sharp", "honest_mid"), ("honest_mid", "honest_weak")):
            m, lo, _ = diff(no_spam, scheme, a, b)
            REPORT[f"S3_{name}_{a}_minus_{b}"] = m
            assert lo > 0, (name, a, b, m, lo)


def test_s2_s3_with_spam_are_recorded(with_spam: fbsim.Results) -> None:
    # With 3 spammers most points come from correcting spam and the skill gaps shrink to a few
    # dollars per account, inside the noise at 2,000 questions. Recorded, not asserted.
    for a, b in (("extremizer", "honest_mid"), ("honest_sharp", "honest_mid"), ("honest_mid", "honest_weak")):
        REPORT[f"spam_reference_{a}_minus_{b}"] = diff(with_spam, with_spam.reference, a, b)
    assert REPORT


def test_s4_spam_loses(with_spam: fbsim.Results) -> None:
    spam = with_spam.per_account(with_spam.reference, "spam") * with_spam.n_accounts["spam"]
    m, _, hi = ci(spam)
    REPORT["S4_spam_group_net"] = m
    REPORT["S4_p_spam_net_negative"] = float(np.mean(spam < 0))
    assert hi < 0
    assert np.mean(spam < 0) >= 0.9


def test_s5_budget_and_cash_caps_hold_in_every_question(with_spam: fbsim.Results, no_spam: fbsim.Results) -> None:
    for res in (with_spam, no_spam):
        eps = 1e-9 * res.reference_bound
        assert max(res.reference_cost) <= res.reference_bound + eps
        assert max(res.production_net_cost) <= res.production_bound + eps
        assert max(res.production_cash_paid) <= fbsim.POOL_MINOR / 100


def test_s6_delivered_beats_msr_beats_prior(with_spam: fbsim.Results) -> None:
    deliv, msr, prior = (
        np.asarray(v) for v in (with_spam.crps_delivered, with_spam.crps_msr_final, with_spam.crps_prior)
    )
    REPORT["S6_crps"] = {"delivered": float(deliv.mean()), "msr_final": float(msr.mean()), "prior": float(prior.mean())}
    assert ci(deliv - msr)[2] < 0
    assert ci(msr - prior)[2] < 0


def test_s7_production_settlement_no_spam(no_spam: fbsim.Results) -> None:
    assert not any(no_spam.haircut)
    cash = no_spam.production_cash
    REPORT["S7_cash_per_account"] = {t: float(no_spam.per_account(cash, t).mean()) for t in fbsim.TYPES[:4]}
    assert diff(no_spam, cash, "honest_sharp", "honest_mid")[1] > 0
    assert diff(no_spam, cash, "honest_mid", "honest_weak")[1] > 0
    assert diff(no_spam, cash, "extremizer", "honest_mid")[2] < 0


def test_s8_msr_vs_closest_wins(with_spam: fbsim.Results) -> None:
    assert all(v == 0.0 for v in with_spam.reference["copier"])
    m, lo, _ = ci(np.asarray(with_spam.naive["copier"]))
    REPORT["S8_naive_copier_group"] = m
    assert lo > 0


def test_spam_regime_is_recorded_not_asserted(with_spam: fbsim.Results) -> None:
    cash = with_spam.production_cash
    REPORT["spam_regime"] = {
        "haircut_frequency": float(np.mean(with_spam.haircut)),
        "cash_per_account": {t: float(with_spam.per_account(cash, t).mean()) for t in fbsim.TYPES},
    }
    assert max(with_spam.production_cash_paid) <= fbsim.POOL_MINOR / 100  # the cap still holds


# ---- the original reference scripts reproduce the TDD tables exactly (seed 0) -------------


def _run_script(name: str, args: list[str], cwd: Path) -> str:
    shutil.copy(SIMS / name, cwd / name)
    env = {**os.environ, "MPLBACKEND": "Agg"}
    out = subprocess.run([sys.executable, name, *args], cwd=cwd, capture_output=True, text=True, check=True, env=env)
    return out.stdout


def test_reference_sim_reproduces_tdd_baselines(tmp_path: Path) -> None:
    out = _run_script("forecast_bounty_sim.py", [], tmp_path)
    rows = {
        ln.split()[0]: [float(v) for v in ln.split()[1:]]
        for ln in out.splitlines()
        if re.match(r"^(honest_\w+|extremizer|copier|spam)\s+-?\d", ln)
    }
    # columns: naive group, naive/acct, tournament group, tournament/acct, msr group, msr/acct
    assert rows["honest_sharp"][5] == 120.8
    assert rows["honest_mid"][5] == 115.7
    assert rows["honest_weak"][5] == 105.6
    assert rows["extremizer"][5] == 104.4
    assert rows["copier"][5] == 0.0
    assert rows["spam"][5] == -464.9
    assert rows["copier"][0] == 355.4
    assert "60.4" in out and "5.84" in out and "8.38" in out and "2.66" in out


def test_settlement_variants_reproduce_tdd_tables(tmp_path: Path) -> None:
    out = _run_script("settlement_variants.py", ["1500"], tmp_path)
    blocks = out.split("k=")[1:]
    defaults = next(b for b in blocks if b.startswith("1.0 full=False spam=0 delta=0.5 reserve=0.1 prior=gauss"))
    for label, value in (("sharp", "26.2"), ("mid", "15.5"), ("weak", "8.7"), ("extrem", "14.6")):
        assert re.search(rf"{label}\s+cash/acct\s+{re.escape(value)}\b", defaults), (label, defaults)
    assert "haircut freq 0.00" in defaults
    k2 = next(b for b in blocks if b.startswith("2.0"))
    assert "cash paid/P mean 0.41" in k2 and "haircut freq 0.04" in k2


def teardown_module() -> None:
    path = os.environ.get("SIM_REPORT")
    if path:
        Path(path).write_text(json.dumps(REPORT, indent=2, sort_keys=True) + "\n")
