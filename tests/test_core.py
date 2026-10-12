import numpy as np
import pandas as pd
import pytest

from credit_risk_monitor import commentary, metrics, portfolio
from credit_risk_monitor.woe import WoEBinner


def test_ks_and_gini_on_perfect_ranking():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert metrics.ks(y, scores) == pytest.approx(1.0)
    assert metrics.gini(y, scores) == pytest.approx(1.0)


def test_psi_is_zero_for_identical_and_large_for_shifted():
    rng = np.random.default_rng(0)
    base = rng.uniform(0, 1, 5000)
    assert metrics.psi(base, base) == pytest.approx(0.0, abs=1e-6)
    assert metrics.psi(base, np.clip(base + 0.3, 0, 1)) > 0.25


def test_woe_marks_the_risky_bin_negative():
    x = pd.DataFrame({"util": np.r_[np.full(500, 0.1), np.full(500, 0.9)]})
    y = pd.Series(np.r_[np.zeros(450), np.ones(50), np.zeros(250), np.ones(250)].astype(int))
    binner = WoEBinner(max_bins=2).fit(x, y)
    woe = binner.transform(x)["util"]
    assert woe.iloc[0] > 0 > woe.iloc[-1]
    assert binner.iv_table()["information_value"].iloc[0] > 0


def test_stress_pd_stays_a_probability_and_increases():
    pd_scores = np.array([0.01, 0.2, 0.9, 0.99])
    stressed = portfolio.stress_pd(pd_scores, 1.75)
    assert np.all(stressed <= 1) and np.all(stressed > pd_scores)


def test_expected_loss_matches_pd_lgd_ead():
    raw = pd.DataFrame({"LIMIT_BAL": [100_000], "BILL_AMT1": [40_000]})
    ead = portfolio.exposure_at_default(raw).iloc[0]
    assert ead == pytest.approx(40_000 + portfolio.CCF * 60_000)


FACTS = {"gini": 0.484, "ks": 0.393, "expected_loss_baseline": 497700249, "severe_stress_increase_pct": 44.3}


def test_verify_accepts_numbers_from_the_facts():
    text = "Gini is 0.484, KS 0.393, and expected loss of NT$497,700,249 rises 44.3% in the severe case."
    assert commentary.verify(text, FACTS) == []


def test_verify_rejects_an_invented_number():
    assert commentary.verify("Expected loss rises 60% in the severe case.", FACTS) == [60.0]


def test_generate_falls_back_to_template_when_draft_has_bad_numbers(monkeypatch):
    facts = {
        "gini": 0.484, "ks": 0.393, "auc": 0.742, "psi_dev_vs_holdout": 0.001,
        "share_d_e_pct": 23.0, "el_share_d_e_pct": 43.2, "default_rate_current_pct": 13.8,
        "default_rate_2plus_late_pct": 69.6, "expected_loss_baseline": 497700249,
        "expected_loss_severe": 718429226, "severe_stress_increase_pct": 44.3,
        "accounts_in_d_e_baseline": 6892, "accounts_in_d_e_severe": 9467,
    }
    monkeypatch.setattr(commentary, "draft_with_claude", lambda f: "Losses could triple to 1.5 billion.")
    text, source = commentary.generate(facts)
    assert source == "template"
    assert commentary.verify(text, facts) == []


def test_missing_credentials_falls_back_instead_of_crashing(monkeypatch):
    pytest.importorskip("anthropic")
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", "/nonexistent")  # no `ant auth login` profile either
    assert commentary.draft_with_claude({"gini": 0.484}) is None


def test_groq_without_key_falls_back(monkeypatch):
    pytest.importorskip("groq")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert commentary.draft_with_groq({"gini": 0.484}) is None


def test_generate_uses_the_named_provider(monkeypatch):
    monkeypatch.setitem(commentary.DRAFTERS, "groq", lambda f: "The Gini is 0.484.")
    assert commentary.generate({"gini": 0.484}, provider="groq") == ("The Gini is 0.484.", "groq")
