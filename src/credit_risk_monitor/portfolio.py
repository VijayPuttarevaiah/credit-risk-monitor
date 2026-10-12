"""Portfolio views: risk grades, delinquency, roll rates, expected loss, stress.

Expected loss uses the standard EL = PD x LGD x EAD. The dataset gives no
recovery or drawdown history, so LGD and the credit conversion factor (CCF)
are stated assumptions, not estimates. PD here is the model's probability of
default on next month's payment, so EL is a next-month figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LGD = 0.85   # assumed loss given default for unsecured card balances
CCF = 0.50   # assumed share of the unused limit drawn before default

GRADES = ["A", "B", "C", "D", "E"]
GRADE_PD_CUTS = [0.0, 0.10, 0.20, 0.35, 0.55, 1.0]

# Each scenario multiplies every account's odds of default. Shocking odds rather
# than PD keeps probabilities below 1 and hits riskier accounts harder in
# absolute terms, which is closer to how a downturn behaves.
STRESS_SCENARIOS = {"baseline": 1.00, "moderate": 1.30, "severe": 1.75}

DELINQUENCY_BUCKETS = {0: "current", 1: "1 month late", 2: "2 months late"}


def assign_grades(pd_scores: np.ndarray) -> pd.Series:
    return pd.cut(pd_scores, GRADE_PD_CUTS, labels=GRADES, include_lowest=True).astype(str)


def exposure_at_default(df: pd.DataFrame) -> pd.Series:
    limit = df["LIMIT_BAL"].astype(float)
    balance = df["BILL_AMT1"].clip(lower=0).astype(float).clip(upper=limit)
    return balance + CCF * (limit - balance)


def grade_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """frame needs columns: grade, pd, default, ead, expected_loss."""
    out = (
        frame.groupby("grade")
        .agg(
            accounts=("default", "size"),
            avg_pd=("pd", "mean"),
            observed_bad_rate=("default", "mean"),
            exposure=("ead", "sum"),
            expected_loss=("expected_loss", "sum"),
        )
        .reindex(GRADES)
        .dropna(how="all")
        .reset_index()
    )
    out["share_of_accounts"] = out["accounts"] / out["accounts"].sum()
    return out


def delinquency_buckets(df: pd.DataFrame) -> pd.DataFrame:
    bucket = df["PAY_1"].map(DELINQUENCY_BUCKETS).fillna("3+ months late")
    out = (
        pd.DataFrame({"bucket": bucket, "default": df["default"]})
        .groupby("bucket")
        .agg(accounts=("default", "size"), next_month_default_rate=("default", "mean"))
        .reset_index()
    )
    order = list(DELINQUENCY_BUCKETS.values()) + ["3+ months late"]
    return out.set_index("bucket").reindex(order).dropna(how="all").reset_index()


def roll_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Share of accounts moving between delinquency states from month 2 to month 1."""
    prev = df["PAY_2"].clip(upper=3)
    curr = df["PAY_1"].clip(upper=3)
    labels = {0: "current", 1: "1 late", 2: "2 late", 3: "3+ late"}
    table = pd.crosstab(prev.map(labels), curr.map(labels), normalize="index")
    order = [v for v in labels.values() if v in table.index]
    return table.reindex(index=order, columns=[c for c in labels.values() if c in table.columns])


def stress_pd(pd_scores: np.ndarray, odds_multiplier: float) -> np.ndarray:
    odds = pd_scores / (1 - pd_scores)
    stressed = odds * odds_multiplier
    return stressed / (1 + stressed)


def stress_test(pd_scores: np.ndarray, ead: np.ndarray) -> pd.DataFrame:
    """Recompute expected loss and grade mix under each odds-shock scenario."""
    rows = []
    for name, multiplier in STRESS_SCENARIOS.items():
        stressed_pd = stress_pd(pd_scores, multiplier)
        grades = assign_grades(stressed_pd)
        rows.append(
            {
                "scenario": name,
                "odds_multiplier": multiplier,
                "avg_pd": float(stressed_pd.mean()),
                "accounts_in_d_e": int(grades.isin(["D", "E"]).sum()),
                "expected_loss": float((stressed_pd * LGD * ead).sum()),
            }
        )
    out = pd.DataFrame(rows)
    out["change_vs_baseline"] = out["expected_loss"] / out.loc[0, "expected_loss"] - 1
    return out
