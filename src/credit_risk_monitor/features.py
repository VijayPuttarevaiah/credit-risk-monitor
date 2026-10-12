"""Behavioural features a credit card risk team would look at.

Demographic columns (SEX, MARRIAGE, AGE) are deliberately left out of the
model. They stay in the data so the portfolio can still be reviewed for
fairness, but a lending decision should not depend on them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import BILL_AMT, PAY_AMT, PAY_STATUS

FEATURES = [
    "credit_limit",
    "utilization",
    "avg_utilization_6m",
    "worst_delinquency_6m",
    "months_delinquent_6m",
    "current_delinquency",
    "payment_ratio",
    "avg_payment_ratio_6m",
    "balance_trend",
]


def _safe_ratio(num: pd.Series, den: pd.Series, cap: float) -> pd.Series:
    out = num / den.where(den > 0)
    return out.fillna(0).clip(lower=0, upper=cap)


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame of model features, one row per account."""
    limit = df["LIMIT_BAL"].astype(float)
    bills = df[BILL_AMT].clip(lower=0).astype(float)
    pays = df[PAY_AMT].astype(float)
    status = df[PAY_STATUS]

    util_by_month = bills.div(limit, axis=0).clip(upper=2.0)
    pay_ratio_by_month = pd.DataFrame(
        {i: _safe_ratio(pays.iloc[:, i - 1], bills.iloc[:, i], cap=1.0) for i in range(1, 6)}
    )

    feats = pd.DataFrame(index=df.index)
    feats["credit_limit"] = limit
    feats["utilization"] = util_by_month["BILL_AMT1"]
    feats["avg_utilization_6m"] = util_by_month.mean(axis=1)
    feats["worst_delinquency_6m"] = status.max(axis=1)
    feats["months_delinquent_6m"] = (status > 0).sum(axis=1)
    feats["current_delinquency"] = status["PAY_1"]
    # PAY_AMT1 (paid in month 1) settles the statement from month 2.
    feats["payment_ratio"] = pay_ratio_by_month[1]
    feats["avg_payment_ratio_6m"] = pay_ratio_by_month.mean(axis=1)
    feats["balance_trend"] = (bills["BILL_AMT1"] - bills["BILL_AMT6"]) / limit
    return feats.replace([np.inf, -np.inf], 0)
