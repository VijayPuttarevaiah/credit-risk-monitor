"""Model performance and stability metrics used by credit risk teams."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-6


def auc(y_true, pd_scores) -> float:
    return float(roc_auc_score(y_true, pd_scores))


def gini(y_true, pd_scores) -> float:
    """Gini = 2 * AUC - 1. 0 is a coin flip, 1 is perfect ranking."""
    return 2 * auc(y_true, pd_scores) - 1


def ks(y_true, pd_scores) -> float:
    """Kolmogorov-Smirnov: the widest gap between the cumulative bad and good curves."""
    fpr, tpr, _ = roc_curve(y_true, pd_scores)
    return float(np.max(tpr - fpr))


def psi(expected, actual, bins: int = 10) -> float:
    """Population Stability Index between a reference and a new score distribution.

    Rule of thumb: below 0.10 stable, 0.10-0.25 worth watching, above 0.25 a real shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e_share = np.histogram(expected, edges)[0] / len(expected) + EPS
    a_share = np.histogram(actual, edges)[0] / len(actual) + EPS
    return float(np.sum((a_share - e_share) * np.log(a_share / e_share)))


def calibration_table(y_true, pd_scores, bins: int = 10) -> pd.DataFrame:
    """Predicted versus observed default rate by score decile."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(pd_scores)})
    frame["decile"] = pd.qcut(frame["pd"], bins, labels=False, duplicates="drop") + 1
    return (
        frame.groupby("decile")
        .agg(accounts=("y", "size"), predicted_pd=("pd", "mean"), observed_default_rate=("y", "mean"))
        .reset_index()
    )


def summary(y_true, pd_scores) -> dict[str, float]:
    return {"auc": auc(y_true, pd_scores), "gini": gini(y_true, pd_scores), "ks": ks(y_true, pd_scores)}
