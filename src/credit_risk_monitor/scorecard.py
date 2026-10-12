"""Points-based scorecard built on a logistic regression over WoE features.

Scaling follows the usual convention: a score of 600 at good:bad odds of
50:1, and every 20 points (PDO) doubles the odds. Higher score = safer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .woe import WoEBinner

BASE_SCORE = 600
BASE_ODDS = 50
PDO = 20


@dataclass
class Scorecard:
    binner: WoEBinner
    model: LogisticRegression
    dropped: list[str] = field(default_factory=list)

    @property
    def factor(self) -> float:
        return PDO / np.log(2)

    @property
    def offset(self) -> float:
        return BASE_SCORE - self.factor * np.log(BASE_ODDS)

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.binner.transform(X))[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Convert each account's log-odds of being good into scorecard points."""
        log_odds_bad = self.model.decision_function(self.binner.transform(X))
        return np.round(self.offset - self.factor * log_odds_bad).astype(int)

    def points_table(self) -> pd.DataFrame:
        """Points contributed by every bin of every feature: the scorecard itself."""
        n = len(self.binner.bins)
        intercept = self.model.intercept_[0]
        rows = []
        for coef, (feature, fb) in zip(self.model.coef_[0], self.binner.bins.items()):
            for i, woe in enumerate(fb.woe):
                points = self.offset / n - self.factor * (coef * woe + intercept / n)
                rows.append(
                    {
                        "feature": feature,
                        "lower": fb.edges[i],
                        "upper": fb.edges[i + 1],
                        "woe": woe,
                        "points": round(points),
                    }
                )
        return pd.DataFrame(rows)


def fit(X: pd.DataFrame, y: pd.Series, max_bins: int = 6) -> Scorecard:
    """Fit WoE bins and a logistic regression, removing wrong-sign features.

    On WoE inputs every coefficient should be negative: a safer bin (higher WoE)
    must lower the log-odds of default. A positive coefficient usually means the
    feature overlaps heavily with another one, so the worst offender is dropped
    and the model refitted until every sign makes business sense.
    """
    binner = WoEBinner(max_bins=max_bins).fit(X, y)
    woe = binner.transform(X)
    kept = list(X.columns)
    dropped: list[str] = []
    while True:
        model = LogisticRegression(max_iter=1000).fit(woe[kept], y)
        coefs = pd.Series(model.coef_[0], index=kept)
        wrong_sign = coefs[coefs > 0]
        if wrong_sign.empty:
            break
        worst = wrong_sign.idxmax()
        kept.remove(worst)
        dropped.append(worst)
    binner.bins = {col: binner.bins[col] for col in kept}
    return Scorecard(binner=binner, model=model, dropped=dropped)
