"""Weight of Evidence (WoE) binning and Information Value (IV).

WoE = ln(share of goods in the bin / share of bads in the bin).
A positive WoE means the bin is safer than average, a negative one riskier.
IV sums (good share - bad share) * WoE over the bins and ranks how much a
feature separates good accounts from bad ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SMOOTHING = 0.5  # added to bin counts so an empty bin never produces log(0)


@dataclass
class FeatureBins:
    edges: np.ndarray
    woe: np.ndarray
    iv: float
    table: pd.DataFrame


@dataclass
class WoEBinner:
    max_bins: int = 6
    bins: dict[str, FeatureBins] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WoEBinner":
        for col in X.columns:
            self.bins[col] = self._fit_feature(X[col], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col, fb in self.bins.items():
            idx = np.searchsorted(fb.edges[1:-1], X[col].to_numpy(), side="right")
            out[col] = fb.woe[idx]
        return out

    def iv_table(self) -> pd.DataFrame:
        rows = [(col, fb.iv) for col, fb in self.bins.items()]
        return (
            pd.DataFrame(rows, columns=["feature", "information_value"])
            .sort_values("information_value", ascending=False)
            .reset_index(drop=True)
        )

    def _fit_feature(self, x: pd.Series, y: pd.Series) -> FeatureBins:
        quantiles = np.linspace(0, 1, self.max_bins + 1)
        inner = np.unique(np.quantile(x, quantiles[1:-1]))
        edges = np.concatenate([[-np.inf], inner, [np.inf]])
        bin_idx = np.searchsorted(edges[1:-1], x.to_numpy(), side="right")

        counts = (
            pd.DataFrame({"bin": bin_idx, "bad": y.to_numpy()})
            .groupby("bin")["bad"]
            .agg(accounts="size", bads="sum")
            .reindex(range(len(edges) - 1), fill_value=0)
        )
        counts["goods"] = counts["accounts"] - counts["bads"]
        good_share = (counts["goods"] + SMOOTHING) / (counts["goods"].sum() + SMOOTHING)
        bad_share = (counts["bads"] + SMOOTHING) / (counts["bads"].sum() + SMOOTHING)
        woe = np.log(good_share / bad_share)
        iv = float(((good_share - bad_share) * woe).sum())

        table = counts.assign(
            lower=edges[:-1],
            upper=edges[1:],
            bad_rate=counts["bads"] / counts["accounts"].where(counts["accounts"] > 0),
            woe=woe,
        )
        return FeatureBins(edges=edges, woe=woe.to_numpy(), iv=iv, table=table)
