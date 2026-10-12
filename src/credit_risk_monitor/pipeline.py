"""End-to-end run: data -> features -> models -> portfolio views -> reports.

Usage:
    uv run credit-risk-monitor            # with AI commentary if credentials exist
    uv run credit-risk-monitor --provider groq   # draft with a Groq-hosted model
    uv run credit-risk-monitor --no-ai    # deterministic template only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from . import commentary, data, db, features, metrics, portfolio, scorecard

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
SEED = 42


def run(use_ai: bool = True, provider: str = "claude") -> dict:
    raw = data.load(data.download(DATA_DIR))
    X = features.build(raw)
    y = raw[data.TARGET]

    X_dev, X_hold, y_dev, y_hold = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)

    card = scorecard.fit(X_dev, y_dev)
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=SEED)
    gbm.fit(X_dev, y_dev)

    pd_dev = card.predict_pd(X_dev)
    pd_hold = card.predict_pd(X_hold)
    model_metrics = {
        "scorecard_holdout": metrics.summary(y_hold, pd_hold),
        "gradient_boosting_holdout": metrics.summary(y_hold, gbm.predict_proba(X_hold)[:, 1]),
        "scorecard_development": metrics.summary(y_dev, pd_dev),
        "psi_dev_vs_holdout": metrics.psi(pd_dev, pd_hold),
        "scorecard_features": list(card.binner.bins),
        "dropped_wrong_sign": card.dropped,
    }

    # Portfolio views use the scorecard on every account, as a monthly run would.
    pd_all = card.predict_pd(X)
    scored = pd.DataFrame(
        {
            "account_id": raw["ID"],
            "pd": pd_all,
            "score": card.score(X),
            "grade": portfolio.assign_grades(pd_all),
            "default": y,
            "ead": portfolio.exposure_at_default(raw),
        }
    ).join(X)
    scored["expected_loss"] = scored["pd"] * portfolio.LGD * scored["ead"]

    grades = portfolio.grade_summary(scored)
    delinquency = portfolio.delinquency_buckets(raw)
    rolls = portfolio.roll_rates(raw)
    stress = portfolio.stress_test(pd_all, scored["ead"].to_numpy())
    calibration = metrics.calibration_table(y_hold, pd_hold)

    facts = _facts(model_metrics, grades, delinquency, stress)
    text, source = commentary.generate(facts, use_ai=use_ai, provider=provider)

    _write_reports(card, model_metrics, grades, delinquency, rolls, stress, calibration, facts, text, source)
    db.write(scored, REPORTS_DIR / "portfolio.db")
    return {"metrics": model_metrics, "facts": facts, "commentary_source": source}


def _facts(model_metrics, grades, delinquency, stress) -> dict:
    hold = model_metrics["scorecard_holdout"]
    d_e = grades[grades["grade"].isin(["D", "E"])]
    by_bucket = delinquency.set_index("bucket")
    late_2plus = by_bucket.loc[["2 months late", "3+ months late"]]
    late_2plus_rate = (late_2plus["accounts"] * late_2plus["next_month_default_rate"]).sum() / late_2plus["accounts"].sum()
    severe = stress.set_index("scenario").loc["severe"]
    return {
        "gini": round(hold["gini"], 3),
        "ks": round(hold["ks"], 3),
        "auc": round(hold["auc"], 3),
        "psi_dev_vs_holdout": round(model_metrics["psi_dev_vs_holdout"], 4),
        "share_d_e_pct": round(100 * d_e["accounts"].sum() / grades["accounts"].sum(), 1),
        "el_share_d_e_pct": round(100 * d_e["expected_loss"].sum() / grades["expected_loss"].sum(), 1),
        "default_rate_current_pct": round(100 * by_bucket.loc["current", "next_month_default_rate"], 1),
        "default_rate_2plus_late_pct": round(100 * late_2plus_rate, 1),
        "expected_loss_baseline": int(round(stress.loc[0, "expected_loss"])),
        "expected_loss_severe": int(round(severe["expected_loss"])),
        "severe_stress_increase_pct": round(100 * severe["change_vs_baseline"], 1),
        "accounts_in_d_e_baseline": int(stress.loc[0, "accounts_in_d_e"]),
        "accounts_in_d_e_severe": int(severe["accounts_in_d_e"]),
    }


def _write_reports(card, model_metrics, grades, delinquency, rolls, stress, calibration, facts, text, source):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(model_metrics, indent=2))
    (REPORTS_DIR / "facts.json").write_text(json.dumps(facts, indent=2))
    card.binner.iv_table().to_csv(REPORTS_DIR / "information_value.csv", index=False)
    card.points_table().to_csv(REPORTS_DIR / "scorecard_points.csv", index=False)
    grades.to_csv(REPORTS_DIR / "grade_summary.csv", index=False)
    delinquency.to_csv(REPORTS_DIR / "delinquency_buckets.csv", index=False)
    rolls.to_csv(REPORTS_DIR / "roll_rates.csv")
    stress.to_csv(REPORTS_DIR / "stress_test.csv", index=False)
    calibration.to_csv(REPORTS_DIR / "calibration.csv", index=False)
    (REPORTS_DIR / "commentary.md").write_text(f"<!-- source: {source} -->\n\n{text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the credit card portfolio risk monitor.")
    parser.add_argument("--no-ai", action="store_true", help="skip the AI commentary draft")
    parser.add_argument(
        "--provider",
        choices=sorted(commentary.DRAFTERS),
        default="claude",
        help="which model drafts the commentary (default: claude)",
    )
    args = parser.parse_args()
    result = run(use_ai=not args.no_ai, provider=args.provider)
    hold = result["metrics"]["scorecard_holdout"]
    gbm = result["metrics"]["gradient_boosting_holdout"]
    print(f"Scorecard holdout:         AUC {hold['auc']:.3f}  Gini {hold['gini']:.3f}  KS {hold['ks']:.3f}")
    print(f"Gradient boosting holdout: AUC {gbm['auc']:.3f}  Gini {gbm['gini']:.3f}  KS {gbm['ks']:.3f}")
    print(f"PSI dev vs holdout: {result['metrics']['psi_dev_vs_holdout']:.4f}")
    print(f"Dropped for wrong sign: {', '.join(result['metrics']['dropped_wrong_sign']) or 'none'}")
    print(f"Commentary source: {result['commentary_source']}")
    print(f"Reports written to {REPORTS_DIR}")


if __name__ == "__main__":
    np.seterr(divide="ignore", invalid="ignore")
    main()
