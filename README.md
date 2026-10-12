# Credit Risk Monitor

A proof of concept for monitoring a credit card portfolio the way a credit risk
team would: predict which accounts will default, turn that into risk grades and
expected loss, check whether the model is still stable, and see what a downturn
would do to the numbers.

It runs on public data. Nothing here touches a real lender's customers.

## What it does

| Step | What happens | Where |
|---|---|---|
| Data | Downloads the UCI *Default of Credit Card Clients* dataset and fixes its undocumented category codes | `data.py` |
| Features | Builds behavioural features: utilization, delinquency history, payment ratios, balance trend | `features.py` |
| Scorecard | Bins each feature, converts bins to Weight of Evidence, fits a logistic regression, and scales it to points (600 at 50:1 odds, 20 points to double the odds) | `woe.py`, `scorecard.py` |
| Benchmark | Fits a gradient boosting model on the same features to show what the explainable scorecard gives up | `pipeline.py` |
| Validation | AUC, Gini, KS, calibration by decile, and PSI between development and holdout samples | `metrics.py` |
| Portfolio | Risk grades A to E, delinquency buckets, roll rates, exposure at default and expected loss | `portfolio.py` |
| Stress test | Multiplies every account's odds of default under moderate and severe scenarios and recomputes expected loss and grade mix | `portfolio.py` |
| SQL | Writes scored accounts to SQLite with views for the grade summary, utilization bands and a high-risk watchlist | `db.py` |
| Commentary | Drafts the monthly commentary with Claude, then rejects the draft if any number in it is not one the pipeline computed | `commentary.py` |
| Dashboard | Streamlit view over all of the above | `app/dashboard.py` |

## Design choices worth knowing

- **Demographics are not model inputs.** `SEX`, `MARRIAGE` and `AGE` stay in the
  data but are never used to score an account.
- **Every scorecard coefficient must point the right way.** On WoE inputs a safer
  bin has to lower the predicted risk. If a coefficient comes out positive, the
  feature is dropped and the model refitted. On this data that removes
  `utilization` and `payment_ratio`, which overlap 0.91 and 0.81 with their
  six-month averages.
- **The AI only writes words around numbers it was given.** The model gets a JSON
  of computed facts. Its draft is scanned for numbers, and if any of them cannot
  be traced back to a fact, the draft is thrown away and a fixed template is
  used. The model never changes a figure or a decision.
- **Expected loss uses stated assumptions.** EL = PD x LGD x EAD, with LGD 85% and
  a credit conversion factor of 50% on the unused limit. The dataset has no
  recovery or drawdown history, so these are assumptions, not estimates.

## Results on the current run

| Model (30% holdout) | AUC | Gini | KS |
|---|---|---|---|
| WoE scorecard | 0.742 | 0.484 | 0.393 |
| Gradient boosting | 0.774 | 0.548 | 0.420 |

- PSI between development and holdout scores: 0.001 (stable, as expected for a random split).
- Grades D and E hold 23.0% of accounts and 43.2% of expected loss.
- Accounts already two or more months late default next month at 69.6%, against 13.8% for current accounts.
- The severe scenario (odds of default x 1.75) raises expected loss by 44.3% and grows the D and E population from 6,892 to 9,467 accounts.

The gradient boosting model ranks better. The scorecard is still the one used for
grading, because every point it assigns can be explained to a customer or a
regulator.

## Running it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run credit-risk-monitor --no-ai        # deterministic commentary
uv run streamlit run app/dashboard.py     # open the dashboard
uv run pytest                             # tests
```

To let Claude draft the commentary, install the optional extra and make Anthropic
credentials available (`ANTHROPIC_API_KEY` or an `ant auth login` profile):

```bash
uv sync --extra ai
uv run credit-risk-monitor
```

Outputs land in `reports/`: metrics, information value, scorecard points, grade
summary, delinquency, roll rates, stress test, calibration, commentary, and
`portfolio.db`.

## Limitations

- The data is 30,000 Taiwanese accounts from April to September 2005, in NT dollars. It says nothing about any current lender's portfolio.
- The target is default on the next month's payment, so PD and expected loss are next-month figures, not the 12-month or lifetime figures a provisioning model would use.
- PSI compares two random samples of the same period. In production it would compare this month's applicants or accounts against the development sample.
- Portfolio views score every account, including the development sample, as a monthly run would. Model quality is judged on the holdout only.
- The dataset's repayment status codes are known to be inconsistent. Very few accounts sit at "1 month late" in month 2 (28 of 30,000), so that row of the roll-rate table is not meaningful.

## Data

Yeh, I-Cheng (2009). *Default of Credit Card Clients*. UCI Machine Learning
Repository. https://doi.org/10.24432/C55S3H. Licensed CC BY 4.0. The raw file
is downloaded on first run and is not committed.
