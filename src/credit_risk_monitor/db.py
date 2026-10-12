"""SQLite store for scored accounts, with SQL views for the portfolio reports."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

VIEWS = {
    "v_grade_summary": """
        SELECT grade,
               COUNT(*)                    AS accounts,
               ROUND(AVG(pd), 4)           AS avg_pd,
               ROUND(AVG("default"), 4)    AS observed_bad_rate,
               ROUND(SUM(ead), 0)          AS exposure,
               ROUND(SUM(expected_loss), 0) AS expected_loss
        FROM scored_accounts
        GROUP BY grade
        ORDER BY grade
    """,
    "v_high_risk_watchlist": """
        SELECT account_id, grade, ROUND(pd, 4) AS pd, score, credit_limit,
               ROUND(utilization, 3) AS utilization, current_delinquency,
               ROUND(expected_loss, 0) AS expected_loss
        FROM scored_accounts
        WHERE grade IN ('D', 'E') AND "default" = 0
        ORDER BY expected_loss DESC
    """,
    "v_utilization_bands": """
        SELECT CASE
                 WHEN utilization < 0.3 THEN '0-30%'
                 WHEN utilization < 0.7 THEN '30-70%'
                 WHEN utilization < 1.0 THEN '70-100%'
                 ELSE 'over limit'
               END AS utilization_band,
               COUNT(*)                 AS accounts,
               ROUND(AVG(pd), 4)        AS avg_pd,
               ROUND(AVG("default"), 4) AS observed_bad_rate
        FROM scored_accounts
        GROUP BY utilization_band
        ORDER BY MIN(utilization)
    """,
}


def write(scored: pd.DataFrame, db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        scored.to_sql("scored_accounts", conn, if_exists="replace", index=False)
        for name, sql in VIEWS.items():
            conn.execute(f"DROP VIEW IF EXISTS {name}")
            conn.execute(f"CREATE VIEW {name} AS {sql}")


def query(db_path: Path, sql: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(sql, conn)
