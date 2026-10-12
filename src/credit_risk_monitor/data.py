"""Download and clean the UCI "Default of Credit Card Clients" dataset.

Source: Yeh, I-Cheng (2009). UCI Machine Learning Repository, dataset 350.
30,000 Taiwanese credit card accounts, Apr-Sep 2005, amounts in NT dollars.
Licensed CC BY 4.0. The target is whether the account defaulted on the
following month's payment.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
XLS_NAME = "default of credit card clients.xls"

PAY_STATUS = [f"PAY_{i}" for i in range(1, 7)]   # repayment status, month 1 = most recent
BILL_AMT = [f"BILL_AMT{i}" for i in range(1, 7)]  # statement balance
PAY_AMT = [f"PAY_AMT{i}" for i in range(1, 7)]    # amount paid
TARGET = "default"


def download(dest_dir: Path) -> Path:
    """Fetch the dataset into dest_dir and return the path to the .xls file."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    xls_path = dest_dir / XLS_NAME
    if xls_path.exists():
        return xls_path
    with urllib.request.urlopen(UCI_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extract(XLS_NAME, dest_dir)
    return xls_path


def load(xls_path: Path) -> pd.DataFrame:
    """Read the raw sheet and return a cleaned frame with consistent column names."""
    df = pd.read_excel(xls_path, header=1)
    df = df.rename(columns={"PAY_0": "PAY_1", "default payment next month": TARGET})
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the undocumented category codes the dataset is known to contain."""
    df = df.copy()
    # EDUCATION is documented as 1-4; 0, 5 and 6 appear and are folded into "other" (4).
    df["EDUCATION"] = df["EDUCATION"].where(df["EDUCATION"].between(1, 4), 4)
    # MARRIAGE is documented as 1-3; 0 is folded into "other" (3).
    df["MARRIAGE"] = df["MARRIAGE"].where(df["MARRIAGE"].between(1, 3), 3)
    # Repayment status -2 (no consumption) and -1 (paid in full) both mean "not delinquent".
    for col in PAY_STATUS:
        df[col] = df[col].clip(lower=0)
    df[TARGET] = df[TARGET].astype(int)
    return df
