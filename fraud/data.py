"""Load the credit card fraud dataset, add features and split it by time."""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "creditcard.csv"

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = PCA_COLUMNS + ["log_amount", "hour"]
TARGET = "Class"


def validate(df: pd.DataFrame) -> None:
    """Fail early if the file is not the dataset the pipeline expects."""
    expected = {"Time", "Amount", TARGET, *PCA_COLUMNS}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df[list(expected)].isna().any().any():
        raise ValueError("Dataset contains missing values")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError("Class must contain only 0 (genuine) and 1 (fraud)")
    if (df["Amount"] < 0).any():
        raise ValueError("Amount must not be negative")


def load_raw(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_data.py")
    df = pd.read_csv(path)
    validate(df)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Amounts are heavily skewed, so use log(1 + amount); Time is seconds since the first transaction."""
    out = df.copy()
    out["log_amount"] = np.log1p(out["Amount"])
    out["hour"] = (out["Time"] // 3600) % 24
    return out


def time_split(df: pd.DataFrame, val_frac: float = 0.2, test_frac: float = 0.2) -> dict:
    """Train on the earliest transactions, tune on the next block, test on the latest.

    A random split would let the model see the future, which a deployed fraud model never can.
    """
    ordered = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * (1 - val_frac - test_frac))
    val_end = int(n * (1 - test_frac))
    return {
        "train": ordered.iloc[:train_end],
        "val": ordered.iloc[train_end:val_end],
        "test": ordered.iloc[val_end:],
    }


def make_synthetic(n: int = 6000, fraud_rate: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Small dataset with the same columns, used by the tests and the CI smoke run."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    df = pd.DataFrame(rng.normal(size=(n, len(PCA_COLUMNS))), columns=PCA_COLUMNS)
    # Shift a few components for fraud rows so the problem is learnable but not trivial.
    for col, shift in {"V4": 2.0, "V12": -2.5, "V14": -3.0, "V17": -2.0}.items():
        df[col] += shift * y
    df["Time"] = np.sort(rng.uniform(0, 172_800, n))
    df["Amount"] = np.round(rng.lognormal(mean=3.5, sigma=1.2, size=n), 2)
    df[TARGET] = y
    return df
