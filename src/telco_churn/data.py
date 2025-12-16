from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def fetch_telco(openml_id: int = 42178) -> pd.DataFrame:
    """Fetch Telco churn dataset from OpenML via sklearn."""
    bunch = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")

    # Usually bunch.frame exists (features + target). Be defensive:
    if hasattr(bunch, "frame") and bunch.frame is not None:
        return bunch.frame.copy()

    # Fallback: combine data + target
    df = bunch.data.copy()
    df["Churn"] = bunch.target
    return df


def clean_telco_df(
    df: pd.DataFrame,
    target_col: str = "Churn",
    drop_cols: Iterable[str] = ("customerID",),
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns (X, y) where:
      - y is {0,1}
      - TotalCharges is coerced to numeric if present
    """
    df = df.copy()

    # Drop ID-like columns if present
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in columns: {list(df.columns)}")

    # Target to {0,1}
    y_raw = df[target_col].astype(str).str.strip().str.lower()
    y = y_raw.map({"yes": 1, "no": 0})
    if y.isna().any():
        # If labels differ, fallback to factorize (still stable within this dataset)
        y = pd.Series(pd.factorize(y_raw)[0], index=df.index)

    X = df.drop(columns=[target_col])

    # Common gotcha: TotalCharges sometimes arrives as string with blanks
    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")

    return X, y


def split_stratified(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitData:
    """Stratified train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
