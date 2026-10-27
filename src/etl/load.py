"""Extract: load and validate CSV data."""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.config import (
    FAILURE_FLAGS,
    ID_COL,
    PRODUCT_ID_COL,
    RAW_NUMERIC_FEATURES,
    TARGET_COL,
    TEST_CSV,
    TRAIN_CSV,
    TYPE_COL,
)


def _required_columns(require_target: bool) -> list[str]:
    cols = [ID_COL, PRODUCT_ID_COL, TYPE_COL] + RAW_NUMERIC_FEATURES + FAILURE_FLAGS
    if require_target:
        cols.append(TARGET_COL)
    return cols


def load_train(path=None) -> pd.DataFrame:
    path = path or TRAIN_CSV
    df = pd.read_csv(path)
    validate_schema(df, require_target=True)
    return df


def load_test(path=None) -> pd.DataFrame:
    path = path or TEST_CSV
    df = pd.read_csv(path)
    validate_schema(df, require_target=False)
    return df


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    missing = [c for c in _required_columns(require_target) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("Dataset must not be empty")

    if require_target and df[TARGET_COL].isna().any():
        raise ValueError(f"Null values in target column {TARGET_COL}")

    for col in RAW_NUMERIC_FEATURES + ([TARGET_COL] if require_target else []):
        if col in df.columns and df[col].isna().any():
            raise ValueError(f"Null values in column {col}")

    if TYPE_COL in df.columns and df[TYPE_COL].isna().any():
        raise ValueError(f"Null values in column {TYPE_COL}")

    for col in FAILURE_FLAGS:
        if col in df.columns and df[col].isna().any():
            raise ValueError(f"Null values in failure flag column {col}")

    for col in RAW_NUMERIC_FEATURES:
        if not is_numeric_dtype(df[col]) or not np.isfinite(df[col]).all():
            raise ValueError(f"Expected finite numeric values in column {col}")
    for col in FAILURE_FLAGS + ([TARGET_COL] if require_target else []):
        if not df[col].isin([0, 1]).all():
            raise ValueError(f"Expected binary values in column {col}")
    if not df[TYPE_COL].isin(["L", "M", "H"]).all():
        raise ValueError("Type must be one of L, M, H")
    for col in [ID_COL, PRODUCT_ID_COL]:
        if df[col].isna().any() or df[col].astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing identifier in column {col}")
    if df[ID_COL].duplicated().any():
        raise ValueError("Duplicate row id")
