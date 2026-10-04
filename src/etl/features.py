"""Transform: feature engineering from case7.ipynb."""

import pandas as pd

from src.config import (
    FAILURE_FLAGS,
    MODEL_FEATURES,
    PRODUCT_ID_COL,
    TARGET_COL,
    TYPE_COL,
)


def clean_contradictory_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where Machine failure=0 but failure flags are set."""
    mask = (df[TARGET_COL] == 0) & (df[FAILURE_FLAGS].sum(axis=1) > 0)
    return df.loc[~mask].copy()


def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Apply ETL transformations and engineered features."""
    out = df.copy()

    if is_train:
        out = clean_contradictory_rows(out)

    out["delta_temperature [K]"] = (
        out["Process temperature [K]"] - out["Air temperature [K]"]
    )
    out["Power [kW]"] = out["Torque [Nm]"] * out["Rotational speed [rpm]"] / 9550
    out["air_mass"] = 14.4 * out["Power [kW]"]
    out["air_heat_power [kW]"] = (
        out["air_mass"] * 1.005 * out["delta_temperature [K]"] / 60
    )
    denominator = out["air_heat_power [kW]"] + out["Power [kW]"]
    out["efficiency [%]"] = (out["Power [kW]"] / denominator.replace(0, pd.NA)) * 100
    out["efficiency [%]"] = out["efficiency [%]"].fillna(0.0).clip(0.0, 100.0)

    flags = out[FAILURE_FLAGS].fillna(0).astype(int)
    out[FAILURE_FLAGS] = flags
    out = out.sort_values([TYPE_COL, PRODUCT_ID_COL, "Tool wear [min]"])
    out["failures_sum"] = flags.sum(axis=1)
    cumsum = out.groupby([TYPE_COL, PRODUCT_ID_COL], observed=True)[
        "failures_sum"
    ].cumsum()
    # вычитаем текущую строку, чтобы считать только прошлые отказы
    out["total_failures_cum"] = cumsum - out["failures_sum"]
    out = out.drop(columns=["failures_sum"])

    return out.reset_index(drop=True)


def get_feature_matrix(
    df: pd.DataFrame, include_target: bool = True
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return X and optional y for modeling."""
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Features not built: {missing}")

    X = df[MODEL_FEATURES].copy()
    for col in FAILURE_FLAGS:
        X[col] = X[col].astype(int)
    X[TYPE_COL] = X[TYPE_COL].astype(str)

    y = None
    if include_target and TARGET_COL in df.columns:
        y = df[TARGET_COL].astype(int)

    return X, y
