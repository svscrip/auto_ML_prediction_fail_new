import pandas as pd
import pytest

from src.etl.features import build_features, get_feature_matrix


def _synthetic_row() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 0,
                "Product ID": "L50096",
                "Type": "L",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 309.0,
                "Rotational speed [rpm]": 1600,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 100,
                "Machine failure": 0,
                "TWF": 0,
                "HDF": 0,
                "PWF": 0,
                "OSF": 0,
                "RNF": 0,
            }
        ]
    )


def test_power_and_efficiency_formulas():
    df = build_features(_synthetic_row(), is_train=True)
    expected_power = 40.0 * 1600 / 9550
    assert abs(df["Power [kW]"].iloc[0] - expected_power) < 1e-6
    assert df["delta_temperature [K]"].iloc[0] == 9.0
    assert df["air_mass"].iloc[0] == pytest.approx(14.4 * expected_power)
    assert 0 <= df["efficiency [%]"].iloc[0] <= 100


def test_clean_contradictory_rows_removes_bad():
    row = _synthetic_row()
    row["HDF"] = 1  # флаг активен, но Machine failure = 0 — противоречие
    cleaned = build_features(row, is_train=True)
    assert len(cleaned) == 0


def test_clean_contradictory_rows_keeps_consistent():
    row = _synthetic_row()  # Machine failure=0, все флаги 0 — всё ок
    cleaned = build_features(row, is_train=True)
    assert len(cleaned) == 1


def test_efficiency_clipped_to_valid_range():
    row = _synthetic_row()
    # Process temp < Air temp -> delta отрицательный -> без clip efficiency уходит < 0
    row["Air temperature [K]"] = 320.0
    row["Process temperature [K]"] = 300.0
    df = build_features(row, is_train=True)
    assert df["efficiency [%]"].iloc[0] >= 0.0
    assert df["efficiency [%]"].iloc[0] <= 100.0


def test_total_failures_cum_excludes_current_row():
    row = _synthetic_row()
    row["TWF"] = 1  # текущая строка имеет отказ
    df = build_features(row, is_train=False)
    # первая строка в группе не должна считать свой собственный отказ
    assert df["total_failures_cum"].iloc[0] == 0


def test_get_feature_matrix_shape():
    df = build_features(_synthetic_row(), is_train=True)
    X, y = get_feature_matrix(df, include_target=True)
    assert len(X.columns) >= 10
    assert y is not None
    assert y.iloc[0] == 0
