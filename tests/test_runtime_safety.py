"""Regression tests for boundary cases in inference, input and monitoring."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src import mlflow_setup
from src.etl.load import load_train, validate_schema
from src.monitoring import interpret_psi, population_stability_index
from src.predict import assign_risk_level_vector


def test_risk_ties_preserve_index_and_do_not_invent_high_risk():
    scores = pd.Series([0.5, 0.5, 0.5], index=[10, 20, 30])
    result = assign_risk_level_vector(scores)
    assert result.index.equals(scores.index)
    assert result.tolist() == ["Низкий"] * 3


def test_risk_matches_original_policy_for_distinct_quantiles():
    scores = pd.Series(np.linspace(0, 1, 101))
    result = assign_risk_level_vector(scores)
    assert result.iloc[50] == "Низкий"
    assert result.iloc[90] == "Средний"
    assert result.iloc[91] == "Высокий"
    assert assign_risk_level_vector(pd.Series([], dtype=float)).empty


@pytest.mark.parametrize("score", [np.nan, np.inf, -0.1, 1.1])
def test_risk_rejects_invalid_scores(score):
    with pytest.raises(ValueError):
        assign_risk_level_vector(pd.Series([score]))


def test_psi_includes_minimum_and_handles_empty_bins():
    # Equal support, reversed frequencies: both endpoint bins must count.
    expected = pd.Series([0] * 9 + [1])
    actual = pd.Series([0] + [1] * 9)
    assert population_stability_index(expected, actual, bins=2) == pytest.approx(
        1.6 * np.log(9)
    )
    shifted = population_stability_index(pd.Series([0, 0]), pd.Series([1, 1]))
    assert np.isfinite(shifted) and shifted > 0.25
    assert population_stability_index(expected, expected) == pytest.approx(0)
    assert population_stability_index(pd.Series([5, 5]), pd.Series([5])) == 0


@pytest.mark.parametrize("values", [[], [np.nan], [np.inf]])
def test_psi_rejects_unusable_data(values):
    with pytest.raises(ValueError):
        population_stability_index(pd.Series([1, 2]), pd.Series(values))


@pytest.mark.parametrize("value", [np.nan, np.inf, -1])
def test_invalid_psi_is_not_silently_classified(value):
    with pytest.raises(ValueError):
        interpret_psi(value)


@pytest.mark.parametrize(
    "column,value",
    [("Torque [Nm]", np.inf), ("TWF", 2), ("Machine failure", -1), ("Type", "X")],
)
def test_schema_rejects_invalid_values(column, value):
    data = load_train().head(3).copy()
    data.loc[data.index[0], column] = value
    with pytest.raises(ValueError):
        validate_schema(data)


def test_schema_rejects_duplicate_row_ids():
    data = load_train().head(2).copy()
    data.loc[data.index[1], "id"] = data.iloc[0]["id"]
    with pytest.raises(ValueError, match="Duplicate row id"):
        validate_schema(data)


def test_mlflow_mismatch_preserves_store(tmp_path, monkeypatch):
    store = tmp_path / "mlflow.db"
    store.write_bytes(b"existing history must survive")
    fake = Mock()
    fake.get_experiment_by_name.return_value = SimpleNamespace(
        artifact_location="file:///another-project/artifacts/mlartifacts"
    )
    monkeypatch.setattr(mlflow_setup, "mlflow", fake)
    monkeypatch.setattr(mlflow_setup, "MLFLOW_ARTIFACTS_DIR", tmp_path / "mlartifacts")
    monkeypatch.setattr(mlflow_setup, "MLFLOW_ARTIFACTS_URI", tmp_path.as_uri())
    with pytest.raises(RuntimeError, match="History was preserved"):
        mlflow_setup.setup_mlflow()
    assert store.read_bytes() == b"existing history must survive"
    fake.create_experiment.assert_not_called()
    fake.set_experiment.assert_not_called()


def test_mlflow_reuses_matching_experiment(tmp_path, monkeypatch):
    fake = Mock()
    fake.get_experiment_by_name.return_value = SimpleNamespace(
        artifact_location=tmp_path.as_uri()
    )
    monkeypatch.setattr(mlflow_setup, "mlflow", fake)
    monkeypatch.setattr(mlflow_setup, "MLFLOW_ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(mlflow_setup, "MLFLOW_ARTIFACTS_URI", tmp_path.as_uri())
    mlflow_setup.setup_mlflow()
    fake.create_experiment.assert_not_called()
    fake.set_experiment.assert_called_once()
