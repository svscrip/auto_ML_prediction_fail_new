"""Regression tests for boundary cases in inference, input and monitoring."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml

from src import mlflow_setup
from src.etl.load import load_train, validate_schema
from src.monitoring import (
    compute_data_quality_report,
    interpret_psi,
    population_stability_index,
    save_monitoring_report,
)
from src.predict import assign_risk_level_vector
from src.train import train_model


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


def test_single_row_quality_report_is_valid_json(tmp_path):
    report = compute_data_quality_report(load_train().head(1))
    assert report["std_Torque [Nm]"] is None
    path = tmp_path / "report.json"
    save_monitoring_report(report, path)
    assert json.loads(path.read_text(encoding="utf-8")) == report
    json.dumps(report, allow_nan=False)


def test_invalid_monitoring_report_does_not_overwrite_previous(tmp_path):
    path = tmp_path / "report.json"
    path.write_text('{"previous": true}', encoding="utf-8")
    with pytest.raises(ValueError):
        save_monitoring_report({"psi": np.nan}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"previous": True}


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_sample_size_fails_before_side_effects(size, tmp_path):
    output = tmp_path / "training"
    with pytest.raises(ValueError, match="positive integer"):
        train_model(sample_size=size, output_dir=output)
    assert not output.exists()


def test_compose_services_do_not_reuse_the_host_mlflow_store():
    root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((root / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    for name in ("train", "predict"):
        env = services[name]["environment"]
        store = env["ML_ARTIFACTS_DIR"].rstrip("/")
        assert store != "/app/artifacts", name
        assert env["MLFLOW_TRACKING_URI"] == f"sqlite:///{store}/mlflow.db", name
        assert env["MLFLOW_ARTIFACTS_URI"] == f"file://{store}/mlartifacts", name
    server = services["mlflow"]["command"].split()
    train_env = services["train"]["environment"]
    backend = server[server.index("--backend-store-uri") + 1]
    artifact_root = server[server.index("--default-artifact-root") + 1]
    assert backend == train_env["MLFLOW_TRACKING_URI"]
    assert artifact_root == train_env["MLFLOW_ARTIFACTS_URI"]
