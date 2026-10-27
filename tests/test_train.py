import json
import time

import pytest

from src import mlflow_setup
from src.config import ARTIFACTS_DIR, MLFLOW_TRACKING_URI, SMOKE_SAMPLE_SIZE
from src.train import train_model


@pytest.fixture
def isolated_mlflow(tmp_path, monkeypatch):
    """Point MLflow at a throwaway store so tests never touch real history."""
    store = tmp_path / "mlflow"
    artifacts = store / "mlartifacts"
    monkeypatch.setattr(
        mlflow_setup,
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{(store / 'mlflow.db').as_posix()}",
    )
    monkeypatch.setattr(mlflow_setup, "MLFLOW_ARTIFACTS_DIR", artifacts)
    monkeypatch.setattr(mlflow_setup, "MLFLOW_ARTIFACTS_URI", artifacts.as_uri())
    yield store
    # setup_mlflow sets a process-wide tracking URI; restore the configured one.
    mlflow_setup.mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@pytest.mark.slow
def test_smoke_train_metrics(tmp_path, isolated_mlflow):
    output = tmp_path / "training"
    metrics = train_model(sample_size=SMOKE_SAMPLE_SIZE, output_dir=output)
    assert metrics["roc_auc"] >= 0.85
    assert metrics["recall"] >= 0.5
    assert (output / "model.cbm").exists()
    assert (output / "metrics.json").exists()


def test_smoke_train_fast(tmp_path, isolated_mlflow):
    """Quick smoke without slow marker for CI default run."""
    output = tmp_path / "smoke_test"
    model_before = ARTIFACTS_DIR / "model.cbm"
    mtime_before = model_before.stat().st_mtime_ns if model_before.exists() else None
    t0 = time.perf_counter()
    metrics = train_model(sample_size=2000, output_dir=output)
    elapsed = time.perf_counter() - t0
    assert elapsed < 120
    assert metrics["roc_auc"] > 0.7
    metrics_path = output / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path, encoding="utf-8") as f:
        saved = json.load(f)
    assert "f1" in saved
    assert (isolated_mlflow / "mlflow.db").exists()
    mtime_after = model_before.stat().st_mtime_ns if model_before.exists() else None
    assert mtime_after == mtime_before, "test must not replace the working model"
