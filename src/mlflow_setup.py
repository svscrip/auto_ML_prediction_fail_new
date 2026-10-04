"""Single MLflow backend for this project (SQLite + local artifacts)."""

import mlflow
from mlflow.exceptions import MlflowException

from src.config import (
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_ARTIFACTS_URI,
    MLFLOW_DB,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)


def _normalize_artifact_uri(uri: str) -> str:
    return uri.replace("\\", "/").rstrip("/").removeprefix("file:").removeprefix("//")


def _artifact_uri_matches(actual: str, expected: str) -> bool:
    actual_norm = _normalize_artifact_uri(actual)
    expected_norm = _normalize_artifact_uri(expected)
    return actual_norm == expected_norm or actual_norm.endswith(
        "/artifacts/mlartifacts"
    )


def _reset_tracking_store() -> None:
    """Drop local SQLite store if experiment paths belong to another machine."""
    if MLFLOW_DB.exists():
        MLFLOW_DB.unlink()


def setup_mlflow() -> None:
    """Configure tracking URI and ensure experiment uses project artifact paths."""
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is not None and not _artifact_uri_matches(
        experiment.artifact_location, MLFLOW_ARTIFACTS_URI
    ):
        _reset_tracking_store()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        try:
            mlflow.create_experiment(
                MLFLOW_EXPERIMENT_NAME,
                artifact_location=MLFLOW_ARTIFACTS_URI,
            )
        except MlflowException as exc:
            if "already exists" not in str(exc).lower():
                raise
            _reset_tracking_store()
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.create_experiment(
                MLFLOW_EXPERIMENT_NAME,
                artifact_location=MLFLOW_ARTIFACTS_URI,
            )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
