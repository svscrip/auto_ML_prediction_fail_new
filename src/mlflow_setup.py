"""Configure MLflow without destroying existing experiment history."""

from urllib.parse import unquote

import mlflow
from mlflow.exceptions import MlflowException

from src.config import (
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_ARTIFACTS_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)


def _normalize_artifact_uri(uri: str) -> str:
    return unquote(uri).replace("\\", "/").rstrip("/")


def setup_mlflow() -> None:
    """Create or reuse the configured experiment; fail safely on path mismatch."""
    MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        try:
            mlflow.create_experiment(
                MLFLOW_EXPERIMENT_NAME,
                artifact_location=MLFLOW_ARTIFACTS_URI,
            )
        except MlflowException as exc:
            # Another process may have created it concurrently. Re-read and check.
            if exc.error_code != "RESOURCE_ALREADY_EXISTS":
                raise
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("MLflow experiment could not be created or retrieved")
    if _normalize_artifact_uri(experiment.artifact_location) != _normalize_artifact_uri(
        MLFLOW_ARTIFACTS_URI
    ):
        raise RuntimeError(
            "MLflow artifact location differs from configuration. History was "
            "preserved. Restore the original location or set a new "
            "MLFLOW_EXPERIMENT_NAME / MLFLOW_TRACKING_URI after backing up the store."
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
