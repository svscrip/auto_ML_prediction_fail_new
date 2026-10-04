"""Project configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "keis7-main"
TRAIN_CSV = DATA_DIR / "train.csv"
TEST_CSV = DATA_DIR / "test.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MLFLOW_DB = ARTIFACTS_DIR / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = ARTIFACTS_DIR / "mlartifacts"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.resolve().as_posix()}"
MLFLOW_ARTIFACTS_URI = f"file:///{MLFLOW_ARTIFACTS_DIR.resolve().as_posix()}"
MLFLOW_EXPERIMENT_NAME = "machine_failure_prediction"

TARGET_COL = "Machine failure"
ID_COL = "id"
PRODUCT_ID_COL = "Product ID"
TYPE_COL = "Type"

FAILURE_FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

RAW_NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

ENGINEERED_FEATURES = [
    "delta_temperature [K]",
    "Power [kW]",
    "air_mass",
    "air_heat_power [kW]",
    "efficiency [%]",
    "total_failures_cum",
]

MODEL_FEATURES = RAW_NUMERIC_FEATURES + FAILURE_FLAGS + ENGINEERED_FEATURES + [TYPE_COL]
CAT_FEATURES = [TYPE_COL, *FAILURE_FLAGS]

CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.03,
    "depth": 8,
    "l2_leaf_reg": 5,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": 42,
    "verbose": 100,
    "early_stopping_rounds": 100,
    "auto_class_weights": "Balanced",
}

TRAIN_TEST_SIZE = 0.2
RANDOM_STATE = 42
SMOKE_SAMPLE_SIZE = 5000

RISK_THRESHOLDS = {
    "low": 0.45,
    "medium": 0.50,
}
