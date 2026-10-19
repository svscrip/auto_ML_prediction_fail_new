"""Model card metrics generator and Markdown renderer.

The module evaluates the trained CatBoost model on exactly the same stratified
hold-out split as :mod:`src.train`, enriches the result with calibration,
subgroup, environment and git metadata, and stores it as JSON.  The JSON
snapshot is used to render the tables in ``docs/model-card.md`` between
``<!-- model-card:<name>:start -->`` markers.

Examples
--------
Compute a fresh snapshot after training::

    poetry run python -m src.model_card --output artifacts/model_card_metrics.json

Render the documentation from the snapshot::

    poetry run python -m src.model_card --write docs/model-card.md

Verify that the documentation is up to date (used by tests and CI)::

    poetry run python -m src.model_card --check docs/model-card.md
"""

from __future__ import annotations

import argparse
import difflib
import importlib.metadata
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import psutil
import yaml
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS_DIR,
    CAT_FEATURES,
    CATBOOST_PARAMS,
    DECISION_THRESHOLD,
    FAILURE_FLAGS,
    MODEL_FEATURES,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET_COL,
    TRAIN_TEST_SIZE,
    TYPE_COL,
)
from src.etl.features import build_features, get_feature_matrix
from src.etl.load import load_train

THRESHOLD_SWEEP = (0.10, 0.25, 0.50, 0.75, 0.90)
ECE_BINS = 10

DEFAULT_METRICS_PATH = ARTIFACTS_DIR / "model_card_metrics.json"
EXAMPLE_METRICS_PATH = ARTIFACTS_DIR / "example_model_card_metrics.json"
OUTDATED_METRICS_PATH = ARTIFACTS_DIR / "example_metrics.json"
DEFAULT_CARD_PATH = PROJECT_ROOT / "docs" / "model-card.md"
DEFAULT_CARD_DATA_PATH = PROJECT_ROOT / "docs" / "model-card.yaml"

FAILURE_MODE_LABELS = {
    "TWF": "Tool Wear Failure — износ инструмента",
    "HDF": "Heat Dissipation Failure — перегрев",
    "PWF": "Power Failure — проблема питания",
    "OSF": "Overstrain Failure — перегрузка",
    "RNF": "Random Failure — случайный отказ",
    "NO_FLAG": "Отказ без указанного режима",
}


def _run_git(*args: str) -> str | None:
    """Return stripped stdout of a git command, or ``None`` on failure."""
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def git_metadata() -> dict[str, Any]:
    """Return commit SHA, branch and working-tree state."""
    sha = _run_git("rev-parse", "HEAD") or "unknown"
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    status = _run_git("status", "--porcelain")
    return {
        "sha": sha,
        "short_sha": sha[:7] if sha != "unknown" else sha,
        "branch": branch,
        "dirty": bool(status),
    }


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def environment_metadata() -> dict[str, Any]:
    """Return Python, OS, hardware and dependency versions."""
    memory = psutil.virtual_memory()
    packages = {
        name: _package_version(name)
        for name in ("pandas", "numpy", "scikit-learn", "catboost", "mlflow")
    }
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_total_gb": round(memory.total / (1024**3), 2),
        "packages": packages,
    }


def expected_calibration_error(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    n_bins: int = ECE_BINS,
) -> float:
    """Compute equal-width Expected Calibration Error (ECE).

    Probabilities are split into ``n_bins`` equal-width bins on ``[0, 1]``.
    ECE is the weighted average of ``|accuracy - confidence|`` over bins, where
    the weight is the fraction of observations in the bin.
    """
    targets = np.asarray(y_true, dtype=float)
    probabilities = np.asarray(y_proba, dtype=float)
    if probabilities.size == 0:
        return 0.0

    boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probabilities, boundaries[1:-1])
    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        weight = mask.sum() / probabilities.size
        accuracy = targets[mask].mean()
        confidence = probabilities[mask].mean()
        ece += weight * abs(accuracy - confidence)
    return float(ece)


def classification_metrics(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, Any]:
    """Return classification, ranking and calibration metrics for a subset."""
    targets = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_proba, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    tp = int(((targets == 1) & (predictions == 1)).sum())
    fp = int(((targets == 0) & (predictions == 1)).sum())
    tn = int(((targets == 0) & (predictions == 0)).sum())
    fn = int(((targets == 1) & (predictions == 0)).sum())

    metrics: dict[str, Any] = {
        "support": int(targets.size),
        "positives": int(targets.sum()),
        "negatives": int(targets.size - targets.sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "roc_auc": None,
        "brier": float(brier_score_loss(targets, probabilities)),
        "log_loss": float(log_loss(targets, probabilities, labels=[0, 1])),
        "ece": expected_calibration_error(targets, probabilities),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }
    if np.unique(targets).size > 1:
        metrics["roc_auc"] = float(roc_auc_score(targets, probabilities))
    return metrics


def threshold_sweep(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    thresholds: tuple[float, ...] = THRESHOLD_SWEEP,
) -> list[dict[str, Any]]:
    """Evaluate the classifier for every threshold in ``thresholds``."""
    targets = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_proba, dtype=float)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        confusion = classification_metrics(targets, probabilities, threshold)[
            "confusion_matrix"
        ]
        rows.append(
            {
                "threshold": float(threshold),
                "precision": float(
                    precision_score(targets, predictions, zero_division=0)
                ),
                "recall": float(recall_score(targets, predictions, zero_division=0)),
                "f1": float(f1_score(targets, predictions, zero_division=0)),
                "accuracy": float(accuracy_score(targets, predictions)),
                "predicted_positive_rate": float(predictions.mean()),
                **confusion,
            }
        )
    return rows


def _failure_mode_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    meta: pd.DataFrame,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Report detection quality for every recorded failure mode.

    A mode flag marks an actual failure cause, so for each flag we report how
    many of those failures the model detected at the decision threshold.  A
    prediction is not attributed to a concrete mode, therefore precision is not
    reported per mode; the overall precision is available in ``overall``.
    """
    failures_total = int(y_true.sum())
    result: dict[str, dict[str, Any]] = {}
    for flag in FAILURE_FLAGS:
        mask = (meta[flag].astype(int) == 1).to_numpy()
        support = int(mask.sum())
        detected = int((y_proba[mask] >= threshold).sum()) if support else 0
        result[flag] = {
            "support": support,
            "share_of_failures": (
                float(support / failures_total) if failures_total else 0.0
            ),
            "detected": detected,
            "missed": support - detected,
            "recall": float(detected / support) if support else 0.0,
            "mean_probability": float(y_proba[mask].mean()) if support else 0.0,
        }

    no_flag_mask = ((y_true == 1) & (meta[FAILURE_FLAGS].sum(axis=1) == 0)).to_numpy()
    support = int(no_flag_mask.sum())
    detected = int((y_proba[no_flag_mask] >= threshold).sum()) if support else 0
    result["NO_FLAG"] = {
        "support": support,
        "share_of_failures": (
            float(support / failures_total) if failures_total else 0.0
        ),
        "detected": detected,
        "missed": support - detected,
        "recall": float(detected / support) if support else 0.0,
        "mean_probability": float(y_proba[no_flag_mask].mean()) if support else 0.0,
    }
    return result


def build_model_card_metrics(
    model_path: Path | None = None,
    threshold: float = DECISION_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate the trained model and return a JSON-serialisable model card."""
    model_path = model_path or ARTIFACTS_DIR / "model.cbm"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python -m src.train"
        )

    df_raw = load_train()
    df = build_features(df_raw, is_train=True)
    X, y = get_feature_matrix(df, include_target=True)
    if y is None:
        raise ValueError(f"Target column {TARGET_COL} is missing")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TRAIN_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    del X_train, y_train

    meta = df.loc[X_val.index, [TYPE_COL, *FAILURE_FLAGS]].reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    cat_idx = [X_val.columns.get_loc(c) for c in CAT_FEATURES if c in X_val.columns]
    probabilities = model.predict_proba(Pool(X_val, cat_features=cat_idx))[:, 1]

    y_array = y_val.to_numpy()
    overall = classification_metrics(y_array, probabilities, threshold)

    by_type: dict[str, dict[str, Any]] = {}
    for type_value in sorted(meta[TYPE_COL].astype(str).unique()):
        mask = (meta[TYPE_COL].astype(str) == type_value).to_numpy()
        by_type[type_value] = classification_metrics(
            y_array[mask], probabilities[mask], threshold
        )

    by_failure_mode = _failure_mode_metrics(y_array, probabilities, meta, threshold)

    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "decision_threshold": float(threshold),
        "model": {
            "algorithm": "CatBoostClassifier",
            "path": (
                model_path.relative_to(PROJECT_ROOT).as_posix()
                if model_path.is_relative_to(PROJECT_ROOT)
                else model_path.as_posix()
            ),
            "n_features": len(MODEL_FEATURES),
            "features": MODEL_FEATURES,
            "params": CATBOOST_PARAMS,
        },
        "dataset": {
            "source": "keis7-main/train.csv",
            "rows_raw": int(len(df_raw)),
            "target_rate_raw": float(df_raw[TARGET_COL].mean()),
            "rows_after_cleaning": int(len(df)),
            "target_rate_after_cleaning": float(y.mean()),
            "split": {
                "test_size": float(TRAIN_TEST_SIZE),
                "random_state": int(RANDOM_STATE),
                "stratified": True,
            },
            "validation": {
                "rows": int(len(y_val)),
                "positives": int(y_val.sum()),
                "negatives": int(len(y_val) - y_val.sum()),
                "positive_rate": float(y_val.mean()),
            },
        },
        "overall": overall,
        "calibration": {
            "bins": ECE_BINS,
            "brier": overall["brier"],
            "log_loss": overall["log_loss"],
            "ece": overall["ece"],
        },
        "by_type": by_type,
        "by_failure_mode": by_failure_mode,
        "threshold_sweep": threshold_sweep(y_array, probabilities),
        "environment": environment_metadata(),
        "git": git_metadata(),
    }


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def _source_path(sources: dict[str, str], key: str, default: str) -> Path:
    return PROJECT_ROOT / sources.get(key, default)


def load_render_context(
    card_data: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Load the YAML card, metrics snapshot and monitoring JSON references."""
    sources = card_data.get("sources", {})
    return {
        "card": card_data,
        "metrics": metrics,
        "smoke": _load_json(
            _source_path(
                sources,
                "smoke_monitoring",
                "artifacts/example_monitoring_summary.json",
            )
        ),
        "cv": _load_json(
            _source_path(
                sources,
                "cv_baseline",
                "reference-material/keis7-research/model_metadata.json",
            )
        ),
        "inference": _load_json(
            _source_path(
                sources,
                "inference_monitoring",
                "artifacts/example_inference_monitoring.json",
            )
        ),
        "outdated": _load_json(
            _source_path(
                sources,
                "outdated_metrics",
                "artifacts/example_metrics.json",
            )
        ),
    }


def _format_value(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _format_percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(item) for item in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(item) for item in row) + " |")
    return "\n".join(lines)


def _render_full_metrics(context: dict[str, Any]) -> str:
    overall = context["metrics"]["overall"]
    calibration = context["metrics"]["calibration"]
    confusion = overall["confusion_matrix"]
    rows = [
        ["Порог решения", _format_value(overall["threshold"], 2)],
        ["Accuracy", _format_value(overall["accuracy"])],
        ["Precision", _format_value(overall["precision"])],
        ["Recall", _format_value(overall["recall"])],
        ["F1", _format_value(overall["f1"])],
        ["ROC-AUC", _format_value(overall["roc_auc"])],
        ["Brier score", _format_value(overall["brier"])],
        ["Log loss", _format_value(overall["log_loss"])],
        [f"ECE ({calibration['bins']} бинов)", _format_value(overall["ece"])],
        ["Поддержка (validation)", overall["support"]],
        ["Отказы / норма", f"{overall['positives']} / {overall['negatives']}"],
        [
            "TP / FP / TN / FN",
            (
                f"{confusion['tp']} / {confusion['fp']} / "
                f"{confusion['tn']} / {confusion['fn']}"
            ),
        ],
    ]
    return _table(["Метрика", "Полный прогон"], rows)


def _render_smoke_metrics(context: dict[str, Any]) -> str:
    metrics = context["smoke"]["model_metrics"]
    rows = [
        ["ROC-AUC", _format_value(metrics["roc_auc"])],
        ["Recall", _format_value(metrics["recall"])],
        ["Precision", _format_value(metrics["precision"])],
        ["F1", _format_value(metrics["f1"])],
        ["Время обучения, с", _format_value(metrics["train_time_sec"], 1)],
    ]
    return _table(["Метрика", "Smoke-прогон (5 000 строк)"], rows)


def _render_cv_baseline(context: dict[str, Any]) -> str:
    metrics = context["cv"]
    rows = [
        [
            "ROC-AUC",
            _format_value(metrics["roc_auc_mean"]),
            _format_value(metrics["roc_auc_std"]),
        ],
        [
            "Recall",
            _format_value(metrics["recall_mean"]),
            _format_value(metrics["recall_std"]),
        ],
        [
            "Precision",
            _format_value(metrics["precision_mean"]),
            _format_value(metrics["precision_std"]),
        ],
        [
            "F1",
            _format_value(metrics["f1_mean"]),
            _format_value(metrics["f1_std"]),
        ],
    ]
    return _table(["Метрика", "Mean", "Std"], rows)


def _render_by_type(context: dict[str, Any]) -> str:
    metrics = context["metrics"]["by_type"]
    labels = context["card"].get("render", {}).get("type_labels", {})
    rows = []
    for key in sorted(metrics):
        item = metrics[key]
        rows.append(
            [
                labels.get(key, key),
                item["support"],
                item["positives"],
                _format_value(item["precision"]),
                _format_value(item["recall"]),
                _format_value(item["f1"]),
                _format_value(item["roc_auc"]),
            ]
        )
    return _table(
        [
            "Тип",
            "Support",
            "Отказы",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC",
        ],
        rows,
    )


def _render_by_failure_mode(context: dict[str, Any]) -> str:
    metrics = context["metrics"]["by_failure_mode"]
    labels = {
        **FAILURE_MODE_LABELS,
        **context["card"].get("render", {}).get("failure_mode_labels", {}),
    }
    rows = []
    for flag in [*FAILURE_FLAGS, "NO_FLAG"]:
        item = metrics.get(flag)
        if item is None:
            continue
        rows.append(
            [
                flag,
                labels.get(flag, flag),
                item["support"],
                _format_percent(item["share_of_failures"]),
                f"{item['detected']} / {item['missed']}",
                _format_value(item["recall"]),
                _format_value(item["mean_probability"]),
            ]
        )
    return _table(
        [
            "Режим",
            "Описание",
            "Support",
            "Доля отказов",
            "Обнаружено / пропущено",
            "Recall",
            "Средняя вероятность",
        ],
        rows,
    )


def _render_threshold_sweep(context: dict[str, Any]) -> str:
    rows = []
    for item in context["metrics"]["threshold_sweep"]:
        rows.append(
            [
                _format_value(item["threshold"], 2),
                _format_value(item["precision"]),
                _format_value(item["recall"]),
                _format_value(item["f1"]),
                _format_value(item["accuracy"]),
                _format_percent(item["predicted_positive_rate"]),
                f"{item['tp']} / {item['fp']} / {item['fn']}",
            ]
        )
    return _table(
        [
            "Порог",
            "Precision",
            "Recall",
            "F1",
            "Accuracy",
            "Доля предсказанных отказов",
            "TP / FP / FN",
        ],
        rows,
    )


def _render_calibration(context: dict[str, Any]) -> str:
    calibration = context["metrics"]["calibration"]
    rows = [
        ["Brier score", _format_value(calibration["brier"])],
        ["Log loss", _format_value(calibration["log_loss"])],
        ["ECE", _format_value(calibration["ece"])],
        ["Бинов ECE", calibration["bins"]],
    ]
    return _table(["Метрика калибровки", "Значение"], rows)


def _render_dataset_summary(context: dict[str, Any]) -> str:
    dataset = context["metrics"]["dataset"]
    split = dataset["split"]
    validation = dataset["validation"]
    rows = [
        ["Источник", dataset["source"]],
        ["Строк в исходном CSV", dataset["rows_raw"]],
        ["Доля отказов (исходный CSV)", _format_percent(dataset["target_rate_raw"])],
        ["Строк после ETL", dataset["rows_after_cleaning"]],
        [
            "Доля отказов после ETL",
            _format_percent(dataset["target_rate_after_cleaning"]),
        ],
        ["Признаков модели", context["metrics"]["model"]["n_features"]],
        ["Validation, строк", validation["rows"]],
        ["Validation, отказов", validation["positives"]],
        ["Validation, норма", validation["negatives"]],
        [
            "Validation, доля отказов",
            _format_percent(validation["positive_rate"]),
        ],
        ["Test size", _format_value(split["test_size"], 2)],
        ["Random state", split["random_state"]],
        ["Стратификация", _format_value(split["stratified"])],
    ]
    return _table(["Параметр", "Значение"], rows)


def _render_training_monitoring(context: dict[str, Any]) -> str:
    smoke = context["smoke"]
    quality = smoke["data_quality"]
    infrastructure = smoke.get("infrastructure", {})
    before = infrastructure.get("before", {})
    after = infrastructure.get("after", {})
    type_distribution = quality.get("type_distribution") or {}
    type_text = ", ".join(f"{key}: {value}" for key, value in type_distribution.items())
    rows = [
        ["Строк после ETL", quality["rows"]],
        ["Доля отказов", _format_percent(quality["target_rate"])],
        ["Пропусков всего", quality["null_total"]],
        ["Распределение Type", type_text or "—"],
        [
            "CPU до / после, %",
            f"{_format_value(before.get('cpu_percent'), 1)} / "
            f"{_format_value(after.get('cpu_percent'), 1)}",
        ],
        [
            "RAM до / после, %",
            f"{_format_value(before.get('ram_used_percent'), 1)} / "
            f"{_format_value(after.get('ram_used_percent'), 1)}",
        ],
        ["RAM всего, GB", _format_value(before.get("ram_total_gb"), 2)],
        [
            "Время обучения, с",
            _format_value(smoke["model_metrics"]["train_time_sec"], 1),
        ],
    ]
    return _table(["Показатель", "Smoke-прогон"], rows)


def _render_inference_monitoring(context: dict[str, Any]) -> str:
    inference = context["inference"]
    performance = inference.get("performance", {})
    rows = [
        ["Строк предсказано", inference["predictions_rows"]],
        ["risk_level=Высокий", inference["high_risk_count"]],
        ["Доля Высокий", _format_percent(inference["high_risk_rate"])],
        ["Время модели, с", _format_value(performance.get("inference_time_sec"), 4)],
        ["Строк в секунду", _format_value(performance.get("rows_per_sec"), 1)],
        ["Время pipeline, с", _format_value(performance.get("pipeline_time_sec"), 3)],
    ]
    return _table(["Показатель", "Инференс на test.csv"], rows)


def _render_psi(context: dict[str, Any]) -> str:
    drift = context["inference"]["drift"]
    rows = []
    for feature, stats in drift["features"].items():
        rows.append(
            [
                feature,
                _format_value(stats["mean_shift"], 4),
                _format_value(stats["psi"], 5),
                stats["status"],
            ]
        )
    return _table(["Признак", "Сдвиг среднего", "PSI", "Статус"], rows)


def _render_environment(context: dict[str, Any]) -> str:
    environment = context["metrics"]["environment"]
    packages = environment["packages"]
    rows = [
        ["Python", environment["python"]],
        ["Реализация", environment["implementation"]],
        ["Платформа", environment["platform"]],
        ["Машина", environment["machine"]],
        ["Процессор", environment["processor"]],
        ["Логических CPU", environment["cpu_count"]],
        ["RAM всего, GB", _format_value(environment["ram_total_gb"], 2)],
    ]
    for name, version in packages.items():
        rows.append([f"Пакет {name}", version])
    return _table(["Параметр", "Значение"], rows)


def _render_git(context: dict[str, Any]) -> str:
    metrics = context["metrics"]
    git = metrics["git"]
    rows = [
        ["Commit SHA", git["sha"]],
        ["Short SHA", git["short_sha"]],
        ["Ветка", git["branch"]],
        ["Незакоммиченные изменения", _format_value(git["dirty"])],
        ["Снимок метрик (UTC)", metrics["generated_at"]],
    ]
    return _table(["Параметр", "Значение"], rows)


def _render_outdated_metrics(context: dict[str, Any]) -> str:
    rows = [[name, _format_value(value)] for name, value in context["outdated"].items()]
    return _table(["Метрика", "Устаревший пример"], rows)


RENDERERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "full-metrics": _render_full_metrics,
    "smoke-metrics": _render_smoke_metrics,
    "cv-baseline": _render_cv_baseline,
    "by-type": _render_by_type,
    "by-failure-mode": _render_by_failure_mode,
    "threshold-sweep": _render_threshold_sweep,
    "calibration": _render_calibration,
    "dataset-summary": _render_dataset_summary,
    "training-monitoring": _render_training_monitoring,
    "inference-monitoring": _render_inference_monitoring,
    "psi": _render_psi,
    "environment": _render_environment,
    "git": _render_git,
    "outdated-metrics": _render_outdated_metrics,
}


def _block_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        rf"(<!--\s*model-card:{re.escape(name)}:start\s*-->)"
        rf"(.*?)"
        rf"(<!--\s*model-card:{re.escape(name)}:end\s*-->)",
        re.DOTALL,
    )


def render_markdown(text: str, context: dict[str, Any]) -> str:
    """Replace every model-card marker block with freshly rendered content."""
    for name, renderer in RENDERERS.items():
        pattern = _block_pattern(name)
        match = pattern.search(text)
        if match is None:
            raise ValueError(f"Marker block not found in Markdown: {name}")
        content = renderer(context).strip()

        def replace(
            matched: re.Match[str],
            block: str = content,
        ) -> str:
            return f"{matched.group(1)}\n{block}\n{matched.group(3)}"

        text = pattern.sub(replace, text, count=1)
    return text


def find_mismatches(
    text: str,
    context: dict[str, Any],
) -> list[tuple[str, str, str]]:
    """Return ``(block, expected, actual)`` for every stale marker block."""
    mismatches: list[tuple[str, str, str]] = []
    for name, renderer in RENDERERS.items():
        expected = renderer(context).strip()
        match = _block_pattern(name).search(text)
        if match is None:
            mismatches.append((name, expected, "<!-- block not found -->"))
            continue
        actual = match.group(2).strip()
        if actual != expected:
            mismatches.append((name, expected, actual))
    return mismatches


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def _resolve_metrics_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    if DEFAULT_METRICS_PATH.exists():
        return DEFAULT_METRICS_PATH
    return EXAMPLE_METRICS_PATH


def _load_card_data(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build model card metrics and render docs/model-card.md"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Metrics JSON path (default: {DEFAULT_METRICS_PATH})",
    )
    parser.add_argument(
        "--write",
        type=Path,
        nargs="?",
        const=DEFAULT_CARD_PATH,
        default=None,
        metavar="PATH",
        help="Render marker blocks in the Markdown card",
    )
    parser.add_argument(
        "--check",
        type=Path,
        nargs="?",
        const=DEFAULT_CARD_PATH,
        default=None,
        metavar="PATH",
        help="Exit with code 1 if a marker block is stale",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Metrics JSON used by --write/--check (default: auto-detect)",
    )
    parser.add_argument(
        "--card-data",
        type=Path,
        default=DEFAULT_CARD_DATA_PATH,
        help=f"YAML card data (default: {DEFAULT_CARD_DATA_PATH})",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Trained model path (default: artifacts/model.cbm)",
    )
    args = parser.parse_args(argv)

    if args.write is not None and args.check is not None:
        parser.error("--write and --check are mutually exclusive")

    if args.write is not None or args.check is not None:
        if args.output is not None:
            metrics = build_model_card_metrics(args.model, DECISION_THRESHOLD)
            _write_json(args.output, metrics)
        else:
            metrics = _load_json(_resolve_metrics_path(args.metrics))

        card_data = _load_card_data(args.card_data)
        context = load_render_context(card_data, metrics)
        target = args.write or args.check
        text = target.read_text(encoding="utf-8")

        if args.write is not None:
            rendered = render_markdown(text, context)
            if rendered != text:
                target.write_text(rendered, encoding="utf-8")
            print(f"Model card rendered: {target}")
            return 0

        mismatches = find_mismatches(text, context)
        if not mismatches:
            print(f"Model card is up to date: {target}")
            return 0

        print(f"Model card is stale: {target} ({len(mismatches)} block(s))")
        for name, expected, actual in mismatches:
            print(f"\n[{name}]")
            diff = difflib.unified_diff(
                actual.splitlines(),
                expected.splitlines(),
                fromfile="current",
                tofile="expected",
                lineterm="",
            )
            print("\n".join(list(diff)[:40]))
        return 1

    output = args.output or DEFAULT_METRICS_PATH
    metrics = build_model_card_metrics(args.model, DECISION_THRESHOLD)
    _write_json(output, metrics)
    print(f"Model card metrics saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
