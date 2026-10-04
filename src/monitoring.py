import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from src.config import RAW_NUMERIC_FEATURES, TARGET_COL

PSI_STABLE = 0.1
PSI_WARNING = 0.25


def compute_data_quality_report(df: pd.DataFrame, label: str = "train") -> dict:
    """
    Формирует отчет о качестве данных переданного датасета.

    Возвращает число строк (rows), количество пустых значений (null_counts),
    арифметическое среднее и стандартное отклонение для числовых колонок.
    """
    report = {
        "dataset": label,
        "rows": len(df),
        "null_counts": df.isnull().sum().to_dict(),
        "target_rate": (
            float(df[TARGET_COL].mean()) if TARGET_COL in df.columns else None
        ),
        "type_distribution": (
            df["Type"].value_counts().to_dict() if "Type" in df.columns else {}
        ),
    }
    for col in RAW_NUMERIC_FEATURES:
        if col in df.columns:
            report[f"mean_{col}"] = float(df[col].mean())
            report[f"std_{col}"] = float(df[col].std())
    return report


def population_stability_index(
    expected: pd.Series, actual: pd.Series, bins: int = 10
) -> float:
    """
    Рассчитывает индекс стабильности популяции (PSI) для оценки смещения распределений.

    PSI (Population Stability Index) измеряет, насколько распределение
    фактических данных (например, за текущий период) отличается от
    ожидаемого распределения (например, за базовый период). Метрика широко
    используется в кредитном скоринге и мониторинге моделей машинного обучения
    для обнаружения дрейфа признаков.

    Интерпретация результатов (эмпирическое правило):
        - PSI < 0.1   : Распределения практически идентичны (изменений нет).
        - 0.1 <= PSI < 0.25 : Небольшой сдвиг (требуется внимание).
        - PSI >= 0.25  : Значительный сдвиг, модель требует переобучения.
    """
    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1,
    )
    expected_pct = pd.cut(expected, breakpoints, duplicates="drop").value_counts(
        normalize=True
    )
    actual_pct = pd.cut(actual, breakpoints, duplicates="drop").value_counts(
        normalize=True
    )
    aligned = pd.concat([expected_pct, actual_pct], axis=1, join="outer").fillna(0.0001)
    aligned.columns = ["expected", "actual"]
    psi = (
        (aligned["actual"] - aligned["expected"])
        * np.log(aligned["actual"] / aligned["expected"])
    ).sum()
    return float(psi)


def compare_distributions(reference: pd.DataFrame, current: pd.DataFrame) -> dict:
    """
    Выполняет сравнение распределений целевого датасета с новым датасетом
    с помощью разницы между средними значениями соответствующих колонок и метрики PSI
    """
    drift = {}
    for col in RAW_NUMERIC_FEATURES:
        if col not in reference.columns or col not in current.columns:
            continue
        drift[col] = {
            "mean_shift": float(current[col].mean() - reference[col].mean()),
            "psi": population_stability_index(reference[col], current[col]),
        }
    return drift


def infrastructure_snapshot() -> dict:
    """
    Возвращает информацию об используемых ресурсах
    """
    mem = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_total_gb": round(mem.total / (1024**3), 2),
        "ram_used_percent": mem.percent,
        "swap_memory_used": psutil.swap_memory().used,
        "swap_memory_free": psutil.swap_memory().free,
    }


def interpret_psi(psi: float) -> str:
    """Classify drift severity by Population Stability Index."""
    if psi < PSI_STABLE:
        return "stable"
    if psi < PSI_WARNING:
        return "warning"
    return "critical"


def evaluate_drift(drift: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Summarize per-feature PSI drift with overall status and alerts."""
    features = {}
    alerts: list[dict[str, Any]] = []
    worst_status = "stable"
    status_rank = {"stable": 0, "warning": 1, "critical": 2}

    for feature, stats in drift.items():
        psi = float(stats["psi"])
        status = interpret_psi(psi)
        features[feature] = {
            "mean_shift": float(stats["mean_shift"]),
            "psi": psi,
            "status": status,
        }
        if status_rank[status] > status_rank[worst_status]:
            worst_status = status
        if status != "stable":
            alerts.append(
                {
                    "feature": feature,
                    "psi": psi,
                    "status": status,
                    "message": f"PSI={psi:.4f} — дрейф признака {feature}",
                }
            )

    return {
        "overall_status": worst_status,
        "features": features,
        "alerts": alerts,
        "thresholds": {
            "stable": f"< {PSI_STABLE}",
            "warning": f"{PSI_STABLE} – {PSI_WARNING}",
            "critical": f">= {PSI_WARNING}",
        },
    }


def build_training_monitoring_summary(
    metrics: dict[str, Any],
    data_report: dict[str, Any],
) -> dict[str, Any]:
    """Consolidated training-time monitoring snapshot."""
    infra = data_report.get("infrastructure", {})
    return {
        "stage": "training",
        "model_metrics": {
            "roc_auc": metrics.get("roc_auc"),
            "recall": metrics.get("recall"),
            "precision": metrics.get("precision"),
            "f1": metrics.get("f1"),
            "train_time_sec": metrics.get("train_time_sec"),
        },
        "data_quality": {
            "rows": data_report.get("rows"),
            "target_rate": data_report.get("target_rate"),
            "null_total": sum(data_report.get("null_counts", {}).values()),
            "type_distribution": data_report.get("type_distribution"),
        },
        "infrastructure": infra,
    }


def build_inference_monitoring_summary(
    predictions_rows: int,
    high_risk_count: int,
    drift: dict[str, dict[str, float]],
    test_quality: dict[str, Any],
    infrastructure: dict[str, Any] | None = None,
    inference_time_sec: float | None = None,
    pipeline_time_sec: float | None = None,
) -> dict[str, Any]:
    """Consolidated inference-time monitoring snapshot."""
    drift_eval = evaluate_drift(drift)
    performance: dict[str, Any] = {}
    if inference_time_sec is not None:
        performance["inference_time_sec"] = round(inference_time_sec, 3)
        if predictions_rows:
            performance["rows_per_sec"] = round(
                predictions_rows / inference_time_sec, 1
            )
    if pipeline_time_sec is not None:
        performance["pipeline_time_sec"] = round(pipeline_time_sec, 3)

    return {
        "stage": "inference",
        "predictions_rows": predictions_rows,
        "high_risk_count": high_risk_count,
        "high_risk_rate": (
            round(high_risk_count / predictions_rows, 4) if predictions_rows else 0
        ),
        "performance": performance,
        "drift": drift_eval,
        "test_quality": {
            "rows": test_quality.get("rows"),
            "null_total": sum(test_quality.get("null_counts", {}).values()),
            "type_distribution": test_quality.get("type_distribution"),
        },
        "infrastructure": infrastructure,
    }


def save_monitoring_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
