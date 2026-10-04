"""Inference and business recommendations."""

import argparse
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool

from src.config import (
    ARTIFACTS_DIR,
    CAT_FEATURES,
    PRODUCT_ID_COL,
    TYPE_COL,
)
from src.etl.features import build_features, get_feature_matrix
from src.etl.load import load_test, load_train
from src.monitoring import compare_distributions, compute_data_quality_report


def assign_risk_level_vector(probs: pd.Series) -> pd.Series:
    quart_50 = probs.quantile(0.50)
    quart_90 = probs.quantile(0.90)
    return pd.cut(
        probs,
        bins=[-0.01, quart_50, quart_90, 1.01],
        labels=["Низкий", "Средний", "Высокий"],
    ).astype(str)


def predict(
    model_path: Path | None = None,
    output_dir: Path | None = None,
    use_train_for_drift: bool = True,
) -> pd.DataFrame:
    """
    Выполняет предсказание вероятности отказа оборудования и присваивает риск
    из перечисления ["Низкий", "Средний", "Высокий"].
    Формирует рекомендации по улучшению качества модели и, при необходимости,
    рассчитывает дрейф данных.
    Для презентации работоспособности используются тестовые данные
    """
    output_dir = output_dir or ARTIFACTS_DIR
    model_path = model_path or output_dir / "model.cbm"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python -m src.train"
        )

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    test_raw = load_test()
    test_df = build_features(test_raw, is_train=False)
    X, _ = get_feature_matrix(test_df, include_target=False)

    cat_idx = [X.columns.get_loc(c) for c in CAT_FEATURES if c in X.columns]
    pool = Pool(X, cat_features=cat_idx)
    probabilities = model.predict_proba(pool)[:, 1]

    result = test_df[
        [c for c in ["id", PRODUCT_ID_COL, TYPE_COL] if c in test_df.columns]
    ].copy()
    if "id" not in result.columns:
        result = test_df[[PRODUCT_ID_COL, TYPE_COL]].copy()

    result["failure_probability"] = probabilities
    for col in [
        "efficiency [%]",
        "Tool wear [min]",
        "delta_temperature [K]",
        "Power [kW]",
        "air_mass",
    ]:
        if col in test_df.columns:
            result[col] = test_df[col].values
    if "Power [kW]" in result.columns:
        result["mechanical_power [kW]"] = result["Power [kW]"]
    if "air_mass" in result.columns:
        result["air_mass [kg/s]"] = result["air_mass"]

    result["risk_level"] = assign_risk_level_vector(pd.Series(probabilities))

    predictions_path = output_dir / "predictions.csv"
    result.to_csv(predictions_path, index=False)

    recommendations = build_recommendations(result, test_df)
    rec_path = output_dir / "maintenance_recommendations.csv"
    recommendations.to_csv(rec_path, index=False)

    if use_train_for_drift:
        train_df = build_features(load_train(), is_train=True)
        drift = compare_distributions(train_df, test_df)
        report = {
            "predictions_rows": len(result),
            "high_risk_count": int((result["risk_level"] == "Высокий").sum()),
            "drift": drift,
            "test_quality": compute_data_quality_report(test_df, "test"),
        }
        with open(output_dir / "inference_monitoring.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return result


def build_recommendations(
    predictions: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """
    На основе предсказаний формирует рекомендации по улучшению качества модели.
    """
    rows = []

    if "efficiency [%]" in predictions.columns:
        low_eff = (predictions["efficiency [%]"] < 50).mean()
        if low_eff > 0.1:
            rows.append(
                {
                    "Проблема": "Низкая эффективность системы",
                    "Решение": "Настроить систему при КПД < 50.0%",
                    "Приоритет": "Высокий",
                }
            )

    if "Tool wear [min]" in predictions.columns:
        wear_threshold = predictions["Tool wear [min]"].quantile(0.90)
        high_wear = (predictions["Tool wear [min]"] > wear_threshold).sum()
        if high_wear > 0:
            rows.append(
                {
                    "Проблема": "Критический износ инструмента",
                    "Решение": f"Плановые замены при износе > {wear_threshold:.0f} мин",
                    "Приоритет": "Высокий",
                }
            )

    high_risk = (predictions["risk_level"] == "Высокий").sum()
    if high_risk > 0:
        rows.append(
            {
                "Проблема": "Высокий прогноз отказа",
                "Решение": "Целевое обслуживание оборудования с risk_level=Высокий",
                "Приоритет": "Высокий",
            }
        )

    if not rows:
        rows.append(
            {
                "Проблема": "Стабильное состояние",
                "Решение": "Продолжить мониторинг KPI",
                "Приоритет": "Низкий",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = predict(model_path=args.model, output_dir=args.output_dir)
    print(f"Predictions saved: {len(result)} rows")


if __name__ == "__main__":
    main()
