"""Train CatBoost model with MLflow tracking."""

import argparse
import json
import time
from pathlib import Path

import mlflow
import mlflow.catboost
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    ARTIFACTS_DIR,
    CAT_FEATURES,
    CATBOOST_PARAMS,
    RANDOM_STATE,
    SMOKE_SAMPLE_SIZE,
    TRAIN_TEST_SIZE,
)
from src.etl.features import build_features, get_feature_matrix
from src.etl.load import load_train
from src.mlflow_setup import setup_mlflow
from src.monitoring import (
    build_training_monitoring_summary,
    compute_data_quality_report,
    infrastructure_snapshot,
    save_monitoring_report,
)
from src.plots import (
    save_confusion_matrix,
    save_feature_importance,
    save_infrastructure_chart,
    save_model_metrics_chart,
    save_roc_curve,
)


def train_model(
    sample_size: int | None = None,
    output_dir: Path | None = None,
) -> dict:
    output_dir = output_dir or ARTIFACTS_DIR
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_mlflow()

    df = load_train()
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_STATE)

    df = build_features(df, is_train=True)
    X, y = get_feature_matrix(df, include_target=True)

    cat_idx = [X.columns.get_loc(c) for c in CAT_FEATURES if c in X.columns]
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TRAIN_TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)

    infra_before = infrastructure_snapshot()
    t0 = time.perf_counter()

    with mlflow.start_run(run_name="catboost_train"):
        for key, value in CATBOOST_PARAMS.items():
            mlflow.log_param(key, value)
        mlflow.log_param("sample_size", sample_size or len(df))
        mlflow.log_param("n_features", len(X.columns))

        model = CatBoostClassifier(**CATBOOST_PARAMS)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        train_time = time.perf_counter() - t0
        mlflow.log_metric("train_time_sec", train_time)

        y_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, y_proba)),
            "train_time_sec": train_time,
        }
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, value)

        infra_after = infrastructure_snapshot()
        mlflow.log_metric("cpu_percent_before", infra_before["cpu_percent"])
        mlflow.log_metric("cpu_percent_after", infra_after["cpu_percent"])
        mlflow.log_metric("ram_used_percent_before", infra_before["ram_used_percent"])
        mlflow.log_metric("ram_used_percent_after", infra_after["ram_used_percent"])

        model_path = output_dir / "model.cbm"
        model.save_model(str(model_path))
        mlflow.log_artifact(str(model_path))

        importance = dict(zip(X.columns, model.get_feature_importance().tolist()))
        metrics["feature_importance_top5"] = dict(
            sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        )

        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(str(metrics_path))
        save_confusion_matrix(y_val, y_pred, plots_dir / "confusion_matrix.png")
        save_roc_curve(y_val, y_proba, plots_dir / "roc_curve.png")
        save_feature_importance(importance, plots_dir / "feature_importance.png")
        for plot in plots_dir.glob("*.png"):
            mlflow.log_artifact(str(plot))

        data_report = compute_data_quality_report(df, "train_processed")
        data_report["infrastructure"] = {
            "before": infra_before,
            "after": infra_after,
        }
        save_monitoring_report(data_report, output_dir / "data_quality.json")
        mlflow.log_artifact(str(output_dir / "data_quality.json"))

        monitoring_summary = build_training_monitoring_summary(metrics, data_report)
        summary_path = output_dir / "monitoring_summary.json"
        save_monitoring_report(monitoring_summary, summary_path)
        mlflow.log_artifact(str(summary_path))

        save_model_metrics_chart(plots_dir / "model_metrics.png", metrics)
        save_infrastructure_chart(
            plots_dir / "infrastructure_training.png",
            stage="training",
            before=infra_before,
            after=infra_after,
            duration_sec=train_time,
            duration_label="Training",
        )
        mlflow.log_artifact(str(plots_dir / "model_metrics.png"))
        mlflow.log_artifact(str(plots_dir / "infrastructure_training.png"))

        mlflow.catboost.log_model(model, name="model")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train machine failure model")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Use subset for smoke tests (default: full dataset)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Train on {SMOKE_SAMPLE_SIZE} rows",
    )
    args = parser.parse_args()
    sample = args.sample_size or (SMOKE_SAMPLE_SIZE if args.smoke else None)
    metrics = train_model(sample_size=sample)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
