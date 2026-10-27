"""Isolated feature ablation; never updates the working model or MLflow store.

Run with ``python -m src.ablation``. Both variants use the same raw 60/20/20
split. No label/flag-based cleaning is applied. Early stopping sees validation
only; the final holdout is scored after both models have been fitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import (
    CAT_FEATURES,
    CATBOOST_PARAMS,
    DECISION_THRESHOLD,
    FAILURE_FLAGS,
    ID_COL,
    MODEL_FEATURES,
    PRODUCT_ID_COL,
    PROJECT_ROOT,
    RANDOM_STATE,
    RAW_NUMERIC_FEATURES,
    TARGET_COL,
    TRAIN_CSV,
    TYPE_COL,
)
from src.etl.features import build_features
from src.etl.load import load_train

EXCLUDED_FEATURES = [*FAILURE_FLAGS, "total_failures_cum"]
SENSOR_FEATURES = [c for c in MODEL_FEATURES if c not in EXCLUDED_FEATURES]
VARIANTS = {
    "with_failure_signals": True,
    "without_failure_signals": False,
}
LABELS = {
    "with_failure_signals": "17 признаков, с флагами и счётчиком",
    "without_failure_signals": "11 признаков, без флагов и счётчика",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Freeze one stratified split before any feature construction or cleaning."""
    development, holdout = train_test_split(
        df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET_COL]
    )
    train, validation = train_test_split(
        development,
        test_size=0.25,
        random_state=RANDOM_STATE + 1,
        stratify=development[TARGET_COL],
    )
    return {"train": train, "validation": validation, "holdout": holdout}


def feature_matrix(df: pd.DataFrame, include_signals: bool) -> pd.DataFrame:
    """Reuse the baseline's formulas without inspecting labels or other splits.

    The sensor variant never reads the supplied failure flags. Zero placeholders
    satisfy the legacy feature builder's interface; neither these placeholders
    nor their cumulative sum enter the sensor model. Reindexing by row ID keeps
    the labels and both variants aligned after the legacy builder's sorting.
    """
    columns = [ID_COL, PRODUCT_ID_COL, TYPE_COL, *RAW_NUMERIC_FEATURES]
    if include_signals:
        columns += FAILURE_FLAGS
    inputs = df.loc[:, columns].copy()
    if not include_signals:
        inputs[FAILURE_FLAGS] = 0
    built = build_features(inputs, is_train=False).set_index(ID_COL)
    selected = MODEL_FEATURES if include_signals else SENSOR_FEATURES
    result = built.loc[df[ID_COL].tolist(), selected].copy()
    result[TYPE_COL] = result[TYPE_COL].astype(str)
    return result


def evaluate(targets: np.ndarray, scores: np.ndarray) -> dict:
    """Fixed-threshold metrics and ranking scores; no tuning on these labels."""
    targets, scores = np.asarray(targets), np.asarray(scores)
    if (
        targets.ndim != 1
        or scores.shape != targets.shape
        or set(np.unique(targets)) != {0, 1}
        or not np.isfinite(scores).all()
        or not ((scores >= 0) & (scores <= 1)).all()
    ):
        raise ValueError("Need both binary classes and finite scores in [0, 1]")
    predicted = (scores >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, predicted, labels=[0, 1]).ravel()
    return {
        "rows": int(len(targets)),
        "positives": int(targets.sum()),
        "threshold": DECISION_THRESHOLD,
        "roc_auc": float(roc_auc_score(targets, scores)),
        "average_precision": float(average_precision_score(targets, scores)),
        "precision": float(precision_score(targets, predicted, zero_division=0)),
        "recall": float(recall_score(targets, predicted, zero_division=0)),
        "f1": float(f1_score(targets, predicted, zero_division=0)),
        "brier": float(brier_score_loss(targets, scores)),
        "log_loss": float(log_loss(targets, scores, labels=[0, 1])),
        "confusion_matrix": {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
        },
    }


def result_table(report: dict) -> str:
    """A single table source for the experiment note and the model card."""
    lines = [
        "| Вариант | ROC-AUC | AP | Recall | Precision | F1 | TP / FP / FN |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in VARIANTS:
        metrics = report["variants"][name]["holdout_metrics"]
        values = " | ".join(
            f"{metrics[key]:.4f}"
            for key in ("roc_auc", "average_precision", "recall", "precision", "f1")
        )
        counts = metrics["confusion_matrix"]
        lines.append(
            f"| {LABELS[name]} | {values} | "
            f"{counts['tp']} / {counts['fp']} / {counts['fn']} |"
        )
    return "\n".join(lines)


def run_experiment(output: Path, data_path: Path = TRAIN_CSV) -> dict:
    """Train two independent artifacts in a new directory, then evaluate once."""
    # Refuse an existing location before reading data or fitting any model.
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    df = load_train(data_path)
    parts = split_data(df)
    params = {**CATBOOST_PARAMS, "allow_writing_files": False, "thread_count": 4}
    assignments = pd.concat(
        [part[[ID_COL]].assign(split=name) for name, part in parts.items()]
    ).sort_values(ID_COL)
    assignments.to_csv(output / "split_assignments.csv", index=False)
    split_summary = {}
    for name, part in parts.items():
        ids = sorted(int(value) for value in part[ID_COL])
        split_summary[name] = {
            "rows": len(part),
            "positives": int(part[TARGET_COL].sum()),
            "contradictory_rows_retained": int(
                ((part[TARGET_COL] == 0) & (part[FAILURE_FLAGS].sum(axis=1) > 0)).sum()
            ),
            "sorted_ids_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }

    models = {}
    results = {}
    # Neither this loop nor feature_matrix receives the holdout's labels.
    for name, include_signals in VARIANTS.items():
        print(f"Training {name}...", flush=True)
        cats = CAT_FEATURES if include_signals else [TYPE_COL]
        train = Pool(
            feature_matrix(parts["train"], include_signals),
            parts["train"][TARGET_COL].to_numpy(),
            cat_features=cats,
        )
        validation = Pool(
            feature_matrix(parts["validation"], include_signals),
            parts["validation"][TARGET_COL].to_numpy(),
            cat_features=cats,
        )
        model = CatBoostClassifier(**params)
        start = time.perf_counter()
        model.fit(train, eval_set=validation, use_best_model=True)
        duration = time.perf_counter() - start
        model.save_model(str(output / f"{name}.cbm"))
        models[name] = model
        results[name] = {
            "features": MODEL_FEATURES if include_signals else SENSOR_FEATURES,
            "categorical_features": cats,
            "tree_count": model.tree_count_,
            "training_seconds": duration,
            "validation_metrics": evaluate(
                parts["validation"][TARGET_COL].to_numpy(),
                model.predict_proba(validation)[:, 1],
            ),
        }

    print("Both models fitted; evaluating the frozen final holdout...", flush=True)
    holdout = parts["holdout"]
    predictions = holdout[[ID_COL, TARGET_COL]].copy()
    for name, include_signals in VARIANTS.items():
        pool = Pool(
            feature_matrix(holdout, include_signals),
            cat_features=results[name]["categorical_features"],
        )
        scores = models[name].predict_proba(pool)[:, 1]
        results[name]["holdout_metrics"] = evaluate(
            holdout[TARGET_COL].to_numpy(), scores
        )
        predictions[name] = scores
    predictions.to_csv(output / "holdout_predictions.csv", index=False)

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    sources = [
        "src/ablation.py",
        "src/config.py",
        "src/etl/features.py",
        "src/etl/load.py",
        "poetry.lock",
    ]
    report = {
        "schema_version": "1.0",
        "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": {p: file_hash(PROJECT_ROOT / p) for p in sources},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                p: version(p) for p in ("catboost", "pandas", "numpy", "scikit-learn")
            },
        },
        "data": {
            "file": data_path.name,
            "sha256": file_hash(data_path),
            "rows": len(df),
        },
        "protocol": {
            "split": "stratified random 60/20/20, frozen before feature construction",
            "holdout_seed": RANDOM_STATE,
            "validation_seed": RANDOM_STATE + 1,
            "label_or_flag_based_row_filtering": False,
            "cumulative_feature": (
                "calculated separately within each split; not temporal history"
            ),
            "selection": "early stopping on validation only; no holdout tuning",
            "threshold": DECISION_THRESHOLD,
            "catboost_params": params,
            "working_model_replaced": False,
            "mlflow_used": False,
        },
        "splits": split_summary,
        "variants": results,
        "evidence_sha256": {
            p.name: file_hash(p) for p in output.iterdir() if p.is_file()
        },
        "limits": [
            "Synthetic data, one split and seed; no temporal prediction claim",
            "Joint removal of six features cannot isolate each feature's effect",
            "Reference has batch-dependent cumulative feature and post-event flags",
            "Reference is a new controlled run, not the historical AUC 0.9701 snapshot",
            "Balanced class weights remain; scores are not calibrated probabilities",
        ],
    }
    (output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "results.md").write_text(result_table(report) + "\n", encoding="utf-8")
    print(result_table(report), flush=True)
    print(f"Results: {output.resolve()}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or PROJECT_ROOT / "artifacts/experiments" / (
        f"feature-ablation-{stamp}-{uuid.uuid4().hex[:8]}"
    )
    run_experiment(output)


if __name__ == "__main__":
    main()
