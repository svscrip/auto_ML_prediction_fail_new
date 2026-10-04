"""Generate evaluation plots for artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix


def save_confusion_matrix(y_true, y_pred, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_roc_curve(y_true, y_proba, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_feature_importance(importances: dict, path: Path, top_n: int = 15) -> None:
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*items) if items else ([], [])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_model_metrics_chart(path: Path, metrics: dict) -> None:
    """Validation metrics chart for monitoring reports."""
    labels = ["ROC-AUC", "Recall", "Precision", "F1"]
    keys = ["roc_auc", "recall", "precision", "f1"]
    values = [float(metrics.get(key, 0)) for key in keys]
    colors = ["#27ae60", "#2980b9", "#8e44ad", "#d35400"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, values, color=colors, height=0.55)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score")
    train_time = metrics.get("train_time_sec")
    title = "Model Metrics (validation)"
    if train_time is not None:
        title += f" · training {train_time:.1f} s"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=10,
        )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_infrastructure_chart(
    path: Path,
    stage: str,
    before: dict,
    after: dict,
    duration_sec: float,
    duration_label: str = "Duration",
) -> None:
    """CPU/RAM and stage duration chart."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    fig.suptitle(f"Infrastructure — {stage}", fontsize=12, fontweight="bold")

    cpu_vals = [before.get("cpu_percent", 0), after.get("cpu_percent", 0)]
    ram_vals = [before.get("ram_used_percent", 0), after.get("ram_used_percent", 0)]
    x = ["Before", "After"]

    axes[0].bar(x, cpu_vals, color=["#5dade2", "#2874a6"], width=0.55)
    axes[0].set_ylim(0, max(100, max(cpu_vals) * 1.2))
    axes[0].set_ylabel("%")
    axes[0].set_title("CPU load")
    for i, value in enumerate(cpu_vals):
        axes[0].text(i, value + 1, f"{value:.1f}%", ha="center", fontsize=9)

    axes[1].bar(x, ram_vals, color=["#58d68d", "#239b56"], width=0.55)
    axes[1].set_ylim(0, max(100, max(ram_vals) * 1.2))
    axes[1].set_ylabel("%")
    ram_total = after.get("ram_total_gb") or before.get("ram_total_gb")
    axes[1].set_title("RAM used")
    if ram_total:
        axes[1].set_xlabel(f"{ram_total:.1f} GB total", fontsize=9)
    for i, value in enumerate(ram_vals):
        axes[1].text(i, value + 1, f"{value:.1f}%", ha="center", fontsize=9)

    if duration_sec < 0.1:
        display_value = duration_sec * 1000
        ylabel = "Milliseconds"
        value_text = f"{display_value:.1f} ms"
    else:
        display_value = duration_sec
        ylabel = "Seconds"
        value_text = f"{duration_sec:.2f} s"

    axes[2].bar([duration_label], [display_value], color="#e67e22", width=0.45)
    axes[2].set_title("Stage time")
    axes[2].set_ylabel(ylabel)
    axes[2].text(
        0,
        display_value + max(display_value * 0.08, 0.5),
        value_text,
        ha="center",
        fontsize=10,
    )

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_drift_chart(drift_features: dict, path: Path) -> None:
    """PSI drift chart for inference monitoring."""
    features = list(drift_features.keys())
    psi_values = [drift_features[f]["psi"] for f in features]
    colors = [
        "#2ecc71" if v < 0.1 else "#f1c40f" if v < 0.25 else "#e74c3c"
        for v in psi_values
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(features, psi_values, color=colors)
    ax.axvline(0.1, color="orange", linestyle="--", label="PSI warning (0.1)")
    ax.axvline(0.25, color="red", linestyle="--", label="PSI critical (0.25)")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Feature Drift: train -> test")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
