"""Tests for the generated model card and its machine-readable sources."""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.metrics import brier_score_loss, log_loss

from src.model_card import (
    DEFAULT_CARD_DATA_PATH,
    DEFAULT_CARD_PATH,
    RENDERERS,
    constant_prior_baseline,
    expected_calibration_error,
    find_mismatches,
    load_render_context,
    render_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "artifacts" / "example_model_card_metrics.json"

with open(DEFAULT_CARD_DATA_PATH, encoding="utf-8") as file:
    CARD = yaml.safe_load(file)
with open(METRICS_PATH, encoding="utf-8") as file:
    METRICS = json.load(file)
with open(
    ROOT / "artifacts" / "example_monitoring_summary.json", encoding="utf-8"
) as file:
    SMOKE = json.load(file)
with open(
    ROOT / "reference-material" / "keis7-research" / "model_metadata.json",
    encoding="utf-8",
) as file:
    CV = json.load(file)
with open(
    ROOT / "artifacts" / "example_inference_monitoring.json", encoding="utf-8"
) as file:
    INFERENCE = json.load(file)

MITCHELL_SECTIONS = [
    "model_details",
    "intended_use",
    "factors",
    "metrics",
    "evaluation_data",
    "training_data",
    "quantitative_analyses",
    "ethical_considerations",
    "caveats_and_recommendations",
]


def test_card_data_has_mitchell_sections():
    assert [key for key in CARD if key in MITCHELL_SECTIONS] == MITCHELL_SECTIONS
    for section in MITCHELL_SECTIONS:
        assert CARD[section], section


def test_card_sources_exist():
    assert CARD["sources"]
    for key, relative_path in CARD["sources"].items():
        assert (ROOT / relative_path).exists(), f"{key}: {relative_path}"


def test_model_details_match_snapshot():
    assert CARD["model_details"]["features"] == METRICS["model"]["n_features"]
    assert CARD["model_details"]["artifact"] == METRICS["model"]["path"]
    assert CARD["model_details"]["decision_threshold"] == pytest.approx(
        METRICS["decision_threshold"]
    )
    assert (
        CARD["metrics"]["decision_threshold"]
        == CARD["model_details"]["decision_threshold"]
    )


@pytest.mark.parametrize(
    "key",
    ["roc_auc", "recall", "precision", "f1", "accuracy", "ece"],
)
def test_full_run_metrics_match_snapshot(key):
    assert CARD["metrics"]["full_run"][key] == pytest.approx(
        METRICS["overall"][key], abs=1e-4
    )


def test_dataset_summary_matches_snapshot():
    dataset = METRICS["dataset"]
    validation = dataset["validation"]
    assert CARD["training_data"]["rows_raw"] == dataset["rows_raw"]
    assert CARD["training_data"]["rows_after_etl"] == dataset["rows_after_cleaning"]
    assert CARD["training_data"]["target_rate_raw"] == pytest.approx(
        dataset["target_rate_raw"], abs=1e-4
    )
    assert CARD["evaluation_data"]["validation_rows"] == validation["rows"]
    assert CARD["evaluation_data"]["validation_positives"] == validation["positives"]
    assert CARD["evaluation_data"]["validation_negatives"] == validation["negatives"]


def test_smoke_and_cv_references_match_sources():
    smoke = SMOKE["model_metrics"]
    for key in ("roc_auc", "recall", "precision", "f1"):
        assert CARD["metrics"]["smoke_run"][key] == pytest.approx(smoke[key])
        assert CARD["metrics"]["cv_baseline"][f"{key}_mean"] == pytest.approx(
            CV[f"{key}_mean"], abs=1e-4
        )
        assert CARD["metrics"]["cv_baseline"][f"{key}_std"] == pytest.approx(
            CV[f"{key}_std"], abs=1e-4
        )


def test_inference_reference_matches_source():
    assert CARD["evaluation_data"]["inference_rows"] == INFERENCE["predictions_rows"]
    assert CARD["quantitative_analyses"]["monitoring_findings"]
    assert INFERENCE["high_risk_rate"] == pytest.approx(0.1)


def test_threshold_sweep_contains_decision_threshold():
    rows = METRICS["threshold_sweep"]
    assert [row["threshold"] for row in rows] == [0.1, 0.25, 0.5, 0.75, 0.9]
    selected = next(row for row in rows if row["threshold"] == 0.5)
    assert selected["recall"] == pytest.approx(METRICS["overall"]["recall"])
    assert selected["precision"] == pytest.approx(METRICS["overall"]["precision"])


def test_failure_modes_cover_all_flags_and_no_flag():
    assert list(METRICS["by_failure_mode"]) == [
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
        "NO_FLAG",
    ]
    flagged = [mode for mode in METRICS["by_failure_mode"] if mode != "NO_FLAG"]
    for mode in flagged:
        assert METRICS["by_failure_mode"][mode]["recall"] == pytest.approx(1.0)


def test_markdown_blocks_are_up_to_date():
    context = load_render_context(CARD, METRICS)
    text = DEFAULT_CARD_PATH.read_text(encoding="utf-8")
    assert find_mismatches(text, context) == []


def test_render_is_idempotent_and_markers_are_unique():
    context = load_render_context(CARD, METRICS)
    text = DEFAULT_CARD_PATH.read_text(encoding="utf-8")
    for name in RENDERERS:
        assert text.count(f"<!-- model-card:{name}:start -->") == 1
        assert text.count(f"<!-- model-card:{name}:end -->") == 1
    assert render_markdown(text, context) == text


def test_stale_block_is_detected():
    context = load_render_context(CARD, METRICS)
    text = DEFAULT_CARD_PATH.read_text(encoding="utf-8")
    stale = text.replace("| ROC-AUC | 0.9701 |", "| ROC-AUC | 0.0000 |")
    assert stale != text
    assert [name for name, _, _ in find_mismatches(stale, context)] == ["full-metrics"]


def test_markdown_internal_links_resolve():
    text = DEFAULT_CARD_PATH.read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)]+)\)", text)
    assert targets
    for target in targets:
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = target.split("#", 1)[0]
        assert (DEFAULT_CARD_PATH.parent / path).exists(), target


def test_constant_baseline_matches_sklearn_and_beats_snapshot_model():
    rate = METRICS["dataset"]["validation"]["positive_rate"]
    rows = METRICS["dataset"]["validation"]["rows"]
    positives = METRICS["dataset"]["validation"]["positives"]
    targets = np.r_[np.ones(positives), np.zeros(rows - positives)]
    constant = np.full(rows, rate)
    baseline = constant_prior_baseline(rate)
    assert baseline["brier"] == pytest.approx(brier_score_loss(targets, constant))
    assert baseline["log_loss"] == pytest.approx(log_loss(targets, constant))
    # Balanced class weights make the scores worse probabilities than the prior.
    assert METRICS["calibration"]["brier"] > baseline["brier"]
    assert METRICS["calibration"]["log_loss"] > baseline["log_loss"]


def test_expected_calibration_error_is_zero_for_perfect_scores():
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(targets, probabilities) == pytest.approx(0.0)
