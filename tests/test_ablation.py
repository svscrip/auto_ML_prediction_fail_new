"""Check isolation, label alignment and the interpretation of ablation metrics."""

import numpy as np
import pandas as pd
import pytest

from src.ablation import (
    EXCLUDED_FEATURES,
    SENSOR_FEATURES,
    evaluate,
    feature_matrix,
    run_experiment,
    split_data,
)
from src.config import FAILURE_FLAGS, ID_COL, MODEL_FEATURES, TARGET_COL
from src.etl.features import build_features
from src.etl.load import load_train


@pytest.fixture
def data():
    frame = load_train().head(100).copy()
    # Make both classes and contradictory observations explicit in the fixture.
    frame[TARGET_COL] = np.arange(len(frame)) % 2
    frame[FAILURE_FLAGS] = 1
    return frame.sample(frac=1, random_state=7)


def test_splits_are_disjoint_lossless_and_reproducible(data):
    parts = split_data(data)
    ids = [set(part[ID_COL]) for part in parts.values()]
    assert [len(part) for part in parts.values()] == [60, 20, 20]
    assert set.union(*ids) == set(data[ID_COL])
    assert sum(map(len, ids)) == len(set.union(*ids))
    assert sum((part[TARGET_COL] == 0).sum() for part in parts.values()) == 50
    for name, part in parts.items():
        pd.testing.assert_frame_equal(part, split_data(data)[name])
        assert part[TARGET_COL].nunique() == 2


def test_sensor_features_ignore_labels_flags_and_batch_composition(data):
    expected = feature_matrix(data, include_signals=False)
    changed = data.copy()
    changed[FAILURE_FLAGS] = 0
    changed[TARGET_COL] = 1 - changed[TARGET_COL]
    pd.testing.assert_frame_equal(expected, feature_matrix(changed, False))
    # The variant also works when the unavailable labels/flags are absent.
    absent = data.drop(columns=[TARGET_COL, *FAILURE_FLAGS])
    pd.testing.assert_frame_equal(expected, feature_matrix(absent, False))
    row = data.iloc[[7]]
    pd.testing.assert_frame_equal(expected.loc[row[ID_COL]], feature_matrix(row, False))
    assert list(expected.columns) == SENSOR_FEATURES
    assert len(expected.columns) == 11
    assert not set(EXCLUDED_FEATURES) & set(expected.columns)


def test_reference_preserves_rows_and_alignment_without_cleaning(data):
    matrix = feature_matrix(data, include_signals=True)
    assert matrix.index.tolist() == data[ID_COL].tolist()
    assert len(matrix) == len(data)
    assert list(matrix.columns) == MODEL_FEATURES
    expected = build_features(data, is_train=False).set_index(ID_COL)
    pd.testing.assert_frame_equal(matrix, expected.loc[data[ID_COL], MODEL_FEATURES])


def test_fixed_threshold_counts_and_ranking_metrics():
    result = evaluate(np.array([0, 0, 1, 1]), np.array([0.1, 0.8, 0.7, 0.9]))
    assert result["confusion_matrix"] == {"tp": 2, "fp": 1, "tn": 1, "fn": 0}
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1
    assert result["roc_auc"] == pytest.approx(0.75)
    assert result["average_precision"] == pytest.approx(5 / 6)


@pytest.mark.parametrize("scores", [[np.nan, 0.5], [-0.1, 1], [0.1, 1.1]])
def test_metrics_reject_invalid_scores(scores):
    with pytest.raises(ValueError):
        evaluate(np.array([0, 1]), np.array(scores))


def test_existing_output_is_never_reused(tmp_path):
    model = tmp_path / "model.cbm"
    model.write_bytes(b"existing working model")
    with pytest.raises(FileExistsError):
        run_experiment(tmp_path, data_path=tmp_path / "missing.csv")
    assert model.read_bytes() == b"existing working model"
