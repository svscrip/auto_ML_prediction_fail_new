import json
import time

import pytest

from src.config import ARTIFACTS_DIR, SMOKE_SAMPLE_SIZE
from src.train import train_model


@pytest.mark.slow
def test_smoke_train_metrics():
    metrics = train_model(sample_size=SMOKE_SAMPLE_SIZE, output_dir=ARTIFACTS_DIR)
    assert metrics["roc_auc"] >= 0.85
    assert metrics["recall"] >= 0.5
    assert (ARTIFACTS_DIR / "model.cbm").exists()
    assert (ARTIFACTS_DIR / "metrics.json").exists()


def test_smoke_train_fast():
    """Quick smoke without slow marker for CI default run."""
    t0 = time.perf_counter()
    metrics = train_model(sample_size=2000, output_dir=ARTIFACTS_DIR / "smoke_test")
    elapsed = time.perf_counter() - t0
    assert elapsed < 120
    assert metrics["roc_auc"] > 0.7
    metrics_path = ARTIFACTS_DIR / "smoke_test" / "metrics.json"
    assert metrics_path.exists()
    with open(metrics_path, encoding="utf-8") as f:
        saved = json.load(f)
    assert "f1" in saved
