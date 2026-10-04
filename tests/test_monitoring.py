from src.monitoring import evaluate_drift, interpret_psi


def test_interpret_psi_stable():
    assert interpret_psi(0.05) == "stable"
    assert interpret_psi(0.0) == "stable"


def test_interpret_psi_warning():
    assert interpret_psi(0.15) == "warning"


def test_interpret_psi_critical():
    assert interpret_psi(0.3) == "critical"


def test_evaluate_drift_all_stable():
    drift = {
        "Torque [Nm]": {"mean_shift": -0.01, "psi": 0.0002},
        "Tool wear [min]": {"mean_shift": -0.12, "psi": 0.00018},
    }
    result = evaluate_drift(drift)
    assert result["overall_status"] == "stable"
    assert result["alerts"] == []


def test_evaluate_drift_with_alert():
    drift = {
        "Torque [Nm]": {"mean_shift": 0.5, "psi": 0.3},
    }
    result = evaluate_drift(drift)
    assert result["overall_status"] == "critical"
    assert len(result["alerts"]) == 1
