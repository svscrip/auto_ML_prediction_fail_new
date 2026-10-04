import pytest

from src.config import TRAIN_CSV
from src.etl.load import load_train, validate_schema


def test_train_file_exists():
    assert TRAIN_CSV.exists()


def test_load_train_schema():
    df = load_train()
    assert len(df) > 1000
    assert "Machine failure" in df.columns
    assert df["Machine failure"].isin([0, 1]).all()


def test_validate_schema_rejects_missing_column():
    df = load_train().drop(columns=["Torque [Nm]"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_schema(df)
