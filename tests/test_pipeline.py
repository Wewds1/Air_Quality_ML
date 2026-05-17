import tempfile
from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, LABEL_COLS
from src.data_loading import load_raw_data
from src.preprocessing import preprocess_raw
from src.features import build_feature_frame


def test_data_loading():
    df = load_raw_data(str(RAW_DATA_PATH))
    assert len(df) > 0, "No rows loaded"
    assert "timestamp" in df.columns, "Missing timestamp column"
    print("✓ test_data_loading passed")


def test_preprocessing():
    df = load_raw_data(str(RAW_DATA_PATH))
    clean_df = preprocess_raw(df)
    assert "season" in clean_df.columns, "Season column missing after cleanup"
    assert clean_df["season"].notna().all(), "Null season values after cleanup"
    assert clean_df["station_type"].notna().all(), "Null station_type values after cleanup"
    print("✓ test_preprocessing passed")


def test_features():
    df = load_raw_data(str(RAW_DATA_PATH))
    clean_df = preprocess_raw(df)
    feature_df = build_feature_frame(clean_df)
    
    expected_features = ["hour_sin", "hour_cos", "pm_fine_ratio", "pm25_roll3_mean", "pm25_trend"]
    for feat in expected_features:
        assert feat in feature_df.columns, f"Missing expected feature: {feat}"
    print("✓ test_features passed")


def test_labels_present():
    df = load_raw_data(str(RAW_DATA_PATH))
    for label in LABEL_COLS:
        assert label in df.columns, f"Missing label column: {label}"
    print("✓ test_labels_present passed")


if __name__ == "__main__":
    test_data_loading()
    test_preprocessing()
    test_features()
    test_labels_present()
    print("\nAll tests passed!")