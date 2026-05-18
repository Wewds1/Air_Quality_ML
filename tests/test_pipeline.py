from fastapi.testclient import TestClient

import app as app_module
from src.config import MODELS_DIR, OUTPUTS_DIR, RAW_DATA_PATH
from src.data_loading import load_raw_data
from src.features import build_feature_frame
from src.monitoring import build_dashboard_snapshot
from src.preprocessing import MISSING_COLS, preprocess_raw
from src.validation import validation_summary


def test_validation_summary_tracks_known_sensor_issues():
    df = load_raw_data(str(RAW_DATA_PATH))
    summary = validation_summary(df)

    assert summary["duplicates_excluding_reading_id"] == 28
    assert summary["physical_checks"]["pm10_less_than_pm25"] == 409
    assert summary["physical_checks"]["zero_wind_speed"] == 129


def test_preprocessing_repairs_and_imputes_sensor_values():
    df = load_raw_data(str(RAW_DATA_PATH))
    clean_df = preprocess_raw(df)

    assert len(clean_df) == 10000
    assert clean_df["season"].isin(["Winter", "Spring", "Summer", "Fall", "Unknown"]).all()
    assert (clean_df["pm10"] >= clean_df["pm25"]).all()
    assert clean_df[MISSING_COLS].isna().sum().sum() == 0
    assert {"pm10_adjusted_flag", "aqi_gap_flag", "wind_speed_zero_flag"}.issubset(clean_df.columns)


def test_features_cover_operational_risk_context():
    df = load_raw_data(str(RAW_DATA_PATH))
    feature_df = build_feature_frame(preprocess_raw(df))

    expected_features = {
        "pm25_cardio_zone",
        "multi_label_event",
        "pollutants_above_who",
        "pm25_roll3_mean",
        "aqi_trend",
    }

    assert expected_features.issubset(feature_df.columns)
    assert "wind_dir_deg" not in feature_df.columns
    assert "hour" not in feature_df.columns
    assert "month" not in feature_df.columns


def test_dashboard_snapshot_includes_live_sections():
    snapshot = build_dashboard_snapshot(
        OUTPUTS_DIR / "smoke_test_predictions.csv",
        metrics_path=MODELS_DIR / "metrics_summary.json",
    )

    assert snapshot["status"] == "ok"
    assert {"overview", "alerts", "stations", "timeline", "labels", "monitoring", "model"}.issubset(snapshot.keys())
    assert len(snapshot["labels"]) == 5


def test_dashboard_api_returns_json(monkeypatch):
    monkeypatch.setattr(app_module, "PREDICTIONS_PATH", OUTPUTS_DIR / "smoke_test_predictions.csv")
    monkeypatch.setattr(app_module, "METRICS_PATH", MODELS_DIR / "metrics_summary.json")

    client = TestClient(app_module.app)
    response = client.get("/api/dashboard")
    home_response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "overview" in payload
    assert home_response.status_code == 200
    assert "Air Quality Command Center" in home_response.text
