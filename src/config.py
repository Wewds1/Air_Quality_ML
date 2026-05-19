from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
STATIC_DIR = ROOT / "static"

RAW_DATA_PATH = DATA_RAW_DIR / "air_quality_readings.csv"
ENGINEERED_SAMPLE_PATH = DATA_PROCESSED_DIR / "feature_engineering_sample.csv"
FEATURE_COLUMNS_PATH = OUTPUTS_DIR / "feature_engineering_columns.json"
MONITORING_SNAPSHOT_PATH = OUTPUTS_DIR / "monitoring_snapshot.json"
LIVE_READINGS_PATH = OUTPUTS_DIR / "live_readings.csv"
LIVE_PREDICTIONS_PATH = OUTPUTS_DIR / "live_predictions.csv"

LABEL_COLS = [
    "label_respiratory_risk",
    "label_cardiovascular_risk",
    "label_vulnerable_alert",
    "label_outdoor_warning",
    "label_industrial_event",
]

ID_COLS = ["reading_id", "timestamp"]
CATEGORICAL_COLS = ["season", "station_id", "station_type"]

RANDOM_SEED = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

WHO_GUIDELINES = {
    "pm25": 15.0,
    "pm10": 45.0,
    "no2": 40.0,
    "o3": 100.0,
    "so2": 40.0,
}

LIVE_BASE_COLUMNS = [
    "reading_id",
    "timestamp",
    "station_id",
    "station_type",
    "elevation_m",
    "near_highway",
    "near_industry",
    "temp_c",
    "humidity_pct",
    "wind_speed_ms",
    "wind_dir_deg",
    "pressure_hpa",
    "precipitation_mm",
    "visibility_km",
    "temp_inversion",
    "pm25",
    "pm10",
    "no2",
    "o3",
    "so2",
    "co",
    "benzene",
    "aqi",
]

LIVE_REQUIRED_COLUMNS = [
    "station_id",
    "station_type",
    "elevation_m",
    "near_highway",
    "near_industry",
    "temp_c",
    "humidity_pct",
    "wind_speed_ms",
    "precipitation_mm",
    "temp_inversion",
    "pm25",
    "pm10",
    "no2",
    "o3",
    "so2",
    "co",
    "aqi",
]
