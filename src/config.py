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
