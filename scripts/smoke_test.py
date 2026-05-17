from pathlib import Path

from src.config import RAW_DATA_PATH, OUTPUTS_DIR, MODELS_DIR
from src.data_loading import load_raw_data
from src.preprocessing import preprocess_raw
from src.features import build_feature_frame
from src.score_batch import score_batch
from src.monitoring import build_monitoring_report


def run_smoke_test():
    print("=== Smoke Test: Raw CSV to Batch Scoring to Monitoring ===\n")

    print("1. Loading raw data...")
    raw_df = load_raw_data(str(RAW_DATA_PATH))
    print(f"   ✓ Loaded {len(raw_df)} rows, {len(raw_df.columns)} columns\n")

    print("2. Preprocessing categories and missing flags...")
    clean_df = preprocess_raw(raw_df)
    print(f"   ✓ Cleaned {len(clean_df)} rows\n")

    print("3. Building engineered features...")
    feature_df = build_feature_frame(clean_df)
    print(f"   ✓ Created {len(feature_df.columns)} total columns\n")

    print("4. Running batch scoring...")
    predictions_path = OUTPUTS_DIR / "smoke_test_predictions.csv"
    scored_df = score_batch(str(RAW_DATA_PATH), str(predictions_path))
    print(f"   ✓ Scored {len(scored_df)} rows\n")

    print("5. Generating monitoring report...")
    report = build_monitoring_report(str(predictions_path))
    print(f"   ✓ Monitoring report:\n{report}\n")

    print("=== Smoke Test PASSED ===")
    return True


if __name__ == "__main__":
    run_smoke_test()