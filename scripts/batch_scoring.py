from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "feature_engineering_sample.csv"


OUTPUT_PATH =  BASE_DIR / "outputs" / "batch_predictions.csv"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_PATH)
print("Loaded rows:", len(df))

df.to_csv(OUTPUT_PATH, index=False)
print("Saved:", OUTPUT_PATH)