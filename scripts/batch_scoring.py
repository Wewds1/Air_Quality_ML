from pathlib import Path
import pandas as pd

INPUT_PATH = Path("../data/processed/feature_engineering_sample.csv")
OUTPUT_PATH = Path("../outputs/batch_predictions.csv")

df = pd.read_csv(INPUT_PATH)
print("Loaded rows:", len(df))

df.to_csv(OUTPUT_PATH, index=False)
print("Saved:", OUTPUT_PATH)