from pathlib import Path
import json
import pandas as pd

DATA_PATH = Path("../data/processed/feature_engineering_sample.csv")
MODELS_DIR = Path("../models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(df.head())

print("Training script placeholder: wire this to notebook 04 logic or src/train.py next.")