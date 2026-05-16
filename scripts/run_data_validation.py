from pathlib import Path
import pandas as pd

DATA_PATH = Path("../data/raw/air_quality_readings.csv")

df = pd.read_csv(DATA_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

print("Rows:", len(df))
print("Columns:", df.shape[1])
print("Missing values:")
print(df.isna().sum().sort_values(ascending=False).head(15))