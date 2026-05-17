from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
latest_path = OUTPUTS_DIR / "batch_predictions.csv"

if latest_path.exists():
    df = pd.read_csv(latest_path)
    print(df.describe(include="all"))
else:
    print("No batch predictions found yet.")