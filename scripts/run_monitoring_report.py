from pathlib import Path
import pandas as pd

OUTPUTS_DIR = Path("../outputs")
latest_path = OUTPUTS_DIR / "batch_predictions.csv"

if latest_path.exists():
    df = pd.read_csv(latest_path)
    print(df.describe(include="all"))
else:
    print("No batch predictions found yet.")