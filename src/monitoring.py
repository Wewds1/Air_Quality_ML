from pathlib import Path
import pandas as pd

from src.config import LABEL_COLS


def build_monitoring_report(predictions_path: str, output_path: str | None = None) -> pd.DataFrame:
    predictions_path = Path(predictions_path)
    output_path = Path(output_path) if output_path else predictions_path.parent / "monitoring_report.csv"

    df = pd.read_csv(predictions_path)

    alert_cols = [f"alert_{label.replace('label_', '')}" for label in LABEL_COLS]

    report = pd.DataFrame({
        "metric": [
            *[f"rate_{col}" for col in alert_cols],
            "avg_total_alerts",
        ],
        "value": [
            *[df[col].mean() for col in alert_cols],
            df["total_alerts"].mean(),
        ],
    })

    report.to_csv(output_path, index=False)
    return report