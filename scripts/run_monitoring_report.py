from src.config import OUTPUTS_DIR
from src.monitoring import build_monitoring_report

report = build_monitoring_report(OUTPUTS_DIR / "batch_predictions.csv")
print(report)