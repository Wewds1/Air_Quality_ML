import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.config import OUTPUTS_DIR, MODELS_DIR

app = FastAPI(title="Air Quality Alerts API", version="1.0")

PREDICTIONS_PATH = OUTPUTS_DIR / "batch_predictions.csv"
METRICS_PATH = MODELS_DIR / "metrics_summary.json"


class HealthResponse(BaseModel):
    status: str
    latest_predictions_count: int


class AlertReading(BaseModel):
    reading_id: str
    station_id: str
    aqi: float
    total_alerts: int
    scored_at: str


class CurrentAlertsResponse(BaseModel):
    status: str
    stations_with_alerts: int
    readings: list[AlertReading]


@app.get("/health", response_model=HealthResponse)
def health():
    if not PREDICTIONS_PATH.exists():
        return {"status": "no_predictions_yet", "latest_predictions_count": 0}

    df = pd.read_csv(PREDICTIONS_PATH)
    return {"status": "healthy", "latest_predictions_count": len(df)}


@app.get("/current_alerts", response_model=CurrentAlertsResponse)
def current_alerts(limit: int = 10):
    if not PREDICTIONS_PATH.exists():
        return {"status": "no_predictions", "stations_with_alerts": 0, "readings": []}

    df = pd.read_csv(PREDICTIONS_PATH)
    df_alerts = df[df["total_alerts"] > 0].sort_values("aqi", ascending=False).head(limit)

    readings = [
        AlertReading(
            reading_id=row["reading_id"],
            station_id=row["station_id"],
            aqi=row["aqi"],
            total_alerts=row["total_alerts"],
            scored_at=row["scored_at"],
        )
        for _, row in df_alerts.iterrows()
    ]

    return {
        "status": "ok",
        "stations_with_alerts": len(df_alerts),
        "readings": readings,
    }


@app.get("/metrics")
def get_metrics():
    if not METRICS_PATH.exists():
        return {"status": "no_metrics_yet"}

    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)
    return metrics