import asyncio
import json
from typing import Any

from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.config import LIVE_PREDICTIONS_PATH, MODELS_DIR, OUTPUTS_DIR, STATIC_DIR
from src.live_data import (
    LiveReadingValidationError,
    active_predictions_path,
    persist_live_predictions,
    persist_live_readings,
    prepare_live_readings,
)
from src.monitoring import build_dashboard_snapshot, load_metrics_summary
from src.score_batch import score_frame

app = FastAPI(
    title="Air Quality Command Center",
    version="2.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

PREDICTIONS_PATH = OUTPUTS_DIR / "batch_predictions.csv"
METRICS_PATH = MODELS_DIR / "metrics_summary.json"
INDEX_PATH = STATIC_DIR / "index.html"
PREDICTIONS_HISTORY_PATH = OUTPUTS_DIR / "predictions_history.csv"

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _dashboard_snapshot(alert_limit: int = 8, timeline_points: int = 24) -> dict:
    return build_dashboard_snapshot(
        predictions_path=active_predictions_path(PREDICTIONS_PATH),
        metrics_path=METRICS_PATH,
        alert_limit=alert_limit,
        timeline_points=timeline_points,
    )
    
    
def _append_to_history(predictions_df: pd.DataFrame):
    """Append predictions to history CSV."""
    if PREDICTIONS_HISTORY_PATH.exists():
        existing = pd.read_csv(PREDICTIONS_HISTORY_PATH)
        combined = pd.concat([existing, predictions_df], ignore_index=True)
        combined.to_csv(PREDICTIONS_HISTORY_PATH, index=False)
    else:
        predictions_df.to_csv(PREDICTIONS_HISTORY_PATH, index=False)

def _get_prediction_history(limit: int = 20) -> list[dict]:
    """Retrieve latest predictions from history."""
    if not PREDICTIONS_HISTORY_PATH.exists():
        return []
    df = pd.read_csv(PREDICTIONS_HISTORY_PATH).tail(limit)
    return df.to_dict(orient="records")


@app.get("/", include_in_schema=False)
def index():
    if not INDEX_PATH.exists():
        return JSONResponse({"status": "frontend_missing"}, status_code=503)
    return FileResponse(INDEX_PATH)


@app.get("/api/health")
def api_health():
    snapshot = _dashboard_snapshot(alert_limit=1, timeline_points=1)
    return {
        "status": snapshot["status"],
        "latest_predictions_count": snapshot["overview"]["readings_scored"],
        "latest_scored_at": snapshot["overview"]["latest_scored_at"],
    }


@app.get("/health")
def health():
    return api_health()


@app.get("/api/dashboard")
def api_dashboard(
    alert_limit: int = Query(default=8, ge=1, le=20),
    timeline_points: int = Query(default=24, ge=6, le=72),
):
    return _dashboard_snapshot(alert_limit=alert_limit, timeline_points=timeline_points)


@app.get("/api/alerts")
def api_alerts(limit: int = Query(default=10, ge=1, le=25)):
    snapshot = _dashboard_snapshot(alert_limit=limit, timeline_points=12)
    return {
        "status": snapshot["status"],
        "stations_with_alerts": len(snapshot["stations"]),
        "readings": snapshot["alerts"],
    }


@app.get("/current_alerts")
def current_alerts(limit: int = Query(default=10, ge=1, le=25)):
    return api_alerts(limit=limit)


@app.get("/api/metrics")
def api_metrics():
    return load_metrics_summary(METRICS_PATH)


@app.get("/metrics")
def metrics():
    return api_metrics()


@app.post("/api/readings")
def ingest_readings(payload: dict[str, Any] | list[dict[str, Any]]):
    try:
        readings_df = prepare_live_readings(payload)
    except LiveReadingValidationError as exc:
        return JSONResponse({"status": "invalid_reading", "detail": str(exc)}, status_code=422)

    persist_live_readings(readings_df)
    scored_df = score_frame(readings_df)
    persist_live_predictions(scored_df)

    snapshot = build_dashboard_snapshot(
        predictions_path=LIVE_PREDICTIONS_PATH,
        metrics_path=METRICS_PATH,
        alert_limit=min(len(scored_df), 8),
        timeline_points=24,
    )
    latest_rows = scored_df[["reading_id", "station_id", "total_alerts", "max_risk_prob", "dominant_risk"]].to_dict(orient="records")

    return {
        "status": "ingested",
        "ingested_count": int(len(scored_df)),
        "active_predictions_path": str(LIVE_PREDICTIONS_PATH),
        "latest_rows": latest_rows,
        "dashboard_overview": snapshot["overview"],
    }


@app.get("/api/stream/dashboard")
async def stream_dashboard(
    request: Request,
    alert_limit: int = Query(default=8, ge=1, le=20),
    timeline_points: int = Query(default=24, ge=6, le=72),
):
    async def event_generator():
        last_signature = None

        while True:
            if await request.is_disconnected():
                break

            snapshot = _dashboard_snapshot(alert_limit=alert_limit, timeline_points=timeline_points)
            signature = snapshot.get("data_signature")

            if signature != last_signature:
                payload = json.dumps(snapshot)
                yield f"event: dashboard\ndata: {payload}\n\n"
                last_signature = signature
            else:
                yield "event: heartbeat\ndata: {}\n\n"

            await asyncio.sleep(5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/predict")
def predict_single(payload: dict[str, Any]):
    """Predict for a single row of sensor data."""
    try:
        df = pd.DataFrame([payload])
        df["reading_id"] = "USR_" + str(int(datetime.now(timezone.utc).timestamp()))
        df["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Reindex to required columns (add missing columns with defaults)
        required_cols = [
            "reading_id", "timestamp", "station_id", "station_type", "elevation_m",
            "near_highway", "near_industry", "temp_c", "humidity_pct", "wind_speed_ms",
            "pressure_hpa", "precipitation_mm", "visibility_km", "temp_inversion",
            "pm25", "pm10", "no2", "o3", "so2", "co", "benzene", "aqi",
            "year", "month", "hour", "day_of_week", "is_weekend", "is_rush_hour", "season"
        ]
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        # Add derived temporal columns
        ts = pd.to_datetime(df["timestamp"].iloc[0])
        df["year"] = ts.year
        df["month"] = ts.month
        df["hour"] = ts.hour
        df["day_of_week"] = ts.dayofweek
        df["is_weekend"] = int(ts.dayofweek >= 5)
        df["is_rush_hour"] = int(ts.hour in [7, 8, 9, 17, 18, 19])
        df["wind_dir_deg"] = payload.get("wind_dir_deg", 180)
        df["benzene"] = payload.get("benzene", 0)
        
        scored_df = score_frame(df)
        _append_to_history(scored_df)
        
        result = scored_df.iloc[0].to_dict()
        return {
            "status": "success",
            "prediction": result,
            "message": f"Predicted with alerts: {result.get('total_alerts', 0)}"
        }
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "detail": str(exc)},
            status_code=400
        )

@app.post("/api/predict-batch")
def predict_batch(payload: dict[str, Any]):
    """Predict for a batch of rows (max 100)."""
    try:
        rows = payload.get("rows", [])
        
        if not rows:
            return JSONResponse(
                {"status": "error", "detail": "No rows provided"},
                status_code=400
            )
        
        if len(rows) > 100:
            return JSONResponse(
                {"status": "error", "detail": "Batch limit is 100 rows"},
                status_code=400
            )
        
        df = pd.DataFrame(rows)
        df["reading_id"] = [f"BATCH_{i:04d}" for i in range(len(df))]
        df["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Add required columns
        required_cols = [
            "reading_id", "timestamp", "station_id", "station_type", "elevation_m",
            "near_highway", "near_industry", "temp_c", "humidity_pct", "wind_speed_ms",
            "pressure_hpa", "precipitation_mm", "visibility_km", "temp_inversion",
            "pm25", "pm10", "no2", "o3", "so2", "co", "benzene", "aqi",
            "year", "month", "hour", "day_of_week", "is_weekend", "is_rush_hour", "season"
        ]
        
        for col in required_cols:
            if col not in df.columns:
                df[col] = None
        
        # Add temporal columns
        ts = pd.to_datetime(df["timestamp"].iloc[0])
        df["year"] = ts.year
        df["month"] = ts.month
        df["hour"] = ts.hour
        df["day_of_week"] = ts.dayofweek
        df["is_weekend"] = int(ts.dayofweek >= 5)
        df["is_rush_hour"] = int(ts.hour in [7, 8, 9, 17, 18, 19])
        df["wind_dir_deg"] = df.get("wind_dir_deg", 180)
        df["benzene"] = df.get("benzene", 0)
        
        scored_df = score_frame(df)
        _append_to_history(scored_df)
        
        results = scored_df[["reading_id", "station_id", "total_alerts", "max_risk_prob", "dominant_risk", "scored_at"]].to_dict(orient="records")
        
        return {
            "status": "success",
            "rows_processed": len(scored_df),
            "predictions": results,
            "message": f"Processed {len(scored_df)} rows"
        }
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "detail": str(exc)},
            status_code=400
        )

@app.get("/api/predictions-history")
def get_predictions_history(limit: int = Query(default=20, ge=1, le=100)):
    """Retrieve latest predictions from history."""
    history = _get_prediction_history(limit=limit)
    return {
        "status": "success",
        "total_records": len(history),
        "records": history
    }