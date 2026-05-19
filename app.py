import asyncio
import json
from typing import Any

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

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _dashboard_snapshot(alert_limit: int = 8, timeline_points: int = 24) -> dict:
    return build_dashboard_snapshot(
        predictions_path=active_predictions_path(PREDICTIONS_PATH),
        metrics_path=METRICS_PATH,
        alert_limit=alert_limit,
        timeline_points=timeline_points,
    )


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
