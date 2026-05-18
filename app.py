import asyncio
import json

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.config import MODELS_DIR, OUTPUTS_DIR, STATIC_DIR
from src.monitoring import build_dashboard_snapshot, load_metrics_summary

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
        predictions_path=PREDICTIONS_PATH,
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
