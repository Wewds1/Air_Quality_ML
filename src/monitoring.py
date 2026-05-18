import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config import LABEL_COLS, MODELS_DIR, MONITORING_SNAPSHOT_PATH

ALERT_LABEL_MAP = {
    "label_respiratory_risk": "Respiratory Risk",
    "label_cardiovascular_risk": "Cardiovascular Risk",
    "label_vulnerable_alert": "Vulnerable Alert",
    "label_outdoor_warning": "Outdoor Warning",
    "label_industrial_event": "Industrial Event",
}


def load_metrics_summary(metrics_path: str | Path | None = None) -> dict:
    metrics_path = Path(metrics_path) if metrics_path else MODELS_DIR / "metrics_summary.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def load_predictions(predictions_path: str | Path) -> pd.DataFrame:
    predictions_path = Path(predictions_path)
    df = pd.read_csv(predictions_path)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "scored_at" in df.columns:
        df["scored_at"] = pd.to_datetime(df["scored_at"], errors="coerce")

    probability_cols = [f"prob_{label}" for label in LABEL_COLS if f"prob_{label}" in df.columns]
    alert_cols = [f"alert_{label.replace('label_', '')}" for label in LABEL_COLS if f"alert_{label.replace('label_', '')}" in df.columns]

    if probability_cols and "max_risk_prob" not in df.columns:
        df["max_risk_prob"] = df[probability_cols].max(axis=1).round(4)

    if probability_cols and "dominant_risk" not in df.columns:
        df["dominant_risk"] = (
            df[probability_cols]
            .idxmax(axis=1)
            .str.replace("prob_label_", "", regex=False)
        )

    if alert_cols and "alert_signature" not in df.columns:
        df["alert_signature"] = df[alert_cols].astype(str).agg("".join, axis=1)

    return df


def _latest_timestamp(df: pd.DataFrame, column: str) -> str | None:
    if column not in df.columns or df[column].dropna().empty:
        return None
    return df[column].dropna().max().isoformat()


def _safe_round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def build_overview(df: pd.DataFrame) -> dict:
    active_df = df[df["total_alerts"] > 0] if "total_alerts" in df.columns else df.iloc[0:0]
    latest_timestamp = _latest_timestamp(df, "timestamp")
    latest_scored_at = _latest_timestamp(df, "scored_at")

    return {
        "readings_scored": int(len(df)),
        "stations_seen": int(df["station_id"].nunique()) if "station_id" in df.columns else 0,
        "active_alert_rows": int(len(active_df)),
        "latest_timestamp": latest_timestamp,
        "latest_scored_at": latest_scored_at,
        "avg_aqi": _safe_round(df["aqi"].mean(), 2) if "aqi" in df.columns else 0.0,
        "avg_total_alerts": _safe_round(df["total_alerts"].mean(), 2) if "total_alerts" in df.columns else 0.0,
        "max_risk_prob": _safe_round(df["max_risk_prob"].max(), 4) if "max_risk_prob" in df.columns else 0.0,
    }


def build_recent_alerts(df: pd.DataFrame, limit: int = 8) -> list[dict]:
    if "total_alerts" not in df.columns:
        return []

    alert_frame = (
        df[df["total_alerts"] > 0]
        .sort_values(["max_risk_prob", "aqi", "total_alerts"], ascending=[False, False, False])
        .head(limit)
        .copy()
    )

    alerts = []
    for _, row in alert_frame.iterrows():
        alerts.append(
            {
                "reading_id": row.get("reading_id"),
                "station_id": row.get("station_id"),
                "station_type": row.get("station_type"),
                "aqi": _safe_round(row.get("aqi", 0), 1),
                "total_alerts": int(row.get("total_alerts", 0)),
                "dominant_risk": str(row.get("dominant_risk", "")).replace("_", " ").title(),
                "max_risk_prob": _safe_round(row.get("max_risk_prob", 0), 4),
                "timestamp": row.get("timestamp").isoformat() if pd.notna(row.get("timestamp")) else None,
                "scored_at": row.get("scored_at").isoformat() if pd.notna(row.get("scored_at")) else None,
            }
        )

    return alerts


def build_station_summary(df: pd.DataFrame) -> list[dict]:
    if "timestamp" not in df.columns or df["timestamp"].dropna().empty:
        return []

    latest_timestamp = df["timestamp"].dropna().max()
    latest_df = df[df["timestamp"] == latest_timestamp].copy()
    stations = []

    for station_id, station_df in latest_df.groupby("station_id"):
        dominant_risk = station_df["dominant_risk"].mode().iloc[0] if "dominant_risk" in station_df.columns else "unknown"
        quality_flags = int(
            station_df[
                [
                    column
                    for column in [
                        "pm10_adjusted_flag",
                        "aqi_gap_flag",
                        "wind_speed_zero_flag",
                        "benzene_missing",
                        "visibility_km_missing",
                        "wind_dir_deg_missing",
                        "pressure_hpa_missing",
                    ]
                    if column in station_df.columns
                ]
            ].sum(axis=1).sum()
        )

        stations.append(
            {
                "station_id": station_id,
                "station_type": station_df["station_type"].iloc[0] if "station_type" in station_df.columns else None,
                "aqi": _safe_round(station_df["aqi"].mean(), 1),
                "total_alerts": int(station_df["total_alerts"].max()) if "total_alerts" in station_df.columns else 0,
                "max_risk_prob": _safe_round(station_df["max_risk_prob"].max(), 4) if "max_risk_prob" in station_df.columns else 0.0,
                "dominant_risk": str(dominant_risk).replace("_", " ").title(),
                "quality_flags": quality_flags,
            }
        )

    return sorted(stations, key=lambda station: (station["max_risk_prob"], station["aqi"]), reverse=True)


def build_timeline(df: pd.DataFrame, points: int = 24) -> list[dict]:
    if "timestamp" not in df.columns or df["timestamp"].dropna().empty:
        return []

    recent_timestamps = sorted(df["timestamp"].dropna().unique())[-points:]
    timeline_df = df[df["timestamp"].isin(recent_timestamps)].copy()

    grouped = (
        timeline_df.groupby("timestamp")
        .agg(
            avg_aqi=("aqi", "mean"),
            avg_total_alerts=("total_alerts", "mean"),
            max_risk_prob=("max_risk_prob", "max"),
            active_rows=("total_alerts", lambda values: int((values > 0).sum())),
        )
        .reset_index()
        .sort_values("timestamp")
    )

    return [
        {
            "timestamp": row["timestamp"].isoformat(),
            "avg_aqi": _safe_round(row["avg_aqi"], 2),
            "avg_total_alerts": _safe_round(row["avg_total_alerts"], 2),
            "max_risk_prob": _safe_round(row["max_risk_prob"], 4),
            "active_rows": int(row["active_rows"]),
        }
        for _, row in grouped.iterrows()
    ]


def build_label_summary(df: pd.DataFrame, metrics_summary: dict) -> list[dict]:
    training_rates = metrics_summary.get("training_label_rates", {})
    label_summary = []

    for label in LABEL_COLS:
        alert_column = f"alert_{label.replace('label_', '')}"
        if alert_column not in df.columns:
            continue

        current_rate = df[alert_column].mean()
        training_rate = training_rates.get(label, 0.0)
        label_summary.append(
            {
                "label": label,
                "display_name": ALERT_LABEL_MAP[label],
                "current_rate": _safe_round(current_rate, 4),
                "training_rate": _safe_round(training_rate, 4),
                "rate_gap": _safe_round(current_rate - training_rate, 4),
            }
        )

    return label_summary


def build_monitoring_summary(df: pd.DataFrame, metrics_summary: dict) -> dict:
    quality_columns = [
        column
        for column in [
            "pm10_adjusted_flag",
            "aqi_gap_flag",
            "wind_speed_zero_flag",
            "benzene_missing",
            "visibility_km_missing",
            "wind_dir_deg_missing",
            "pressure_hpa_missing",
        ]
        if column in df.columns
    ]

    quality_rate = 0.0
    if quality_columns:
        quality_rate = df[quality_columns].sum(axis=1).gt(0).mean()

    cooccurrence_gap = 0.0
    training_cooccurrence = metrics_summary.get("training_label_cooccurrence", {})
    alert_columns = [f"alert_{label.replace('label_', '')}" for label in LABEL_COLS]
    if training_cooccurrence and not df.empty and all(column in df.columns for column in alert_columns):
        current_cooccurrence = (
            df[alert_columns]
            .rename(columns={f"alert_{label.replace('label_', '')}": label for label in LABEL_COLS})
            .astype(float)
        )
        current_matrix = current_cooccurrence.T.dot(current_cooccurrence) / len(current_cooccurrence)
        deltas = []
        for row_label in LABEL_COLS:
            for column_label in LABEL_COLS:
                baseline = training_cooccurrence[row_label][column_label]
                current = float(current_matrix.loc[row_label, column_label])
                deltas.append(abs(current - baseline))
        cooccurrence_gap = sum(deltas) / len(deltas)

    return {
        "quality_flag_rate": _safe_round(quality_rate, 4),
        "cooccurrence_gap": _safe_round(cooccurrence_gap, 4),
        "model": metrics_summary.get("selected_model"),
        "test_hamming_loss": _safe_round(metrics_summary.get("selected_metrics", {}).get("hamming_loss", 0.0), 4),
        "test_f1_macro": _safe_round(metrics_summary.get("selected_metrics", {}).get("f1_macro", 0.0), 4),
        "test_exact_match": _safe_round(metrics_summary.get("selected_metrics", {}).get("exact_match", 0.0), 4),
    }


def build_dashboard_snapshot(
    predictions_path: str | Path,
    metrics_path: str | Path | None = None,
    alert_limit: int = 8,
    timeline_points: int = 24,
) -> dict:
    predictions_path = Path(predictions_path)
    metrics_summary = load_metrics_summary(metrics_path)

    if not predictions_path.exists():
        return {
            "status": "no_predictions",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overview": build_overview(pd.DataFrame()),
            "alerts": [],
            "stations": [],
            "timeline": [],
            "labels": [],
            "monitoring": build_monitoring_summary(pd.DataFrame(), metrics_summary),
            "model": {
                "selected_model": metrics_summary.get("selected_model"),
                "feature_count": metrics_summary.get("feature_count"),
                "metrics": metrics_summary.get("selected_metrics", {}),
            },
        }

    df = load_predictions(predictions_path)
    snapshot = {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_signature": f"{predictions_path.stat().st_mtime_ns}:{len(df)}",
        "overview": build_overview(df),
        "alerts": build_recent_alerts(df, limit=alert_limit),
        "stations": build_station_summary(df),
        "timeline": build_timeline(df, points=timeline_points),
        "labels": build_label_summary(df, metrics_summary),
        "monitoring": build_monitoring_summary(df, metrics_summary),
        "model": {
            "selected_model": metrics_summary.get("selected_model"),
            "feature_count": metrics_summary.get("feature_count"),
            "metrics": metrics_summary.get("selected_metrics", {}),
            "thresholds": metrics_summary.get("candidate_metrics", {})
            .get(metrics_summary.get("selected_model", ""), {})
            .get("thresholds", {}),
        },
    }
    return snapshot


def build_monitoring_report(
    predictions_path: str,
    output_path: str | None = None,
    metrics_path: str | None = None,
) -> pd.DataFrame:
    predictions_path = Path(predictions_path)
    output_path = Path(output_path) if output_path else predictions_path.parent / "monitoring_report.csv"
    snapshot = build_dashboard_snapshot(predictions_path, metrics_path=metrics_path)

    rows = []
    for metric, value in snapshot["overview"].items():
        rows.append({"section": "overview", "metric": metric, "value": value})

    for label in snapshot["labels"]:
        rows.extend(
            [
                {"section": "label", "metric": f"{label['label']}_current_rate", "value": label["current_rate"]},
                {"section": "label", "metric": f"{label['label']}_training_rate", "value": label["training_rate"]},
                {"section": "label", "metric": f"{label['label']}_rate_gap", "value": label["rate_gap"]},
            ]
        )

    for metric, value in snapshot["monitoring"].items():
        rows.append({"section": "monitoring", "metric": metric, "value": value})

    for station in snapshot["stations"][:6]:
        rows.append(
            {
                "section": "station",
                "metric": f"{station['station_id']}_max_risk_prob",
                "value": station["max_risk_prob"],
            }
        )

    report = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    MONITORING_SNAPSHOT_PATH.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return report
