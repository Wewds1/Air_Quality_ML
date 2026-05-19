from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from src.config import (
    LABEL_COLS,
    LIVE_BASE_COLUMNS,
    LIVE_PREDICTIONS_PATH,
    LIVE_READINGS_PATH,
    LIVE_REQUIRED_COLUMNS,
)


class LiveReadingValidationError(ValueError):
    pass


def _season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def _ensure_timestamp(series: pd.Series) -> pd.Series:
    if series.isna().all():
        return pd.Series(
            [datetime.now(timezone.utc).replace(microsecond=0) for _ in range(len(series))],
            index=series.index,
        )

    filled = series.copy()
    fallback = datetime.now(timezone.utc).replace(microsecond=0)
    filled = filled.fillna(fallback)
    return filled


def _generate_reading_ids(count: int) -> list[str]:
    return [f"LIVE-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:8]}" for _ in range(count)]


def prepare_live_readings(payload: list[dict] | dict) -> pd.DataFrame:
    records = payload if isinstance(payload, list) else [payload]
    if not records:
        raise LiveReadingValidationError("At least one reading is required.")

    df = pd.DataFrame(records)
    missing_required = [column for column in LIVE_REQUIRED_COLUMNS if column not in df.columns]
    if missing_required:
        raise LiveReadingValidationError(f"Missing required fields: {', '.join(missing_required)}")

    for column in LIVE_BASE_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[LIVE_BASE_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["timestamp"] = _ensure_timestamp(df["timestamp"])
    df["reading_id"] = df["reading_id"].fillna(pd.Series(_generate_reading_ids(len(df)), index=df.index))

    df["year"] = df["timestamp"].dt.year.astype(int)
    df["month"] = df["timestamp"].dt.month.astype(int)
    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    df["season"] = df["month"].map(_season_from_month)

    numeric_columns = [
        column
        for column in LIVE_BASE_COLUMNS
        if column not in {"reading_id", "timestamp", "station_id", "station_type"}
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for label in LABEL_COLS:
        if label not in df.columns:
            df[label] = pd.NA

    ordered_columns = [
        "reading_id",
        "timestamp",
        "year",
        "month",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_rush_hour",
        "season",
        "station_id",
        "station_type",
        "elevation_m",
        "near_highway",
        "near_industry",
        "temp_c",
        "humidity_pct",
        "wind_speed_ms",
        "wind_dir_deg",
        "pressure_hpa",
        "precipitation_mm",
        "visibility_km",
        "temp_inversion",
        "pm25",
        "pm10",
        "no2",
        "o3",
        "so2",
        "co",
        "benzene",
        "aqi",
        *LABEL_COLS,
    ]
    return df[ordered_columns]


def _merge_persistent_csv(path: Path, rows: pd.DataFrame) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_csv(path)
        merged = pd.concat([existing, rows], ignore_index=True)
        merged = merged.drop_duplicates(subset=["reading_id"], keep="last")
    else:
        merged = rows.copy()

    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return merged


def persist_live_readings(readings_df: pd.DataFrame) -> pd.DataFrame:
    return _merge_persistent_csv(LIVE_READINGS_PATH, readings_df)


def persist_live_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    return _merge_persistent_csv(LIVE_PREDICTIONS_PATH, predictions_df)


def active_predictions_path(default_path: Path) -> Path:
    if LIVE_PREDICTIONS_PATH.exists() and LIVE_PREDICTIONS_PATH.stat().st_size > 0:
        return LIVE_PREDICTIONS_PATH
    return default_path
