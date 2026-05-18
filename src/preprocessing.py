import numpy as np
import pandas as pd

from src.validation import aqi_from_pm25

MISSING_COLS = ["benzene", "visibility_km", "wind_dir_deg", "pressure_hpa"]
SEASON_MAP = {
    "winter": "Winter",
    "spring": "Spring",
    "summer": "Summer",
    "fall": "Fall",
    "autumn": "Fall",
}


def drop_sensor_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    subset = [column for column in df.columns if column != "reading_id"]
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned["season"] = (
        cleaned["season"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(SEASON_MAP)
        .fillna("Unknown")
    )

    cleaned["station_type"] = cleaned["station_type"].astype(str).str.strip().str.title()

    return cleaned


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    flagged = df.copy()
    for column in MISSING_COLS:
        flagged[f"{column}_missing"] = flagged[column].isna().astype(int)
    return flagged


def add_sensor_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    flagged = df.copy()
    flagged["pm10_adjusted_flag"] = (flagged["pm10"] < flagged["pm25"]).astype(int)
    flagged["aqi_reference_pm25"] = aqi_from_pm25(flagged["pm25"])
    flagged["aqi_gap"] = (flagged["aqi"] - flagged["aqi_reference_pm25"]).round(2)
    flagged["aqi_gap_flag"] = flagged["aqi_gap"].abs().gt(30).astype(int)
    flagged["wind_speed_zero_flag"] = flagged["wind_speed_ms"].eq(0).astype(int)
    return flagged


def repair_physical_values(df: pd.DataFrame) -> pd.DataFrame:
    repaired = df.copy()
    repaired["pm10"] = np.where(
        repaired["pm10"] < repaired["pm25"],
        (repaired["pm25"] * 1.5).round(2),
        repaired["pm10"],
    )
    return repaired


def _fill_group_median(df: pd.DataFrame, column: str, group_column: str) -> pd.Series:
    grouped_median = df.groupby(group_column)[column].transform("median")
    return df[column].fillna(grouped_median).fillna(df[column].median())


def impute_domain_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    imputed = df.copy()
    imputed["benzene"] = _fill_group_median(imputed, "benzene", "station_type")
    imputed["visibility_km"] = imputed["visibility_km"].fillna(1.0)
    imputed["wind_dir_deg"] = imputed["wind_dir_deg"].fillna(180.0)
    imputed["pressure_hpa"] = _fill_group_median(imputed, "pressure_hpa", "month")
    return imputed


def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = drop_sensor_duplicates(df)
    cleaned = clean_categories(cleaned)
    cleaned = add_missing_flags(cleaned)
    cleaned = add_sensor_quality_flags(cleaned)
    cleaned = repair_physical_values(cleaned)
    cleaned = impute_domain_missing_values(cleaned)
    return cleaned
