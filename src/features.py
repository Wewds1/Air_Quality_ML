import numpy as np
import pandas as pd

from src.config import WHO_GUIDELINES


def add_circular_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    featured["hour_sin"] = np.sin(2 * np.pi * featured["hour"] / 24).round(4)
    featured["hour_cos"] = np.cos(2 * np.pi * featured["hour"] / 24).round(4)
    featured["month_sin"] = np.sin(2 * np.pi * featured["month"] / 12).round(4)
    featured["month_cos"] = np.cos(2 * np.pi * featured["month"] / 12).round(4)

    wind_rad = np.deg2rad(featured["wind_dir_deg"].fillna(180))
    featured["wind_sin"] = np.sin(wind_rad).round(4)
    featured["wind_cos"] = np.cos(wind_rad).round(4)

    return featured


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    featured["pm_fine_ratio"] = (featured["pm25"] / featured["pm10"].clip(lower=0.1)).round(3)
    featured["oxidant_load"] = (featured["o3"] + featured["no2"]).round(1)
    featured["combustion_index"] = (featured["no2"] * 0.6 + featured["co"] * 10).round(2)
    featured["industrial_signature"] = (
        featured["so2"] * 0.5 + featured["benzene"].fillna(0) * 3
    ).round(2)
    featured["aqi_excess_150"] = (featured["aqi"] - 150).clip(lower=0)
    featured["aqi_excess_100"] = (featured["aqi"] - 100).clip(lower=0)

    for column in ["pm25", "pm10", "no2", "so2", "co", "benzene", "aqi"]:
        featured[f"log_{column}"] = np.log1p(featured[column].fillna(0))

    return featured


def add_weather_station_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    featured["dispersion_index"] = (featured["wind_speed_ms"] / (1 + featured["temp_inversion"] * 3)).round(3)
    featured["precipitation_flag"] = (featured["precipitation_mm"] > 0).astype(int)
    featured["heavy_rain"] = (featured["precipitation_mm"] > 5).astype(int)
    featured["heat_ozone_risk"] = ((featured["temp_c"] > 25) & (featured["o3"] > 100)).astype(int)
    featured["inversion_severity"] = (
        featured["temp_inversion"]
        * (1 / featured["wind_speed_ms"].clip(lower=0.5))
        * (featured["humidity_pct"] / 100)
    ).round(3)
    featured["haze_flag"] = (featured["visibility_km"].fillna(1.0) < 5).astype(int)
    featured["urban_heat"] = ((featured["station_type"] == "Urban") & (featured["temp_c"] > 28)).astype(int)
    featured["industrial_downwind"] = (
        (featured["near_industry"] == 1)
        & (featured["wind_dir_deg"].fillna(180).between(180, 315))
    ).astype(int)
    featured["urban_rush"] = ((featured["station_type"] == "Urban") & (featured["is_rush_hour"] == 1)).astype(int)

    return featured


def add_risk_context_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    featured["pm25_cardio_zone"] = (featured["pm25"] > 55.4).astype(int)
    featured["multi_label_event"] = (featured["aqi"] > 150).astype(int)
    featured["pollutants_above_who"] = (
        (featured["pm25"] > WHO_GUIDELINES["pm25"]).astype(int)
        + (featured["pm10"] > WHO_GUIDELINES["pm10"]).astype(int)
        + (featured["no2"] > WHO_GUIDELINES["no2"]).astype(int)
        + (featured["o3"] > WHO_GUIDELINES["o3"]).astype(int)
        + (featured["so2"] > WHO_GUIDELINES["so2"]).astype(int)
    ).clip(0, 5)

    return featured


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    rolling_df = df.sort_values(["station_id", "timestamp"]).copy()

    for column in ["pm25", "no2", "aqi"]:
        rolling_df[f"{column}_roll3_mean"] = (
            rolling_df.groupby("station_id")[column]
            .transform(lambda values: values.rolling(3, min_periods=1).mean())
            .round(2)
        )
        rolling_df[f"{column}_lag1"] = rolling_df.groupby("station_id")[column].shift(1)
        rolling_df[f"{column}_trend"] = (rolling_df[column] - rolling_df[f"{column}_roll3_mean"]).round(2)

    rolling_df["pm25_rising"] = (rolling_df["pm25_trend"] > 5).astype(int)
    return rolling_df


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    featured = add_circular_features(df)
    featured = add_composite_features(featured)
    featured = add_weather_station_features(featured)
    featured = add_risk_context_features(featured)
    featured = add_rolling_features(featured)
    drop_columns = [column for column in ["wind_dir_deg", "hour", "month"] if column in featured.columns]
    return featured.drop(columns=drop_columns)
