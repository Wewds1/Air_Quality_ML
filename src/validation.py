import numpy as np
import pandas as pd


AQI_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400),
    (350.5, 500.4, 401, 500),
]


def aqi_from_pm25(pm25: pd.Series | np.ndarray) -> np.ndarray:
    values = np.asarray(pm25, dtype=float)
    clipped = np.nan_to_num(values, nan=0.0)
    aqi = np.zeros_like(clipped, dtype=float)

    for conc_lo, conc_hi, aqi_lo, aqi_hi in AQI_BREAKPOINTS:
        mask = (clipped >= conc_lo) & (clipped <= conc_hi)
        if mask.any():
            aqi[mask] = ((aqi_hi - aqi_lo) / (conc_hi - conc_lo)) * (clipped[mask] - conc_lo) + aqi_lo

    aqi[clipped > AQI_BREAKPOINTS[-1][1]] = 500
    return np.rint(aqi).astype(int)


def validation_summary(df: pd.DataFrame) -> dict:
    duplicate_subset = [column for column in df.columns if column != "reading_id"]
    aqi_reference = aqi_from_pm25(df["pm25"])

    return {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicates_excluding_reading_id": int(df.duplicated(subset=duplicate_subset).sum()),
        "missing_values": df.isna().sum().to_dict(),
        "missing_rates": df.isna().mean().round(4).to_dict(),
        "physical_checks": {
            "pm10_less_than_pm25": int((df["pm10"] < df["pm25"]).sum()),
            "aqi_discrepancy_over_30": int((df["aqi"].sub(aqi_reference).abs() > 30).sum()),
            "zero_wind_speed": int((df["wind_speed_ms"] == 0).sum()),
            "extreme_o3_no2": int(((df["o3"] > 200) & (df["no2"] > 200)).sum()),
        },
    }
