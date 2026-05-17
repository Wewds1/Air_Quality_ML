import pandas as pd


def validation_summary(df: pd.DataFrame) -> dict:
    return {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "duplicate_rows": df.duplicated().sum(),
        "missing_values": df.isna().sum().to_dict(),
        "pm10_less_than_pm25": (df["pm10"] < df["pm25"]).sum(),
    }