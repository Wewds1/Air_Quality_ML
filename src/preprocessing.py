import pandas as pd

MISSING_COLS = ["benzene", "visibility_km", "wind_dir_deg", "pressure_hpa"]


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    cleaned["season"] = (
        cleaned["season"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"autumn": "fall"})
        .str.title()
    )

    cleaned["station_type"] = cleaned["station_type"].astype(str).str.strip().str.title()

    return cleaned


def add_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    flagged = df.copy()
    for column in MISSING_COLS:
        flagged[f"{column}_missing"] = flagged[column].isna().astype(int)
    return flagged


def preprocess_raw(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_categories(df)
    cleaned = add_missing_flags(cleaned)
    return cleaned