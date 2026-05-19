import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

np.random.seed(42)

def generate_sample_data(n_rows: int = 100, output_path: str = "data/sample_synthetic_data.csv") -> pd.DataFrame:
    """Generate synthetic air quality data with realistic and edge-case scenarios."""
    
    stations = ["STN_URBAN_01", "STN_URBAN_02", "STN_SUBURB_01", "STN_SUBURB_02", "STN_INDUSTRIAL_01", "STN_RURAL_01"]
    station_types = ["Urban", "Suburban", "Industrial", "Rural"]
    seasons = ["Winter", "Spring", "Summer", "Fall"]
    
    data = {
        "reading_id": [f"SYN{i:06d}" for i in range(n_rows)],
        "timestamp": [datetime.now() - timedelta(hours=i) for i in range(n_rows)],
        "station_id": np.random.choice(stations, n_rows),
        "station_type": np.random.choice(station_types, n_rows),
        "elevation_m": np.random.uniform(10, 320, n_rows).round(1),
        "near_highway": np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
        "near_industry": np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
        "temp_c": np.random.uniform(-5, 35, n_rows).round(1),
        "humidity_pct": np.random.uniform(30, 95, n_rows).round(1),
        "wind_speed_ms": np.random.exponential(scale=3, size=n_rows).round(1),
        "wind_dir_deg": np.random.uniform(0, 360, n_rows).round(1),
        "pressure_hpa": np.random.uniform(990, 1030, n_rows).round(1),
        "precipitation_mm": np.random.exponential(scale=1, size=n_rows).round(1),
        "visibility_km": np.random.exponential(scale=5, size=n_rows).clip(0.5, 50).round(1),
        "temp_inversion": np.random.choice([0, 1], n_rows, p=[0.85, 0.15]),
    }
    
    # Realistic pollutants with edge cases
    data["pm25"] = np.random.lognormal(mean=2.5, sigma=0.8, size=n_rows).round(1)
    # 20% of rows have unrealistic low pm10 (edge case)
    data["pm10"] = np.where(
        np.random.random(n_rows) < 0.2,
        data["pm25"] * 0.8,  # Edge case: pm10 < pm25
        (data["pm25"] * 1.5 + np.random.normal(0, 10, n_rows)).clip(0).round(1)
    )
    data["no2"] = np.random.lognormal(mean=3.8, sigma=0.7, size=n_rows).round(1)
    data["o3"] = np.random.lognormal(mean=3.5, sigma=0.9, size=n_rows).round(1)
    data["so2"] = np.random.lognormal(mean=2.0, sigma=1.0, size=n_rows).round(1)
    data["co"] = np.random.exponential(scale=1.2, size=n_rows).round(2)
    data["benzene"] = np.random.exponential(scale=0.5, size=n_rows).clip(0, 10).round(2)
    data["aqi"] = (data["pm25"] * 3 + np.random.normal(0, 15, n_rows)).clip(0, 500).round(0).astype(int)
    
    df = pd.DataFrame(data)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"✓ Generated {n_rows} synthetic rows → {output_path}")
    return df

if __name__ == "__main__":
    generate_sample_data(n_rows=100)