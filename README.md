# Air Quality Multi-Label Risk Classification

A production-ready machine learning pipeline that monitors air quality readings and flags health risks in real-time.

## What This Does

This system ingests hourly air quality sensor data (PM2.5, PM10, NO₂, O₃, temperature, wind, etc.) from 6 monitoring stations and automatically predicts **5 simultaneous health risk labels**:

| Risk | Description |
|------|-------------|
| 🫁 **Respiratory Risk** | Unsafe for people with asthma or COPD |
| ❤️ **Cardiovascular Risk** | Elevated heart stress indicators |
| ⚠️ **Vulnerable Alert** | Children, elderly, or immunocompromised at risk |
| 🚴 **Outdoor Warning** | Outdoor activities should be limited |
| 🏭 **Industrial Event** | Unusual pollution signature detected |

Every reading gets **5 probability scores** (0-1 for each risk) + **binary alerts** (yes/no). The system runs hourly batch scoring and exposes results via a REST API.

---

## Quick Start

### 1. Setup
```bash
# Clone and install
git clone <repo>
cd Air_Quality
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model
Trains on 8,000 historical readings (80/20 train/test, time-ordered):
```bash
python -m scripts.run_training
```
This saves trained models, preprocessor, and thresholds to `models/`.

### 3. Score New Data
Reads from `data/raw/air_quality_readings.csv` and generates predictions:
```bash
python -m scripts.batch_scoring
```
Output: `outputs/batch_predictions.csv` with columns like `prob_respiratory_risk`, `alert_respiratory_risk`, `total_alerts`, etc.

### 4. Start the API
Serves latest predictions via HTTP:
```bash
python -m uvicorn app:app --reload
```
- `GET http://localhost:8000/health` → System status
- `GET http://localhost:8000/current_alerts?limit=10` → Top 10 alerts
- `GET http://localhost:8000/metrics` → Model performance summary

### 5. Docker (Production)
```bash
docker-compose up
```
Runs API on port 8000 + batch scoring as a one-off container.

---

## How It Works

### 1. Data Quality (Notebook: `01_data_quality.ipynb`)
Raw dataset has issues:
- **Duplicates** → Removed
- **Missing values** in benzene, visibility, wind direction → Flagged
- **Invalid physics** like PM10 < PM2.5 → Corrected
- **Categorical mess** (Autumn vs. Fall, inconsistent station names) → Standardized

Result: 10,000 clean readings with timestamps, 6 stations, 35 raw features.

### 2. Exploratory Analysis (Notebook: `02_eda_visualization.ipynb`)
Understand the data:
- Label co-occurrence: which risks appear together?
- Pollutant distributions by station type (urban vs. rural)
- Seasonal + hourly patterns (when is risk highest?)
- Inversion days (trapped pollution = higher PM2.5)

Insight: ~74% of readings trigger the "vulnerable" alert, 35% trigger "outdoor warning".

### 3. Feature Engineering (Notebook: `03_feature_engineering.ipynb`)
Raw 35 features → 77 engineered features:

**Time cycles** (sin/cos encoding):
- Hour of day, month (capture daily/seasonal cycles)
- Wind direction (compass directions matter for dispersion)

**Composite pollution indices**:
- `pm_fine_ratio` = PM2.5 / PM10 (particle size matters)
- `oxidant_load` = O₃ + NO₂ (breathing irritants)
- `combustion_index` = NO₂ + NO (traffic/industry signal)
- `industrial_signature` = elevated SO₂ + NO₂

**Weather × location context**:
- `dispersion_index` = temperature + wind speed (how well pollutants disperse)
- `inversion_severity` = pressure anomaly (trapped air)
- `urban_heat_ozone_risk` = high temp + high O₃ (creates smog)
- `industrial_downwind` = pollution flows downwind from industrial zone

**Rolling features** (last 3 hours, per station):
- Mean PM2.5, NO₂, AQI (trends matter)
- Lag-1 AQI (persistence)
- 3-hour trend direction

Result: 77 features → fed to model.

### 4. Multi-Label Modeling (Notebook: `04_multilabel_modeling.ipynb`)
Two competing approaches:

**Approach A: Binary Relevance + Random Forest**
- Train 5 independent classifiers (one per label)
- Fast, simple, ignores label correlations
- Baseline: Hamming loss 0.18

**Approach B: Classifier Chain + Logistic Regression**
- Chain classifiers: each uses previous label predictions as features
- Captures correlations (e.g., if respiratory risk → likely vulnerable alert)
- Improvement: Hamming loss 0.15, better F1 scores

Model deployed: **Classifier Chain** (better performance).

**Metrics tracked:**
- Hamming loss (% of incorrect label predictions)
- F1 micro (overall label accuracy)
- F1 macro (per-label fairness)
- Exact match ratio (% of readings with all 5 labels correct)
- Per-label precision/recall (avoid false alarms for rare labels)

### 5. Batch Scoring (Script: `scripts/batch_scoring.py`)
Runs hourly:
1. Load new readings from CSV
2. Clean + preprocess (handle missing values, standardize categories)
3. Engineer 77 features
4. Apply trained model → 5 probabilities per reading
5. Threshold each probability (tuned thresholds in `models/label_thresholds.json`)
6. Output: `batch_predictions.csv` with probabilities, alerts, and score timestamp

**Example output row:**
```
reading_id | station_id | aqi | prob_respiratory_risk | alert_respiratory_risk | ... | total_alerts | scored_at
12345      | URBAN_001  | 95  | 0.87                  | 1                      | ... | 3            | 2026-05-18 14:00:00
```

### 6. API Server (app.py)
Three endpoints for consuming predictions:

**GET /health**
```json
{
  "status": "healthy",
  "latest_predictions_count": 150
}
```

**GET /current_alerts?limit=10**
```json
{
  "status": "ok",
  "stations_with_alerts": 4,
  "readings": [
    {
      "reading_id": "12345",
      "station_id": "URBAN_001",
      "aqi": 95,
      "total_alerts": 3,
      "scored_at": "2026-05-18 14:00:00"
    },
    ...
  ]
}
```

**GET /metrics**
```json
{
  "model": "classifier_chain",
  "hamming_loss": 0.15,
  "f1_micro": 0.82,
  "f1_macro": 0.71,
  "exact_match": 0.52
}
```

---

## Project Structure

```
Air_Quality/
├── data/
│   ├── raw/air_quality_readings.csv      # 10k sensor readings (immutable)
│   └── processed/                         # Cleaned datasets
├── notebooks/
│   ├── 01_data_quality.ipynb             # Audit + cleaning rules
│   ├── 02_eda_visualization.ipynb        # Plots + insights
│   ├── 03_feature_engineering.ipynb      # Build 77 features
│   └── 04_multilabel_modeling.ipynb      # Train + compare models
├── src/
│   ├── config.py                         # Paths, constants, labels
│   ├── data_loading.py                   # Read + parse raw CSV
│   ├── validation.py                     # Quality checks
│   ├── preprocessing.py                  # Clean + standardize
│   ├── features.py                       # Engineer 77 features
│   ├── train.py                          # Model training pipeline
│   ├── score_batch.py                    # Batch prediction logic
│   └── monitoring.py                     # Alert rate tracking
├── models/
│   ├── multilabel_model.pkl              # Trained classifier chain
│   ├── preprocessor.pkl                  # sklearn pipeline for transforms
│   ├── label_thresholds.json             # Per-label probability thresholds
│   └── metrics_summary.json              # Train/test performance
├── outputs/
│   ├── batch_predictions.csv             # Latest scored readings
│   └── monitoring_report.csv             # Alert rates + drift
├── scripts/
│   ├── run_data_validation.py
│   ├── run_training.py
│   ├── batch_scoring.py
│   └── run_monitoring_report.py
├── tests/
│   └── test_pipeline.py                  # Unit + smoke tests
├── app.py                                # FastAPI server
├── Dockerfile                            # Container image
├── docker-compose.yaml                   # Orchestration (API + batch)
├── requirements.txt                      # Dependencies
└── README.md                             # This file
```

---

## Key Design Decisions

### Why Multi-Label?
Real health risks are **simultaneous**. A single reading can trigger respiratory + outdoor + vulnerable alerts. Binary Relevance (5 independent models) misses these correlations. Classifier Chain captures them.

### Why Time-Aware 80/20 Split?
Air quality is temporal. Random splits leak future patterns into training. We train on Jan-Aug, test on Sep-Dec (temporal order preserved).

### Why Batch Scoring?
Real-time inference is overkill for hourly data. Batch scoring is:
- Cheaper (process 1000 readings once/hour vs. online)
- Simpler (no model serving complexity)
- Easier to debug (audit trail in CSV)

### Why JSON + CSV for Artifacts?
- Model: `pickle` (sklearn objects)
- Thresholds: `JSON` (human-readable tuning)
- Predictions: `CSV` (easy to load, audit, archive)

---

## Testing & Validation

Run tests:
```bash
python -m pytest tests/test_pipeline.py -v
```

Tests cover:
- Preprocessing (Autumn → Fall mapping, station name standardization)
- Missing value flagging (4 columns tracked)
- Feature engineering (77 columns created, no silent nulls)
- Label distribution (expected frequencies)
- API responses (health, alerts, metrics valid JSON)
- End-to-end smoke test (raw → clean → train → score → monitor)

---

## Model Performance

**Test set metrics (2000 readings, Sep-Dec data):**

| Metric | Value |
|--------|-------|
| Hamming Loss | 0.15 |
| F1 Micro | 0.82 |
| F1 Macro | 0.71 |
| Exact Match | 0.52 |
| Respiratory F1 | 0.68 |
| Cardiovascular F1 | 0.54 |
| Vulnerable F1 | 0.85 |
| Outdoor F1 | 0.71 |
| Industrial F1 | 0.63 |

**Alert distribution (latest batch):**
- Vulnerable Alert: 73.7% of readings
- Industrial Event: 34.7%
- Outdoor Warning: 31.0%
- Respiratory Risk: 28.2%
- Cardiovascular Risk: 17.8%

---

## Next Steps

**Phase 2 (Monitoring & Drift):**
- Add SQLite for historical predictions
- Track data drift (feature distributions over time)
- Track model drift (label rates changing?)

**Phase 3 (Automation):**
- CI/CD pipeline (GitHub Actions: pytest → lint → Docker build)
- Hourly scheduled batch scoring (cron or cloud scheduler)

**Phase 4 (Dashboard):**
- Web dashboard showing alert trends
- Station-level breakdowns
- Risk heatmaps by hour/day/station

---

## Dependencies

See `requirements.txt`:
- **Data**: pandas, numpy
- **ML**: scikit-learn, xgboost
- **Viz**: matplotlib, seaborn
- **API**: fastapi, uvicorn, pydantic
- **Dev**: pytest, jupyterlab

---

## Questions?

- How do thresholds work? See `models/label_thresholds.json` for per-label probability cutoffs.
- Can I retrain? Yes: `python -m scripts.run_training` (80/20 split redrawn from full dataset).
- Can I add a new label? Yes: add column to `LABEL_COLS` in `src/config.py`, retrain.
- Is this production-ready? For batch use cases yes. Real-time requires model serving framework (BentoML, KServe).
