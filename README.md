# Air Quality Command Center

Production-oriented multi-label air quality risk scoring with a live monitoring dashboard.

The project now does four things end to end:

1. Cleans and validates raw sensor data with domain-specific repair rules from `scratch.md`.
2. Trains and selects a multi-label model with a time-aware train/validation/test split.
3. Tunes per-label thresholds and scores new batches with operational metadata.
4. Serves a live dashboard and API that update from the latest scored artifacts.

## Current Status

This repo is now aligned with the major implementation items in `scratch.md`:

- Duplicate sensor batches are dropped by all fields except `reading_id`.
- Domain-specific imputations are applied for `benzene`, `visibility_km`, `wind_dir_deg`, and `pressure_hpa`.
- Physical validity issues are tracked and repaired, including `pm10 < pm25`.
- The feature set includes the missing risk-context features from `scratch.md`.
- Training now uses model comparison plus per-label threshold tuning.
- Monitoring includes label-rate drift, co-occurrence gap, and sensor-quality pressure.
- The frontend is no longer empty; `/` serves a live dashboard with SSE updates.

The repo is not fully finished in the cloud-platform sense. Recommended next hardening steps are still:

- scheduled scoring orchestration outside the app process
- auth / access control for the API and dashboard
- persistent historical storage beyond CSV + JSON artifacts
- CI/CD and deployment checks
- feature-distribution drift beyond label and co-occurrence monitoring

## Model Snapshot

Latest trained artifacts in `models/`:

- Selected model: `random_forest`
- Feature count: `75`
- Time split: `7000 train / 1000 validation / 2000 test`
- Hamming loss: `0.0358`
- F1 micro: `0.9505`
- F1 macro: `0.9406`
- Exact match: `0.8340`
- Jaccard samples: `0.7367`

Compared with the previous stored baseline, the optimized pipeline improved every headline metric slightly:

- Hamming loss: `0.03619 -> 0.0358`
- F1 micro: `0.94977 -> 0.95054`
- F1 macro: `0.93963 -> 0.94064`
- Exact match: `0.83350 -> 0.8340`

The main threshold change selected by validation was a looser operating point for `label_industrial_event` (`0.30`) while the other labels stayed at `0.50`.

## Architecture

### Data and preprocessing

`src/preprocessing.py` now handles:

- duplicate removal
- season and station normalization
- missingness flags
- PM10 repair when it falls below PM2.5
- AQI consistency flags against a PM2.5-derived reference
- zero-wind sensor flags
- grouped imputations for the known sensor gaps

### Feature engineering

`src/features.py` now covers:

- cyclical encodings for hour, month, and wind direction
- pollutant composite features
- weather and station interaction features
- rolling and lag features per station
- risk-context features from `scratch.md`:
  - `pm25_cardio_zone`
  - `multi_label_event`
  - `pollutants_above_who`

### Training and model selection

`src/train.py` now:

- uses a time-aware `70/10/20` train/validation/test split
- compares:
  - `random_forest`
  - `extra_trees`
  - `classifier_chain_lr`
- tunes thresholds per label on the validation split
- stores:
  - trained model
  - fitted preprocessor
  - feature column contract
  - threshold map
  - candidate metrics
  - training label-rate and co-occurrence metadata

### Scoring and monitoring

`src/score_batch.py` now writes richer operational output:

- per-label probabilities
- per-label binary alerts
- `total_alerts`
- `max_risk_prob`
- `dominant_risk`
- `alert_signature`
- `model_version`
- sensor-quality and missingness flags

`src/monitoring.py` builds:

- dashboard snapshots
- label-rate drift summaries
- co-occurrence gap summaries
- station ranking cards
- timeline payloads
- CSV and JSON monitoring artifacts

### API and frontend

`app.py` serves:

- `/` - live dashboard
- `/api/dashboard` - full dashboard payload
- `/api/stream/dashboard` - Server-Sent Events live stream
- `/api/health` and `/health`
- `/api/alerts` and `/current_alerts`
- `/api/metrics` and `/metrics`
- `/api/docs`

The frontend in `static/` is a black/red/ivory editorial dashboard rather than a generic analytics template. It uses:

- bold condensed typography
- asymmetrical panels
- a live trajectory section
- station watch cards
- alert rail
- model and monitoring diagnostics
- SSE with polling fallback

## Quick Start

### 1. Install

```bash
python -m venv venv
source venv/Scripts/activate  # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train artifacts

```bash
python -m scripts.run_training
```

Outputs:

- `models/multilabel_model.pkl`
- `models/preprocessor.pkl`
- `models/label_thresholds.json`
- `models/metrics_summary.json`
- `outputs/feature_engineering_columns.json`

### 3. Score a batch

```bash
python -m scripts.batch_scoring
```

Default output:

- `outputs/batch_predictions.csv`

### 4. Build monitoring artifacts

```bash
python -m scripts.run_monitoring_report
```

Outputs:

- `outputs/monitoring_report.csv`
- `outputs/monitoring_snapshot.json`

### 5. Run the app

Local development:

```bash
python -m uvicorn app:app --reload
```

Production-style container entrypoint:

```bash
docker-compose up --build
```

## Testing

The updated test suite checks:

- sensor validation assumptions
- preprocessing repairs and imputations
- feature coverage for the `scratch.md` gaps
- dashboard snapshot generation
- dashboard API response shape

Run:

```bash
pytest tests/test_pipeline.py -v
python -m scripts.smoke_test
```

## Project Layout

```text
Air_Quality/
|-- app.py
|-- Dockerfile
|-- docker-compose.yaml
|-- requirements.txt
|-- README.md
|-- scratch.md
|-- data/
|-- models/
|-- outputs/
|-- scripts/
|-- src/
|-- static/
`-- tests/
```

## Notes On Remaining Gaps

If the goal is "production" in the strict sense rather than "production-ready local deployment", the main missing pieces are operational rather than modeling:

- no scheduler or worker service is appending hourly runs automatically
- no auth, rate limiting, or multi-user tenancy
- no persistent database for long-lived alert history
- no structured application logging or metrics exporter
- no deployment manifests for a target platform

Those are the next places to invest, not more cosmetic model work.
