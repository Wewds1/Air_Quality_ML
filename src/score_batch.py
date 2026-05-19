import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.config import FEATURE_COLUMNS_PATH, LABEL_COLS, MODELS_DIR, OUTPUTS_DIR
from src.features import build_feature_frame
from src.preprocessing import preprocess_raw


def _build_probability_frame(model_wrapper, transformed: pd.DataFrame) -> pd.DataFrame:
    proba = model_wrapper.predict_proba(transformed)

    if isinstance(proba, list):
        if len(proba) != len(LABEL_COLS):
            raise ValueError(f"Expected {len(LABEL_COLS)} probability arrays, got {len(proba)}.")

        return pd.DataFrame(
            {
                f"prob_{label}": label_proba[:, 1]
                for label, label_proba in zip(LABEL_COLS, proba)
            }
        )

    if getattr(proba, "ndim", 0) != 2 or proba.shape[1] != len(LABEL_COLS):
        raise ValueError(f"Unexpected probability output shape: {getattr(proba, 'shape', None)}")

    return pd.DataFrame({f"prob_{label}": proba[:, index] for index, label in enumerate(LABEL_COLS)})


def _load_feature_columns() -> list[str] | None:
    if not FEATURE_COLUMNS_PATH.exists():
        return None
    return json.loads(FEATURE_COLUMNS_PATH.read_text(encoding="utf-8"))


def _load_model_metadata() -> dict:
    metrics_path = MODELS_DIR / "metrics_summary.json"
    if not metrics_path.exists():
        return {}
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def score_frame(df: pd.DataFrame, output_path: str | None = None) -> pd.DataFrame:
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    model = joblib.load(MODELS_DIR / "multilabel_model.pkl")
    thresholds = json.loads((MODELS_DIR / "label_thresholds.json").read_text(encoding="utf-8"))
    model_metadata = _load_model_metadata()

    clean_df = preprocess_raw(df)
    feature_df = build_feature_frame(clean_df)

    trained_columns = _load_feature_columns()
    excluded_cols = {"reading_id", "timestamp", *LABEL_COLS}
    feature_cols = trained_columns or [column for column in feature_df.columns if column not in excluded_cols]
    X = feature_df.reindex(columns=feature_cols)

    transformed = preprocessor.transform(X)
    model_wrapper = model.named_steps["model"]
    proba_matrix = _build_probability_frame(model_wrapper, transformed)

    for label in LABEL_COLS:
        alert_name = f"alert_{label.replace('label_', '')}"
        proba_name = f"prob_{label}"
        proba_matrix[alert_name] = (proba_matrix[proba_name] >= thresholds[label]).astype(int)

    alert_cols = [column for column in proba_matrix.columns if column.startswith("alert_")]
    risk_prob_cols = [column for column in proba_matrix.columns if column.startswith("prob_")]

    proba_matrix["total_alerts"] = proba_matrix[alert_cols].sum(axis=1)
    proba_matrix["max_risk_prob"] = proba_matrix[risk_prob_cols].max(axis=1).round(4)
    proba_matrix["dominant_risk"] = (
        proba_matrix[risk_prob_cols]
        .idxmax(axis=1)
        .str.replace("prob_label_", "", regex=False)
    )
    proba_matrix["alert_signature"] = proba_matrix[alert_cols].astype(str).agg("".join, axis=1)
    proba_matrix["scored_at"] = datetime.now(timezone.utc).isoformat()
    proba_matrix["model_version"] = model_metadata.get("selected_model", "unknown")

    scored = pd.concat([clean_df.reset_index(drop=True), proba_matrix], axis=1)
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scored.to_csv(output_path, index=False)
    return scored


def score_batch(input_path: str, output_path: str | None = None) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else OUTPUTS_DIR / "batch_predictions.csv"

    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return score_frame(df, str(output_path))
