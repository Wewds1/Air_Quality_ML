import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from src.config import LABEL_COLS, MODELS_DIR, OUTPUTS_DIR
from src.features import build_feature_frame
from src.preprocessing import preprocess_raw


def _build_probability_frame(model_wrapper, transformed: pd.DataFrame) -> pd.DataFrame:
    proba = model_wrapper.predict_proba(transformed)

    if isinstance(proba, list):
        if len(proba) != len(LABEL_COLS):
            raise ValueError(
                f"Expected {len(LABEL_COLS)} probability arrays, got {len(proba)}."
            )

        return pd.DataFrame(
            {
                f"prob_{label}": label_proba[:, 1]
                for label, label_proba in zip(LABEL_COLS, proba)
            }
        )

    if getattr(proba, "ndim", 0) != 2:
        raise ValueError(f"Unexpected probability output shape: {getattr(proba, 'shape', None)}")

    if proba.shape[1] != len(LABEL_COLS):
        raise ValueError(
            f"Expected probability matrix with {len(LABEL_COLS)} columns, got {proba.shape[1]}."
        )

    return pd.DataFrame(
        {f"prob_{label}": proba[:, i] for i, label in enumerate(LABEL_COLS)}
    )


def score_batch(input_path: str, output_path: str | None = None) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else OUTPUTS_DIR / "batch_predictions.csv"

    df = pd.read_csv(input_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    model = joblib.load(MODELS_DIR / "multilabel_model.pkl")

    clean_df = preprocess_raw(df)
    feature_df = build_feature_frame(clean_df)

    excluded_cols = {"reading_id", "timestamp", *LABEL_COLS}
    feature_cols = [col for col in feature_df.columns if col not in excluded_cols]
    X = feature_df[feature_cols]

    transformed = preprocessor.transform(X)

    model_wrapper = model.named_steps["model"]
    proba_matrix = _build_probability_frame(model_wrapper, transformed)

    thresholds_path = MODELS_DIR / "label_thresholds.json"
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))

    for label in LABEL_COLS:
        proba_matrix[f"alert_{label.replace('label_', '')}"] = (
            proba_matrix[f"prob_{label}"] >= thresholds[label]
        ).astype(int)

    alert_cols = [c for c in proba_matrix.columns if c.startswith("alert_")]
    proba_matrix["total_alerts"] = proba_matrix[alert_cols].sum(axis=1)
    proba_matrix["scored_at"] = datetime.utcnow().isoformat()

    scored = pd.concat([df.reset_index(drop=True), proba_matrix], axis=1)
    scored.to_csv(output_path, index=False)
    return scored