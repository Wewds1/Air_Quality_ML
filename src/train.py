import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss, jaccard_score, precision_score, recall_score
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLS,
    FEATURE_COLUMNS_PATH,
    ID_COLS,
    LABEL_COLS,
    MODELS_DIR,
    RANDOM_SEED,
    TEST_SIZE,
    VALIDATION_SIZE,
)
from src.features import build_feature_frame
from src.preprocessing import preprocess_raw


def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def make_split(df: pd.DataFrame):
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    total_rows = len(df_sorted)
    test_start = int(total_rows * (1 - TEST_SIZE))
    validation_rows = max(int(total_rows * VALIDATION_SIZE), len(LABEL_COLS) * 25)
    validation_start = max(test_start - validation_rows, 0)

    train_df = df_sorted.iloc[:validation_start].copy()
    validation_df = df_sorted.iloc[validation_start:test_start].copy()
    test_df = df_sorted.iloc[test_start:].copy()
    return train_df, validation_df, test_df


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    categorical_features = [col for col in CATEGORICAL_COLS if col in X_train.columns]
    numeric_features = [col for col in X_train.columns if col not in categorical_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_candidate_models(X_train: pd.DataFrame) -> dict[str, Pipeline]:
    return {
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                (
                    "model",
                    MultiOutputClassifier(
                        RandomForestClassifier(
                            n_estimators=400,
                            max_depth=18,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            random_state=RANDOM_SEED,
                            n_jobs=-1,
                        ),
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                (
                    "model",
                    MultiOutputClassifier(
                        ExtraTreesClassifier(
                            n_estimators=500,
                            max_depth=None,
                            min_samples_leaf=1,
                            class_weight="balanced_subsample",
                            random_state=RANDOM_SEED,
                            n_jobs=-1,
                        ),
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "classifier_chain_lr": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                (
                    "model",
                    ClassifierChain(
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=1500,
                            random_state=RANDOM_SEED,
                        ),
                        order=[2, 0, 1, 3, 4],
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


def probability_frame(model_wrapper, transformed: pd.DataFrame) -> pd.DataFrame:
    proba = model_wrapper.predict_proba(transformed)

    if isinstance(proba, list):
        return pd.DataFrame(
            {
                label: label_proba[:, 1]
                for label, label_proba in zip(LABEL_COLS, proba)
            }
        )

    return pd.DataFrame({label: proba[:, index] for index, label in enumerate(LABEL_COLS)})


def apply_thresholds(probabilities: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            label: (probabilities[label] >= thresholds[label]).astype(int)
            for label in LABEL_COLS
        },
        index=probabilities.index,
    )


def evaluate_multilabel(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:
    metrics = {
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "exact_match": float((y_true.values == y_pred.values).all(axis=1).mean()),
        "jaccard_samples": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "per_label": {},
    }

    for label in LABEL_COLS:
        metrics["per_label"][label] = {
            "precision": float(precision_score(y_true[label], y_pred[label], zero_division=0)),
            "recall": float(recall_score(y_true[label], y_pred[label], zero_division=0)),
            "f1": float(f1_score(y_true[label], y_pred[label], zero_division=0)),
        }

    return metrics


def tune_thresholds(y_true: pd.DataFrame, probabilities: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    candidate_thresholds = np.arange(0.2, 0.85, 0.05)
    best_thresholds = {}
    best_label_scores = {}

    for label in LABEL_COLS:
        best_threshold = 0.5
        best_score = -1.0

        for threshold in candidate_thresholds:
            predictions = (probabilities[label] >= threshold).astype(int)
            score = f1_score(y_true[label], predictions, zero_division=0)
            if score > best_score or (np.isclose(score, best_score) and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
                best_threshold = float(round(threshold, 2))
                best_score = float(score)

        best_thresholds[label] = best_threshold
        best_label_scores[label] = round(best_score, 4)

    return best_thresholds, best_label_scores


def label_cooccurrence(y_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    matrix = y_frame.astype(float).T.dot(y_frame.astype(float)) / len(y_frame)
    return {
        row_label: {
            column_label: round(float(matrix.loc[row_label, column_label]), 4)
            for column_label in LABEL_COLS
        }
        for row_label in LABEL_COLS
    }


def selection_key(candidate_result: dict) -> tuple[float, float, float, float]:
    validation_metrics = candidate_result["validation_tuned_metrics"]
    return (
        -validation_metrics["hamming_loss"],
        validation_metrics["f1_macro"],
        validation_metrics["exact_match"],
        validation_metrics["f1_micro"],
    )


def train_pipeline(raw_csv_path: str):
    raw_df = load_training_data(raw_csv_path)
    clean_df = preprocess_raw(raw_df)
    feature_df = build_feature_frame(clean_df)

    train_df, validation_df, test_df = make_split(feature_df)

    y_train = train_df[LABEL_COLS].astype(int)
    y_validation = validation_df[LABEL_COLS].astype(int)
    y_test = test_df[LABEL_COLS].astype(int)

    drop_cols = ID_COLS + LABEL_COLS
    X_train = train_df.drop(columns=drop_cols)
    X_validation = validation_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)

    feature_columns = X_train.columns.tolist()
    candidate_results = {}
    fitted_models = {}

    for model_name, pipeline in build_candidate_models(X_train).items():
        pipeline.fit(X_train, y_train)

        transformed_validation = pipeline.named_steps["preprocessor"].transform(X_validation)
        validation_proba = probability_frame(pipeline.named_steps["model"], transformed_validation)
        thresholds, tuned_label_scores = tune_thresholds(y_validation, validation_proba)
        validation_pred = apply_thresholds(validation_proba, thresholds)

        transformed_test = pipeline.named_steps["preprocessor"].transform(X_test)
        test_proba = probability_frame(pipeline.named_steps["model"], transformed_test)
        test_pred = apply_thresholds(test_proba, thresholds)

        candidate_results[model_name] = {
            "thresholds": thresholds,
            "validation_tuned_metrics": evaluate_multilabel(y_validation, validation_pred),
            "test_tuned_metrics": evaluate_multilabel(y_test, test_pred),
            "validation_threshold_scores": tuned_label_scores,
        }
        fitted_models[model_name] = pipeline

    selected_model_name = max(candidate_results, key=lambda name: selection_key(candidate_results[name]))
    selected_model = fitted_models[selected_model_name]
    selected_thresholds = candidate_results[selected_model_name]["thresholds"]
    selected_metrics = candidate_results[selected_model_name]["test_tuned_metrics"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(selected_model, MODELS_DIR / "multilabel_model.pkl")
    joblib.dump(selected_model.named_steps["preprocessor"], MODELS_DIR / "preprocessor.pkl")

    FEATURE_COLUMNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_COLUMNS_PATH.write_text(json.dumps(feature_columns, indent=2), encoding="utf-8")

    with open(MODELS_DIR / "label_thresholds.json", "w", encoding="utf-8") as file_handle:
        json.dump(selected_thresholds, file_handle, indent=2)

    summary = {
        "selected_model": selected_model_name,
        "selected_metrics": selected_metrics,
        "candidate_metrics": candidate_results,
        "label_order": LABEL_COLS,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "split_rows": {
            "train": len(train_df),
            "validation": len(validation_df),
            "test": len(test_df),
        },
        "training_label_rates": train_df[LABEL_COLS].mean().round(4).to_dict(),
        "training_label_cooccurrence": label_cooccurrence(train_df[LABEL_COLS]),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(MODELS_DIR / "metrics_summary.json", "w", encoding="utf-8") as file_handle:
        json.dump(summary, file_handle, indent=2)

    return summary
