import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss, jaccard_score
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
)
from src.features import build_feature_frame
from src.preprocessing import preprocess_raw


def load_training_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def make_split(df: pd.DataFrame):
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - TEST_SIZE))
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    return train_df, test_df


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    categorical_features = [col for col in CATEGORICAL_COLS if col in X_train.columns]
    numeric_features = [col for col in X_train.columns if col not in categorical_features]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_multilabel(y_true, y_pred):
    return {
        "hamming_loss": hamming_loss(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "exact_match": (y_true.values == y_pred).all(axis=1).mean(),
        "jaccard_samples": jaccard_score(y_true, y_pred, average="samples", zero_division=0),
    }


def train_pipeline(raw_csv_path: str):
    raw_df = load_training_data(raw_csv_path)
    clean_df = preprocess_raw(raw_df)
    feature_df = build_feature_frame(clean_df)

    train_df, test_df = make_split(feature_df)

    y_train = train_df[LABEL_COLS].astype(int)
    y_test = test_df[LABEL_COLS].astype(int)

    drop_cols = ID_COLS + LABEL_COLS
    X_train = train_df.drop(columns=drop_cols)
    X_test = test_df.drop(columns=drop_cols)

    preprocessor = build_preprocessor(X_train)

    baseline_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=RANDOM_SEED,
                n_jobs=-1,
            ),
            n_jobs=-1,
        )),
    ])
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)

    chain_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", ClassifierChain(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_SEED,
            ),
            order=[2, 0, 1, 3, 4],
            random_state=RANDOM_SEED,
        )),
    ])
    chain_model.fit(X_train, y_train)
    chain_pred = chain_model.predict(X_test)

    baseline_metrics = evaluate_multilabel(y_test, baseline_pred)
    chain_metrics = evaluate_multilabel(y_test, chain_pred)

    selected_model_name = "baseline_rf" if baseline_metrics["hamming_loss"] <= chain_metrics["hamming_loss"] else "classifier_chain_lr"
    selected_model = baseline_model if selected_model_name == "baseline_rf" else chain_model
    selected_metrics = baseline_metrics if selected_model_name == "baseline_rf" else chain_metrics

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(selected_model, MODELS_DIR / "multilabel_model.pkl")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")

    thresholds = {label: 0.5 for label in LABEL_COLS}
    with open(MODELS_DIR / "label_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    summary = {
        "selected_model": selected_model_name,
        "baseline_metrics": baseline_metrics,
        "chain_metrics": chain_metrics,
        "selected_metrics": selected_metrics,
        "label_order": LABEL_COLS,
    }
    with open(MODELS_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary