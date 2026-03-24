"""
Quick-start script: generates synthetic data matching the Kaggle schema,
trains all 3 models via MLflow, and saves artifacts/ for Docker serving.

Usage:
    python scripts/generate_and_train.py
"""

import logging
import os
import sys
from pathlib import Path

# Make sure project root is on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

import joblib
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from pipeline.feature_engineer import (
    build_preprocessor,
    create_domain_features,
    encode_target,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    NOMINAL_FEATURES,
)
from pipeline.data_ingestion import TARGET_COLUMN, YES_NO_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
N_SAMPLES = 3000
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------------

def generate_synthetic_data(n: int = N_SAMPLES, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Generate synthetic data matching the Kaggle online-shopping schema."""
    rng = np.random.default_rng(seed)

    yes_no = lambda p: rng.choice(["Yes", "No"], size=n, p=[p, 1 - p])

    df = pd.DataFrame({
        "country":                      rng.choice(["CANADA", "CHINA", "INDIA"], size=n, p=[0.30, 0.35, 0.35]),
        "online_consumer":              yes_no(0.80),
        "age_group":                    rng.choice(["Gen Z", "Millennials", "Gen X", "Baby Boomers"], size=n, p=[0.20, 0.35, 0.30, 0.15]),
        "annual_salary_band":           rng.choice(["Low", "Medium", "Medium High", "High"], size=n, p=[0.20, 0.30, 0.30, 0.20]),
        "gender":                       rng.choice(["Female", "Male", "Prefer not to say"], size=n, p=[0.48, 0.48, 0.04]),
        "education":                    rng.choice(["Highschool Graduate", "University Graduate", "Masters' Degree", "Doctorate Degree"], size=n, p=[0.15, 0.45, 0.30, 0.10]),
        "payment_method_card":          yes_no(0.65),
        "living_region":                rng.choice(["Metropolitan", "Suburban Areas", "Rural Areas"], size=n, p=[0.45, 0.35, 0.20]),
        "online_service_preference":    yes_no(0.70),
        "ai_endorsement":               yes_no(0.55),
        "ai_privacy_no_trust":          yes_no(0.35),
        "ai_enhance_experience":        yes_no(0.60),
        "ai_tool_chatbots":             yes_no(0.50),
        "ai_tool_virtual_assistant":    yes_no(0.45),
        "ai_tool_voice_photo_search":   yes_no(0.40),
        "payment_method_cod":           yes_no(0.30),
        "payment_method_ewallet":       yes_no(0.55),
        "product_category_appliances":  yes_no(0.30),
        "product_category_electronics": yes_no(0.50),
        "product_category_groceries":   yes_no(0.65),
        "product_category_personal_care": yes_no(0.45),
        "product_category_clothing":    yes_no(0.55),
    })

    # Create domain features to compute a realistic satisfaction label
    df_feat = create_domain_features(df.copy())
    score = (
        df_feat["ai_readiness_score"] * 0.4
        + df_feat["ai_tool_usage_count"] * 0.2
        + df_feat["digital_payment_preference"] * 0.15
        + df_feat["product_category_count"] * 0.1
        + rng.normal(0, 0.5, size=n)
    )
    threshold = np.percentile(score, 45)
    df[TARGET_COLUMN] = np.where(score >= threshold, "Satisfied", "Unsatisfied")

    logger.info("Generated %d synthetic samples. Class distribution:\n%s",
                n, df[TARGET_COLUMN].value_counts().to_string())
    return df


# ---------------------------------------------------------------------------
# 2. Save data splits
# ---------------------------------------------------------------------------

def prepare_splits(df: pd.DataFrame):
    train_val, test = train_test_split(df, test_size=0.20, stratify=df[TARGET_COLUMN], random_state=RANDOM_STATE)
    train, val = train_test_split(train_val, test_size=0.125, stratify=train_val[TARGET_COLUMN], random_state=RANDOM_STATE)
    logger.info("Split sizes — train: %d | val: %d | test: %d", len(train), len(val), len(test))
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    train.to_csv(DATA_DIR / "processed" / "train.csv", index=False)
    val.to_csv(DATA_DIR / "processed" / "val.csv", index=False)
    test.to_csv(DATA_DIR / "processed" / "test.csv", index=False)
    return train, val, test


# ---------------------------------------------------------------------------
# 3. Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(train_df, val_df, test_df):
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = create_domain_features(train_df)
    val_df   = create_domain_features(val_df)
    test_df  = create_domain_features(test_df)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_val   = val_df.drop(columns=[TARGET_COLUMN])
    y_val   = val_df[TARGET_COLUMN]
    X_test  = test_df.drop(columns=[TARGET_COLUMN])
    y_test  = test_df[TARGET_COLUMN]

    y_train_enc, le = encode_target(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc   = preprocessor.transform(X_val)
    X_test_enc  = preprocessor.transform(X_test)

    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.joblib")
    joblib.dump(le,           ARTIFACTS_DIR / "label_encoder.joblib")
    logger.info("Preprocessor + label encoder saved to %s", ARTIFACTS_DIR)
    return X_train_enc, X_val_enc, X_test_enc, y_train_enc, y_val_enc, y_test_enc, preprocessor, le


# ---------------------------------------------------------------------------
# 4. Train & track with MLflow (local file-based tracking)
# ---------------------------------------------------------------------------

MODELS = {
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "params": {"C": 1.0, "solver": "lbfgs"},
    },
    "random_forest": {
        "model": RandomForestClassifier(random_state=RANDOM_STATE),
        "params": {"n_estimators": 100, "max_depth": 8, "min_samples_split": 5},
    },
    "lightgbm": {
        "model": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "num_leaves": 31,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },
}


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


def train_models(X_train, y_train, X_val, y_val, X_test, y_test, class_names, le):
    # Use local file-based MLflow (no server needed for training)
    mlruns_dir = ROOT / "mlruns"
    mlflow.set_tracking_uri(f"file:///{mlruns_dir.as_posix()}")
    mlflow.set_experiment("ai_retail_satisfaction")

    best_run_id  = None
    best_f1      = -1.0
    best_model   = None
    best_name    = None

    for name, config in MODELS.items():
        logger.info("=" * 50)
        logger.info("Training: %s", name)
        model  = config["model"]
        params = config["params"]
        model.set_params(**params)

        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(params)
            mlflow.log_param("model_type", name)
            mlflow.log_param("dataset", "synthetic_3000_samples")

            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
            mlflow.log_metric("cv_f1_mean", float(cv_f1.mean()))
            mlflow.log_metric("cv_f1_std",  float(cv_f1.std()))
            logger.info("  CV F1: %.4f ± %.4f", cv_f1.mean(), cv_f1.std())

            # Full training
            model.fit(X_train, y_train)

            # Validation
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1]
            val_m = compute_metrics(y_val, y_val_pred, y_val_prob)
            for k, v in val_m.items():
                mlflow.log_metric(f"val_{k}", v)
            logger.info("  Val metrics: %s", {k: f"{v:.4f}" for k, v in val_m.items()})

            # Test
            y_test_pred = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)[:, 1]
            test_m = compute_metrics(y_test, y_test_pred, y_test_prob)
            for k, v in test_m.items():
                mlflow.log_metric(f"test_{k}", v)
            logger.info("  Test metrics: %s", {k: f"{v:.4f}" for k, v in test_m.items()})

            # Save report artifact
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            report_path = ARTIFACTS_DIR / f"{name}_report.txt"
            from sklearn.metrics import classification_report
            report_path.write_text(
                classification_report(y_test, y_test_pred, target_names=class_names)
            )
            mlflow.log_artifact(str(report_path))

            # Log model
            if name == "lightgbm":
                mlflow.lightgbm.log_model(model, name="model")
            else:
                mlflow.sklearn.log_model(model, name="model")

            if val_m["f1"] > best_f1:
                best_f1      = val_m["f1"]
                best_run_id  = run.info.run_id
                best_model   = model
                best_name    = name

    logger.info("=" * 50)
    logger.info("Best model: %s  val_f1=%.4f  run_id=%s", best_name, best_f1, best_run_id)

    # Register best model
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(model_uri, "ai_retail_satisfaction_model")
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name="ai_retail_satisfaction_model",
        alias="production",
        version=registered.version,
    )
    logger.info("Registered model version %s → alias 'production'", registered.version)

    # Save locally for Docker
    joblib.dump(best_model, ARTIFACTS_DIR / "model.joblib")
    logger.info("Best model saved to %s/model.joblib", ARTIFACTS_DIR)
    return best_name, best_f1


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Step 1/4 — Generating synthetic data (%d samples) ...", N_SAMPLES)
    df = generate_synthetic_data()

    logger.info("Step 2/4 — Splitting into train / val / test ...")
    train_df, val_df, test_df = prepare_splits(df)

    logger.info("Step 3/4 — Feature engineering ...")
    X_train, X_val, X_test, y_train, y_val, y_test, preprocessor, le = engineer_features(
        train_df, val_df, test_df
    )

    logger.info("Step 4/4 — Training 3 models with MLflow tracking ...")
    best_name, best_f1 = train_models(
        X_train, y_train, X_val, y_val, X_test, y_test,
        list(le.classes_), le
    )

    # Verify artifacts
    needed = ["model.joblib", "preprocessor.joblib", "label_encoder.joblib"]
    for f in needed:
        p = ARTIFACTS_DIR / f
        size_kb = p.stat().st_size // 1024
        logger.info("  ✓ %s  (%d KB)", f, size_kb)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  TRAINING COMPLETE!")
    logger.info("  Best model : %s  (val F1 = %.4f)", best_name, best_f1)
    logger.info("  Artifacts  : %s/", ARTIFACTS_DIR)
    logger.info("  Next step  : docker-compose up --build -d")
    logger.info("=" * 60)
