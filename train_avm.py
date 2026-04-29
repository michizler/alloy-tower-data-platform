"""
AlloyTower AVM Training Pipeline
=================================

End-to-end training pipeline for the AlloyTower Property Valuation models.

Takes raw alloy_data.csv as input and produces:
  - Cleaned dataset (alloy_clean.csv)
  - Model A: Assessment-aware AVM (uses assessed_value as a feature)
  - Model B: Fundamentals-only AVM (excludes assessment data)
  - MLflow tracking for all runs (sqlite:///mlflow.db by default)
  - Saved model artifacts: model_A.pkl, model_B.pkl, target_encoder.pkl

Usage:
    python train_avm.py
    python train_avm.py --input source_data/alloy_data.csv --output-dir model/models/
    python train_avm.py --skip-tuning      # skip hyperparameter search for faster runs

Author: Bright Uzosike
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from itertools import product
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# ============================================================================
# Configuration
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_CV_FOLDS = 5

# Columns dropped during cleaning (see Data Quality Findings report).
DROP_COLUMNS = [
    "id",  # redundant with property_id
    "full_address",
    "street_address",
    "unit",
    "owner_name",
    "assessor_id",  # high-cardinality strings
    "latitude",
    "longitude",
    "zip_code",  # randomly generated (Findings 4-5)
    "price_per_sqft",  # target leakage
    "annual_tax",
    "building_age",  # redundant with assessed_value, year_built
    "last_sale_date",
    "days_since_sale",
    "sale_year",
    "tax_year",  # no temporal signal (Finding 7)
]

# Below this $/sqft, sale prices fall below US construction cost — physically implausible.
MIN_PRICE_PER_SQFT = 50

COMMON_FEATURES = [
    "state",
    "city",
    "county",
    "property_type",
    "owner_occupied",
    "sqft",
    "lot_size_sqft",
    "bedrooms",
    "bathrooms",
    "year_built",
]
MODEL_A_FEATURES = COMMON_FEATURES + ["assessed_value"]
MODEL_B_FEATURES = COMMON_FEATURES.copy()

TARGET_ENCODE_COLS = ["state", "city", "county"]
NATIVE_CATEGORICAL_COLS = ["property_type", "owner_occupied"]

# LightGBM hyperparameter grid (small, focused — not exhaustive).
PARAM_GRID = {
    "num_leaves": [15, 31, 63],
    "min_child_samples": [10, 20, 40],
}

LGBM_BASE_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "max_depth": 6,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "random_state": RANDOM_STATE,
}
NUM_BOOST_ROUND = 500

EXPERIMENT_A = "alloytower_avm_modelA_assessment_aware"
EXPERIMENT_B = "alloytower_avm_modelB_fundamentals_only"

# ============================================================================
# Logging
# ============================================================================


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("avm_pipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s | %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(handler)
    return logger


log = setup_logging()


# ============================================================================
# Data loading and cleaning
# ============================================================================


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw alloy_data.csv with the correct delimiter, encoding, and dtypes."""
    log.info(f"Loading raw data from {path}")
    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",  # strips the BOM
        thousands=",",
        parse_dates=["last_sale_date"],
        dtype={
            "property_id": "string",
            "zip_code": "string",
            "assessor_id": "string",
            "owner_occupied": "boolean",
        },
    )
    # Defensive: scrub any BOM that leaked through into a column name
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions from the Data Quality Findings report."""
    n_before = len(df)

    # Drop physically implausible rows (below US construction cost / sqft).
    df = df[df["price_per_sqft"] >= MIN_PRICE_PER_SQFT].copy()
    n_dropped = n_before - len(df)
    log.info(
        f"Dropped {n_dropped} rows with price_per_sqft < ${MIN_PRICE_PER_SQFT} "
        f"({100 * n_dropped / n_before:.1f}%); {len(df):,} retained"
    )

    # Drop columns that don't carry signal or carry leakage.
    drop_existing = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_existing)
    log.info(
        f"Dropped {len(drop_existing)} columns; {df.shape[1]} retained for modelling"
    )

    return df


# ============================================================================
# Feature engineering
# ============================================================================


def add_interactions(X: pd.DataFrame) -> pd.DataFrame:
    """Add the small set of interaction features that earned their place during EDA."""
    X = X.copy()
    # sqft per bedroom (handle div-by-zero defensively)
    X["sqft_per_bedroom"] = X["sqft"] / X["bedrooms"].replace(0, np.nan)
    X["sqft_per_bedroom"] = X["sqft_per_bedroom"].fillna(X["sqft"])
    # state × sqft: a square foot in CA is worth more than one in ID
    X["state_x_sqft"] = X["state"] * X["sqft"]
    # Re-derive building_age cleanly from year_built (fresh, not stale)
    X["building_age"] = 2024 - X["year_built"]
    return X


def fit_target_encoder(X_train: pd.DataFrame, y_train: pd.Series) -> TargetEncoder:
    """Fit target encoder on training data ONLY. Smoothing pulls small categories toward the mean."""
    encoder = TargetEncoder(cols=TARGET_ENCODE_COLS, smoothing=10)
    encoder.fit(X_train[TARGET_ENCODE_COLS], y_train)
    return encoder


def transform_features(
    X: pd.DataFrame, encoder: TargetEncoder, train_categories: dict | None = None
) -> tuple[pd.DataFrame, dict]:
    """Apply target encoding and prepare native categorical columns for LightGBM."""
    X = X.copy()
    X[TARGET_ENCODE_COLS] = encoder.transform(X[TARGET_ENCODE_COLS])

    # Native categoricals
    captured_categories = {}
    for col in NATIVE_CATEGORICAL_COLS:
        if train_categories is None:
            X[col] = X[col].astype("category")
            captured_categories[col] = X[col].cat.categories
        else:
            X[col] = pd.Categorical(X[col], categories=train_categories[col])

    return X, captured_categories


# ============================================================================
# Metrics & plotting
# ============================================================================


def compute_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    nonzero = y_true != 0
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": float(
            np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100
        ),
    }


def save_residuals_plot(y_true, y_pred, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=10)
    lim = max(max(y_true), max(y_pred))
    axes[0].plot([0, lim], [0, lim], "r--", linewidth=1)
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title("Predicted vs Actual")
    axes[0].ticklabel_format(style="plain")

    residuals = np.array(y_true) - np.array(y_pred)
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(0, color="r", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Residual")
    axes[1].set_title("Residuals vs Predicted")
    axes[1].ticklabel_format(style="plain")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Model training
# ============================================================================


def train_baselines(
    X_train, X_test, y_train, y_test, experiment_name: str, model_label: str
) -> dict:
    """Run the three baselines (mean, state-mean, linear regression) and log to MLflow."""
    mlflow.set_experiment(experiment_name)
    results = {}

    # Mean baseline
    with mlflow.start_run(run_name=f"{model_label}_mean_baseline"):
        pred = np.full(len(y_test), y_train.mean())
        m = compute_metrics(y_test, pred)
        mlflow.log_param("model_type", "mean_baseline")
        mlflow.log_metrics(m)
        results["mean_baseline"] = m
        log.info(
            f"  Mean baseline    | MAE ${m['MAE']:>10,.0f} | "
            f"MAPE {m['MAPE']:>5.1f}% | R² {m['R2']:>5.3f}"
        )

    # State-mean baseline (state column has been target-encoded, so it IS the state mean)
    with mlflow.start_run(run_name=f"{model_label}_state_mean_baseline"):
        pred = X_test["state"].values
        m = compute_metrics(y_test, pred)
        mlflow.log_param("model_type", "state_mean_baseline")
        mlflow.log_metrics(m)
        results["state_mean_baseline"] = m
        log.info(
            f"  State-mean       | MAE ${m['MAE']:>10,.0f} | "
            f"MAPE {m['MAPE']:>5.1f}% | R² {m['R2']:>5.3f}"
        )

    # Linear regression
    with mlflow.start_run(run_name=f"{model_label}_linear_regression"):
        cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) == "category"]
        X_tr = X_train.drop(columns=cat_cols)
        X_te = X_test.drop(columns=cat_cols)

        model = LinearRegression()
        model.fit(X_tr, np.log1p(y_train))
        pred = np.expm1(model.predict(X_te))
        m = compute_metrics(y_test, pred)

        mlflow.log_param("model_type", "linear_regression")
        mlflow.log_param("target_transform", "log1p")
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(model, name="model")
        results["linear_regression"] = m
        log.info(
            f"  Linear reg       | MAE ${m['MAE']:>10,.0f} | "
            f"MAPE {m['MAPE']:>5.1f}% | R² {m['R2']:>5.3f}"
        )

    return results


def cv_tune_lightgbm(
    X_train: pd.DataFrame, y_train: pd.Series, skip_tuning: bool = False
) -> tuple[dict, pd.DataFrame]:
    """Grid-search over PARAM_GRID using N_CV_FOLDS-fold CV. Returns best params + grid results."""
    cat_features = [c for c in X_train.columns if str(X_train[c].dtype) == "category"]
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if skip_tuning:
        log.info("  Skipping hyperparameter tuning (using defaults)")
        return (
            {**LGBM_BASE_PARAMS, "num_leaves": 31, "min_child_samples": 20},
            pd.DataFrame(),
        )

    grid_results = []
    best_score = float("inf")
    best_params = None

    for num_leaves, min_child in product(
        PARAM_GRID["num_leaves"], PARAM_GRID["min_child_samples"]
    ):
        params = {
            **LGBM_BASE_PARAMS,
            "num_leaves": num_leaves,
            "min_child_samples": min_child,
        }
        fold_maes = []

        for tr_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

            dtrain = lgb.Dataset(
                X_tr, label=np.log1p(y_tr), categorical_feature=cat_features
            )
            model = lgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)
            fold_maes.append(mean_absolute_error(y_val, np.expm1(model.predict(X_val))))

        mean_mae = float(np.mean(fold_maes))
        grid_results.append(
            {
                "num_leaves": num_leaves,
                "min_child_samples": min_child,
                "cv_mae_mean": mean_mae,
                "cv_mae_std": float(np.std(fold_maes)),
            }
        )
        if mean_mae < best_score:
            best_score, best_params = mean_mae, params

    grid_df = (
        pd.DataFrame(grid_results).sort_values("cv_mae_mean").reset_index(drop=True)
    )
    log.info(
        f"  Best CV params: num_leaves={best_params['num_leaves']}, "
        f"min_child_samples={best_params['min_child_samples']} (CV MAE: ${best_score:,.0f})"
    )
    return best_params, grid_df


def cv_evaluate(X_train, y_train, params: dict) -> pd.DataFrame:
    """Run N_CV_FOLDS-fold CV with chosen params; return per-metric mean and std."""
    cat_features = [c for c in X_train.columns if str(X_train[c].dtype) == "category"]
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    fold_metrics = []
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        dtrain = lgb.Dataset(
            X_tr, label=np.log1p(y_tr), categorical_feature=cat_features
        )
        model = lgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)
        fold_metrics.append(compute_metrics(y_val, np.expm1(model.predict(X_val))))

    return pd.DataFrame(fold_metrics).agg(["mean", "std"])


def train_final_lightgbm(
    X_train,
    X_test,
    y_train,
    y_test,
    params: dict,
    experiment_name: str,
    model_label: str,
    output_dir: Path,
    cv_summary: pd.DataFrame,
) -> tuple[lgb.Booster, dict]:
    """Train final LightGBM on full training set, evaluate on test, log everything."""
    cat_features = [c for c in X_train.columns if str(X_train[c].dtype) == "category"]

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{model_label}_lightgbm_final") as run:
        dtrain = lgb.Dataset(
            X_train, label=np.log1p(y_train), categorical_feature=cat_features
        )
        model = lgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)
        pred = np.expm1(model.predict(X_test))
        test_metrics = compute_metrics(y_test, pred)

        # Log everything
        mlflow.log_params(
            {
                **params,
                "num_boost_round": NUM_BOOST_ROUND,
                "num_features": X_train.shape[1],
                "model_type": "lightgbm_tuned",
            }
        )
        mlflow.log_metrics(test_metrics)
        for metric in ["MAE", "MAPE", "R2", "RMSE"]:
            mlflow.log_metric(
                f"cv_{metric}_mean", float(cv_summary.loc["mean", metric])
            )
            mlflow.log_metric(f"cv_{metric}_std", float(cv_summary.loc["std", metric]))

        # Feature importance
        imp = pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": model.feature_importance(importance_type="gain"),
            }
        ).sort_values("importance", ascending=False)
        imp_path = output_dir / f"feature_importance_{model_label}.csv"
        imp.to_csv(imp_path, index=False)
        mlflow.log_artifact(str(imp_path))

        # Residuals plot
        plot_path = output_dir / f"residuals_{model_label}.png"
        save_residuals_plot(
            y_test, pred, plot_path, f"{model_label} — LightGBM (final)"
        )
        mlflow.log_artifact(str(plot_path))

        mlflow.lightgbm.log_model(model, name="model")

        log.info(
            f"  LightGBM (final) | MAE ${test_metrics['MAE']:>10,.0f} | "
            f"MAPE {test_metrics['MAPE']:>5.1f}% | R² {test_metrics['R2']:>5.3f}"
        )
        log.info(
            f"  CV mean ± std    | MAE ${cv_summary.loc['mean','MAE']:,.0f} ± ${cv_summary.loc['std','MAE']:,.0f} | "
            f"MAPE {cv_summary.loc['mean','MAPE']:.1f}% ± {cv_summary.loc['std','MAPE']:.1f}%"
        )
        log.info(f"  MLflow run_id    | {run.info.run_id}")

    return model, test_metrics


# ============================================================================
# Pipeline orchestration
# ============================================================================


def run_pipeline(
    input_path: Path, output_dir: Path, mlflow_uri: str, skip_tuning: bool
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # MLflow setup
    mlflow.set_tracking_uri(mlflow_uri)
    log.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # ---- Load and clean ----
    raw = load_raw_data(input_path)
    clean = clean_data(raw)
    clean_path = output_dir / "alloy_clean.csv"
    clean.to_csv(clean_path, index=False)
    log.info(f"Wrote cleaned dataset to {clean_path}")

    # ---- Train/test split (single split, fixed seed) ----
    y = clean["last_sale_price"]
    X = clean[MODEL_A_FEATURES]  # superset; subset per-model below
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    log.info(f"Train: {len(X_train_raw):,} rows | Test: {len(X_test_raw):,} rows")

    # ---- Fit target encoder on training data only ----
    encoder = fit_target_encoder(X_train_raw, y_train)
    X_train_enc, train_cats = transform_features(X_train_raw, encoder)
    X_test_enc, _ = transform_features(X_test_raw, encoder, train_categories=train_cats)

    # Save encoder for inference reuse
    with open(output_dir / "target_encoder.pkl", "wb") as f:
        pickle.dump({"encoder": encoder, "train_categories": train_cats}, f)
    log.info(f"Saved target encoder to {output_dir / 'target_encoder.pkl'}")

    # ---- Feature matrices for both models ----
    X_train_A = add_interactions(X_train_enc[MODEL_A_FEATURES])
    X_test_A = add_interactions(X_test_enc[MODEL_A_FEATURES])
    X_train_B = add_interactions(X_train_enc[MODEL_B_FEATURES])
    X_test_B = add_interactions(X_test_enc[MODEL_B_FEATURES])

    # ============================================================
    # MODEL A
    # ============================================================
    log.info("=" * 70)
    log.info("MODEL A: Assessment-aware AVM")
    log.info("=" * 70)

    log.info("Training baselines for Model A")
    train_baselines(X_train_A, X_test_A, y_train, y_test, EXPERIMENT_A, "modelA")

    log.info("Tuning LightGBM for Model A")
    best_A, grid_A = cv_tune_lightgbm(X_train_A, y_train, skip_tuning)

    log.info("Cross-validating Model A with best params")
    cv_A = cv_evaluate(X_train_A, y_train, best_A)

    log.info("Training final Model A")
    model_A, metrics_A = train_final_lightgbm(
        X_train_A,
        X_test_A,
        y_train,
        y_test,
        best_A,
        EXPERIMENT_A,
        "modelA",
        output_dir,
        cv_A,
    )

    # ============================================================
    # MODEL B
    # ============================================================
    log.info("=" * 70)
    log.info("MODEL B: Fundamentals-only AVM")
    log.info("=" * 70)

    log.info("Training baselines for Model B")
    train_baselines(X_train_B, X_test_B, y_train, y_test, EXPERIMENT_B, "modelB")

    log.info("Tuning LightGBM for Model B")
    best_B, grid_B = cv_tune_lightgbm(X_train_B, y_train, skip_tuning)

    log.info("Cross-validating Model B with best params")
    cv_B = cv_evaluate(X_train_B, y_train, best_B)

    log.info("Training final Model B")
    model_B, metrics_B = train_final_lightgbm(
        X_train_B,
        X_test_B,
        y_train,
        y_test,
        best_B,
        EXPERIMENT_B,
        "modelB",
        output_dir,
        cv_B,
    )

    # ============================================================
    # Save final models
    # ============================================================
    model_A.save_model(str(output_dir / "model_A.txt"))
    model_B.save_model(str(output_dir / "model_B.txt"))
    log.info(f"Saved final models to {output_dir}/model_A.txt and model_B.txt")

    # Pickle bundles for one-step inference
    bundle_A = {
        "model": model_A,
        "feature_cols": list(X_train_A.columns),
        "target_encoder": encoder,
        "train_categories": train_cats,
        "model_label": "Model A (Assessment-aware)",
        "metrics": metrics_A,
        "cv_summary": cv_A.to_dict(),
    }
    bundle_B = {
        "model": model_B,
        "feature_cols": list(X_train_B.columns),
        "target_encoder": encoder,
        "train_categories": train_cats,
        "model_label": "Model B (Fundamentals-only)",
        "metrics": metrics_B,
        "cv_summary": cv_B.to_dict(),
    }
    with open(output_dir / "model_A.pkl", "wb") as f:
        pickle.dump(bundle_A, f)
    with open(output_dir / "model_B.pkl", "wb") as f:
        pickle.dump(bundle_B, f)
    log.info(f"Saved model bundles to {output_dir}/model_A.pkl and model_B.pkl")

    # ============================================================
    # Summary
    # ============================================================
    log.info("=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    log.info(
        f"Model A — Test:  MAE ${metrics_A['MAE']:>10,.0f} | "
        f"MAPE {metrics_A['MAPE']:>5.1f}% | R² {metrics_A['R2']:>5.3f}"
    )
    log.info(
        f"Model A — CV:    MAE ${cv_A.loc['mean','MAE']:,.0f} ± ${cv_A.loc['std','MAE']:,.0f} | "
        f"MAPE {cv_A.loc['mean','MAPE']:.1f}% ± {cv_A.loc['std','MAPE']:.1f}%"
    )
    log.info(
        f"Model B — Test:  MAE ${metrics_B['MAE']:>10,.0f} | "
        f"MAPE {metrics_B['MAPE']:>5.1f}% | R² {metrics_B['R2']:>5.3f}"
    )
    log.info(
        f"Model B — CV:    MAE ${cv_B.loc['mean','MAE']:,.0f} ± ${cv_B.loc['std','MAE']:,.0f} | "
        f"MAPE {cv_B.loc['mean','MAPE']:.1f}% ± {cv_B.loc['std','MAPE']:.1f}%"
    )
    log.info("")
    log.info(f"To view full MLflow UI:  mlflow ui --backend-store-uri {mlflow_uri}")


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AlloyTower AVM models (Model A and Model B) end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("alloy_data.csv"),
        help="Path to raw alloy_data.csv input",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for cleaned data, models, and plots",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default="sqlite:///mlflow.db",
        help="MLflow tracking URI (SQLite default)",
    )
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter grid search; use defaults (faster)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        log.error(f"Input file not found: {args.input}")
        return 1
    try:
        run_pipeline(args.input, args.output_dir, args.mlflow_uri, args.skip_tuning)
    except Exception as e:
        log.exception(f"Pipeline failed: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
