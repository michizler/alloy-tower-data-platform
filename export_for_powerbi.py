"""
AlloyTower Power BI Export
==========================

Produces a single CSV for the DA team to import into the Power BI star schema.

Takes the cleaned dataset and the two trained models (from train_avm.py),
adds the five DS-owned fields described in the Day 9 Validation Response
Section 5, and writes the result to artifacts/alloy_for_powerbi.csv.

Fields added
------------
  avm_value_modelA      Predicted property value, assessment-aware AVM.
  avm_value_modelB      Predicted property value, fundamentals-only AVM.
  assessment_ratio      last_sale_price / assessed_value.
  assessment_flag       Categorical: "Under-assessed", "Over-assessed", or "Within band".
  avm_residual_pct      % difference between actual sale price and Model A prediction.

Usage
-----
    python export_for_powerbi.py
    python export_for_powerbi.py --input artifacts/alloy_clean.csv --output-dir artifacts/

Author: Bright Uzosike
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

# Assessment-flag thresholds. A property is flagged if its sale-to-assessed
# ratio falls outside this band. The dataset's mean ratio is 1.05 with std 0.09,
# so ±0.10 around 1.00 captures the bulk of normal-range properties and
# isolates genuine outliers.
ASSESSMENT_BAND_LOW = 0.90
ASSESSMENT_BAND_HIGH = 1.10

# AVM residual is reported as a percentage of actual price. Stored as a number
# (e.g., 12.5 means the prediction was 12.5% off). Sign indicates direction:
# positive = model under-predicted; negative = model over-predicted.

# ============================================================================
# Logging
# ============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("powerbi_export")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s",
                                               datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    return logger

log = setup_logging()


# ============================================================================
# Feature pipeline (mirrors train_avm.py)
# ============================================================================

def apply_feature_pipeline(df: pd.DataFrame, encoder, train_categories: dict) -> pd.DataFrame:
    """Replicate the feature pipeline used during training so model.predict works."""
    X = df.copy()

    # Target encoding for state, city, county
    target_encode_cols = ["state", "city", "county"]
    X[target_encode_cols] = encoder.transform(X[target_encode_cols])

    # Native categoricals
    for col in ["property_type", "owner_occupied"]:
        X[col] = pd.Categorical(X[col], categories=train_categories[col])

    # Interaction features (same as train_avm.py)
    X["sqft_per_bedroom"] = X["sqft"] / X["bedrooms"].replace(0, np.nan)
    X["sqft_per_bedroom"] = X["sqft_per_bedroom"].fillna(X["sqft"])
    X["state_x_sqft"] = X["state"] * X["sqft"]
    X["building_age"] = 2024 - X["year_built"]

    return X


# ============================================================================
# Field construction
# ============================================================================

def build_assessment_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Compute assessment_ratio and assessment_flag columns."""
    df = df.copy()

    # Avoid div-by-zero defensively (shouldn't occur in cleaned data)
    safe_assessed = df["assessed_value"].replace(0, np.nan)
    df["assessment_ratio"] = (df["last_sale_price"] / safe_assessed).round(4)

    # Categorical flag based on the band
    conditions = [
        df["assessment_ratio"] > ASSESSMENT_BAND_HIGH,
        df["assessment_ratio"] < ASSESSMENT_BAND_LOW,
    ]
    choices = ["Under-assessed", "Over-assessed"]
    df["assessment_flag"] = np.select(conditions, choices, default="Within band")

    # If ratio couldn't be computed (e.g. assessed_value is 0/null), flag accordingly
    df.loc[df["assessment_ratio"].isna(), "assessment_flag"] = "Unknown"

    return df


def build_avm_predictions(df: pd.DataFrame, bundle_A: dict, bundle_B: dict) -> pd.DataFrame:
    """Run both models against the dataset and add prediction columns."""
    df = df.copy()

    # Model A pipeline
    X_A = apply_feature_pipeline(df, bundle_A["target_encoder"], bundle_A["train_categories"])
    X_A = X_A[bundle_A["feature_cols"]]
    df["avm_value_modelA"] = np.expm1(bundle_A["model"].predict(X_A)).round(2)

    # Model B pipeline
    X_B = apply_feature_pipeline(df, bundle_B["target_encoder"], bundle_B["train_categories"])
    X_B = X_B[bundle_B["feature_cols"]]
    df["avm_value_modelB"] = np.expm1(bundle_B["model"].predict(X_B)).round(2)

    # Residual: % difference between actual sale price and Model A prediction.
    # Positive = model under-predicted; negative = model over-predicted.
    df["avm_residual_pct"] = (
        (df["last_sale_price"] - df["avm_value_modelA"]) / df["last_sale_price"] * 100
    ).round(2)

    return df


# ============================================================================
# Pipeline orchestration
# ============================================================================

def export_for_powerbi(input_path: Path, output_dir: Path,
                       model_a_path: Path, model_b_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    log.info(f"Loading cleaned dataset from {input_path}")
    df = pd.read_csv(input_path)
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")

    # ---- Load model bundles ----
    log.info(f"Loading Model A from {model_a_path}")
    with open(model_a_path, "rb") as f:
        bundle_A = pickle.load(f)
    log.info(f"Loading Model B from {model_b_path}")
    with open(model_b_path, "rb") as f:
        bundle_B = pickle.load(f)

    # ---- Build the new fields ----
    log.info("Computing assessment_ratio and assessment_flag")
    df = build_assessment_fields(df)

    log.info("Running Model A and Model B predictions")
    df = build_avm_predictions(df, bundle_A, bundle_B)

    # ---- Reorder columns: keep originals first, new DS fields at the end ----
    new_fields = ["avm_value_modelA", "avm_value_modelB", "assessment_ratio",
                  "assessment_flag", "avm_residual_pct"]
    original_cols = [c for c in df.columns if c not in new_fields]
    df = df[original_cols + new_fields]

    # ---- Write CSV ----
    output_path = output_dir / "alloy_for_powerbi.csv"
    df.to_csv(output_path, index=False)
    log.info(f"Wrote {len(df):,} rows to {output_path}")

    # ---- Summary stats for the SA / DA ----
    log.info("=" * 70)
    log.info("EXPORT SUMMARY — fields added")
    log.info("=" * 70)

    # Assessment flag distribution
    log.info("\nAssessment flag distribution:")
    flag_counts = df["assessment_flag"].value_counts()
    for flag, count in flag_counts.items():
        pct = 100 * count / len(df)
        log.info(f"  {flag:<20} {count:>5,} rows ({pct:.1f}%)")

    # Assessment ratio summary
    log.info(f"\nAssessment ratio (sale / assessed):")
    log.info(f"  Mean:   {df['assessment_ratio'].mean():.3f}")
    log.info(f"  Median: {df['assessment_ratio'].median():.3f}")
    log.info(f"  Std:    {df['assessment_ratio'].std():.3f}")

    # AVM residual summary
    log.info(f"\nAVM Model A residual (% of actual price):")
    log.info(f"  Mean abs residual: {df['avm_residual_pct'].abs().mean():.2f}%")
    log.info(f"  Median abs:        {df['avm_residual_pct'].abs().median():.2f}%")
    log.info(f"  Max abs:           {df['avm_residual_pct'].abs().max():.2f}%")

    # Outliers worth investigating (>20% off)
    big_residuals = df[df["avm_residual_pct"].abs() > 20]
    log.info(f"\nProperties with Model A residual > ±20%: {len(big_residuals):,} "
             f"({100 * len(big_residuals) / len(df):.1f}%)")
    log.info("  These are candidates for the Outlier Detection panel in Power BI.")

    log.info("")
    log.info(f"Hand off {output_path.name} to the DA team for import into the star schema.")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export AlloyTower cleaned dataset + DS fields for Power BI import.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=Path("model/models/alloy_clean.csv"),
                        help="Path to cleaned dataset CSV (output of train_avm.py)")
    parser.add_argument("--output-dir", type=Path, default=Path("model"),
                        help="Directory to write alloy_for_powerbi.csv")
    parser.add_argument("--model-a", type=Path, default=Path("model/models/model_A.pkl"),
                        help="Path to Model A bundle (.pkl)")
    parser.add_argument("--model-b", type=Path, default=Path("model/models/model_B.pkl"),
                        help="Path to Model B bundle (.pkl)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Validate inputs exist
    for path, name in [(args.input, "input CSV"),
                       (args.model_a, "Model A bundle"),
                       (args.model_b, "Model B bundle")]:
        if not path.exists():
            log.error(f"{name} not found: {path}")
            log.error("Did you run train_avm.py first?")
            return 1

    try:
        export_for_powerbi(args.input, args.output_dir, args.model_a, args.model_b)
    except Exception as e:
        log.exception(f"Export failed: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
