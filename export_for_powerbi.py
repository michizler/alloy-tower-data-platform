"""
AlloyTower Power BI Export — Simple
====================================

Takes raw source data (source_data/alloy_data.csv) and produces a CSV for
Power BI import that preserves every original row and column, plus two
derived analytics fields:

  median_price_per_sqft_group  Median price/sqft for properties in the same
                               state and property_type — a benchmark each
                               row can be compared against.
  assessment_flag              "Under-assessed" | "Over-assessed" |
                               "Within band" | "Unknown" — based on the
                               sale-to-assessed ratio.

Usage
-----
    python build_powerbi_export.py
    python build_powerbi_export.py --input source_data/alloy_data.csv \
                                   --output alloy_for_powerbi.csv

Author: Bright Uzosike
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# Configuration
# ============================================================================

# Assessment-flag thresholds. Mean ratio in the dataset is ~1.05, std 0.09.
# Within ±10% of 1.0 is treated as 'within band'; outside is flagged.
# 'Under-assessed' = sale price > 110% of assessed value (assessor undervalued)
# 'Over-assessed'  = sale price <  90% of assessed value (assessor overvalued)
ASSESSMENT_BAND_LOW = 0.90
ASSESSMENT_BAND_HIGH = 1.10


# ============================================================================
# Logging
# ============================================================================


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("powerbi_export")
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
# Data loading
# ============================================================================


def load_raw(path: Path) -> pd.DataFrame:
    """Load raw alloy_data.csv — semicolon-delimited, UTF-8 with BOM."""
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
    df.columns = df.columns.str.replace("\ufeff", "", regex=False).str.strip()
    log.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ============================================================================
# Field construction
# ============================================================================


def build_group_median(df: pd.DataFrame) -> pd.DataFrame:
    """Add median_price_per_sqft_group: median of price_per_sqft within each
    state × property_type combination.

    This gives every row a benchmark it can be compared against in Power BI:
    'this property's price/sqft vs. typical for its category in this state'.
    """
    df = df.copy()
    if "price_per_sqft" not in df.columns:
        log.warning(
            "price_per_sqft not in source data — deriving from "
            "last_sale_price / sqft"
        )
        df["price_per_sqft"] = (df["last_sale_price"] / df["sqft"]).round(2)

    group_medians = df.groupby(["state", "property_type"])["price_per_sqft"].transform(
        "median"
    )
    df["median_price_per_sqft_group"] = group_medians.round(2)
    return df


def build_assessment_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add assessment_flag based on the sale-to-assessed ratio.

    Naming convention:
      Under-assessed = sale price > 110% of assessed value (assessor's value low)
      Over-assessed  = sale price <  90% of assessed value (assessor's value high)
      Within band    = sale within ±10% of assessed value
      Unknown        = assessed_value missing or zero
    """
    df = df.copy()
    safe_assessed = df["assessed_value"].replace(0, np.nan)
    ratio = df["last_sale_price"] / safe_assessed

    conditions = [
        ratio > ASSESSMENT_BAND_HIGH,
        ratio < ASSESSMENT_BAND_LOW,
    ]
    choices = ["Under-assessed", "Over-assessed"]
    df["assessment_flag"] = np.select(conditions, choices, default="Within band")
    df.loc[ratio.isna(), "assessment_flag"] = "Unknown"

    return df


# ============================================================================
# Pipeline orchestration
# ============================================================================


def run_export(input_path: Path, output_path: Path) -> None:
    df = load_raw(input_path)

    log.info("Computing median_price_per_sqft_group (state × property_type)")
    df = build_group_median(df)

    log.info("Computing assessment_flag")
    df = build_assessment_flag(df)

    # Write CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info(f"Wrote {len(df):,} rows × {df.shape[1]} columns to {output_path}")

    # Summary
    log.info("=" * 70)
    log.info("EXPORT SUMMARY")
    log.info("=" * 70)

    log.info("\nAssessment flag distribution:")
    flag_counts = df["assessment_flag"].value_counts()
    for flag, count in flag_counts.items():
        pct = 100 * count / len(df)
        log.info(f"  {flag:<20} {count:>5,} rows ({pct:.1f}%)")

    log.info(f"\nGroup median price/sqft (state × type):")
    n_groups = df.groupby(["state", "property_type"]).ngroups
    log.info(f"  Distinct groups:  {n_groups}")
    log.info(
        f"  Range:            ${df['median_price_per_sqft_group'].min():,.2f} – "
        f"${df['median_price_per_sqft_group'].max():,.2f}"
    )
    log.info(f"  Overall median:   ${df['median_price_per_sqft_group'].median():,.2f}")

    log.info("")
    log.info(f"Hand off {output_path.name} to the DA team for Power BI import.")


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the AlloyTower Power BI export from raw source data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("source_data/alloy_data.csv"),
        help="Path to raw alloy_data.csv input",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("alloy_for_powerbi.csv"),
        help="Path to write the Power BI CSV",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        log.error(f"Input file not found: {args.input}")
        return 1
    try:
        run_export(args.input, args.output)
    except Exception as e:
        log.exception(f"Export failed: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
