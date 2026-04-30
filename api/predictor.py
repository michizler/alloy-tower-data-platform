"""
Model wrapper for the AlloyTower AVM service.

Loads the trained Model A and Model B bundles produced by train_avm.py,
applies the same feature pipeline used during training, and returns
predictions plus SHAP explanations for the API to serve.

Single instance per FastAPI process — load on startup, reuse across requests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import pickle

import numpy as np
import pandas as pd
import shap


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class FeatureContribution:
    """One feature's contribution to a single prediction."""
    feature: str
    display_name: str           # human-readable label
    value: str                  # the feature's value as the user would see it
    contribution: float         # signed dollars: + pushed price up, - pushed it down
    direction: Literal["up", "down", "neutral"]


@dataclass
class Prediction:
    """A single property valuation result."""
    estimated_value: float
    lower_bound: float          # estimated_value × (1 - mape)
    upper_bound: float          # estimated_value × (1 + mape)
    mape: float                 # CV-based MAPE for this model variant, as fraction (e.g. 0.08)
    model_used: Literal["A", "B"]
    contributions: list[FeatureContribution]


# ============================================================================
# Predictor
# ============================================================================

class AVMPredictor:
    """Wraps Model A and Model B; chooses the right one based on input."""

    # Plain-language labels for features the stakeholder will see
    DISPLAY_NAMES = {
        "state": "State",
        "city": "City",
        "county": "County",
        "property_type": "Property type",
        "owner_occupied": "Owner-occupied",
        "sqft": "Living area (sqft)",
        "lot_size_sqft": "Lot size (sqft)",
        "bedrooms": "Bedrooms",
        "bathrooms": "Bathrooms",
        "year_built": "Year built",
        "assessed_value": "Assessed value",
        "sqft_per_bedroom": "Sqft per bedroom",
        "state_x_sqft": "Location × size",
        "building_age": "Building age",
    }

    def __init__(self, model_a_path: Path, model_b_path: Path):
        with open(model_a_path, "rb") as f:
            self.bundle_a = pickle.load(f)
        with open(model_b_path, "rb") as f:
            self.bundle_b = pickle.load(f)

        # Pre-build SHAP explainers — cache on the instance because TreeExplainer
        # construction takes a few seconds and we want every request to be fast.
        self._explainer_a = shap.TreeExplainer(self.bundle_a["model"])
        self._explainer_b = shap.TreeExplainer(self.bundle_b["model"])

        # MAPE values come from the bundle's CV summary; if not stored, fall back
        # to the figures we observed during training.
        self.mape_a = self._safe_get_cv_metric(self.bundle_a, "MAPE", default=8.5) / 100
        self.mape_b = self._safe_get_cv_metric(self.bundle_b, "MAPE", default=44.9) / 100

    @staticmethod
    def _safe_get_cv_metric(bundle: dict, metric: str, default: float) -> float:
        try:
            return float(bundle["cv_summary"]["mean"][metric])
        except (KeyError, TypeError):
            return default

    # ------------------------------------------------------------------
    # Feature pipeline — must mirror train_avm.py exactly
    # ------------------------------------------------------------------
    def _apply_pipeline(self, raw: dict, bundle: dict) -> pd.DataFrame:
        """Take a dict of raw user inputs and produce a single-row DataFrame
        with all engineered features in the order the model expects."""
        df = pd.DataFrame([raw])

        # Target encoding for state, city, county
        df[["state", "city", "county"]] = bundle["target_encoder"].transform(
            df[["state", "city", "county"]]
        )

        # Native categoricals — match training categories
        for col in ["property_type", "owner_occupied"]:
            df[col] = pd.Categorical(df[col], categories=bundle["train_categories"][col])

        # Interactions
        df["sqft_per_bedroom"] = df["sqft"] / df["bedrooms"].replace(0, np.nan)
        df["sqft_per_bedroom"] = df["sqft_per_bedroom"].fillna(df["sqft"])
        df["state_x_sqft"] = df["state"] * df["sqft"]
        df["building_age"] = 2024 - df["year_built"]

        return df[bundle["feature_cols"]]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict(self, raw: dict) -> Prediction:
        """Predict a property's value. Routes to Model A if assessed_value is
        provided and non-null; otherwise uses Model B."""
        use_model_a = raw.get("assessed_value") is not None
        bundle = self.bundle_a if use_model_a else self.bundle_b
        explainer = self._explainer_a if use_model_a else self._explainer_b
        mape = self.mape_a if use_model_a else self.mape_b

        # Drop assessed_value cleanly if Model B is being used
        raw_for_pipeline = {k: v for k, v in raw.items() if not (not use_model_a and k == "assessed_value")}

        # Model B doesn't have assessed_value in its feature list, so we need to
        # be sure we don't pass it through. The pipeline subset by feature_cols
        # handles that correctly.
        X = self._apply_pipeline(raw_for_pipeline, bundle)

        # Predict (in log-space, then exp back)
        y_log = bundle["model"].predict(X)
        y = float(np.expm1(y_log)[0])

        # SHAP contributions in log-space, then convert to dollar contributions
        # by linearising around the prediction. This is an approximation but
        # close enough for stakeholder-facing explanation.
        shap_values = explainer.shap_values(X)
        # shap_values is shape (1, n_features) for regression
        shap_row = shap_values[0]

        # Convert log-space SHAP to dollar-space: each SHAP value represents
        # a delta in log price. We multiply by the prediction to approximate
        # the dollar effect.
        contributions = []
        for feat, shap_val in zip(bundle["feature_cols"], shap_row):
            display_value = self._format_feature_value(feat, raw_for_pipeline)
            dollar_delta = float(shap_val) * y
            direction = "up" if dollar_delta > 1000 else "down" if dollar_delta < -1000 else "neutral"
            contributions.append(FeatureContribution(
                feature=feat,
                display_name=self.DISPLAY_NAMES.get(feat, feat),
                value=display_value,
                contribution=dollar_delta,
                direction=direction,
            ))
        # Sort by absolute contribution, descending
        contributions.sort(key=lambda c: abs(c.contribution), reverse=True)

        return Prediction(
            estimated_value=y,
            lower_bound=y * (1 - mape),
            upper_bound=y * (1 + mape),
            mape=mape,
            model_used="A" if use_model_a else "B",
            contributions=contributions,
        )

    def _format_feature_value(self, feat: str, raw: dict) -> str:
        """Format the raw input value back into a user-readable string."""
        # For target-encoded features (state, city, county), show the category name
        if feat in ("state", "city", "county"):
            return str(raw.get(feat, ""))
        if feat in ("sqft_per_bedroom", "state_x_sqft", "building_age"):
            return ""  # interaction features — no clean user-facing value
        if feat == "owner_occupied":
            return "Yes" if raw.get(feat) else "No"
        v = raw.get(feat)
        if v is None:
            return ""
        if isinstance(v, float):
            return f"{v:,.1f}" if feat == "bathrooms" else f"{v:,.0f}"
        if isinstance(v, int):
            return f"{v:,}"
        return str(v)

    # ------------------------------------------------------------------
    # Metadata for the /metadata endpoint
    # ------------------------------------------------------------------
    def metadata(self) -> dict:
        return {
            "model_a": {
                "label": "Assessment-aware AVM",
                "mape": round(self.mape_a * 100, 2),
                "n_features": len(self.bundle_a["feature_cols"]),
                "use_when": "A recent county assessment is available.",
            },
            "model_b": {
                "label": "Fundamentals-only AVM",
                "mape": round(self.mape_b * 100, 2),
                "n_features": len(self.bundle_b["feature_cols"]),
                "use_when": "No assessment is available (e.g., new builds, recent sales).",
            },
            "training_date": "2026-04-29",
            "training_rows": 1956,
            "data_caveats": [
                "Latitude, longitude, and ZIP fields excluded due to data quality issues.",
                "Sale dates excluded — temporal patterns in the dataset do not reflect real market trends.",
                "Physical attributes (bedrooms, bathrooms, sqft) carry weaker signal than typical AVMs.",
            ],
        }
