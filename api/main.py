"""
AlloyTower AVM API
==================

FastAPI service exposing the trained AVM models and similar-property lookup
to any HTTP client.

Endpoints
---------
  GET  /health         Liveness check
  GET  /metadata       Model versions, MAPE, training info
  POST /predict        Single property valuation + SHAP explanation
  POST /predict/batch  Batch scoring (up to 100 rows)
  POST /comparables    Similar-property lookup (returns N most similar)

Run locally:
    uvicorn api.main:app --reload --port 8000

Run in production (Render handles this via the Dockerfile CMD):
    uvicorn api.main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict

from api.predictor import AVMPredictor

# Make the similar_properties module importable when the API runs from the
# project root. It sits alongside the api/ directory.
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from similar_properties import SimilarPropertiesFinder
except ImportError:
    SimilarPropertiesFinder = None  # graceful degradation if module missing


# ============================================================================
# Configuration
# ============================================================================


def find_artifacts_dir() -> Path:
    """Search common locations for model bundles. Works whether artifacts/
    sits at project root or under model/models/ (the actual repo layout)."""
    project_root = Path(__file__).resolve().parents[1]
    for sub in ["artifacts", "model/models", "models", "data"]:
        candidate = project_root / sub
        if (candidate / "model_A.pkl").exists():
            return candidate
    # Fallback — Render will fail health check with a clear error
    return project_root / "artifacts"


ARTIFACTS_DIR = find_artifacts_dir()
MODEL_A_PATH = ARTIFACTS_DIR / "model_A.pkl"
MODEL_B_PATH = ARTIFACTS_DIR / "model_B.pkl"
CLEAN_CSV_PATH = ARTIFACTS_DIR / "alloy_clean.csv"

# CORS — allow the Streamlit Cloud URL plus localhost for local dev.
# Add additional origins as comma-separated values in the EXTRA_CORS_ORIGINS env var.
DEFAULT_ORIGINS = [
    "https://alloy-avm.streamlit.app",  # production Streamlit Cloud frontend
    "http://localhost:8501",  # local Streamlit
    "http://127.0.0.1:8501",  # local Streamlit alt
]
extra = os.environ.get("EXTRA_CORS_ORIGINS", "").strip()
ALLOWED_ORIGINS = DEFAULT_ORIGINS + (
    [o.strip() for o in extra.split(",") if o.strip()] if extra else []
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("avm_api")


# ============================================================================
# Request / response schemas
# ============================================================================


class PropertyInput(BaseModel):
    """A single property the user wants to value."""

    state: str = Field(..., examples=["CA"], description="2-letter US state code")
    city: str = Field(..., examples=["San Francisco"])
    county: str = Field(..., examples=["San Francisco"])
    property_type: str = Field(
        ...,
        examples=["Condo"],
        description="Single Family, Condo, Townhouse, or Multi Family",
    )
    sqft: int = Field(..., gt=200, lt=30000, examples=[1200])
    lot_size_sqft: int = Field(..., ge=0, examples=[0])
    bedrooms: int = Field(..., ge=0, le=15, examples=[2])
    bathrooms: float = Field(..., ge=0, le=15, examples=[2.0])
    year_built: int = Field(..., ge=1800, le=2026, examples=[2010])
    owner_occupied: bool = Field(..., examples=[True])
    assessed_value: Optional[float] = Field(
        None,
        ge=0,
        description="Optional: if provided, Model A (assessment-aware) is used.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "state": "CA",
                "city": "San Francisco",
                "county": "San Francisco",
                "property_type": "Condo",
                "sqft": 1200,
                "lot_size_sqft": 0,
                "bedrooms": 2,
                "bathrooms": 2.0,
                "year_built": 2010,
                "owner_occupied": True,
                "assessed_value": 1100000,
            }
        }
    )


class FeatureContributionOut(BaseModel):
    feature: str
    display_name: str
    value: str
    contribution: float
    direction: str


class PredictionOut(BaseModel):
    estimated_value: float
    lower_bound: float
    upper_bound: float
    mape: float
    model_used: str
    contributions: list[FeatureContributionOut]


class BatchPredictionRequest(BaseModel):
    properties: list[PropertyInput] = Field(..., max_length=100)


class ComparablesRequest(BaseModel):
    query: PropertyInput
    n: int = Field(5, ge=1, le=20)
    target_price: Optional[float] = Field(
        None, gt=0, description="Optional: restricts to ±20% band"
    )
    price_tolerance: float = Field(0.20, ge=0.05, le=1.0)


class ComparableOut(BaseModel):
    property_id: str
    city: str
    state: str
    property_type: str
    sqft: int
    bedrooms: int
    bathrooms: float
    year_built: int
    last_sale_price: float
    similarity_score: float


# ============================================================================
# App
# ============================================================================

app = FastAPI(
    title="AlloyTower AVM API",
    description="Property valuation and similar-property lookup for the AlloyTower platform.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globals — set on startup
predictor: Optional[AVMPredictor] = None
finder: Optional[SimilarPropertiesFinder] = None


@app.on_event("startup")
def load_artifacts() -> None:
    global predictor, finder

    log.info(f"Looking for artifacts in: {ARTIFACTS_DIR}")
    log.info(f"CORS allowed origins: {ALLOWED_ORIGINS}")

    if not MODEL_A_PATH.exists() or not MODEL_B_PATH.exists():
        log.error(f"Model bundles not found at {MODEL_A_PATH} or {MODEL_B_PATH}.")
        raise RuntimeError("Model bundles missing — check your artifacts directory")

    log.info("Loading AVM models...")
    predictor = AVMPredictor(MODEL_A_PATH, MODEL_B_PATH)
    log.info(
        f"Loaded Model A (MAPE {predictor.mape_a*100:.1f}%) "
        f"and Model B (MAPE {predictor.mape_b*100:.1f}%)"
    )

    if SimilarPropertiesFinder is not None and CLEAN_CSV_PATH.exists():
        log.info(f"Loading comparables index from {CLEAN_CSV_PATH}...")
        finder = SimilarPropertiesFinder.from_csv(CLEAN_CSV_PATH)
        log.info(f"Indexed {len(finder.df):,} properties for similarity lookup")
    else:
        log.warning(
            "Comparables service disabled: similar_properties module or "
            "alloy_clean.csv not available."
        )


# ============================================================================
# Endpoints
# ============================================================================


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "predictor_loaded": predictor is not None,
        "comparables_loaded": finder is not None,
    }


@app.get("/metadata")
def metadata() -> dict:
    if predictor is None:
        raise HTTPException(503, "Predictor not loaded")
    return predictor.metadata()


@app.post("/predict", response_model=PredictionOut)
def predict(prop: PropertyInput) -> dict:
    if predictor is None:
        raise HTTPException(503, "Predictor not loaded")
    try:
        result = predictor.predict(prop.model_dump())
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(500, f"Prediction failed: {e}")

    return {
        "estimated_value": result.estimated_value,
        "lower_bound": result.lower_bound,
        "upper_bound": result.upper_bound,
        "mape": result.mape,
        "model_used": result.model_used,
        "contributions": [
            {
                "feature": c.feature,
                "display_name": c.display_name,
                "value": c.value,
                "contribution": c.contribution,
                "direction": c.direction,
            }
            for c in result.contributions
        ],
    }


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest) -> dict:
    if predictor is None:
        raise HTTPException(503, "Predictor not loaded")
    results = []
    for prop in request.properties:
        try:
            r = predictor.predict(prop.model_dump())
            results.append(
                {
                    "estimated_value": r.estimated_value,
                    "lower_bound": r.lower_bound,
                    "upper_bound": r.upper_bound,
                    "model_used": r.model_used,
                }
            )
        except Exception as e:
            results.append({"error": str(e)})
    return {"results": results}


@app.post("/comparables", response_model=list[ComparableOut])
def comparables(request: ComparablesRequest) -> list[dict]:
    if finder is None:
        raise HTTPException(503, "Comparables service not available")
    query = request.query.model_dump()
    matches = finder.find_similar_by_attrs(
        query,
        n=request.n,
        target_price=request.target_price,
        price_tolerance=request.price_tolerance,
    )
    return [
        {
            "property_id": m.property_id,
            "city": m.city,
            "state": m.state,
            "property_type": m.property_type,
            "sqft": m.sqft,
            "bedrooms": m.bedrooms,
            "bathrooms": m.bathrooms,
            "year_built": m.year_built,
            "last_sale_price": m.last_sale_price,
            "similarity_score": round(1 / (1 + m.distance), 3),
        }
        for m in matches
    ]
