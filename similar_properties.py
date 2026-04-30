"""
AlloyTower Similar Properties Lookup
=====================================

Content-based property similarity search using attribute matching.
Implements FR-009 (personalised property recommendations) as the optional
Phase 1 stretch deliverable.

Approach
--------
Given a query property, returns the N most similar properties in the dataset
based on a weighted similarity score over:
  - Categorical match (state, city, property_type, owner_occupied)
  - Numeric distance (sqft, bedrooms, bathrooms, year_built, lot_size_sqft)
  - Optional price-band constraint (e.g., within ±20% of a target price)

This is content-based, not user-behaviour-based, because the dataset has no
user activity log. With user data in Phase 2, this could be augmented with
collaborative filtering.

Usage
-----
    from similar_properties import SimilarPropertiesFinder

    finder = SimilarPropertiesFinder.from_csv("model/models/alloy_clean.csv")

    # Find 5 properties similar to property_id "P-100695"
    matches = finder.find_similar_by_id("P-100695", n=5)

    # Find 5 properties similar to a custom property spec
    query = {
        "state": "CA", "city": "San Francisco", "property_type": "Condo",
        "sqft": 1200, "bedrooms": 2, "bathrooms": 2.0, "year_built": 2010,
        "lot_size_sqft": 0, "owner_occupied": True,
    }
    matches = finder.find_similar_by_attrs(query, n=5)

    # Constrained by price band
    matches = finder.find_similar_by_attrs(
        query, n=5, target_price=1_500_000, price_tolerance=0.20
    )

Run as a script for a quick demonstration:
    python similar_properties.py

Author: Bright Uzosike
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================================
# Configuration
# ============================================================================

# Numeric features used in the similarity calculation. Each contributes
# equally after standardisation; weights below adjust their relative pull.
NUMERIC_FEATURES = ["sqft", "lot_size_sqft", "bedrooms", "bathrooms", "year_built"]

# Categorical features used in the similarity calculation.
# Match = 0 distance contribution; mismatch = penalty (set in CATEGORICAL_PENALTIES).
CATEGORICAL_FEATURES = ["state", "city", "property_type", "owner_occupied"]

# How much each numeric feature counts (after standardisation).
# Higher = matters more for similarity. Sqft is weighted highest because it's
# the most concrete physical signal.
NUMERIC_WEIGHTS = {
    "sqft": 2.0,
    "lot_size_sqft": 1.0,
    "bedrooms": 1.5,
    "bathrooms": 1.5,
    "year_built": 1.0,
}

# Categorical mismatch penalties. Larger = more important that they match.
# State and property_type are the strongest filters — a 3-bed condo in CA
# is fundamentally different from a 3-bed single-family in TX.
CATEGORICAL_PENALTIES = {
    "state": 5.0,
    "city": 2.5,
    "property_type": 4.0,
    "owner_occupied": 0.5,
}


# ============================================================================
# Data class
# ============================================================================

@dataclass
class SimilarMatch:
    """A single property match with its similarity score."""
    property_id: str
    city: str
    state: str
    property_type: str
    sqft: int
    bedrooms: int
    bathrooms: float
    year_built: int
    last_sale_price: float
    distance: float  # lower = more similar

    def to_dict(self) -> dict:
        return {
            "property_id": self.property_id,
            "location": f"{self.city}, {self.state}",
            "property_type": self.property_type,
            "sqft": self.sqft,
            "bedrooms": self.bedrooms,
            "bathrooms": self.bathrooms,
            "year_built": self.year_built,
            "last_sale_price": f"${self.last_sale_price:,.0f}",
            "similarity_score": round(1 / (1 + self.distance), 3),
        }


# ============================================================================
# Finder class
# ============================================================================

class SimilarPropertiesFinder:
    """Index property attributes and return the most similar properties to a query."""

    def __init__(self, df: pd.DataFrame):
        # Defensive: make sure required columns exist
        required = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["property_id", "last_sale_price"])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        self.df = df.reset_index(drop=True).copy()

        # Standardise numeric features so they contribute on a comparable scale
        self.scaler = StandardScaler()
        self.numeric_matrix = self.scaler.fit_transform(self.df[NUMERIC_FEATURES])

        # Apply weights AFTER standardisation
        weights = np.array([NUMERIC_WEIGHTS[f] for f in NUMERIC_FEATURES])
        self.numeric_matrix = self.numeric_matrix * weights

    # ---- Public API ----

    @classmethod
    def from_csv(cls, csv_path: str | Path) -> "SimilarPropertiesFinder":
        df = pd.read_csv(csv_path)
        return cls(df)

    def find_similar_by_id(self, property_id: str, n: int = 5) -> list[SimilarMatch]:
        """Return the n most similar properties to the given property_id."""
        match = self.df[self.df["property_id"] == property_id]
        if match.empty:
            raise KeyError(f"property_id not found: {property_id}")
        query = match.iloc[0].to_dict()
        return self.find_similar_by_attrs(query, n=n, exclude_property_id=property_id)

    def find_similar_by_attrs(
        self,
        query: dict,
        n: int = 5,
        target_price: float | None = None,
        price_tolerance: float = 0.20,
        exclude_property_id: str | None = None,
    ) -> list[SimilarMatch]:
        """Return the n most similar properties to the query attributes.

        Parameters
        ----------
        query : dict
            Dict with at minimum the columns listed in NUMERIC_FEATURES and
            CATEGORICAL_FEATURES. Missing keys default to dataset means.
        n : int
            Number of matches to return.
        target_price : float, optional
            If provided, restrict matches to within `price_tolerance` of this price.
        price_tolerance : float
            Fractional band around target_price (default 0.20 = ±20%).
        exclude_property_id : str, optional
            Useful when searching by ID — excludes the property itself from results.
        """
        # ---- Build query vectors ----
        # Numeric: scale and weight the same way as the index
        q_numeric_raw = np.array([[query.get(f, self.df[f].mean()) for f in NUMERIC_FEATURES]])
        q_numeric = self.scaler.transform(q_numeric_raw)
        weights = np.array([NUMERIC_WEIGHTS[f] for f in NUMERIC_FEATURES])
        q_numeric = q_numeric * weights

        # Numeric distance: Euclidean (this is L2; could swap for L1 if outliers worry you)
        numeric_dist = np.sqrt(((self.numeric_matrix - q_numeric) ** 2).sum(axis=1))

        # Categorical: 0 if match, penalty if mismatch
        cat_dist = np.zeros(len(self.df))
        for feat in CATEGORICAL_FEATURES:
            if feat in query:
                mismatch_mask = self.df[feat].astype(str) != str(query[feat])
                cat_dist = cat_dist + mismatch_mask.values * CATEGORICAL_PENALTIES[feat]

        total_dist = numeric_dist + cat_dist

        # ---- Apply optional price band ----
        if target_price is not None:
            lo = target_price * (1 - price_tolerance)
            hi = target_price * (1 + price_tolerance)
            in_band = (self.df["last_sale_price"] >= lo) & (self.df["last_sale_price"] <= hi)
            total_dist = np.where(in_band, total_dist, np.inf)

        # ---- Exclude the query property if searching by ID ----
        if exclude_property_id is not None:
            same = (self.df["property_id"] == exclude_property_id).values
            total_dist = np.where(same, np.inf, total_dist)

        # ---- Pick top-n ----
        top_idx = np.argsort(total_dist)[:n]
        results = []
        for idx in top_idx:
            if not np.isfinite(total_dist[idx]):
                continue  # exhausted matches (e.g., price band too tight)
            row = self.df.iloc[idx]
            results.append(SimilarMatch(
                property_id=str(row["property_id"]),
                city=str(row["city"]),
                state=str(row["state"]),
                property_type=str(row["property_type"]),
                sqft=int(row["sqft"]),
                bedrooms=int(row["bedrooms"]),
                bathrooms=float(row["bathrooms"]),
                year_built=int(row["year_built"]),
                last_sale_price=float(row["last_sale_price"]),
                distance=float(total_dist[idx]),
            ))
        return results


# ============================================================================
# Demonstration script
# ============================================================================

def _demo() -> None:
    """Quick demo when run as a script. Expects artifacts/alloy_clean.csv."""
    csv_path = Path("model/models/alloy_clean.csv")
    if not csv_path.exists():
        # Fall back to a few common locations
        for alt in [Path("alloy_clean.csv"), Path("../model/alloy_data_cleaned.csv")]:
            if alt.exists():
                csv_path = alt
                break
        else:
            print(f"Could not find alloy_clean.csv. Run train_avm.py first.")
            return

    print(f"Loading {csv_path}...")
    finder = SimilarPropertiesFinder.from_csv(csv_path)
    print(f"Indexed {len(finder.df):,} properties.\n")

    # Demo 1: find similar to a specific property by ID
    sample_id = finder.df.iloc[0]["property_id"]
    sample_row = finder.df.iloc[0]
    print(f"--- Demo 1: properties similar to {sample_id} ---")
    print(f"Query: {sample_row['city']}, {sample_row['state']} | {sample_row['property_type']} | "
          f"{sample_row['sqft']} sqft | {sample_row['bedrooms']}bd/{sample_row['bathrooms']}ba | "
          f"${sample_row['last_sale_price']:,.0f}\n")

    matches = finder.find_similar_by_id(sample_id, n=5)
    for i, m in enumerate(matches, 1):
        d = m.to_dict()
        print(f"  {i}. {d['property_id']} | {d['location']} | {d['property_type']} | "
              f"{d['sqft']} sqft | {d['bedrooms']}bd/{d['bathrooms']}ba | "
              f"{d['last_sale_price']} | similarity={d['similarity_score']}")

    # Demo 2: find similar to a custom query, with price band
    print("\n--- Demo 2: custom query + price band ---")
    query = {
        "state": "CA", "city": "San Francisco", "property_type": "Condo",
        "sqft": 1200, "bedrooms": 2, "bathrooms": 2.0, "year_built": 2010,
        "lot_size_sqft": 0, "owner_occupied": True,
    }
    print(f"Query: CA Condo, 1200 sqft, 2bd/2ba, target ~$1.5M ± 20%\n")
    matches = finder.find_similar_by_attrs(
        query, n=5, target_price=1_500_000, price_tolerance=0.20
    )
    if not matches:
        print("  No matches in the requested price band. Try widening the tolerance.")
    else:
        for i, m in enumerate(matches, 1):
            d = m.to_dict()
            print(f"  {i}. {d['property_id']} | {d['location']} | {d['property_type']} | "
                  f"{d['sqft']} sqft | {d['bedrooms']}bd/{d['bathrooms']}ba | "
                  f"{d['last_sale_price']} | similarity={d['similarity_score']}")


if __name__ == "__main__":
    _demo()