"""
About this model — the honesty page.

Plain-language explanation of how the AVM works, what its limitations are,
and how stakeholders should interpret its outputs. Builds trust by
admitting weaknesses upfront rather than hiding them.
"""

from __future__ import annotations

import streamlit as st

from app.lib import Colors, page_header, get_metadata, get_health, ApiError

# ============================================================================
# Page setup
# ============================================================================

st.set_page_config(page_title="About | AlloyTower AVM", page_icon="ℹ️", layout="wide")

page_header(
    "About this model",
    "How the AlloyTower AVM works, what it can do, and what it can't.",
)


# ============================================================================
# Live metadata from the API
# ============================================================================

try:
    meta = get_metadata()
    api_ok = True
except ApiError as e:
    api_ok = False
    meta = {}
    st.warning(
        f"Could not load live model metadata: {e.message}. Showing static information only."
    )


# ============================================================================
# Section 1: Plain-language overview
# ============================================================================

st.markdown("#### How it works, in plain terms")

st.markdown("""
The AlloyTower AVM (Automated Valuation Model) estimates a property's market
value from its attributes — location, size, type, age, and (when available)
the county's most recent assessment.

It is built using **gradient-boosted decision trees** — the same family of
models used by Zillow's Zestimate and most professional AVM providers. The
model was trained on **1,956 US residential property records** spanning
2016–2026, drawn from the AlloyTower platform's Phase 1 dataset.

The system actually contains **two model variants**, and the right one is
chosen automatically based on what you provide:
""")

# Two cards side by side describing each model
col1, col2 = st.columns(2, gap="medium")

with col1:
    if api_ok and "model_a" in meta:
        a = meta["model_a"]
        st.markdown(
            f'<div style="border: 1px solid {Colors.border}; border-radius: 8px; '
            f'padding: 16px 20px; height: 100%;">'
            f'<div style="font-size: 11px; color: {Colors.success}; '
            f"background: {Colors.success_bg}; padding: 2px 8px; border-radius: 10px; "
            f'display: inline-block; margin-bottom: 8px; font-weight: 600;">'
            f"MODEL A — RECOMMENDED"
            f"</div>"
            f'<h4 style="margin: 4px 0 8px;">{a["label"]}</h4>'
            f'<p style="font-size: 14px; color: {Colors.neutral}; margin: 0 0 12px;">'
            f'<b>When used:</b> {a["use_when"]}'
            f"</p>"
            f'<div style="font-size: 28px; font-weight: 600; color: {Colors.primary};">'
            f'±{a["mape"]:.1f}%'
            f"</div>"
            f'<div style="font-size: 12px; color: {Colors.neutral}; margin-top: 4px;">'
            f'typical prediction accuracy ({a["n_features"]} input features)'
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

with col2:
    if api_ok and "model_b" in meta:
        b = meta["model_b"]
        st.markdown(
            f'<div style="border: 1px solid {Colors.border}; border-radius: 8px; '
            f'padding: 16px 20px; height: 100%;">'
            f'<div style="font-size: 11px; color: {Colors.warning}; '
            f"background: {Colors.warning_bg}; padding: 2px 8px; border-radius: 10px; "
            f'display: inline-block; margin-bottom: 8px; font-weight: 600;">'
            f"MODEL B — FALLBACK"
            f"</div>"
            f'<h4 style="margin: 4px 0 8px;">{b["label"]}</h4>'
            f'<p style="font-size: 14px; color: {Colors.neutral}; margin: 0 0 12px;">'
            f'<b>When used:</b> {b["use_when"]}'
            f"</p>"
            f'<div style="font-size: 28px; font-weight: 600; color: {Colors.primary};">'
            f'±{b["mape"]:.1f}%'
            f"</div>"
            f'<div style="font-size: 12px; color: {Colors.neutral}; margin-top: 4px;">'
            f'typical prediction accuracy ({b["n_features"]} input features)'
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("")


# ============================================================================
# Section 2: What "MAPE" means
# ============================================================================

st.markdown('#### What does "±8% accuracy" actually mean?')

st.markdown(f"""
The number above each model is its **Mean Absolute Percentage Error (MAPE)** —
a measure of how far off, on average, the model's predictions are.

**Plain example:** if a property actually sold for $500,000 and Model A
predicted $460,000, the prediction was off by 8%. If you ran the model on
hundreds of properties and averaged the percentage errors, you'd get the
MAPE figure shown above.

**Industry benchmarks for AVMs:**

| AVM type | Typical MAPE |
|---|---|
| On-market (with full listing data) | 2–7% |
| Off-market (no listing data) | 7–10% |
| Fundamentals-only (sparse features) | 25–45% |

Model A's ±8% places it in the upper-end on-market range. Model B sits
firmly in fundamentals-only territory because of dataset constraints
(see *Known limitations* below).
""")


# ============================================================================
# Section 3: Known limitations — the honesty section
# ============================================================================

st.markdown("#### Known limitations of this dataset")

st.warning(
    "**This is a Phase 1 prototype.** During data validation, several issues "
    "were identified in the underlying dataset. They are documented here in "
    "the spirit of transparency. None affect the AVM's core predictions, but "
    "they shape what the model can and cannot do.",
    icon="⚠️",
)

if api_ok and "data_caveats" in meta:
    for caveat in meta["data_caveats"]:
        st.markdown(f"- {caveat}")

st.markdown("""
**Specifically:**

- **Latitude, longitude, and ZIP fields were excluded** from the model. Validation
  showed only 3% of these matched the property's stated state, suggesting they
  were randomly generated. Location signal comes from **state, city, and county**
  fields, which are clean.
- **No time-series features.** Sale dates in the dataset don't follow real US
  housing market trends (the median price barely moved across 2016–2026, and
  2022 — a real-market peak — is the lowest year here). The AVM is therefore
  built as a **cross-sectional** model and doesn't try to predict price changes
  over time.
- **Physical attributes carry weaker signal than typical AVMs.** In real US
  property data, square footage correlates with sale price at 0.6–0.8. In this
  dataset, it's 0.22. As a result, **location features dominate the predictions**;
  bedroom and bathroom counts contribute only marginally.

These constraints are why Model B's MAPE is around 45% rather than the 10%
that would be typical for fundamentals-only AVMs trained on richer data. Phase 2
is expected to address this by onboarding a substantially larger and richer
dataset.
""")


# ============================================================================
# Section 4: Data lineage
# ============================================================================

st.markdown("#### Data lineage")

if api_ok:
    st.markdown(f"""
- **Training rows:** {meta.get("training_rows", "—"):,}
- **Last trained:** {meta.get("training_date", "—")}
- **Source:** AlloyTower Phase 1 property dataset (cleaned)
- **Cleaning rules applied:** dropped 134 rows with implausible price-per-sqft
  pairings (below US construction cost); dropped randomly generated geographic
  identifiers; excluded date-based columns due to lack of temporal signal.
""")
else:
    st.markdown("Live metadata unavailable — start the API service to see lineage.")


# ============================================================================
# Section 5: When to be sceptical of a prediction
# ============================================================================

st.markdown("#### When to be sceptical of a prediction")

st.markdown("""
The model will produce a number for any input, but that doesn't mean every
prediction is equally reliable. Take results with extra caution when:

- **The property is in an unusual location** (a state or city not heavily
  represented in the training data). The model may default to the regional mean.
- **The property is a new build (year built > 2020)** with no assessed value.
  Model B handles this case but cannot account for new-construction premiums
  or local supply dynamics.
- **The property is unusual in size** (very small studios under 500 sqft,
  or large estates over 6,000 sqft). The model has fewer training examples
  in these tails.
- **The estimated value is far from comparable sales** shown on the home page.
  When the comparables grid shows 5 properties at $500K and the AVM says $1.2M,
  trust the comparables.
""")


# ============================================================================
# Section 6: Phase 2 roadmap
# ============================================================================

st.markdown("#### What's next — Phase 2 roadmap")

st.markdown("""
Subject to data acquisition, Phase 2 priorities are:

1. **Onboard verified geographic identifiers** — real lat/long, real ZIP,
   real census-tract joinable data. Unlocks neighbourhood-level features
   (school ratings, walkability, demographics).
2. **Onboard a real listings feed** — list price, days-on-market, listing
   status, listing photos. Enables on-market AVM (target: <7% MAPE).
3. **Onboard a substantially larger dataset** — 10,000+ records is typical
   for production AVMs. Improves model stability and reduces tail-case error.
4. **Add a real time-series component** — once verified sale dates are
   available, time-aware features can capture genuine market dynamics
   (appreciation rates, seasonal patterns, market-cycle indicators).
""")

st.markdown("---")
st.caption(
    "Model card last refreshed: live metadata from the API. "
    "For technical questions, see the project's data quality findings report and training pipeline."
)
