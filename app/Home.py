"""
AlloyTower AVM — Home (Property Valuation)

The primary page non-technical stakeholders interact with. Three sections:
  1. Property entry form (left)
  2. Valuation result with confidence band (right top)
  3. Feature contributions + comparables (below, full-width)

Stakeholder demo flow:
  - Click "Try a sample property" → form fills in
  - Click "Estimate value" → result appears with explanation
  - Scroll down for comparables grid

Design ethos: every claim the app makes (the price, the range, the why)
should be visually honest. No false precision, no hidden caveats.
"""

from __future__ import annotations

import streamlit as st

from app.lib import (
    Colors, demo_banner, page_header, api_unavailable_message,
    render_stat, fmt_money, fmt_money_short,
    predict, comparables, get_health, ApiError,
)


# ============================================================================
# Page setup
# ============================================================================

st.set_page_config(
    page_title="AlloyTower AVM",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Health check on first load — fail fast with a friendly message
try:
    health = get_health()
    api_ok = health.get("predictor_loaded", False)
except ApiError:
    api_ok = False

page_header(
    "AlloyTower property valuation",
    "Estimate the market value of any US residential property using the Phase 1 AVM."
)

if not api_ok:
    api_unavailable_message()
    st.stop()

demo_banner()


# ============================================================================
# Sample properties for the demo dropdown
# ============================================================================

SAMPLES = {
    "—": None,
    "San Francisco condo (with assessment)": {
        "state": "CA", "city": "San Francisco", "county": "San Francisco",
        "property_type": "Condo", "sqft": 1200, "lot_size_sqft": 0,
        "bedrooms": 2, "bathrooms": 2.0, "year_built": 2010,
        "owner_occupied": True, "assessed_value": 1100000.0,
    },
    "Austin single-family (no assessment)": {
        "state": "TX", "city": "Austin", "county": "Travis",
        "property_type": "Single Family", "sqft": 2400, "lot_size_sqft": 7500,
        "bedrooms": 4, "bathrooms": 3.0, "year_built": 2005,
        "owner_occupied": True, "assessed_value": None,
    },
    "Manhattan condo (with assessment)": {
        "state": "NY", "city": "Manhattan", "county": "New York",
        "property_type": "Condo", "sqft": 950, "lot_size_sqft": 0,
        "bedrooms": 1, "bathrooms": 1.5, "year_built": 1995,
        "owner_occupied": False, "assessed_value": 1450000.0,
    },
}

# Initialise session state from sample if not already present
if "form_values" not in st.session_state:
    st.session_state.form_values = SAMPLES["San Francisco condo (with assessment)"].copy()


# ============================================================================
# Sidebar — quick-fill samples + global controls
# ============================================================================

with st.sidebar:
    st.markdown("### Quick fill")
    sample_choice = st.selectbox(
        "Try a sample property", list(SAMPLES.keys()),
        help="Pre-fills the form below with a realistic example.",
    )
    if sample_choice != "—" and st.button("Load sample", use_container_width=True):
        st.session_state.form_values = SAMPLES[sample_choice].copy()
        st.rerun()

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "This app uses two AVM variants: **Model A** (assessment-aware, "
        "~8% MAPE) when an assessed value is provided, and **Model B** "
        "(fundamentals-only, ~45% MAPE) when not. See the **About this model** "
        "page for the full picture."
    )


# ============================================================================
# Two-column layout: form (left) + result (right)
# ============================================================================

col_form, col_result = st.columns([1, 1.2], gap="large")

# ---- LEFT: input form ----
with col_form:
    st.markdown("#### Property details")

    fv = st.session_state.form_values

    state = st.text_input("State (2-letter code)", value=fv["state"], max_chars=2).upper()
    city = st.text_input("City", value=fv["city"])
    county = st.text_input("County", value=fv["county"])

    property_type = st.selectbox(
        "Property type",
        ["Single Family", "Condo", "Townhouse", "Multi Family"],
        index=["Single Family", "Condo", "Townhouse", "Multi Family"].index(fv["property_type"]),
    )

    c1, c2 = st.columns(2)
    with c1:
        sqft = st.number_input("Living area (sqft)", min_value=200, max_value=30000,
                                value=fv["sqft"], step=50)
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=15,
                                    value=fv["bedrooms"], step=1)
        year_built = st.number_input("Year built", min_value=1800, max_value=2026,
                                      value=fv["year_built"], step=1)
    with c2:
        lot_size = st.number_input("Lot size (sqft)", min_value=0, max_value=1_000_000,
                                    value=fv["lot_size_sqft"], step=100)
        bathrooms = st.number_input("Bathrooms", min_value=0.0, max_value=15.0,
                                     value=float(fv["bathrooms"]), step=0.5)
        owner_occupied = st.checkbox("Owner-occupied", value=fv["owner_occupied"])

    st.markdown("###### Optional: include a recent county assessment")
    use_assessment = st.checkbox(
        "Property has been recently assessed",
        value=fv["assessed_value"] is not None,
        help="Enables Model A (more accurate). If unchecked, Model B is used.",
    )
    assessed_value = None
    if use_assessment:
        assessed_value = st.number_input(
            "Assessed value ($)", min_value=10_000.0, max_value=50_000_000.0,
            value=float(fv["assessed_value"] or 500_000.0), step=10_000.0, format="%.0f",
        )

    st.markdown("")
    estimate_clicked = st.button("Estimate value", type="primary", use_container_width=True)


# ---- RIGHT: result ----
with col_result:
    st.markdown("#### Estimated value")

    if not estimate_clicked:
        st.markdown(
            f'<div style="border: 1px dashed {Colors.border}; border-radius: 8px; '
            f'padding: 40px 20px; text-align: center; color: {Colors.neutral}; '
            f'background: {Colors.neutral_bg};">'
            f'Fill in the details on the left and click <b>Estimate value</b>.'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        # Build the request payload
        payload = {
            "state": state, "city": city, "county": county,
            "property_type": property_type,
            "sqft": int(sqft), "lot_size_sqft": int(lot_size),
            "bedrooms": int(bedrooms), "bathrooms": float(bathrooms),
            "year_built": int(year_built), "owner_occupied": owner_occupied,
            "assessed_value": float(assessed_value) if assessed_value is not None else None,
        }

        try:
            with st.spinner("Computing valuation..."):
                result = predict(payload)
        except ApiError as e:
            st.error(f"Prediction failed: {e.message}")
            st.stop()

        # Persist for the comparables section below
        st.session_state.last_result = result
        st.session_state.last_payload = payload

        # ---- Headline number ----
        render_stat(
            "Estimated market value",
            fmt_money(result["estimated_value"]),
            sublabel=f"Likely range: {fmt_money(result['lower_bound'])} – {fmt_money(result['upper_bound'])}",
            color=Colors.primary,
        )

        # ---- Confidence range bar ----
        mape_pct = result["mape"] * 100
        model_label = "Model A (assessment-aware)" if result["model_used"] == "A" else "Model B (fundamentals-only)"

        st.markdown(
            f'<div style="margin-top: 8px; padding: 12px 16px; '
            f'background: {Colors.neutral_bg}; border-radius: 8px; border-left: 3px solid {Colors.accent};">'
            f'<div style="font-size: 12px; color: {Colors.neutral}; text-transform: uppercase; letter-spacing: 0.5px;">Confidence</div>'
            f'<div style="font-size: 14px; margin-top: 4px;">'
            f'Predictions from <b>{model_label}</b> are typically within '
            f'<b>±{mape_pct:.1f}%</b> of the actual sale price.'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        # ---- Range visual: simple horizontal bar ----
        lo = result["lower_bound"]
        hi = result["upper_bound"]
        mid = result["estimated_value"]
        # Marker as a percentage of the band
        marker_pos = ((mid - lo) / (hi - lo)) * 100 if hi > lo else 50
        st.markdown(
            f'<div style="margin-top: 16px;">'
            f'<div style="display: flex; justify-content: space-between; font-size: 12px; color: {Colors.neutral}; margin-bottom: 4px;">'
            f'<span>{fmt_money_short(lo)}</span><span>{fmt_money_short(hi)}</span>'
            f'</div>'
            f'<div style="position: relative; height: 24px; background: linear-gradient(to right, {Colors.warning_bg}, {Colors.success_bg}, {Colors.warning_bg}); border-radius: 4px;">'
            f'<div style="position: absolute; left: {marker_pos}%; top: -4px; transform: translateX(-50%); width: 4px; height: 32px; background: {Colors.primary}; border-radius: 2px;"></div>'
            f'<div style="position: absolute; left: {marker_pos}%; top: -22px; transform: translateX(-50%); font-size: 11px; font-weight: 600; color: {Colors.primary}; white-space: nowrap;">{fmt_money_short(mid)}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )


# ============================================================================
# Below the fold: feature contributions + comparables
# ============================================================================

if "last_result" not in st.session_state:
    st.stop()

result = st.session_state.last_result
payload = st.session_state.last_payload

st.markdown("")
st.markdown("---")

# Two-column section: contributions | comparables note
contrib_col, comp_col = st.columns([1, 1.2], gap="large")

with contrib_col:
    st.markdown("#### What drove this estimate")
    st.caption("Top factors influencing the prediction, ordered by impact.")

    # Show top 5 contributions, skipping interaction-feature noise
    notable = [
        c for c in result["contributions"]
        if c["feature"] not in ("sqft_per_bedroom", "state_x_sqft", "building_age")
    ][:5]

    for c in notable:
        direction_icon = "▲" if c["direction"] == "up" else "▼" if c["direction"] == "down" else "—"
        direction_color = {"up": Colors.success, "down": Colors.danger, "neutral": Colors.neutral}[c["direction"]]
        bar_color = {"up": Colors.success_bg, "down": Colors.danger_bg, "neutral": Colors.neutral_bg}[c["direction"]]
        contrib_pct = abs(c["contribution"]) / result["estimated_value"] * 100
        bar_width = min(contrib_pct * 2, 100)  # cap at 100% width

        st.markdown(
            f'<div style="margin: 10px 0; padding: 8px 12px; border-radius: 6px; background: {bar_color};">'
            f'<div style="display: flex; justify-content: space-between; align-items: center;">'
            f'<div style="font-size: 14px;">'
            f'<b>{c["display_name"]}</b>'
            f'{": " + c["value"] if c["value"] else ""}'
            f'</div>'
            f'<div style="font-size: 13px; color: {direction_color}; font-weight: 600;">'
            f'{direction_icon} {fmt_money_short(abs(c["contribution"]))}'
            f'</div>'
            f'</div>'
            f'<div style="margin-top: 4px; height: 4px; background: rgba(255,255,255,0.6); border-radius: 2px;">'
            f'<div style="height: 100%; width: {bar_width}%; background: {direction_color}; border-radius: 2px;"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.caption(
        "▲ = pushed price up | ▼ = pushed price down. "
        "Estimates assume the property is in average condition for its category."
    )


with comp_col:
    st.markdown("#### Comparable recent sales")
    st.caption("Five properties from the dataset most similar to your input, sorted by similarity.")

    try:
        with st.spinner("Finding comparables..."):
            comps = comparables(
                payload, n=5,
                target_price=result["estimated_value"],
                tolerance=0.30,
            )
    except ApiError as e:
        st.warning(f"Could not load comparables: {e.message}")
        comps = []

    if not comps:
        st.info(
            "No comparable properties found in the requested price band. "
            "This typically means the input combination is unusual for the dataset."
        )
    else:
        # Card grid — 5 cards in a 1-column layout (full width readability)
        # Each card: location strip, property details, price, similarity badge
        for i, c in enumerate(comps):
            similarity_pct = int(c["similarity_score"] * 100)
            type_color_map = {
                "Single Family": Colors.success, "Condo": Colors.accent,
                "Townhouse": Colors.warning, "Multi Family": Colors.primary,
            }
            type_color = type_color_map.get(c["property_type"], Colors.neutral)
            st.markdown(
                f'<div style="display: flex; gap: 12px; padding: 14px 16px; '
                f'border: 1px solid {Colors.border}; border-radius: 8px; margin-bottom: 10px; '
                f'background: white;">'

                # Left: location strip with a coloured accent based on type
                f'<div style="width: 6px; border-radius: 3px; background: {type_color};"></div>'

                # Middle: details
                f'<div style="flex: 1;">'
                f'<div style="font-size: 14px; font-weight: 600; color: {Colors.primary};">'
                f'{c["city"]}, {c["state"]}'
                f'</div>'
                f'<div style="font-size: 13px; color: {Colors.neutral}; margin-top: 2px;">'
                f'{c["property_type"]} · {c["sqft"]:,} sqft · '
                f'{c["bedrooms"]}bd / {c["bathrooms"]:.1f}ba · built {c["year_built"]}'
                f'</div>'
                f'<div style="font-size: 11px; color: {Colors.neutral}; margin-top: 4px; '
                f'text-transform: uppercase; letter-spacing: 0.5px;">'
                f'ID {c["property_id"]}'
                f'</div>'
                f'</div>'

                # Right: price + similarity
                f'<div style="text-align: right;">'
                f'<div style="font-size: 18px; font-weight: 600; color: {Colors.primary};">'
                f'{fmt_money_short(c["last_sale_price"])}'
                f'</div>'
                f'<div style="font-size: 11px; color: {Colors.success}; margin-top: 2px; '
                f'background: {Colors.success_bg}; padding: 2px 8px; border-radius: 10px; '
                f'display: inline-block;">'
                f'{similarity_pct}% match'
                f'</div>'
                f'</div>'

                f'</div>',
                unsafe_allow_html=True,
            )
