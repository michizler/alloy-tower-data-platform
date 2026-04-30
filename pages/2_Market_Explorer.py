"""
Market Explorer — secondary page for analysts who want to poke around the data.

Filter the cleaned dataset on state, property type, and price range; see
summary statistics and a scatter plot. This is the panel that surfaces
patterns ('show me all SF condos under $1M built after 2010').

Loads the cleaned CSV directly rather than going through the API — the data
is small enough (1,956 rows) and this avoids needing yet another endpoint.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.lib import Colors, page_header, fmt_money, fmt_money_short, demo_banner


# ============================================================================
# Page setup
# ============================================================================

st.set_page_config(page_title="Market Explorer | AlloyTower AVM",
                   page_icon="📊", layout="wide")

page_header(
    "Market explorer",
    "Filter and explore the Phase 1 dataset of 1,956 properties."
)

demo_banner()


# ============================================================================
# Data loading
# ============================================================================

CLEAN_CSV_PATH = Path("model/models/alloy_clean.csv")


@st.cache_data
def load_data() -> pd.DataFrame:
    if not CLEAN_CSV_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CLEAN_CSV_PATH)


df = load_data()

if df.empty:
    st.error(
        f"Cleaned dataset not found at `{CLEAN_CSV_PATH}`. "
        "Run `python train_avm.py` first to produce it."
    )
    st.stop()


# ============================================================================
# Sidebar filters
# ============================================================================

with st.sidebar:
    st.markdown("### Filters")

    states = sorted(df["state"].unique())
    sel_states = st.multiselect("State", states, default=[],
                                 help="Leave empty to include all states.")

    types = sorted(df["property_type"].unique())
    sel_types = st.multiselect("Property type", types, default=[],
                                help="Leave empty to include all types.")

    price_min, price_max = float(df["last_sale_price"].min()), float(df["last_sale_price"].max())
    sel_price = st.slider(
        "Price range",
        min_value=int(price_min), max_value=int(price_max),
        value=(int(price_min), int(price_max)),
        step=10000, format="$%d",
    )

    year_min, year_max = int(df["year_built"].min()), int(df["year_built"].max())
    sel_year = st.slider(
        "Year built",
        min_value=year_min, max_value=year_max,
        value=(year_min, year_max),
    )

    sqft_min, sqft_max = int(df["sqft"].min()), int(df["sqft"].max())
    sel_sqft = st.slider(
        "Living area (sqft)",
        min_value=sqft_min, max_value=sqft_max,
        value=(sqft_min, sqft_max), step=100,
    )


# Apply filters
filtered = df.copy()
if sel_states:
    filtered = filtered[filtered["state"].isin(sel_states)]
if sel_types:
    filtered = filtered[filtered["property_type"].isin(sel_types)]
filtered = filtered[
    (filtered["last_sale_price"].between(sel_price[0], sel_price[1])) &
    (filtered["year_built"].between(sel_year[0], sel_year[1])) &
    (filtered["sqft"].between(sel_sqft[0], sel_sqft[1]))
]


# ============================================================================
# Summary stats — top row of metric cards
# ============================================================================

st.markdown(f"#### {len(filtered):,} properties match your filters")

if filtered.empty:
    st.info("No properties match the current filters. Try widening them.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Median price", fmt_money_short(filtered["last_sale_price"].median()))
with c2:
    st.metric("Median price/sqft", f"${filtered['price_per_sqft'].median() if 'price_per_sqft' in filtered.columns else (filtered['last_sale_price'] / filtered['sqft']).median():.0f}")
with c3:
    st.metric("Median size", f"{filtered['sqft'].median():,.0f} sqft")
with c4:
    st.metric("Median age", f"{2024 - filtered['year_built'].median():.0f} years")


# ============================================================================
# Scatter plot: sqft vs price, coloured by property type
# ============================================================================

st.markdown("#### Price vs. living area")
st.caption("Each dot is one property. Hover for details. Coloured by property type.")

import altair as alt

# Limit to a sane number of dots for rendering
plot_df = filtered.copy()
if len(plot_df) > 1500:
    plot_df = plot_df.sample(1500, random_state=42)

chart = alt.Chart(plot_df).mark_circle(size=40, opacity=0.55).encode(
    x=alt.X("sqft:Q", title="Living area (sqft)",
            scale=alt.Scale(zero=False)),
    y=alt.Y("last_sale_price:Q", title="Sale price ($)",
            axis=alt.Axis(format="$,.0f")),
    color=alt.Color("property_type:N", title="Property type",
                     scale=alt.Scale(range=[
                         Colors.success, Colors.accent,
                         Colors.warning, Colors.primary,
                     ])),
    tooltip=[
        alt.Tooltip("city:N"), alt.Tooltip("state:N"),
        alt.Tooltip("property_type:N", title="Type"),
        alt.Tooltip("sqft:Q", title="Sqft", format=","),
        alt.Tooltip("bedrooms:Q", title="Bd"),
        alt.Tooltip("bathrooms:Q", title="Ba"),
        alt.Tooltip("last_sale_price:Q", title="Sale price", format="$,.0f"),
    ],
).properties(height=380).interactive()

st.altair_chart(chart, use_container_width=True)


# ============================================================================
# Sortable table
# ============================================================================

st.markdown("#### Properties")
st.caption("Sort by clicking any column header. Showing up to 200 rows.")

table_cols = ["property_id", "city", "state", "property_type",
              "sqft", "bedrooms", "bathrooms", "year_built", "last_sale_price"]
display_df = filtered[table_cols].head(200).copy()
display_df.columns = ["ID", "City", "State", "Type", "Sqft", "Bd", "Ba", "Built", "Sale price"]

st.dataframe(
    display_df,
    use_container_width=True, hide_index=True,
    column_config={
        "Sale price": st.column_config.NumberColumn(format="$%d"),
        "Sqft": st.column_config.NumberColumn(format="%d"),
    },
)
