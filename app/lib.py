"""
Shared helpers for the Streamlit frontend.

Centralised so all three pages share the same API client, color tokens,
and small UI components. Saves duplication and keeps look consistent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import requests
import streamlit as st


API_BASE = os.environ.get("AVM_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 30


# ============================================================================
# Color tokens — centralised so the cards & charts stay coherent
# ============================================================================

class Colors:
    primary = "#1F3864"     # deep navy
    accent = "#2E5EAA"      # mid blue
    success = "#27500A"     # dark green
    success_bg = "#EAF3DE"
    warning = "#633806"
    warning_bg = "#FAEEDA"
    danger = "#791F1F"
    danger_bg = "#FCEBEB"
    neutral = "#5F5E5A"
    neutral_bg = "#F2F2F2"
    border = "#D3D1C7"


# ============================================================================
# API client — thin wrapper with friendly error handling
# ============================================================================

@dataclass
class ApiError(Exception):
    status: int
    message: str


def _api_get(path: str) -> dict:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        raise ApiError(0, f"Cannot reach API at {API_BASE}. Is the FastAPI server running? ({e})")
    if r.status_code != 200:
        raise ApiError(r.status_code, r.text)
    return r.json()


def _api_post(path: str, payload: dict) -> dict | list:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.RequestException as e:
        raise ApiError(0, f"Cannot reach API at {API_BASE}. Is the FastAPI server running? ({e})")
    if r.status_code != 200:
        raise ApiError(r.status_code, r.text)
    return r.json()


def get_health() -> dict:
    return _api_get("/health")


def get_metadata() -> dict:
    return _api_get("/metadata")


def predict(payload: dict) -> dict:
    return _api_post("/predict", payload)


def comparables(query: dict, n: int = 5,
                target_price: Optional[float] = None,
                tolerance: float = 0.20) -> list[dict]:
    payload = {"query": query, "n": n, "price_tolerance": tolerance}
    if target_price is not None:
        payload["target_price"] = target_price
    return _api_post("/comparables", payload)


# ============================================================================
# Reusable UI components
# ============================================================================

def page_header(title: str, subtitle: str | None = None) -> None:
    """Consistent page heading across the app."""
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")


def demo_banner() -> None:
    """Persistent banner reminding stakeholders this is a prototype."""
    st.info(
        "**Prototype — Phase 1.** Predictions use a 1,956-property training set. "
        "Production deployment would use a substantially larger dataset. "
        "See the **About this model** page for details.",
        icon="ℹ️",
    )


def api_unavailable_message() -> None:
    """What we show when the FastAPI backend isn't reachable."""
    st.error(
        "**API service unavailable.** This frontend connects to a FastAPI backend "
        "at `" + API_BASE + "`. Start the backend with:\n\n"
        "```\nuvicorn api.main:app --port 8000\n```\n\n"
        "Then refresh this page."
    )


def render_stat(label: str, value: str, sublabel: str | None = None,
                color: str = Colors.primary) -> None:
    """A single inline stat — used on the prediction page."""
    sub_html = f'<div style="font-size: 13px; color: {Colors.neutral}; margin-top: 2px;">{sublabel}</div>' if sublabel else ""
    st.markdown(f'''
        <div style="padding: 8px 0;">
          <div style="font-size: 12px; color: {Colors.neutral}; text-transform: uppercase; letter-spacing: 0.5px;">{label}</div>
          <div style="font-size: 28px; font-weight: 600; color: {color}; margin-top: 4px;">{value}</div>
          {sub_html}
        </div>
    ''', unsafe_allow_html=True)


def fmt_money(value: float) -> str:
    return f"${value:,.0f}"


def fmt_money_short(value: float) -> str:
    """Compact currency: $1.2M, $850K. Used in cards where space is tight."""
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.0f}K"
    return f"${value:,.0f}"
