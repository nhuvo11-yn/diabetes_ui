"""utils.py - Streamlit helpers shared by every page."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import ui

ROOT = Path(__file__).resolve().parent
DISCLAIMER = "Educational project — not for clinical use."


def setup_page(title: str, icon: str = "🩺") -> None:
    """Must be the first Streamlit call on each page. Puts the disclaimer on every sidebar."""
    st.set_page_config(page_title=f"{title} · Diabetes Screening Assistant", page_icon=icon, layout="wide")
    st.sidebar.warning(DISCLAIMER)
    ui.inject()


@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load the trained sklearn Pipeline once per server process (never train inside the app)."""
    return joblib.load(ROOT / "models" / "pipeline.joblib")


@st.cache_data(show_spinner=False)
def load_meta() -> dict:
    return json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_report(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / name)


@st.cache_data(show_spinner=False)
def load_eval_predictions() -> pd.DataFrame:
    return pd.read_csv(ROOT / "reports" / "eval_predictions.csv.gz")


def active_threshold(meta: dict) -> tuple[float, bool]:
    """Threshold used by the Risk Screening page. Returns (value, is_user_override)."""
    override = st.session_state.get("threshold_override")
    if override is not None:
        return float(override), True
    return float(meta["threshold"]), False
