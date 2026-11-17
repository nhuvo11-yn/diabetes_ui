"""ui.py - look & feel only (colors, cards, chart style). No page wording lives here.

Palette: calm teal for the interface; orange = "raises the score / missed", blue = "lowers it"
(kept identical to the chart colors the pages already used).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st

TEAL, TEAL_DARK = "#0F766E", "#115E59"
ORANGE, BLUE = "#d95f02", "#1b6ca8"
INK, MUTED, LINE = "#12312F", "#5B7371", "#D7E5E5"

_CSS = """
:root{--teal:#0F766E;--teal-d:#115E59;--ink:#12312F;--muted:#5B7371;--line:#D7E5E5;--soft:#F3F8F8;}
div[data-testid="stElementContainer"]:has(.ui-css){display:none;}
.stApp{background:linear-gradient(180deg,#EEF6F5 0%,#FFFFFF 380px);}
.block-container{max-width:1120px;padding-top:2.2rem;padding-bottom:4rem;}
h1{color:var(--ink);font-weight:800;letter-spacing:-0.015em;}
h2,h3{color:var(--ink);font-weight:700;letter-spacing:-0.01em;}
[data-testid="stCaptionContainer"]{color:var(--muted);}
hr{border-color:var(--line);}

/* sidebar */
[data-testid="stSidebar"]{background:#E6F1F0;border-right:1px solid var(--line);}
[data-testid="stSidebar"] [data-testid="stAlert"]{border-radius:12px;}
[data-testid="stSidebarNav"] a{border-radius:10px;}

/* cards (containers created with key="card_...") */
[class*="st-key-card"]{background:#fff;border:1px solid var(--line);border-radius:18px;
  padding:1.1rem 1.3rem;box-shadow:0 1px 2px rgba(15,118,110,.06),0 8px 24px rgba(15,118,110,.07);}
[class*="st-key-card"] [data-testid="stMetric"]{background:var(--soft);box-shadow:none;}

/* hero banner on the screening page */
.st-key-hero{background:linear-gradient(135deg,#0F766E 0%,#0E8F8A 100%);border-radius:22px;
  padding:1.7rem 1.9rem;box-shadow:0 10px 30px rgba(15,118,110,.25);margin-bottom:.8rem;}
.st-key-hero h1,.st-key-hero p,.st-key-hero strong{color:#fff;}
.st-key-hero h1{padding:0;margin-bottom:.3rem;}
.st-key-hero [data-testid="stCaptionContainer"],.st-key-hero [data-testid="stCaptionContainer"] *{color:rgba(255,255,255,.88);}

/* form card */
[data-testid="stForm"]{background:#fff;border:1px solid var(--line);border-radius:20px;padding:1.5rem 1.6rem;
  box-shadow:0 1px 2px rgba(15,118,110,.06),0 8px 24px rgba(15,118,110,.07);}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-testid="stNumberInputContainer"]{border-radius:12px;}

/* radio -> soft pill choices */
[data-testid="stRadio"] [role="radiogroup"]{gap:.5rem;}
[data-testid="stRadio"] [role="radiogroup"]>label{background:var(--soft);border:1px solid var(--line);
  border-radius:12px;padding:.3rem .95rem;margin:0;transition:all .15s;}
[data-testid="stRadio"] [role="radiogroup"]>label:hover{border-color:var(--teal);}
[data-testid="stRadio"] [role="radiogroup"]>label:has(input:checked){background:#DDF1EF;border-color:var(--teal);}

/* buttons */
button[kind="primary"],button[kind="primaryFormSubmit"],[data-testid^="stBaseButton-primary"]{
  background:linear-gradient(135deg,#0F766E,#0E8F8A);border:none;border-radius:12px;color:#fff;
  font-weight:700;padding:.6rem 1.7rem;box-shadow:0 6px 16px rgba(15,118,110,.28);}
button[kind="primary"]:hover,button[kind="primaryFormSubmit"]:hover,[data-testid^="stBaseButton-primary"]:hover{
  filter:brightness(1.06);box-shadow:0 8px 20px rgba(15,118,110,.36);}
button[kind="secondary"],[data-testid="stBaseButton-secondary"]{border-radius:12px;border:1px solid var(--line);background:#fff;}

/* metrics */
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:14px 16px;
  box-shadow:0 1px 2px rgba(15,118,110,.05),0 6px 18px rgba(15,118,110,.06);}
[data-testid="stMetricLabel"]{color:var(--muted);}
[data-testid="stMetricValue"]{color:var(--teal-d);font-weight:800;}

/* alerts (result message, notes) */
[data-testid="stAlert"]{border-radius:16px;border:1px solid transparent;}
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]){background:#FFF3E6;border-color:#F5C899;}
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]){background:#E7F6F0;border-color:#A8DCC5;}
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]){background:#EAF4FB;border-color:#B6D6EC;}

/* tables, expanders, images */
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden;}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:16px;background:#fff;}
[data-testid="stImage"] img{border-radius:12px;}

/* summary card with accent edge */
.st-key-card_summary{border-left:6px solid var(--teal);}

@media (max-width:640px){
  .block-container{padding-left:1rem;padding-right:1rem;}
  .st-key-hero{padding:1.2rem 1.2rem;}
  [class*="st-key-card"],[data-testid="stForm"]{padding:.9rem 1rem;}
}
"""


def inject() -> None:
    """Apply the shared style (CSS + chart defaults). Called once per page from utils.setup_page."""
    st.markdown(f"<div class='ui-css'></div><style>{_CSS}</style>", unsafe_allow_html=True)
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": LINE, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#E6EEED", "grid.linewidth": 0.8, "axes.axisbelow": True,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "legend.frameon": False, "font.size": 10,
    })
