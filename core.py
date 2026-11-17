"""core.py - pure helpers shared by train.py and the Streamlit app (no Streamlit import here).

Keeping these functions in one place guarantees that the app uses exactly the same
threshold / metric logic that train.py used when it produced the saved results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Column definitions (BRFSS 2015 "diabetes_binary_health_indicators" file)
# --------------------------------------------------------------------------- #
TARGET = "Diabetes_binary"

FULL_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
    "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education", "Income",
]

# Short questionnaire: only things an adult can answer at home without a lab test.
# PhysActivity was dropped: validation PR-AUC was identical (0.4316) with and without it.
MINIMAL_FEATURES = [
    "Age", "BMI", "HighBP", "HighChol", "GenHlth",
    "DiffWalk", "HeartDiseaseorAttack",
]

NUMERIC_FEATURES = ["BMI", "Age", "GenHlth", "MentHlth", "PhysHlth"]  # scaled inside the pipeline
NON_BINARY = {"BMI", "Age", "GenHlth", "MentHlth", "PhysHlth", "Education", "Income"}

FEATURE_LABELS = {
    "Age": "Age group",
    "BMI": "Body mass index (BMI)",
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "GenHlth": "Self-rated general health",
    "PhysActivity": "Physical activity (past 30 days)",
    "DiffWalk": "Difficulty walking / climbing stairs",
    "HeartDiseaseorAttack": "Heart disease or heart attack",
    "CholCheck": "Cholesterol check in past 5 years",
    "Smoker": "Smoked 100+ cigarettes in lifetime",
    "Stroke": "Ever had a stroke",
    "Fruits": "Eats fruit daily",
    "Veggies": "Eats vegetables daily",
    "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Has health care coverage",
    "NoDocbcCost": "Skipped doctor because of cost",
    "MentHlth": "Poor mental health days (past 30)",
    "PhysHlth": "Poor physical health days (past 30)",
    "Sex": "Sex",
    "Education": "Education level",
    "Income": "Income level",
}

AGE_BANDS = {
    1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44", 6: "45–49",
    7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69", 11: "70–74", 12: "75–79", 13: "80+",
}
GENHLTH_LABELS = {1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"}

INCOME_ORDER = ["Low (< $25k)", "Middle ($25k–$75k)", "High ($75k+)"]
EDU_ORDER = ["Less than high school", "High school graduate", "Some college", "College graduate"]


def income_group(s: pd.Series) -> pd.Series:
    """BRFSS Income 1-4 = under $25k, 5-7 = $25k to under $75k, 8 = $75k or more."""
    v = pd.Series(s).astype(float)
    return pd.Series(
        np.select([v <= 4, v <= 7], [INCOME_ORDER[0], INCOME_ORDER[1]], default=INCOME_ORDER[2]),
        index=v.index,
    )


def edu_group(s: pd.Series) -> pd.Series:
    """BRFSS Education 1-3 merged (level 1 alone has only ~170 rows)."""
    v = pd.Series(s).astype(float)
    return pd.Series(
        np.select([v <= 3, v == 4, v == 5], EDU_ORDER[:3], default=EDU_ORDER[3]),
        index=v.index,
    )


def format_value(feature: str, value: float) -> str:
    if feature == "Age":
        return AGE_BANDS[int(round(value))]
    if feature == "GenHlth":
        return GENHLTH_LABELS[int(round(value))]
    if feature == "BMI":
        return f"{float(value):.1f}"
    if feature in NON_BINARY:
        return f"{float(value):g}"
    return "Yes" if value >= 0.5 else "No"


# --------------------------------------------------------------------------- #
# Confusion-matrix metrics
# --------------------------------------------------------------------------- #
def confusion_counts(y, p, threshold: float):
    y = np.asarray(y).astype(bool)
    yh = np.asarray(p) >= threshold
    tp = int((y & yh).sum())
    fp = int((~y & yh).sum())
    fn = int((y & ~yh).sum())
    tn = int((~y & ~yh).sum())
    return tp, fp, fn, tn


def _div(a, b):
    return float(a) / float(b) if b else float("nan")


def rates(tp: int, fp: int, fn: int, tn: int) -> dict:
    n = tp + fp + fn + tn
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": _div(tp, tp + fn),
        "precision": _div(tp, tp + fp),
        "fnr": _div(fn, tp + fn),
        "fpr": _div(fp, fp + tn),
        "accuracy": _div(tp + tn, n),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "flag_rate": _div(tp + fp, n),
    }


# --------------------------------------------------------------------------- #
# Threshold / cost analysis
# --------------------------------------------------------------------------- #
DEFAULT_GRID = np.round(np.linspace(0.01, 0.99, 197), 4)


def cost_curve(y, p, cost_fn: float, cost_fp: float, grid=DEFAULT_GRID) -> pd.DataFrame:
    """Expected-cost table for every threshold in `grid` (fast, sort-based)."""
    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=float)
    pos = np.sort(p[y])
    neg = np.sort(p[~y])
    grid = np.asarray(grid, dtype=float)
    fn = np.searchsorted(pos, grid, side="left")          # positives scored below t
    fp = len(neg) - np.searchsorted(neg, grid, side="left")  # negatives scored >= t
    tp = len(pos) - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
        recall = tp / len(pos)
    return pd.DataFrame({
        "threshold": grid, "fn": fn, "fp": fp, "tp": tp,
        "cost": cost_fn * fn + cost_fp * fp,
        "recall": recall, "precision": precision,
        "flag_rate": (tp + fp) / len(y),
    })


def choose_threshold(y, p, cost_fn: float, cost_fp: float, grid=DEFAULT_GRID) -> float:
    curve = cost_curve(y, p, cost_fn, cost_fp, grid)
    return float(curve.loc[curve["cost"].idxmin(), "threshold"])


# --------------------------------------------------------------------------- #
# Bootstrap confidence intervals for subgroup metrics
# --------------------------------------------------------------------------- #
def bootstrap_ci(y, p, threshold: float, n_boot: int = 200, seed: int = 42) -> dict:
    y = np.asarray(y).astype(bool)
    yh = np.asarray(p) >= threshold
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n), dtype=np.int32)
    Y, H = y[idx], yh[idx]
    tp = (Y & H).sum(1).astype(float)
    fp = (~Y & H).sum(1).astype(float)
    fn = (Y & ~H).sum(1).astype(float)
    tn = (~Y & ~H).sum(1).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "recall": tp / (tp + fn), "precision": tp / (tp + fp),
            "fnr": fn / (tp + fn), "fpr": fp / (fp + tn),
        }
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in out.items()}


# --------------------------------------------------------------------------- #
# Local explanation (model-agnostic "replace by typical answer" analysis)
# --------------------------------------------------------------------------- #
def local_contributions(pipeline, x_row: pd.DataFrame, features: list[str], reference: dict) -> pd.DataFrame:
    """For each feature: score(user) - score(user with ONE answer replaced by the typical answer).

    Positive = this answer pushes the score up compared with a typical respondent.
    This is an approximation of the model's behaviour, not a causal effect.
    """
    base = float(pipeline.predict_proba(x_row[features])[0, 1])
    variants = pd.concat([x_row[features]] * len(features), ignore_index=True)
    for i, f in enumerate(features):
        variants.loc[i, f] = reference[f]
    p_alt = pipeline.predict_proba(variants)[:, 1]
    return pd.DataFrame({
        "feature": features,
        "your_value": [x_row.iloc[0][f] for f in features],
        "typical_value": [reference[f] for f in features],
        "effect": base - p_alt,
    })
