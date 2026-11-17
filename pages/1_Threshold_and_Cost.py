"""Page 2 - Threshold & cost simulator."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import core
import utils

utils.setup_page("Threshold & Cost", "⚖️")
meta = utils.load_meta()
ev = utils.load_eval_predictions()
val, test = ev[ev.split == "val"], ev[ev.split == "test"]
default_t = meta["threshold"]
ca = meta["cost_assumptions"]

st.title("Threshold & cost simulator")
st.markdown(
    "The model gives a score between 0 and 1. A **threshold** turns that score into a recommendation. "
    "Lowering it catches more people who may need a test (fewer **false negatives**) but sends more people "
    "to testing they did not need (more **false positives**)."
)
st.info(
    "Costs below are **assumed numbers for simulation only**, not measured medical costs. "
    "They express how much worse a missed case is judged to be than an unnecessary test."
)

left, right = st.columns([1, 2])
with left:
    with st.container(key="card_controls"):
        ratio = st.slider("Cost of a missed case (FN) relative to an unnecessary test (FP = 1)",
                          1.0, 30.0, float(ca["cost_fn"] / ca["cost_fp"]), 0.5)
        manual = st.checkbox("Pick the threshold manually")
        best_t = core.choose_threshold(val["y"], val["p_final"], ratio, 1.0)
        if manual:
            t = st.slider("Threshold", 0.02, 0.80, float(best_t), 0.01)
        else:
            t = best_t
            st.metric("Cost-minimizing threshold", f"{t:.2f}", help="Chosen on the validation set.")
        st.caption(f"Default used by the app: {default_t:.2f} (cost ratio {ca['cost_fn'] / ca['cost_fp']:g}:1).")

        b1, b2 = st.columns(2)
        if b1.button("Use on Screening page", type="primary"):
            st.session_state["threshold_override"] = float(t)
            st.success(f"Screening page now uses {t:.2f}.")
        if b2.button("Reset to default"):
            st.session_state.pop("threshold_override", None)
            st.success("Back to the default threshold.")

# ---- metrics on the held-out test set
tp, fp, fn, tn = core.confusion_counts(test["y"], test["p_final"], t)
r = core.rates(tp, fp, fn, tn)
scale = 1000 / r["n"]
with right:
    with st.container(key="card_perf"):
        m = st.columns(4)
        m[0].metric("Recall (cases caught)", f"{r['recall']:.1%}")
        m[1].metric("Precision", f"{r['precision']:.1%}")
        m[2].metric("False-positive rate", f"{r['fpr']:.1%}")
        m[3].metric("Flagged overall", f"{r['flag_rate']:.1%}")
        st.markdown("**Per 1,000 adults screened** (held-out test data, survey prevalence)")
        st.dataframe(pd.DataFrame(
            {"Model flags": [round(tp * scale), round(fp * scale)],
             "Model does not flag": [round(fn * scale), round(tn * scale)]},
            index=["Reported diabetes/prediabetes", "Did not report it"]), width="stretch")
        st.caption(f"Missed cases (false negatives): **{fn * scale:.0f}** · Unnecessary flags (false positives): **{fp * scale:.0f}**")

# ---- charts
curve_val = core.cost_curve(val["y"], val["p_final"], ratio, 1.0)
curve_test = core.cost_curve(test["y"], test["p_final"], ratio, 1.0)
f1, ax1 = plt.subplots(figsize=(5.5, 3.4))
ax1.plot(curve_val["threshold"], curve_val["cost"] / len(val), color="#1b6ca8")
ax1.axvline(t, color="#d95f02", ls="--")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Assumed cost per person (validation)")
ax1.set_title("Expected cost vs. threshold")
f2, ax2 = plt.subplots(figsize=(5.5, 3.4))
ax2.plot(curve_test["threshold"], curve_test["recall"], label="Recall", color="#d95f02")
ax2.plot(curve_test["threshold"], curve_test["precision"], label="Precision", color="#1b6ca8")
ax2.plot(curve_test["threshold"], curve_test["flag_rate"], label="Share flagged", color="#777", ls=":")
ax2.axvline(t, color="#333", ls="--")
ax2.set_xlabel("Threshold")
ax2.set_title("Recall, precision and share flagged (test)")
ax2.legend()
for ax in (ax1, ax2):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
c1, c2 = st.columns(2)
with c1.container(key="card_chart_cost"):
    st.pyplot(f1, clear_figure=True)
with c2.container(key="card_chart_pr"):
    st.pyplot(f2, clear_figure=True)

st.markdown(
    "**How to read this:** with a high cost ratio the best threshold moves down, the model flags more people and "
    "recall rises, while precision falls. Choosing a threshold is a value judgement about "
    "missed cases versus unnecessary tests and testing capacity, not something the data can decide alone."
)
