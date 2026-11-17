"""Page 1 - Risk Screening (+ explanation of the result). Run with:  streamlit run app.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import core
import utils

utils.setup_page("Risk Screening")
meta = utils.load_meta()
pipe = utils.load_pipeline()
features = meta["features"]
threshold, overridden = utils.active_threshold(meta)

with st.container(key="hero"):
    st.title("Diabetes Screening Assistant")
    st.markdown(
        "**Who is this for?** An adult answering a few health questions at home who wants to decide "
        "**whether to ask a health professional about a blood glucose test.**"
    )
    st.caption(
        f"{len(features)} questions · takes about a minute · nothing you enter is stored. "
        "The tool estimates how similar your answers are to people who *reported* diabetes or prediabetes in a US "
        "health survey. It does not diagnose anything."
    )

# ----------------------------------------------------------------------------- form
with st.form("screening_form"):
    left, right = st.columns(2)
    with left:
        age_key = st.selectbox("Age group (years)", list(core.AGE_BANDS), index=5,
                               format_func=lambda k: core.AGE_BANDS[k])
        height = st.number_input("Height (cm)", min_value=120, max_value=220, value=165, step=1)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=65, step=1)
        genhlth = st.selectbox("In general, how would you rate your health?", list(core.GENHLTH_LABELS), index=2,
                               format_func=lambda k: core.GENHLTH_LABELS[k])
    with right:
        highbp = st.radio("Has a health professional ever told you that you have **high blood pressure**?",
                          ["No", "Yes"], horizontal=True)
        highchol = st.radio("Has a health professional ever told you that you have **high cholesterol**?",
                            ["No", "Yes"], horizontal=True)
        heart = st.radio("Have you ever been told you have **coronary heart disease** or had a **heart attack**?",
                         ["No", "Yes"], horizontal=True)
        diffwalk = st.radio("Do you have **serious difficulty walking or climbing stairs**?",
                            ["No", "Yes"], horizontal=True)
    submitted = st.form_submit_button("Analyze", type="primary")

if submitted:
    bmi = float(np.clip(weight / (height / 100) ** 2, meta["input_ranges"]["BMI"]["min"], meta["input_ranges"]["BMI"]["max"]))
    row = {
        "Age": float(age_key), "BMI": bmi, "HighBP": float(highbp == "Yes"), "HighChol": float(highchol == "Yes"),
        "GenHlth": float(genhlth), "DiffWalk": float(diffwalk == "Yes"), "HeartDiseaseorAttack": float(heart == "Yes"),
    }
    x = pd.DataFrame([row])[features]
    score = float(pipe.predict_proba(x)[0, 1])
    contrib = core.local_contributions(pipe, x, features, meta["reference_values"])
    st.session_state["result"] = {"x": x, "score": score, "contrib": contrib}

res = st.session_state.get("result")
if res is None:
    st.info("Fill in the form and press **Analyze**.")
    st.stop()

# ----------------------------------------------------------------------------- result
score, x, contrib = res["score"], res["x"], res["contrib"]
flagged = score >= threshold
base_rate = meta["data"]["prevalence_used"]

st.divider()
st.subheader("Your result")
c1, c2, c3 = st.columns(3)
c1.metric("Model score", f"{score:.1%}",
          help="Share of survey respondents with answers like yours who reported diabetes/prediabetes, "
               "as estimated by the model. It is NOT your personal probability of having diabetes.")
c2.metric("Flagging threshold", f"{threshold:.1%}" + (" (changed in simulator)" if overridden else ""),
          help="Scores at or above this value are flagged. Set on the Threshold & Cost page.")
c3.metric("Survey average", f"{base_rate:.1%}", help="Share of respondents reporting diabetes/prediabetes in the data.")
st.caption(f"Your BMI from height and weight: **{x.iloc[0]['BMI']:.1f}**")

if flagged:
    st.warning(
        "**The model flagged this response.** Consider contacting a healthcare professional to discuss a "
        "blood glucose test (for example fasting glucose or HbA1c)."
    )
else:
    st.success(
        "**The model did not flag this response as positive.** Continue routine health care and consult a "
        "healthcare professional if you have symptoms or concerns."
    )
st.caption("A result below the threshold is not a clean bill of health, and a flag is not a diagnosis. "
           "Only a blood test ordered by a health professional can confirm diabetes or prediabetes.")

# ----------------------------------------------------------------------------- explanation
st.subheader("What pushed your score up or down?")
shown = contrib[contrib["effect"].abs() >= 0.001].copy()
shown["abs"] = shown["effect"].abs()
shown = shown.sort_values("abs", ascending=False).head(5).iloc[::-1]

with st.container(key="card_explain"):
    if shown.empty:
        st.write("Your answers are all close to the typical respondent, so no single answer moves the score noticeably.")
    else:
        fig, ax = plt.subplots(figsize=(7, 0.55 * len(shown) + 1))
        labels = [f"{meta['feature_labels'][f]}: {core.format_value(f, v)}" for f, v in zip(shown["feature"], shown["your_value"])]
        colors = ["#d95f02" if e > 0 else "#1b6ca8" for e in shown["effect"]]
        ax.barh(labels, shown["effect"] * 100, color=colors)
        ax.axvline(0, color="#444", lw=0.8)
        ax.set_xlabel("Change in model score (percentage points) vs. a typical respondent's answer")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for i, e in enumerate(shown["effect"] * 100):
            ax.text(e, i, f" {e:+.1f}", va="center", ha="left" if e >= 0 else "right", fontsize=9)
        st.pyplot(fig, clear_figure=True)
        st.caption("Orange = raises the score, blue = lowers it.")

table = contrib.assign(
    Question=[meta["feature_labels"][f] for f in contrib["feature"]],
    **{"Your answer": [core.format_value(f, v) for f, v in zip(contrib["feature"], contrib["your_value"])],
       "Typical respondent": [core.format_value(f, v) for f, v in zip(contrib["feature"], contrib["typical_value"])],
       "Effect (pp)": (contrib["effect"] * 100).round(1)},
)[["Question", "Your answer", "Typical respondent", "Effect (pp)"]]
with st.expander("All answers and how to read this"):
    st.dataframe(table, hide_index=True, width="stretch")
    st.markdown(
        "- Each bar shows how the score would change if **only that one answer** were replaced by the typical "
        "respondent's answer (median or most common answer in the training data).\n"
        "- Answers interact, so the bars do not add up exactly to the total score.\n"
        "- This describes how the **model behaves**, not what causes diabetes."
    )
