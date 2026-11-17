"""Page 4 - About the model: data, models, metrics, threshold, limitations."""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import core
import utils

utils.setup_page("About the Model", "ℹ️")
meta = utils.load_meta()
comp = utils.load_report("model_comparison.csv")
perm = utils.load_report("permutation_importance.csv")
perm_full = utils.load_report("permutation_importance_full.csv")
odds = utils.load_report("logreg_odds_ratios.csv")
data, tm, ca = meta["data"], meta["test_metrics"], meta["cost_assumptions"]
at = tm["at_threshold"]

st.title("About the model")
st.warning(meta["disclaimer"] + " Results describe patterns in a 2015 US survey and are not medical advice.")

with st.container(key="card_decision"):
    st.header("Decision this tool supports")
    st.markdown(
        "*An adult answering health-related questions at home uses the application to assess their likelihood of "
        "reporting diabetes or prediabetes and decide whether to seek a blood glucose test.*"
    )

with st.container(key="card_data"):
    st.header("Data")
    st.markdown(
        f"- **Source:** CDC Behavioral Risk Factor Surveillance System (BRFSS) 2015, "
        f"*Diabetes Health Indicators* (`{data['file']}`), {data['n_raw']:,} survey responses, 21 health indicators.\n"
        f"- **Target:** `{meta['target']}` = 1 if the respondent **reported** diabetes or prediabetes (self-reported label).\n"
        f"- **Cleaning:** {data['n_duplicate_rows']:,} exact duplicate rows "
        f"{'removed' if data['duplicates_removed'] else 'kept'} → {data['n_used']:,} rows; "
        f"{data['prevalence_used']:.1%} have the positive label ({data['prevalence_raw']:.1%} before de-duplication). No missing values.\n"
        f"- **Split:** stratified 60 / 20 / 20 (train {meta['split']['train']:,} · validation {meta['split']['val']:,} · "
        f"test {meta['split']['test']:,}). The threshold is chosen on validation; all reported metrics use the untouched test set.\n"
        f"- **Pipeline:** preprocessing (`ColumnTransformer`) and model are saved together in one scikit-learn Pipeline."
    )

with st.container(key="card_models"):
    st.header("Models compared")
    show = comp.copy()
    show["Questions"] = show["n_features"]
    show = show[["model", "Questions", "pr_auc_train", "pr_auc_test", "roc_auc_test", "recall", "precision", "f1"]]
    show.columns = ["Model", "Questions", "PR-AUC (train)", "PR-AUC (test)", "ROC-AUC (test)", "Recall", "Precision", "F1"]
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns[2:]}), hide_index=True, width="stretch")
    st.caption(
        f"Recall / precision / F1 use each model's own threshold (cost ratio {ca['cost_fn'] / ca['cost_fp']:g}:1, tuned on validation). "
        f"PR-AUC of a random scorer would be about {tm['prevalence_in_test']:.2f} (the positive rate), so {tm['pr_auc']:.2f} is a clear but modest signal."
    )
    st.markdown(
        f"**Model in the app:** {meta['model']['algorithm']} using only **{len(meta['features'])} easy questions** "
        f"({', '.join(meta['feature_labels'].values())}). Rule: {meta['model']['selection_rule']}. "
        "Using the short questionnaire instead of all 21 variables costs only a small amount of PR-AUC, "
        "and every question can be answered at home without a lab test."
    )

with st.container(key="card_perf"):
    st.header("Performance of the app model (test set)")
    m = st.columns(5)
    m[0].metric("PR-AUC", f"{tm['pr_auc']:.3f}")
    m[1].metric("ROC-AUC", f"{tm['roc_auc']:.3f}")
    m[2].metric("Recall", f"{at['recall']:.1%}")
    m[3].metric("Precision", f"{at['precision']:.1%}")
    m[4].metric("False-positive rate", f"{at['fpr']:.1%}")
    st.caption(f"At threshold {meta['threshold']:.2f}. Accuracy ({at['accuracy']:.1%}) is shown for completeness only: "
               "with an imbalanced label it hides how many cases are missed.")

with st.container(key="card_thr"):
    st.header("Threshold and cost assumptions")
    st.markdown(
        f"Default threshold **{meta['threshold']:.2f}** minimizes an *assumed* cost where a missed case (false negative) "
        f"counts **{ca['cost_fn']:g}×** an unnecessary test (false positive = {ca['cost_fp']:g}). "
        "These weights are assumptions for decision simulation, not measured medical costs. "
        "Change them on the Threshold & Cost page."
    )

with st.container(key="card_importance"):
    st.header("Which answers matter most?")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5.2, 3.4))
        p = perm.sort_values("importance_mean")
        ax.barh(p["label"], p["importance_mean"], xerr=p["importance_std"], color="#1b6ca8")
        ax.set_xlabel("Drop in PR-AUC when the answer is shuffled")
        ax.set_title("App model: permutation importance", fontsize=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        st.pyplot(fig, clear_figure=True)
    with c2:
        st.markdown("**Logistic regression odds ratios** (simple, readable baseline; per 1 unit, other answers fixed)")
        o = odds[["label", "odds_ratio"]].rename(columns={"label": "Answer", "odds_ratio": "Odds ratio"})
        st.dataframe(o.style.format({"Odds ratio": "{:.2f}"}), hide_index=True, width="stretch")
    st.caption("Importance describes what the model relies on. It does not show that any factor causes diabetes.")
    with st.expander("Why these 7 questions? (importance in the full 21-variable model)"):
        pf = perm_full.sort_values("importance_mean", ascending=False).head(10)[["label", "importance_mean"]]
        pf.columns = ["Variable", "Importance"]
        st.dataframe(pf.style.format({"Importance": "{:.4f}"}), hide_index=True, width="stretch")
        st.markdown("The short questionnaire keeps the strongest at-home answers. Physical activity was tested and "
                    "dropped because it did not change validation PR-AUC.")

with st.container(key="card_limits"):
    st.header("Limitations")
    st.markdown(
        "1. **Self-reported label and answers.** Nobody's glucose was measured; \"diabetes or prediabetes\" is what respondents reported.\n"
        "2. **US adults, 2015.** The data do not represent Vietnam or other countries, and health patterns have changed since 2015. "
        "In Asian populations, risk can appear at lower BMI than in this US sample, so scores may under-flag some people.\n"
        "3. **Association, not cause.** The model learns statistical patterns; importance and \"what pushed your score\" are not causal claims.\n"
        f"4. **Moderate discrimination.** About {at['precision']:.0%} of flagged respondents report the label (precision) and about {at['fnr']:.0%} of cases are missed at the default threshold.\n"
        "5. **Uneven performance across groups.** See the Health Equity page; a single threshold does not give every group the same error rates.\n"
        "6. **Survey, not a clinic.** Treat the output as a prompt to talk to a professional, never as a diagnosis."
    )

with st.expander("Reproducibility"):
    st.json({"created_utc": meta["created_utc"], "library_versions": meta["library_versions"],
             "random_state": meta["split"]["random_state"]})
    st.markdown("Everything above is regenerated from raw data with `python train.py`.")
