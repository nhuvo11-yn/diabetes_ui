"""Page 3 - Health equity: does the model behave similarly across income and education groups?"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import core
import utils

utils.setup_page("Health Equity", "🌍")
meta = utils.load_meta()
fair = utils.load_report("fairness.csv")
ev = utils.load_eval_predictions()
test = ev[ev.split == "test"]

st.title("Health equity check")
st.markdown(
    "Does the screening tool behave similarly for people with different incomes and education levels? "
    "We evaluate on held-out test data, split by group, using the **same threshold for everyone**. "
    "Error bars are 95% bootstrap confidence intervals: small groups are less certain."
)

model_labels = {
    "final": f"App model ({len(meta['features'])} questions, no income/education inputs)",
    "full": f"Full model ({meta['full_model_reference']['n_features']} variables, includes income & education)",
}
with st.container(key="card_filters"):
    c1, c2 = st.columns(2)
    model_key = c1.radio("Model", list(model_labels), format_func=model_labels.get)
    grouping = c2.radio("Group by", ["Income", "Education"], horizontal=True)

d = fair[(fair.model == model_key) & (fair.grouping == grouping)].reset_index(drop=True)
overall = fair[(fair.model == model_key) & (fair.grouping == "Overall")].iloc[0]
thr = float(d["threshold"].iloc[0])

# ---- table
def ci(row, m):
    return f"{row[m]:.1%} ({row[m + '_lo']:.1%}–{row[m + '_hi']:.1%})"

table = pd.DataFrame({
    "Group": d["group"], "n": d["n"], "Share with label": d["prevalence"].map("{:.1%}".format),
    "Recall (cases caught)": d.apply(lambda r: ci(r, "recall"), axis=1),
    "Miss rate (FNR)": d.apply(lambda r: ci(r, "fnr"), axis=1),
    "False-positive rate": d.apply(lambda r: ci(r, "fpr"), axis=1),
    "Precision": d.apply(lambda r: ci(r, "precision"), axis=1),
    "Flagged": d["flag_rate"].map("{:.1%}".format),
})
with st.container(key="card_table"):
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption(f"Threshold {thr:.2f} · Overall recall {overall['recall']:.1%}, false-positive rate {overall['fpr']:.1%}.")

# ---- chart
with st.container(key="card_chart"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, m, title, color in ((axes[0], "recall", "Recall: share of cases the model flags", "#d95f02"),
                                (axes[1], "fpr", "False-positive rate: non-cases flagged", "#1b6ca8")):
        yerr = np.array([d[m] - d[m + "_lo"], d[m + "_hi"] - d[m]])
        ax.bar(range(len(d)), d[m], yerr=yerr, color=color, capsize=4)
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([g.split(" (")[0].replace("Less than high school", "< High school") for g in d["group"]],
                           rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        ax.axhline(overall[m], color="#333", ls="--", lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    st.pyplot(fig, clear_figure=True)
    st.caption("Dashed line = overall rate.")

# ---- neutral, data-driven summary
lo_r, hi_r = d.loc[d["recall"].idxmin()], d.loc[d["recall"].idxmax()]
lo_f, hi_f = d.loc[d["fpr"].idxmin()], d.loc[d["fpr"].idxmax()]
overlap = hi_r["recall_lo"] <= lo_r["recall_hi"]
with st.container(key="card_summary"):
    st.subheader("What the numbers say")
    st.markdown(
        f"- **Cases caught (recall)** ranges from **{lo_r['recall']:.1%}** ({lo_r['group']}) to **{hi_r['recall']:.1%}** "
        f"({hi_r['group']}); the confidence intervals {'overlap' if overlap else 'do not overlap'}.\n"
        f"- **False alarms (false-positive rate)** range from **{lo_f['fpr']:.1%}** ({lo_f['group']}) to "
        f"**{hi_f['fpr']:.1%}** ({hi_f['group']}).\n"
        f"- The share of people in each group who reported diabetes/prediabetes also differs "
        f"({d['prevalence'].min():.1%} to {d['prevalence'].max():.1%}). With one shared threshold, groups with a higher "
        f"underlying rate tend to be flagged more often, which raises both recall and false alarms for that group."
    )

with st.expander("How to interpret this (and what it does not show)"):
    st.markdown(
        "- A difference in recall or false-positive rate between groups is a **signal to investigate**, not proof of "
        "discrimination. It can come from different base rates, how the survey measured the label, group size, or "
        "model limits.\n"
        "- Self-reported diabetes in a survey depends on who has had access to testing, so the label itself can be "
        "less complete for some groups.\n"
        "- The app model does not use income or education as inputs, yet performance still differs by those "
        "groups (leaving a variable out does not guarantee equal performance). The full model uses them; compare both above.\n"
        "- Income and education were grouped (3 income bands, 4 education groups) so every group has enough people. "
        "No single metric settles whether a tool is fair; we report several."
    )

# ---- live threshold exploration (point estimates only)
with st.expander("Explore a different threshold (point estimates, no confidence intervals)"):
    pcol = "p_final" if model_key == "final" else "p_full"
    gcol = "income_group" if grouping == "Income" else "edu_group"
    order = core.INCOME_ORDER if grouping == "Income" else core.EDU_ORDER
    t2 = st.slider("Threshold", 0.05, 0.60, float(thr), 0.01, key="eq_thr")
    rows = []
    for g in order:
        sub = test[test[gcol] == g]
        r = core.rates(*core.confusion_counts(sub["y"], sub[pcol], t2))
        rows.append({"Group": g, "Recall": f"{r['recall']:.1%}", "Miss rate": f"{r['fnr']:.1%}",
                     "False-positive rate": f"{r['fpr']:.1%}", "Precision": f"{r['precision']:.1%}",
                     "Flagged": f"{r['flag_rate']:.1%}"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
