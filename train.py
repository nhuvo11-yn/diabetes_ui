"""train.py - rebuild every model artifact from the raw BRFSS 2015 CSV.

    python train.py                       # default: remove duplicate rows, cost ratio FN:FP = 5:1
    python train.py --keep-duplicates     # sensitivity check
    python train.py --cost-fn 8 --cost-fp 1

Outputs
    models/pipeline.joblib             final sklearn Pipeline (ColumnTransformer + model)
    models/model_meta.json             features, input ranges, threshold, cost assumptions, test metrics
    reports/model_comparison.csv       4 models (2 algorithms x 2 feature sets)
    reports/fairness.csv               subgroup metrics with bootstrap 95% CIs
    reports/permutation_importance*.csv, reports/logreg_odds_ratios.csv
    reports/eval_predictions.csv.gz    validation + test scores (used by the app's simulators)
    reports/results_summary.md         tables you can paste into the README
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import core

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42
ALGO_NAMES = {"logreg": "Logistic Regression", "hgb": "Histogram Gradient Boosting"}


# --------------------------------------------------------------------------- #
# 1. Data
# --------------------------------------------------------------------------- #
def load_data(path: Path, keep_duplicates: bool):
    df = pd.read_csv(path)
    missing = set(core.FULL_FEATURES + [core.TARGET]) - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {path.name}: {sorted(missing)}. "
                         "Use diabetes_binary_health_indicators_BRFSS2015.csv (not the 5050 / 012 file).")
    df = df[[core.TARGET] + core.FULL_FEATURES]
    assert df.isna().sum().sum() == 0, "unexpected missing values"
    assert set(df[core.TARGET].unique()) <= {0, 1}, "target must be 0/1"

    n_raw = len(df)
    n_dup = int(df.duplicated().sum())
    prev_raw = float(df[core.TARGET].mean())
    if not keep_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    info = {
        "file": path.name, "n_raw": n_raw, "n_duplicate_rows": n_dup,
        "duplicates_removed": not keep_duplicates, "n_used": len(df),
        "prevalence_raw": prev_raw, "prevalence_used": float(df[core.TARGET].mean()),
    }
    return df, info


# --------------------------------------------------------------------------- #
# 2. Models: ColumnTransformer + estimator inside ONE Pipeline (no leakage)
# --------------------------------------------------------------------------- #
def build_pipeline(algo: str, features: list[str]) -> Pipeline:
    num = [c for c in core.NUMERIC_FEATURES if c in features]
    pre = ColumnTransformer([("num", StandardScaler(), num)], remainder="passthrough")
    if algo == "logreg":
        clf = LogisticRegression(max_iter=2000)
    elif algo == "hgb":
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(algo)
    return Pipeline([("prep", pre), ("model", clf)])


def logreg_odds_ratios(pipe: Pipeline, features: list[str]) -> pd.DataFrame:
    """Odds ratios in ORIGINAL units (undo the StandardScaler on numeric columns)."""
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["model"]
    num = [c for c in core.NUMERIC_FEATURES if c in features]
    names = num + [c for c in features if c not in num]
    scale = dict(zip(num, prep.named_transformers_["num"].scale_))
    rows = []
    for name, coef in zip(names, clf.coef_[0]):
        raw = coef / scale[name] if name in scale else coef
        rows.append({"feature": name, "label": core.FEATURE_LABELS[name],
                     "coef": raw, "odds_ratio": float(np.exp(raw))})
    return pd.DataFrame(rows).sort_values("odds_ratio", ascending=False)


# --------------------------------------------------------------------------- #
# 3. Fairness table
# --------------------------------------------------------------------------- #
def fairness_table(test: pd.DataFrame, models: dict[str, tuple[str, float]], n_boot: int) -> pd.DataFrame:
    rows = []
    groupings = [("Income", "income_group", core.INCOME_ORDER), ("Education", "edu_group", core.EDU_ORDER)]
    for model_key, (pcol, thr) in models.items():
        subsets = [("Overall", "All test respondents", test)]
        for gname, gcol, order in groupings:
            for g in order:
                subsets.append((gname, g, test[test[gcol] == g]))
        for gname, g, sub in subsets:
            r = core.rates(*core.confusion_counts(sub["y"], sub[pcol], thr))
            ci = core.bootstrap_ci(sub["y"], sub[pcol], thr, n_boot=n_boot, seed=RANDOM_STATE)
            row = {"model": model_key, "grouping": gname, "group": g, "threshold": thr,
                   "n": r["n"], "n_pos": r["tp"] + r["fn"], "prevalence": (r["tp"] + r["fn"]) / r["n"],
                   "flag_rate": r["flag_rate"]}
            for m in ("recall", "precision", "fnr", "fpr"):
                row[m] = r[m]
                row[f"{m}_lo"], row[f"{m}_hi"] = ci[m]
            rows.append(row)
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"))
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--cost-fn", type=float, default=5.0, help="assumed cost of a false negative")
    ap.add_argument("--cost-fp", type=float, default=1.0, help="assumed cost of a false positive")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT), help="output root (default: repo root)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df, info = load_data(Path(args.data), args.keep_duplicates)
    print(f"rows raw={info['n_raw']:,} duplicates={info['n_duplicate_rows']:,} used={info['n_used']:,} "
          f"prevalence {info['prevalence_raw']:.3%} -> {info['prevalence_used']:.3%}")

    # 60 / 20 / 20 stratified split. Threshold is chosen on VALIDATION, metrics reported on TEST.
    X, y = df[core.FULL_FEATURES], df[core.TARGET].astype(int)
    X_tv, X_te, y_tv, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=RANDOM_STATE)
    print(f"train={len(X_tr):,} val={len(X_va):,} test={len(X_te):,}")

    feature_sets = {"minimal": core.MINIMAL_FEATURES, "full": core.FULL_FEATURES}
    fitted, scores, rows = {}, {}, []
    for algo in ("logreg", "hgb"):
        for fs_name, feats in feature_sets.items():
            pipe = build_pipeline(algo, feats).fit(X_tr[feats], y_tr)
            p = {"train": pipe.predict_proba(X_tr[feats])[:, 1],
                 "val": pipe.predict_proba(X_va[feats])[:, 1],
                 "test": pipe.predict_proba(X_te[feats])[:, 1]}
            thr = core.choose_threshold(y_va, p["val"], args.cost_fn, args.cost_fp)
            r = core.rates(*core.confusion_counts(y_te, p["test"], thr))
            rows.append({
                "model": f"{ALGO_NAMES[algo]} ({fs_name})", "algo": algo, "feature_set": fs_name,
                "n_features": len(feats),
                "pr_auc_train": average_precision_score(y_tr, p["train"]),
                "pr_auc_val": average_precision_score(y_va, p["val"]),
                "pr_auc_test": average_precision_score(y_te, p["test"]),
                "roc_auc_test": roc_auc_score(y_te, p["test"]),
                "threshold": thr, "recall": r["recall"], "precision": r["precision"], "f1": r["f1"],
                "accuracy": r["accuracy"], "fpr": r["fpr"], "flag_rate": r["flag_rate"],
            })
            fitted[(algo, fs_name)], scores[(algo, fs_name)] = pipe, p
            print(f"  {rows[-1]['model']:<48} PR-AUC val={rows[-1]['pr_auc_val']:.3f} test={rows[-1]['pr_auc_test']:.3f} "
                  f"ROC-AUC={rows[-1]['roc_auc_test']:.3f} thr={thr:.2f} recall={r['recall']:.3f} precision={r['precision']:.3f}")
    comp = pd.DataFrame(rows)
    comp.to_csv(out / "reports" / "model_comparison.csv", index=False)

    # Final model = best VALIDATION PR-AUC among the short-questionnaire models.
    mini = comp[comp.feature_set == "minimal"].sort_values("pr_auc_val", ascending=False)
    final_algo = mini.iloc[0]["algo"]
    final_pipe, full_pipe = fitted[(final_algo, "minimal")], fitted[(final_algo, "full")]
    thr_final = float(mini.iloc[0]["threshold"])
    thr_full = float(comp[(comp.algo == final_algo) & (comp.feature_set == "full")].iloc[0]["threshold"])
    print(f"FINAL MODEL: {ALGO_NAMES[final_algo]} with {len(core.MINIMAL_FEATURES)} questions, threshold={thr_final:.2f}")

    # Evaluation scores for the app (threshold simulator + equity page)
    def frame(split, Xs, ys, key):
        return pd.DataFrame({
            "split": split, "y": ys.values,
            "p_final": scores[(final_algo, "minimal")][key], "p_full": scores[(final_algo, "full")][key],
            "income_group": core.income_group(Xs["Income"]).values,
            "edu_group": core.edu_group(Xs["Education"]).values,
        })
    ev = pd.concat([frame("val", X_va, y_va, "val"), frame("test", X_te, y_te, "test")], ignore_index=True)
    ev.to_csv(out / "reports" / "eval_predictions.csv.gz", index=False, float_format="%.5f", compression="gzip")
    test = ev[ev.split == "test"].reset_index(drop=True)

    # Fairness (bootstrap CIs)
    fair = fairness_table(test, {"final": ("p_final", thr_final), "full": ("p_full", thr_full)}, args.bootstrap)
    fair.to_csv(out / "reports" / "fairness.csv", index=False)

    # Interpretability
    rng = np.random.default_rng(RANDOM_STATE)
    sub = rng.choice(len(X_va), size=min(15000, len(X_va)), replace=False)
    for key, pipe, feats in (("", final_pipe, core.MINIMAL_FEATURES), ("_full", full_pipe, core.FULL_FEATURES)):
        pi = permutation_importance(pipe, X_va.iloc[sub][feats], y_va.iloc[sub], scoring="average_precision",
                                    n_repeats=5, random_state=RANDOM_STATE)
        pd.DataFrame({"feature": feats, "label": [core.FEATURE_LABELS[f] for f in feats],
                      "importance_mean": pi.importances_mean, "importance_std": pi.importances_std}
                     ).sort_values("importance_mean", ascending=False
                     ).to_csv(out / "reports" / f"permutation_importance{key}.csv", index=False)
    logreg_odds_ratios(fitted[("logreg", "minimal")], core.MINIMAL_FEATURES).to_csv(
        out / "reports" / "logreg_odds_ratios.csv", index=False)

    # Typical-respondent reference values (median for ordinal/numeric, mode for yes/no) from TRAIN only
    reference = {}
    for f in core.MINIMAL_FEATURES:
        reference[f] = float(X_tr[f].median()) if f in core.NON_BINARY else float(X_tr[f].mode().iloc[0])

    ranges = {}
    for f in core.FULL_FEATURES:
        ranges[f] = {"min": float(X[f].min()), "max": float(X[f].max()),
                     "type": "numeric/ordinal" if f in core.NON_BINARY else "binary (0/1)"}

    final_row = mini.iloc[0].to_dict()
    r_test = core.rates(*core.confusion_counts(y_te, test["p_final"], thr_final))
    meta = {
        "project": "Diabetes Risk Screening and Health Equity",
        "disclaimer": "Educational project — not for clinical use.",
        "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "target": core.TARGET,
        "target_meaning": "1 = respondent reported diabetes or prediabetes in the 2015 BRFSS survey (self-reported label)",
        "model": {"algorithm": ALGO_NAMES[final_algo], "algo_key": final_algo, "feature_set": "minimal",
                  "selection_rule": "best validation PR-AUC among the short-questionnaire models"},
        "features": core.MINIMAL_FEATURES,
        "feature_labels": {f: core.FEATURE_LABELS[f] for f in core.MINIMAL_FEATURES},
        "input_ranges": {f: ranges[f] for f in core.MINIMAL_FEATURES},
        "all_feature_ranges": ranges,
        "reference_values": reference,
        "threshold": thr_final,
        "cost_assumptions": {"cost_fn": args.cost_fn, "cost_fp": args.cost_fp,
                             "note": "ASSUMED relative costs for decision simulation, not measured medical costs."},
        "data": info,
        "split": {"train": len(X_tr), "val": len(X_va), "test": len(X_te), "random_state": RANDOM_STATE,
                  "scheme": "stratified 60/20/20; threshold tuned on validation, metrics on test"},
        "test_metrics": {
            "pr_auc": float(final_row["pr_auc_test"]), "roc_auc": float(final_row["roc_auc_test"]),
            "prevalence_in_test": float(y_te.mean()),
            "at_threshold": {k: r_test[k] for k in ("recall", "precision", "f1", "accuracy", "fpr", "fnr", "flag_rate",
                                                    "tp", "fp", "fn", "tn")},
        },
        "full_model_reference": {"algorithm": ALGO_NAMES[final_algo], "n_features": len(core.FULL_FEATURES),
                                 "threshold": thr_full},
        "library_versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                             "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out / "models" / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    joblib.dump(final_pipe, out / "models" / "pipeline.joblib", compress=3)

    # Paste-ready summary
    cmp_md = comp[["model", "n_features", "pr_auc_train", "pr_auc_val", "pr_auc_test", "roc_auc_test",
                   "threshold", "recall", "precision", "f1", "flag_rate"]]
    fin = fair[fair.model == "final"].copy()
    for m in ("recall", "fpr"):
        fin[m] = fin.apply(lambda r, m=m: f"{r[m]:.3f} ({r[m + '_lo']:.3f}-{r[m + '_hi']:.3f})", axis=1)
    summary = (
        f"# Results summary (auto-generated by train.py)\n\n"
        f"Data: {info['n_used']:,} rows used ({info['n_duplicate_rows']:,} duplicate rows "
        f"{'removed' if info['duplicates_removed'] else 'kept'}); prevalence {info['prevalence_used']:.1%}.\n\n"
        f"## Model comparison (test set; threshold tuned on validation, cost FN:FP = {args.cost_fn:g}:{args.cost_fp:g})\n\n"
        f"{md_table(cmp_md)}\n\n"
        f"Final app model: **{ALGO_NAMES[final_algo]}, {len(core.MINIMAL_FEATURES)} questions**, threshold {thr_final:.2f}.\n\n"
        f"## Subgroup performance of the final model (test set, 95% bootstrap CI)\n\n"
        f"{md_table(fin[['grouping', 'group', 'n', 'n_pos', 'recall', 'fpr']])}\n"
    )
    (out / "reports" / "results_summary.md").write_text(summary, encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s -> {out / 'models'} and {out / 'reports'}")


if __name__ == "__main__":
    main()
