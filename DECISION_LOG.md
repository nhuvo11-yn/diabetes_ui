# Decision Log

Numbers come from `train.py` (seed 42). Add your own entries as the project evolves.

| # | Decision | Chosen | Alternatives considered | Reason / evidence |
|---|---|---|---|---|
| 1 | Dataset file | `diabetes_binary_health_indicators_BRFSS2015.csv` | 50/50 balanced file; 3-class `diabetes_012` file | Assignment names the imbalanced file; a balanced file hides the real prevalence and distorts precision and thresholds. |
| 2 | Duplicate rows | Remove 24,206 exact duplicates (9.5%) | Keep them | Identical rows could land in both train and test. Caveat: with only 21 coarse variables, identical rows may be different people. **Sensitivity check** (`--keep-duplicates`): PR-AUC test 0.412 (prevalence 13.9%) vs 0.435 (15.3%); ROC-AUC 0.821 vs 0.812. Conclusions unchanged; PR-AUC values are not directly comparable because prevalence differs. |
| 3 | Missing values / outliers | None missing; BMI kept as is (range 12–98) | Clip BMI | Tree model is robust; app limits BMI to the observed range. |
| 4 | Split | Stratified 60/20/20 | 80/20 only | Need a validation set to pick the threshold without touching the test set. |
| 5 | Leakage control | Preprocessing inside the Pipeline; threshold tuned on validation; test used once for reporting | Fit scaler on all data | Prevents information from validation/test reaching training. |
| 6 | Models | Logistic Regression baseline + Histogram Gradient Boosting | Decision Tree, Random Forest | Time constraint; one linear baseline and one flexible model are enough to answer "does complexity help?" (PR-AUC 0.414 → 0.446 on 21 variables). |
| 7 | Tuning | Fixed sensible hyper-parameters, early stopping | Grid search | Train vs test PR-AUC are close (0.433 train vs 0.435 test for the app model; the 21-variable boosting model shows a larger gap, 0.457 vs 0.446), so no sign of overfitting; time-boxed. |
| 8 | Metrics | PR-AUC, ROC-AUC, recall, precision, F1, FPR | Accuracy | Positive class is ~15%; accuracy hides missed cases. |
| 9 | Threshold | Cost-based, 5:1 (FN:FP), ≈0.185 | 0.5; maximize F1 | Screening prefers catching cases. Cost ratio is an **assumption**; the app lets users change it. |
| 10 | Questionnaire | 7 questions: Age, BMI (from height/weight), high BP, high cholesterol, general health, difficulty walking, heart disease | All 21 variables | Only things answerable at home. PR-AUC 0.432 (7) vs 0.440 (21) on validation. |
| 11 | Dropped question | Physical activity | Keep it | Validation PR-AUC identical with and without it (0.4316). |
| 12 | Excluded from the short form | Income, education, sex, CholCheck, AnyHealthcare, NoDocbcCost, lifestyle items, MentHlth/PhysHlth | Include | Not needed for accuracy (adding Income: +0.001, Sex: +0.003 PR-AUC) and keeps the form short; income/education are used only to *evaluate* equity. |
| 13 | Final model | Gradient Boosting, 7 questions | Logistic Regression | Best validation PR-AUC among short-form models (0.432 vs 0.403); logistic odds ratios are still reported for readability. |
| 14 | Explanation method | Replace-one-answer-with-typical-answer effects + permutation importance + odds ratios | SHAP | Model-agnostic, no extra dependency, fast in the app. Not a causal claim. SHAP could be added later. |
| 15 | Fairness groups | Income 3 bands; education 4 groups (levels 1–3 merged) | Raw 8 income / 6 education levels | Education level 1 has only ~170 rows; merging gives usable group sizes. Bootstrap CIs show uncertainty. |
| 16 | Fairness metrics | Recall, miss rate, FPR, precision, flag rate | Single "fairness score" | One metric cannot settle fairness; groups have different base rates. |
| 17 | App scope | 4 pages (screening+explanation, threshold/cost, equity, about) | Separate explanation page; age-group fairness; calibration; user testing | Time-boxed; listed under "future work". |

## Future work (not done)
Calibration curves, SHAP, age-group fairness, user testing with real adults, a Vietnamese-population validation set.
