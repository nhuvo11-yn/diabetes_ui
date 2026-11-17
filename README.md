# Diabetes Risk Screening and Health Equity

A machine-learning decision-support app (Streamlit) that helps an adult decide whether to ask a health professional about a blood glucose test, and checks whether the tool behaves similarly across income and education groups.

> **Educational project — not for clinical use.** The model predicts a *survey label*, not a diagnosis.

-   **Live app:** `<paste Streamlit Community Cloud link>`
-   **Demo video (3 min):** `<paste link>`
-   **Team / course:** `<names, course, date>`

## 1. Problem and user-decision statement

Diabetes and prediabetes often go undetected, and access to testing is uneven. Early screening prompts matter, but a screening tool that works well for some groups and poorly for others can widen health gaps.

**User-decision statement:** *An adult answering health-related questions at home uses the application to assess their likelihood of reporting diabetes or prediabetes and decide whether to seek a blood glucose test.*

The app answers with a score, a flag/no-flag recommendation, the answers that moved the score, and a concrete next step. It never says "you have diabetes" or "you are healthy".

## 2. Data

CDC BRFSS 2015 *Diabetes Health Indicators* (`diabetes_binary_health_indicators_BRFSS2015.csv`): 253,680 survey responses, 21 indicators, target `Diabetes_binary` (1 = respondent **reported** diabetes or prediabetes). No missing values. 24,206 exact duplicate rows were removed → **229,474 rows, 15.3% positive** (13.9% before de-duplication).

## 3. Method

| Step | Choice |
|----|----|
| Split | Stratified 60/20/20 train/validation/test, `random_state=42` |
| Pipeline | `ColumnTransformer` (scale numeric columns) + model in **one** scikit-learn `Pipeline`, saved with Joblib. No scaler is fit outside the training data. |
| Models | Logistic Regression (baseline) and Histogram Gradient Boosting, each on **all 21 variables** and on a **7-question** short form |
| Metrics | PR-AUC and ROC-AUC (threshold-free); recall, precision, F1, false-positive rate at the chosen threshold. Accuracy is not used for selection (label is imbalanced). |
| Threshold | Not 0.5. Chosen on the validation set to minimize an **assumed** cost (a missed case counts 5× an unnecessary test). Adjustable in the app. |
| Explanation | Per-person "what pushed your score" (replace one answer with the typical answer and measure the change), permutation importance, logistic-regression odds ratios |
| Equity | Recall, miss rate, false-positive rate, precision by income (3 groups) and education (4 groups), with 200-sample bootstrap 95% CIs |

## 4. Results (held-out test set, n = 45,895)

| Model                             | Questions | PR-AUC    | ROC-AUC   |
|-----------------------------------|-----------|-----------|-----------|
| Logistic Regression               | 7         | 0.405     | 0.804     |
| Logistic Regression               | 21        | 0.414     | 0.810     |
| **Gradient Boosting (app model)** | **7**     | **0.435** | **0.812** |
| Gradient Boosting                 | 21        | 0.446     | 0.819     |

A random scorer would have PR-AUC ≈ 0.15 (the positive rate). The 7-question model loses only \~0.01 PR-AUC versus all 21 variables, so a short at-home questionnaire is nearly as informative.

**App model at threshold ≈ 0.185:** recall 71.7%, precision 34.0%, false-positive rate 25.1%, 32% of people flagged. Roughly one in three flagged people reports the label, and about 28% of people who do are not flagged.

**Equity finding (same threshold for everyone):**

| Group                 | Recall | False-positive rate |
|-----------------------|--------|---------------------|
| Income \< \$25k       | 79.5%  | 38.5%               |
| Income \$25k–\$75k    | 70.4%  | 25.6%               |
| Income \$75k+         | 59.3%  | 15.3%               |
| College graduate      | 64.0%  | 18.1%               |
| Less than high school | 83.3%  | 44.0%               |

Higher-income and college-educated respondents are **missed more often**, while lower-income and less-educated respondents receive **more false alarms**, even though the app model does not use income or education as inputs. Much of this follows from different underlying rates (23% of the lowest-income group vs 10% of the highest report the label). It is a signal to investigate and to choose thresholds deliberately, not proof of discrimination. See the *Health Equity* page.

## 5. Limitations

1.  Self-reported label and answers; no measured glucose.
2.  US adults in 2015; not representative of Vietnam or today. Asian populations can have risk at lower BMI.
3.  The model finds statistical associations, not causes; explanations are not causal claims.
4.  Moderate discrimination (precision ≈ 34%): many flags are unnecessary, some cases are missed.
5.  Cost weights are assumptions for simulation, not measured costs.
6.  Performance differs across income and education groups (section 4); groups were merged to keep sizes adequate.

## 6. Run it

``` bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train.py            # rebuilds models/ and reports/ from data/ (about 15 s)
streamlit run app.py
```

`train.py --keep-duplicates` and `--cost-fn 8` reproduce the sensitivity checks in `DECISION_LOG.md`.

## 7. Repository

```         
app.py                      Page 1: Risk Screening + explanation
pages/1_Threshold_and_Cost.py   Threshold & cost simulator
pages/2_Health_Equity.py        Subgroup performance with confidence intervals
pages/3_About_Model.py          Data, models, metrics, limitations
core.py  utils.py           Shared logic (no Streamlit in core.py)
train.py                    Rebuilds everything from raw data
models/                     pipeline.joblib, model_meta.json
reports/                    comparison, fairness, importance tables, test scores
data/                       raw CSV + sample_input.csv
notebooks/exploratory_analysis.ipynb
DECISION_LOG.md  AI_USE_APPENDIX.md
```

The app loads the model once with `@st.cache_resource`, runs predictions only when the form is submitted (`st.form`), and shows the disclaimer on every page's sidebar.
