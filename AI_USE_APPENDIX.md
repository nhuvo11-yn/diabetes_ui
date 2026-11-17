# AI Use Appendix

> **Team: edit this file before submitting.** The "What we changed / verified" column must describe what *you* actually did. Delete rows that do not apply and add rows for anything else. Every team member must be able to explain the code.

**Tools used:** Claude (Anthropic) for planning, code drafting and code review. `<add ChatGPT / Copilot if used>`

| Step | AI contribution | What we changed / verified (fill in) |
|---|---|---|
| Project planning | Drafted the project blueprint, scope cuts and decision-log structure | `<e.g. agreed the user-decision statement as a team; removed X>` |
| Data profiling | Ran a script to count duplicates, check ranges and group sizes | `<e.g. re-ran in our notebook and got the same 24,206 duplicates>` |
| `train.py` | Drafted the data loading, 60/20/20 split, Pipeline with `ColumnTransformer`, models, cost-based threshold, bootstrap confidence intervals | `<e.g. read every function, ran it, changed cost ratio, checked no preprocessing is fit outside the Pipeline>` |
| Feature choice | Suggested the 7-question short form after comparing validation PR-AUC with and without each question | `<e.g. we repeated the comparison / agreed with dropping physical activity>` |
| `core.py`, `utils.py` | Drafted shared metric, threshold and loading helpers | `<...>` |
| Streamlit pages | Drafted the 4 pages (form, explanation chart, simulator, equity, about) | `<e.g. edited wording, tested on phone, translated labels>` |
| Testing | Ran headless Streamlit tests on every page and a server smoke test | `<e.g. we also tested the deployed link manually with 5 profiles from data/sample_input.csv>` |
| Documentation | Drafted README, decision log, demo script | `<e.g. we verified every number against reports/results_summary.md>` |

## How we checked the AI output
- Re-ran `python train.py` ourselves and compared the numbers in the README with `reports/results_summary.md`.
- `<Describe any bug you found or any AI suggestion you rejected, and why.>`
- `<Describe how each member verified they understand: pipeline, threshold, fairness table, explanation method.>`

## What the AI did not do
- It did not decide the project topic, the user-decision statement, or the cost assumptions; the team did. `<confirm>`
- It had no access to the assignment brief; the team checked the requirements. `<confirm>`
