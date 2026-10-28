# Predictive maintenance: failure detection on the AI4I 2020 dataset

IND8217E – Final project, Fall 2026, Polytechnique Montréal
Authors: [name 1], [name 2]

## Summary
A random forest classifier detects machine failures in the AI4I 2020 Predictive Maintenance Dataset (10,000 milling jobs, 3.4% failures). On the held-out test set (2,992 jobs, 99 failures) it catches 87 of 99 failures (recall 0.88) with 9 false alarms (precision 0.91), PR-AUC 0.93. All heat dissipation, power and overstrain failures are detected; the 12 missed failures are all tool wear failures, which occur at random within a tool wear window. Under the stated cost assumptions (CAD 1,000 per planned intervention, CAD 10,000 per breakdown) the model costs 78% less than running without predictive maintenance. Hypotheses and their outcomes are listed in `HYPOTHESES.md` and `results/09_hypotheses.csv`.

## Project structure
```
data/        raw dataset (ai4i2020.csv) and files generated from it
src/         notebooks (run in numerical order) and shared Python modules
results/     tables (CSV/JSON/TXT) written by the notebooks
figures/     figures (PNG) written by the notebooks
HYPOTHESES.md  hypotheses and decision rules, fixed before any model was trained
LOG.md         step-by-step record of what was done and why
requirements.txt
```

Shared modules in `src/`:
- `config.py` – paths, random seed (42), hypothesis thresholds, cost assumptions
- `data.py` – train/test loading and failure-mode strata
- `features.py` – feature engineering (power, temperature difference, strain) and preprocessing
- `models.py` – candidate models and hyperparameter grids

## Setup
Tested with Python 3.11.
```
python -m venv .venv
.venv\Scripts\activate          (Windows)   or   source .venv/bin/activate   (macOS/Linux)
pip install -r requirements.txt
```
Open the notebooks in Jupyter or VS Code with this environment as the kernel. The notebooks must be run from the `src/` folder (the default in Jupyter and VS Code), because they import the shared modules from there.

## How to reproduce the results
Run the notebooks in this order (Run All in each). Each one reads the outputs of the previous ones.

| Notebook | Purpose | Main outputs | Runtime |
|---|---|---|---|
| `01_data_check` | Integrity checks, label consistency, cleaning (27 inconsistent rows removed) | `data/ai4i2020_clean.csv` | seconds |
| `02_variables` | Variable categorization (dependent/independent, controllable) | `results/02_variable_categories.csv` | seconds |
| `03_eda` | Exploratory figures, failure mechanisms | `figures/03_*.png` | seconds |
| `04_split` | Stratified 70/30 train/test split (test set locked) | `data/train_ids.csv`, `data/test_ids.csv` | seconds |
| `05_features` | Feature engineering and information content | `figures/05_mutual_information.png` | seconds |
| `07_models` | Nested cross-validation of all candidates, model selection (H2, H3) | `results/07_*` | 2–3 min |
| `08_threshold` | Decision threshold and cost model on out-of-fold predictions | `results/08_*`, `figures/08_*.png` | about 1 min |
| `09_evaluation` | Final evaluation on the test set (H1, H4, H5) | `results/09_*`, `figures/09_*.png` | about 30 s |
| `10_interpretation` | Rules learned by the readable decision tree | `results/10_*` | seconds |

There is no notebook 06: step 6 (hypotheses) is the document `HYPOTHESES.md`.

All random steps use the fixed seed 42, so a full rerun reproduces the same split, models and numbers. The test set is used only in notebook 09 (and for no model decisions); the training and validation data are defined by `data/train_ids.csv` and the cross-validation folds in notebooks 07–08.

## Methods
- Train/test split: 70/30, stratified by failure mode.
- Model selection: nested cross-validation (5 outer folds stratified by failure mode, 3-fold inner grid search), metric PR-AUC (average precision).
- Candidates: dummy classifier, logistic regression, decision tree (depth ≤ 5), random forest; each with raw and physics features.
- Final model: random forest (200 trees, min_samples_leaf 5) with physics features.
- Decision threshold: 0.24, minimising total cost on out-of-fold training predictions.
- Uncertainty: bootstrap (2,000 resamples) and Wilson intervals on the test set.

## References
- Dataset: AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C (licence CC BY 4.0).
- Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications. Third International Conference on Artificial Intelligence for Industries (AI4I).
- Breiman, L. (2001). Random Forests. Machine Learning, 45, 5–32.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
- Ross, B. C. (2014). Mutual Information between Discrete and Continuous Data Sets. PLoS ONE, 9(2), e87357.
- scikit-learn documentation: random forests https://scikit-learn.org/stable/modules/ensemble.html#random-forests, nested cross-validation https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
