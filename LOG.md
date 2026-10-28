# Project log

## Step 1 – Data loading and sanity checks
- Did: Loaded the data and checked missing values, duplicates, product types, class balance, failure modes and label consistency. Removed 27 inconsistent rows and saved the cleaned dataset.
- Why: The documentation states that `Machine failure` = 1 if and only if at least one failure mode is 1. Rows breaking this rule are label noise and would distort both training and per-mode analysis.
- Result:
  - 10,000 rows, 14 columns, no missing values, no duplicates.
  - Product types: L 60.0%, M 30.0%, H 10.0%.
  - 339 failures (3.39%), so the classes are strongly imbalanced.
  - Failure modes: TWF 46, HDF 115, PWF 95, OSF 98, RNF 19; 24 failures have more than one mode.
  - The documented rules reproduce the HDF, PWF and OSF flags with zero mismatches, so these three modes are fully deterministic in this (synthetic) dataset.
  - TWF failures occur at tool wear 198–253 min (documentation says 200–240).
  - Inconsistencies: 9 failures without any mode (none satisfies any documented rule) and 18 RNF rows labelled as non-failure.
  - All 27 removed. Clean dataset: 9,973 rows, 330 failures (3.31%).
  - Only 1 RNF row remains, so RNF cannot be analysed as a separate mode.
- Figures/files: `src/01_data_check.ipynb`, `data/ai4i2020_clean.csv`, `results/01_data_check_summary.csv`
- Open questions / limitations: Removing the rows is a judgement call; keeping them and using the label as given is also defensible. Because three modes follow exact rules, very high model scores are expected and must be discussed as a property of the synthetic data.

## Step 2 – Variable categorization
- Did: Classified every variable as identifier, independent or dependent, and as controllable or non-controllable, and checked the correlations between the sensor variables.
- Why: Required by the assignment. It fixes which columns are model inputs and shows which failure causes can be prevented by changing settings (useful for the business case).
- Result:
  - Inputs (independent): Type, air temperature, process temperature, rotational speed, torque, tool wear. Target (dependent): Machine failure. Excluded: UDI, Product ID (identifiers) and the five failure-mode flags (they define the target, so using them would be leakage).
  - Controllable: Type, rotational speed, torque. Indirectly controllable: tool wear (via the replacement policy). Non-controllable: air and process temperature.
  - Process temperature follows air temperature (r = 0.88, about 10 ± 1 K higher), so it is not a separate setting.
  - Rotational speed and torque are strongly coupled (r = −0.88), so they act as one operating point.
  - Tool wear is uncorrelated with all other variables.
- Figures/files: `src/02_variables.ipynb`, `results/02_variable_categories.csv`
- Open questions / limitations: Controllability is partly a judgement, since the dataset has no information about the real plant. Air temperature could be controlled with climate control, and torque depends on the material being cut.

## Step 3 – Exploratory data analysis
- Did: Created 8 figures: class balance and failure modes, sensor distributions by outcome, correlation matrix, failure rate by product type, and one plot per failure mechanism (power, heat dissipation, overstrain, tool wear) with the documented thresholds drawn in.
- Why: To understand which variables carry information about failures and how each failure mechanism works, which justifies the engineered features and the model choice.
- Result:
  - Strong imbalance: 330 failures out of 9,973 jobs (3.3%).
  - Single variables correlate only weakly with failure (largest: torque, r = 0.19), but the distributions show that failures cluster at low speed, extreme torque and tool wear above about 200 min. Failures therefore depend on thresholds and combinations of variables, which linear correlation does not capture.
  - Failure rate by product type: L 3.9%, M 2.6%, H 2.0%, consistent with the lower overstrain limit for L products.
  - The documented thresholds visibly separate PWF (power outside 3500–9000 W), HDF (temperature difference < 8.6 K and speed < 1380 rpm) and OSF (wear × torque above 11,000 / 12,000 / 13,000 min·Nm for L / M / H).
  - TWF only occurs from 190 min of tool wear upward, and even above 230 min only 16% of jobs fail (7/45), so TWF is only partly predictable.
- Figures/files: src/03_eda.ipynb; figures/03_class_balance.png, 03_distributions.png, 03_correlation.png, 03_failure_rate_by_type.png, 03_mechanism_power.png, 03_mechanism_heat.png, 03_mechanism_overstrain.png, 03_mechanism_tool_wear.png
- Open questions / limitations: The plots use the full cleaned dataset (before the train/test split); this is acceptable because nothing is fitted and the thresholds come from the dataset documentation. With 8 figures and a 10-page limit, the report should probably use only the most informative ones (suggested: distributions, one or two mechanism plots, failure rate by type).

## Step 4 – Train/test split
- Did: Split the cleaned data once into a training set (70%) and a test set (30%), stratified by failure mode, with seed 42. Saved the row IDs to data/train_ids.csv and data/test_ids.csv. Added a helper (src/data.py) that every later notebook uses to load the two sets.
- Why: The assignment requires results on data the model never saw. 70/30 was chosen over 80/20 because failures are rare: the test set now holds 99 failures instead of about 66, which makes the test results and per-mode results more reliable, while 231 training failures is still enough to train the models. Stratifying by failure mode (not only by failure yes/no) guarantees every mode is represented in the test set in the same proportion. Jobs with several modes are assigned to the first mode in column order (TWF, HDF, PWF, OSF, RNF), because some mode combinations occur only once and a group needs at least two jobs to be split. Model comparison will use stratified 5-fold cross-validation on the training set only.
- Result:
  - Train: 6,981 jobs, 231 failures (3.31%). Test: 2,992 jobs, 99 failures (3.31%).
  - Failure modes in the test set: HDF 35, PWF 27, OSF 23, TWF 14 (same shares as in training).
  - Product types: same distribution in both sets (L 60%, M 30%, H 10%).
  - Checks passed: no overlap, all jobs assigned, and re-running produces the identical split.
- Figures/files: src/04_split.ipynb, src/data.py, data/train_ids.csv, data/test_ids.csv
- Open questions / limitations: With 99 test failures, one missed failure changes recall by about 1 percentage point; per-mode results (e.g. TWF with 14 cases) remain uncertain and should be reported with that caveat.

## Step 5 – Preprocessing and feature engineering
- Did: Built the model inputs in a reusable module (src/features.py) and inspected them in 05_features.ipynb on the training set only. Defined two feature sets: raw (product type + five sensor variables) and physics (raw + power, temperature difference, strain). Product type is one-hot encoded; numeric features are standardised inside the model pipeline. Measured the information content of each feature with mutual information.
- Why: Failures depend on combinations of variables (step 3), which single-variable models (trees, linear models) cannot see directly. The engineered features come from basic physics: power = torque × angular speed, temperature difference = process − air temperature, strain = tool wear × torque. Threshold-based features (e.g. "power outside 3500–9000 W") were deliberately not used, because they would copy the dataset's own failure rules into the inputs. One-hot encoding was chosen over ordinal encoding because the failure rates of L, M and H are not equally spaced. Scaling is refitted inside each cross-validation fold to avoid leakage; the engineered features are row-wise calculations and cannot leak. Columns were renamed (e.g. torque, tool_wear) because XGBoost rejects brackets in column names.
- Result:
  - Raw set: 8 model columns; physics set: 11 model columns. No missing or infinite values.
  - Median of failed vs non-failed jobs: power 7,733 vs 6,234 W, temperature difference 9.2 vs 9.8 K, strain 7,682 vs 3,945 min·Nm.
  - Mutual information (training set, mean of 5 seeds): torque 0.046, power 0.045, speed 0.040, strain 0.034, tool wear 0.012, temperature difference 0.011, air temperature 0.007, process temperature 0.006.
  - Strain carries about three times more information than tool wear alone; temperature difference slightly more than either temperature; power about as much as torque alone (torque and speed are already strongly coupled).
- Figures/files: src/features.py, src/05_features.ipynb, figures/05_mutual_information.png
- Open questions / limitations: Mutual information evaluates one feature at a time and cannot show the value of combinations (e.g. HDF needs both temperature difference and speed). Whether the physics features actually improve the models is tested in step 7 (hypothesis H2).

## Step 6 – Hypotheses
- Did: Wrote down the hypotheses, their success thresholds and the model selection rule before training any model (HYPOTHESES.md); stored the thresholds in src/config.py so the evaluation code checks them automatically.
- Why: Accepting or rejecting hypotheses is only honest if the criteria are fixed before the results are known.
- Result:
  - Main comparison metric: PR-AUC (random model ≈ 0.033).
  - Model selection: highest mean PR-AUC in stratified 5-fold cross-validation on the training set; if within 0.01, the simpler model wins. Candidates: dummy, logistic regression, shallow decision tree and random forest, each with raw and physics features. Gradient boosting was left out: a second tree ensemble adds little to the hypotheses, needs more tuning on only 231 training failures, and would take report space.
  - H1: final model reaches recall ≥ 0.80 and precision ≥ 0.60 on the test set. The bar is set below the expected ceiling (about 86% of test failures are rule-based HDF/PWF/OSF; TWF is largely random).
  - H2: physics features raise logistic regression PR-AUC by ≥ 0.05, with a gain in at least 4 of 5 folds.
  - H3: the best random forest beats the best logistic regression by ≥ 0.05 PR-AUC.
  - H4: on the test set, recall ≥ 0.90 for each of HDF, PWF and OSF, and < 0.50 for TWF.
  - H5: the model has a lower expected cost than (a) no predictive maintenance and (b) replacing the tool at 200 min of wear.
- Figures/files: HYPOTHESES.md, src/config.py
- Open questions / limitations: The cost assumptions for H5 are fixed in step 8, still before the test set is used. H4 rests on small test counts (jobs with each mode flag: HDF 35, PWF 29, OSF 31, TWF 14; a job with several modes counts towards each).

## Step 7 – Model training and comparison
- Did: Compared 7 candidates (dummy; logistic regression, shallow decision tree and random forest, each with raw and physics features) with nested cross-validation on the training set: 5 outer folds (stratified by failure mode) for scoring, and a 3-fold grid search inside each outer training part for hyperparameters, including whether to weight failures more heavily. Model definitions and search grids are in src/models.py.
- Why: Nested cross-validation keeps tuning and scoring on separate data, so the comparison is not biased in favour of models with more tuning options. Using the same outer folds for every candidate allows fold-by-fold comparisons (needed for H2). The test set was not used.
- Result:
  - Mean PR-AUC (std): random forest physics 0.909 (0.014), decision tree physics 0.886 (0.010), random forest raw 0.764 (0.083), decision tree raw 0.559 (0.069), logistic regression physics 0.488 (0.042), logistic regression raw 0.460 (0.039), dummy 0.033.
  - H2 rejected: physics features improved logistic regression in all 5 folds, but by only 0.028 on average (threshold 0.05). The linear model cannot represent failures at both extremes (e.g. power too low or too high), even with the physics features.
  - Observation: the physics features helped the tree models far more: +0.33 PR-AUC for the decision tree and +0.14 for the random forest, and they made the random forest much more stable across folds (std 0.083 → 0.014).
  - H3 accepted: the best random forest beats the best logistic regression by 0.42 PR-AUC (threshold 0.05).
  - Class weighting was never selected by the inner search; logistic regression always chose C = 1.
  - Selected model: random forest with physics features (only candidate within 0.01 of the best). Final settings from a 5-fold grid search on the full training set: 200 trees, unlimited depth, min_samples_leaf = 5, no class weighting.
- Figures/files: src/models.py, src/07_models.ipynb, figures/07_cv_comparison.png, results/07_cv_fold_scores.csv, results/07_cv_fold_params.csv, results/07_cv_summary.csv, results/07_selected_model.json
- Open questions / limitations: The decision tree always chose the largest allowed depth (5), so a deeper tree might score higher; the cap is deliberate to keep it readable. The shallow decision tree with physics features is only 0.023 PR-AUC behind the random forest and is fully interpretable, which is worth mentioning in the discussion. PR-AUC values are from cross-validation on the training set; the final, reportable numbers come from the test set in step 9.

## Step 8 – Decision threshold and cost model
- Did: Generated out-of-fold failure probabilities for the training set with the selected random forest (same 5 folds as step 7), defined a cost model, chose the threshold that minimises total cost, compared it with two simple maintenance policies, and ran a sensitivity analysis on the cost ratio. Test set not used.
- Why: The model outputs a probability, so a threshold is needed to turn it into alarms. Minimising cost links the threshold to the business question and fixes the cost assumptions for H5 before the test set is used. Choosing the threshold to hit the H1 targets was rejected because it would bias the H1 test.
- Result:
  - Cost assumptions (no cost data in the dataset): planned intervention CAD 1,000 per alarm, unplanned breakdown CAD 10,000 per missed failure. Total cost = missed × 10,000 + alarms × 1,000. Ties are broken towards the higher threshold (fewer interventions).
  - Pooled out-of-fold PR-AUC: 0.907.
  - Selected threshold: 0.24. At this threshold on the training set: 249 alarms, 200 of 231 failures caught, 49 false alarms; recall 0.866, precision 0.803.
  - The cost curve is flat between about 0.15 and 0.5, so the exact threshold is not critical.
  - Training-set cost: no predictive maintenance CAD 2.31 M, replace tool at 200 min CAD 2.04 M (551 alarms, only 82 failures caught), random forest CAD 0.56 M (76% lower than no predictive maintenance, 73% lower than the tool rule).
  - Recall per mode (training, preview): HDF 0.99, PWF 1.00, OSF 1.00, TWF 0.06 (2 of 32).
  - Sensitivity: the threshold stays at 0.24 for cost ratios 5, 10 and 20; it rises to 0.57 at ratio 2 and drops to 0.02 at ratio 50. The model is cheaper than both simple policies at every ratio tested (saving 39–90% vs no predictive maintenance, 67–85% vs the tool rule).
- Figures/files: src/08_threshold.ipynb, figures/08_precision_recall.png, figures/08_cost_vs_threshold.png, results/08_oof_predictions.csv, results/08_cost_sensitivity.csv, results/08_threshold.json
- Open questions / limitations: The cost figures are assumptions, not data. All numbers here are training-set previews; H1, H4 and H5 are decided on the test set in step 9. The model catches almost no TWF failures, consistent with their largely random nature.

## Step 9 – Final evaluation on the test set
- Did: Trained the selected random forest (physics features, 200 trees, min_samples_leaf 5) once on the full training set, predicted the test set (first and only use), applied the fixed threshold 0.24 and the fixed cost assumptions, and evaluated all hypotheses automatically against the thresholds in config.py. Added 95% intervals: bootstrap (2,000 resamples) for overall metrics, Wilson intervals for per-mode recall.
- Why: These are the only results that count, because every modelling decision was fixed without seeing the test set. Intervals show how much the numbers could vary given only 99 test failures.
- Result:
  - Confusion matrix (2,992 test jobs, 99 failures): 87 failures caught, 12 missed, 9 false alarms, 2,884 correctly left alone.
  - PR-AUC 0.934 (95% interval 0.896–0.968), recall 0.879 (0.816–0.940), precision 0.906 (0.843–0.959). Cross-validated training estimates: 0.909, 0.866, 0.803.
  - Recall per mode: HDF 35/35, PWF 29/29, OSF 31/31 (each 1.00; Wilson lower bounds 0.88–0.90), TWF 2/14 = 0.14 (0.04–0.40). All 12 missed failures are pure tool wear failures (tool wear 206–235 min, predicted probability 0.01–0.19). No RNF failure in the test set.
  - Cost on the test set: no predictive maintenance CAD 990k, replace tool at 200 min CAD 838k (248 alarms, 40 failures caught), random forest CAD 216k (78% lower than no predictive maintenance).
  - Hypotheses: H1 accepted, H2 rejected (from step 7), H3 accepted (from step 7), H4 accepted, H5 accepted.
  - The test scores are slightly better than the training estimates, mainly through fewer false alarms (0.31% of normal test jobs vs 0.73% in training). This is within normal random variation, and the final model was trained on all training data instead of 80%. Leakage check: no job appears in both sets.
- Figures/files: src/09_evaluation.ipynb, figures/09_confusion_matrix.png, figures/09_precision_recall_test.png, figures/09_policy_costs.png, results/09_test_predictions.csv, results/09_test_metrics.csv, results/09_test_recall_by_mode.csv, results/09_test_policies.csv, results/09_hypotheses.csv
- Open questions / limitations: Per-mode recalls rest on small counts: the H4 lower bounds of 0.88–0.90 are at or just below the 0.90 bar, so H4 is accepted on the observed values but would need more data to be certain. The model essentially cannot predict tool wear failures; reducing those requires a different strategy (e.g. a tool replacement policy). Costs are assumptions (step 8).

## Step 10 – Interpretation (decision tree rules)
- Did: Fitted the readable decision tree (physics features, depth 5, at least 10 jobs per leaf, no class weighting, as chosen in every fold of step 7) on the training set, printed its if-then rules and compared its split points with the documented failure thresholds. Not evaluated on the test set. Permutation importance and partial dependence were considered but left out to keep the report within 10 pages; a separate business-case step (step 11) was dropped because steps 8–9 already contain the costs, savings and sensitivity analysis.
- Why: To check that the models learned the real failure mechanisms rather than coincidences, and to explain the model in terms a board can follow. The tree scored close to the random forest (PR-AUC 0.886 vs 0.909 in cross-validation).
- Result:
  - Learned splits vs documented rules: strain 10,998.5 (OSF limit 11,000 for L); power 3,496 and 9,000 W (PWF band 3,500–9,000 W); temperature difference 8.65 K together with speed 1,381 rpm (HDF: < 8.6 K and < 1,380 rpm); tool wear 202.5 min (TWF window about 200 min onwards); type L separated for overstrain (lowest limit).
  - The tree found these thresholds from the data alone, without being given them, so the models rely on the real physics.
  - The M and H overstrain limits (12,000 and 13,000) are only partly resolved at this depth.
  - The leaf for tool wear > 202.5 min holds 345 jobs without and 25 with a failure (7%), which is why tool wear failures do not trigger alarms.
  - Note for the report: the assignment mentions "accuracy of the model". Test accuracy is 99.3% ((87 + 2,884) / 2,992), but accuracy is misleading here because the dummy model reaches 96.7% while catching no failures; recall, precision and PR-AUC are the meaningful metrics.
- Figures/files: src/10_interpretation.ipynb, results/10_tree_rules.txt, results/10_tree_splits.csv
- Open questions / limitations: The tree is a readable stand-in; the deployed model is the random forest, whose internal logic is spread over 200 trees.