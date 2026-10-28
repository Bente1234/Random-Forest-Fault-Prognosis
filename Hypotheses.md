# Hypotheses and decision rules

Fixed on 28 October 2026, before any model was trained. The thresholds are also stored in `src/config.py`, and the evaluation notebook checks them automatically.

## Metrics
- **Recall**: share of actual failures the model flags.
- **Precision**: share of flagged jobs that are actual failures.
- **PR-AUC** (average precision): summary of precision and recall over all decision thresholds; 1.0 is perfect, a random model scores about 0.033 (the failure rate). Used to compare models.

## Model selection rule
Candidates: dummy classifier, logistic regression, shallow decision tree and random forest, each with the raw and the physics feature set. Gradient boosting was left out to keep the comparison focused: a second tree ensemble would add little to the hypotheses, and random forest needs far less tuning on a small number of failures. The final model is the candidate with the highest mean PR-AUC in stratified 5-fold cross-validation on the training set. If candidates differ by less than 0.01, the simpler one is chosen. The decision threshold is chosen on cross-validated training predictions (step 8). The test set is used once, for the final evaluation.

## Hypotheses

| | Hypothesis | Accepted if | Evaluated on |
|---|---|---|---|
| H1 | Failures can be detected reliably. | Final model: recall ≥ 0.80 **and** precision ≥ 0.60 at the chosen threshold | Test set |
| H2 | Physics features (power, temperature difference, strain) improve detection. | Logistic regression: mean PR-AUC gain ≥ 0.05 (physics vs raw) **and** a gain in at least 4 of 5 folds | Cross-validation |
| H3 | A tree ensemble outperforms a linear model. | Best random forest exceeds the best logistic regression by ≥ 0.05 mean PR-AUC | Cross-validation |
| H4 | Rule-based failure modes are more predictable than tool wear failure. | Final model: recall ≥ 0.90 for each of HDF, PWF and OSF **and** recall < 0.50 for TWF | Test set |
| H5 | The model is cheaper than simple maintenance policies. | Expected cost of the model is lower than both (a) no predictive maintenance and (b) replacing the tool at 200 min of tool wear, using the cost assumptions fixed in step 8 | Test set |

## Notes
- Per-mode results in H4 rest on small test counts (jobs with each mode flag: HDF 35, PWF 29, OSF 31, TWF 14; a job with several modes counts towards each) and are reported with that caveat.
- For tree models, the effect of the physics features is reported as an observation, not tested as a hypothesis.
