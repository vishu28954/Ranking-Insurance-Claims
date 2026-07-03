# Ranking Insurance Claims

## Classical ML Ranking & Prioritization Assessment

This project builds a classical machine learning pipeline to rank healthcare insurance claims within analyst worklists. The goal is to help analysts prioritize claims that are most likely to generate meaningful financial recovery while respecting the analyst's limited daily capacity.

Each analyst has an 8-hour workday, and each claim has a different estimated handle time. Because of that, the best ranking is not only the one that puts the highest-dollar claims first. A good operational ranking should prioritize claims that maximize expected recovery within the analyst's available time.

---

## Repository Structure

```text
Ranking-Insurance-Claims/
│
├── data/
│   ├── worklist_claims.csv
│   └── worklists.csv
│
├── outputs/
│   ├── current_worklist_rankings.csv
│   ├── current_worklist_top5_summary.csv
│   └── current_worklist_8hr_capacity_summary.csv
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── evaluation.py
│   ├── features.py
│   ├── models.py
│   ├── plots.py
│   └── train.py
│
├── WriteUp.pages
├── README.md
├── worklist_claims.csv
└── worklists.csv
```

The `src/` folder contains the modular Python code. The `outputs/` folder contains generated prediction and summary files for the current worklists.

---

## Problem Objective

The assignment is to rank claims inside each worklist so analysts can decide which claims to work first.

The business constraint is:

```text
Each analyst has 8 hours per day.
Each claim has a different handle_time_hrs.
```

So the practical objective is:

```text
Maximize estimated recovered dollars within 8 analyst hours.
```

Mathematically:

```text
maximize:    sum(selected_claim_i * predicted_recovery_i)

subject to:  sum(selected_claim_i * handle_time_hrs_i) <= 8
```

This means time matters. A claim with slightly lower recovery but much shorter handle time may be better than a higher-value claim that takes several hours.

---

## Ranking Approaches

### Approach A — Value-Based Ranking

This is the simplest pointwise baseline.

```text
score = final_predicted_recovery
```

Claims are ranked by predicted recovery amount only.

This approach answers:

```text
Which claims are expected to recover the most dollars?
```

---

### Approach B — Time-Adjusted Ranking

This is the operational ranking used for the final current-worklist output.

```text
score = final_predicted_recovery / handle_time_hrs
```

Claims are ranked by predicted recovery per analyst hour.

This approach answers:

```text
Which claims give the highest expected recovery per hour of analyst effort?
```

Example:

```text
Claim A: $2,000 predicted recovery / 0.5 hours = $4,000 per hour
Claim B: $2,500 predicted recovery / 3.0 hours = $833 per hour
```

Even though Claim B has a higher predicted recovery, Claim A is better under an 8-hour capacity constraint.

---

## Model Overview

The final model uses a two-stage approach.

### Stage 1 — Recovery Classification

Stage 1 predicts whether a claim is likely to recover any money.

Target:

```text
is_recovered = 1 if recovered_amount > 0 else 0
```

The classifier outputs:

```text
recovery_probability
```

A threshold of `0.30` is used. This favors recall, because missing a recoverable claim is more costly than sending a few extra false positives to analysts.

---

### Stage 2 — Recovery Amount Regression

Stage 2 predicts how much money a claim may recover, but it is trained only on historical claims that recovered a positive amount.

Target:

```text
log1p(recovered_amount)
```

The log transformation is used because recovered amounts are highly skewed.

Predictions are converted back using:

```text
predicted_amount_if_recovered = expm1(predicted_log_amount)
```

---

### Final Prediction

The final predicted recovery is created using the Stage 1 probability threshold.

```text
if recovery_probability >= 0.30:
    final_predicted_recovery = predicted_amount_if_recovered
else:
    final_predicted_recovery = 0
```

Then claims are ranked using either Approach A or Approach B.

---

## Feature Engineering

The model uses the following feature groups.

### Claim value features

```text
total_billed
expected_payment
actual_payment
payment_variance
amount_yet_to_recover
Percent_amount_yet_to_recover
log_total_billed
log_amount_yet
```

### Timing feature

```text
days_since_payment
```

### Categorical features

```text
payer_id
payer_type
visit_type
contract_version
payer_contract
payer_visit
payer_type_contract
```

These are one-hot encoded.

### Adjustment code features

The `adjustment_codes` field is split into individual codes and multi-hot encoded.

Example:

```text
45|97|B15
```

becomes:

```text
adj_45 = 1
adj_97 = 1
adj_B15 = 1
```

---

## Data Files

The training script expects the input CSV files inside the `data/` folder:

```text
data/worklist_claims.csv
data/worklists.csv
```

The repository also contains copies of these files at the root level. The script uses the files inside `data/`, so make sure the `data/` folder contains both CSVs.

---

## How to Run the Historical Training Pipeline

Run this command from the repository root:

```bash
python -m src.train --data_dir data --outputs_dir outputs --seed 42 --threshold 0.30 --capacity_hours 8
```

Do not run the command from inside the `src/` folder. Running from the repository root allows Python to correctly resolve imports such as:

```python
from src.data import load_raw_data
```

---

## Expected Training Outputs

After running the training pipeline, the following files are written to `outputs/`:

```text
stage1_classification_results.csv
stage2_amount_model_results.csv
approach_a_ranking_metrics.csv
approach_b_ranking_metrics.csv
approach_b_capacity_metrics.csv
test_predictions.csv
summary_metrics.csv
stage1_model.pkl
stage2_amount_model.pkl
adjustment_code_encoder.pkl
```

The pipeline also generates explainability plots such as:

```text
plot_median_recovery_by_relevance_label.png
plot_value_concentration_by_label.png
plot_recovery_vs_handle_time.png
plot_worklist_time_pressure.png
```

---

## Current Worklist Outputs

The repository already contains ranked outputs for the 16 current worklists:

```text
WL-025 through WL-040
```

These are available in the `outputs/` folder.

### 1. `current_worklist_rankings.csv`

This is the claim-level ranked output.

Each row represents one claim in a current worklist.

Important columns:

```text
worklist_id
rank
claim_id
recovery_probability
predicted_amount_if_recovered
final_predicted_recovery
handle_time_hrs
score_value
score_time_adjusted
amount_yet_to_recover
payer_id
payer_type
visit_type
contract_version
adjustment_codes
```

This file answers:

```text
For each current worklist, what order should the analyst work the claims in?
```

The final ranking uses:

```text
score_time_adjusted = final_predicted_recovery / handle_time_hrs
```

---

### 2. `current_worklist_top5_summary.csv`

This summarizes how much estimated recovery is captured if the analyst works only the top 5 ranked claims in each worklist.

Important columns:

```text
worklist_id
estimated_total_recovery
estimated_recovery_top_5
estimated_recovery_capture_top_5
top_5_handle_time_hrs
top_5_claim_ids
```

This file answers:

```text
How much of the estimated total recovery would an analyst capture by working only the top 5 claims?
```

---

### 3. `current_worklist_8hr_capacity_summary.csv`

This simulates the analyst's 8-hour day.

The logic is:

```text
Select ranked claims until cumulative handle_time_hrs reaches 8 hours.
```

Important columns:

```text
worklist_id
claims_selected_in_8hrs
used_handle_time_hrs
estimated_recovery_selected_8hrs
estimated_total_recovery
estimated_recovery_capture_8hrs
selected_claim_ids
```

This file answers:

```text
Given an 8-hour limit, which claims should the analyst actually work?
```

---

## Evaluation Metrics

The model is evaluated using ranking metrics instead of only prediction metrics.

### NDCG@K

NDCG measures whether high-relevance claims are placed near the top.

```text
NDCG@5
NDCG@10
```

These evaluate ranking quality among the top 5 and top 10 claims.

### Recovery Captured@K

This is the business metric.

```text
recovery_captured@K =
actual recovery in top K claims / total actual recovery in the worklist
```

This answers:

```text
If the analyst only works the top K claims, what fraction of recoverable dollars are captured?
```

### 8-Hour Recovery Captured

This evaluates the ranking under the actual analyst capacity constraint.

```text
8_hour_recovery_captured =
actual recovery from claims selected within 8 hours / total actual recovery in the worklist
```

---

## Key Results

The selected model is:

```text
Stage 1: ExtraTreesClassifier
Stage 2: GradientBoostingRegressor trained on log1p(recovered_amount)
Threshold: 0.30
```

Final two-stage prediction performance:

```text
MAE: 593.38
RMSE: 1274.21
R2: 0.492
```

Approach A value-based ranking:

```text
Average NDCG@5: 0.767
Average recovery captured@5: 0.766
Average NDCG@10: 0.749
Average recovery captured@10: 0.888
```

Approach B time-adjusted ranking:

```text
Average NDCG@5: 0.680
Average recovery captured@5: 0.674
Average NDCG@10: 0.717
Average recovery captured@10: 0.885
Average 8-hour recovery captured: 0.777
```

Approach A performs better on pure top-of-list ranking metrics. Approach B is more operationally useful because it accounts for analyst handle time.

---

## Naive Baseline

The naive baseline ranks claims by largest underpayment first:

```text
baseline_score = amount_yet_to_recover
```

This is a reasonable baseline because underpayment is a strong signal. However, it does not learn payer behavior, contract version patterns, adjustment-code signals, or historical recovery outcomes.

The model improves on the naive baseline by combining underpayment information with historical recovery patterns.

---

## Interpretation

The model separates the ranking problem into two questions:

```text
1. Is this claim likely to recover any money?
2. If it recovers, how much is expected?
```

This is useful because many claims recover zero, while a smaller number recover large amounts.

The final ranking uses predicted recovery per analyst hour, making the output more practical for a team that has limited review capacity.

---

## Limitations

The model ranks claims by estimated opportunity, not guaranteed recovery.

A high-ranked claim should be interpreted as:

```text
This claim looks promising based on historical patterns.
```

It should not be interpreted as:

```text
This claim will definitely recover money.
```

The model can struggle when a claim has a large apparent underpayment but the actual dispute outcome depends on information not present in the dataset, such as:

```text
denial reason text
appeal notes
documentation completeness
payer policy details
medical necessity rules
analyst judgment
```

---

## Future Improvements

If more time or data were available, the first improvements would be:

1. Add denial reason text and appeal notes.
2. Add payer-level historical overturn rates.
3. Add analyst feedback after each review.
4. Tune the Stage 1 threshold using a separate validation set.
5. Try learning-to-rank models such as LambdaMART or XGBoost ranker.
6. Add a knapsack optimization layer for exact 8-hour claim selection.

---

## Troubleshooting

### 1. `ModuleNotFoundError: No module named 'src'`

Make sure you are running the command from the repository root:

```bash
python -m src.train --data_dir data --outputs_dir outputs --seed 42 --threshold 0.30 --capacity_hours 8
```

Do not run:

```bash
cd src
python train.py
```

---

### 2. `FileNotFoundError` for CSV files

Check that both files exist here:

```text
data/worklist_claims.csv
data/worklists.csv
```

If they are only present in the root folder, copy them into `data/`.

---

## How to Update This Repository

After replacing or editing files locally:

```bash
git add README.md src/ data/ outputs/
git commit -m "Update README and project files"
git push origin main
```

---

## Summary

This project provides a classical ML solution for ranking insurance claims. It predicts which claims are likely to recover money, estimates the recovery amount, and ranks claims by expected recovery per analyst hour. The final outputs help analysts prioritize current worklists under an 8-hour daily capacity constraint.
