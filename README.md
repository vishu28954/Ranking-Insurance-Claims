# Classical ML Ranking & Prioritization Assessment

## Overview

This project builds a classical machine learning solution for ranking healthcare claims within analyst worklists. The goal is to help analysts prioritize claims that are most likely to generate meaningful financial recovery.

Each analyst has a limited daily capacity of 8 hours, and each claim has a different estimated handle time. Therefore, the project evaluates two ranking strategies:

1. **Approach A — Value-based ranking**  
   Rank claims by predicted recovery amount.

2. **Approach B — Time-adjusted ranking**  
   Rank claims by predicted recovery per analyst hour.

The final model uses a two-stage approach:

- **Stage 1:** Predict whether a claim is likely to recover any amount.
- **Stage 2:** Predict the expected recovered amount for claims that are recoverable.

---

## Project Structure

```text
project/
│
├── data/
│   ├── worklist_claims.csv
│   └── worklists.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   ├── plots.py
│   └── train.py
│
├── outputs/
│
├── requirements.txt
└── README.md
```

---

## Data

Place the two input CSV files inside the `data/` folder:

```text
data/worklist_claims.csv
data/worklists.csv
```

Historical worklists are used for training and testing. Current worklists can later be scored using the same trained pipeline.

---

## Modeling Approach

### Stage 1: Recovery Classification

Stage 1 predicts whether a claim will recover any amount.

Target:

```text
is_recovered = 1 if recovered_amount > 0 else 0
```

The following classifiers are compared:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting

The best Stage 1 model is selected using ROC-AUC.

The selected classifier outputs a `recovery_probability` for each claim. A threshold of `0.30` is used to identify claims likely to recover. This threshold favors high recall, because missing a recoverable claim is more costly than reviewing some false positives.

---

### Stage 2: Recovery Amount Regression

Stage 2 predicts the recovered amount, but it is trained only on claims that actually recovered money in the historical data.

Target:

```text
log1p(recovered_amount)
```

The log transformation is used because recovered amounts are highly skewed. Predictions are converted back to dollar values using:

```text
expm1(predicted_log_amount)
```

The following regressors are compared:

- Gradient Boosting Regressor
- Random Forest Regressor
- Extra Trees Regressor

The selected Stage 2 model is the model with the lowest MAE on recovered claims.

---

## Feature Engineering

The model uses the following feature groups.

### Claim value features

```text
total_billed
expected_payment
actual_payment
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

Categorical features are one-hot encoded.

### Adjustment code features

The `adjustment_codes` field is split into individual codes and multi-hot encoded.

Example:

```text
"45|97|B15"
```

becomes:

```text
adj_45 = 1
adj_97 = 1
adj_B15 = 1
```

---

## Ranking Approaches

### Approach A: Value-Based Ranking

Approach A ranks claims by predicted recovery amount:

```text
score = final_predicted_recovery
```

This approach prioritizes claims expected to recover the most dollars.

---

### Approach B: Time-Adjusted Ranking

Approach B incorporates analyst effort:

```text
score = final_predicted_recovery / handle_time_hrs
```

This ranks claims by expected recovery per analyst hour.

For example:

```text
Claim A: $2,000 predicted recovery / 0.5 hours = $4,000 per hour
Claim B: $2,500 predicted recovery / 3 hours = $833 per hour
```

Although Claim B has a higher predicted recovery amount, Claim A is more efficient under an 8-hour capacity constraint.

---

## Evaluation

The model is evaluated using both prediction and ranking metrics.

### Regression Metrics

```text
MAE
RMSE
R2
```

These measure how close predicted recovery is to actual recovered amount.

### Ranking Metrics

```text
NDCG@5
NDCG@10
Recovery captured@5
Recovery captured@10
```

These evaluate whether high-value claims are placed near the top of each worklist.

### Capacity-Aware Metric

For Approach B, the code also evaluates how much actual recovery is captured within an 8-hour analyst day:

```text
actual_recovery_captured_8hrs
```

---

## Outputs

Running the training command writes the following files to `outputs/`:

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

It also generates the following plots:

```text
plot_median_recovery_by_relevance_label.png
plot_value_concentration_by_label.png
plot_recovery_vs_handle_time.png
plot_worklist_time_pressure.png
```

---

## Key Results

The selected Version 1 model uses:

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

Approach A ranking performance:

```text
Average NDCG@5: 0.767
Average recovery captured@5: 0.766
Average NDCG@10: 0.749
Average recovery captured@10: 0.888
```

Approach B time-adjusted ranking performance:

```text
Average NDCG@5: 0.680
Average recovery captured@5: 0.674
Average NDCG@10: 0.717
Average recovery captured@10: 0.885
Average 8-hour recovery captured: 0.777
```

---

## Interpretation

The model separates recovery prediction into two questions:

1. Is this claim likely to recover any money?
2. If it recovers, how much is expected?

This is useful because many claims recover zero, while a smaller number recover large amounts.

Approach A is better for pure dollar-value ranking. Approach B is better for analyst workflow because it accounts for the 8-hour daily capacity constraint.

---

## Future Improvements

If more time or data were available, the next improvements would be:

1. Tune the Stage 1 threshold using a separate validation set.
2. Try learning-to-rank models such as LambdaMART or XGBoost ranker.
3. Add a knapsack optimization layer to select the optimal set of claims under the exact 8-hour constraint.
4. Add richer historical payer and denial behavior features.
