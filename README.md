# Classical ML Ranking & Prioritization Assessment

## Overview

This project builds a classical machine learning solution to rank healthcare claims inside worklists so analysts can prioritize claims with the highest expected recovery value.

The business problem is not only to predict whether a claim will recover money, but also to rank claims in a way that helps analysts use limited daily working time efficiently. Each analyst has 8 hours per day, and each claim has a different `handle_time_hrs`.

The final solution uses a two-stage modeling approach:

1. **Stage 1 — Recovery Classification**  
   Predict whether a claim is likely to recover any amount.

2. **Stage 2 — Recovery Amount Regression**  
   For claims that are recoverable, predict the recovered amount using a regression model trained on `log1p(recovered_amount)`.

The final predicted recovery is then used for two ranking approaches:

- **Approach A:** Rank by predicted recovery amount.
- **Approach B:** Rank by predicted recovery per analyst hour.

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
