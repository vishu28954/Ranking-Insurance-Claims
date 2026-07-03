import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    ndcg_score
)


def regression_metrics(df, actual_col, prediction_col):
    mae = mean_absolute_error(
        df[actual_col],
        df[prediction_col]
    )

    rmse = mean_squared_error(
        df[actual_col],
        df[prediction_col]
    ) ** 0.5

    r2 = r2_score(
        df[actual_col],
        df[prediction_col]
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


def evaluate_ranking(
    df,
    score_col,
    actual_col="recovered_amount_target",
    relevance_col="relevance_label",
    k_values=[5, 10]
):
    rows = []

    for worklist_id, group in df.groupby("worklist_id"):
        group = group.copy()

        y_true = (
            group[relevance_col]
            .fillna(0)
            .to_numpy()
            .reshape(1, -1)
        )

        y_score = (
            group[score_col]
            .to_numpy()
            .reshape(1, -1)
        )

        total_recovered = group[actual_col].sum()

        result = {
            "worklist_id": worklist_id
        }

        for k in k_values:
            result[f"ndcg@{k}"] = ndcg_score(
                y_true,
                y_score,
                k=k
            )

            ranked = group.sort_values(
                score_col,
                ascending=False
            )

            top_k_recovered = (
                ranked
                .head(k)[actual_col]
                .sum()
            )

            result[f"recovery_captured@{k}"] = (
                top_k_recovered / total_recovered
                if total_recovered > 0
                else np.nan
            )

        rows.append(result)

    return pd.DataFrame(rows)


def evaluate_8_hour_capacity(
    df,
    score_col,
    handle_time_col="handle_time_hrs",
    actual_col="recovered_amount_target",
    prediction_col="final_predicted_recovery",
    capacity_hours=8.0
):
    rows = []

    for worklist_id, group in df.groupby("worklist_id"):
        ranked = group.sort_values(
            score_col,
            ascending=False
        ).copy()

        selected_rows = []
        used_time = 0.0

        for _, row in ranked.iterrows():
            claim_time = row[handle_time_col]

            if used_time + claim_time <= capacity_hours:
                selected_rows.append(row)
                used_time += claim_time

        selected_df = pd.DataFrame(selected_rows)

        total_actual_recovery = group[actual_col].sum()

        selected_actual_recovery = (
            selected_df[actual_col].sum()
            if not selected_df.empty
            else 0
        )

        selected_predicted_recovery = (
            selected_df[prediction_col].sum()
            if not selected_df.empty
            else 0
        )

        rows.append({
            "worklist_id": worklist_id,
            "claims_selected_in_8hrs": selected_df.shape[0],
            "used_handle_time_hrs": used_time,
            "selected_actual_recovery": selected_actual_recovery,
            "selected_predicted_recovery": selected_predicted_recovery,
            "total_actual_recovery": total_actual_recovery,
            "actual_recovery_captured_8hrs": (
                selected_actual_recovery / total_actual_recovery
                if total_actual_recovery > 0
                else np.nan
            )
        })

    return pd.DataFrame(rows)
