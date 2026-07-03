import argparse
from pathlib import Path

import pandas as pd
import joblib

from src.config import Config
from src.data import (
    load_raw_data,
    get_historical_claims,
    split_by_worklist
)

from src.features import (
    add_core_features,
    add_adjustment_code_features,
    get_feature_columns
)

from src.models import (
    train_stage1_classifiers,
    train_stage2_amount_models,
    create_final_predictions
)

from src.evaluation import (
    regression_metrics,
    evaluate_ranking,
    evaluate_8_hour_capacity
)

from src.plots import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data"
    )

    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.30
    )

    parser.add_argument(
        "--capacity_hours",
        type=float,
        default=8.0
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = Config(
        data_dir=Path(args.data_dir),
        outputs_dir=Path(args.outputs_dir),
        seed=args.seed,
        stage1_threshold=args.threshold,
        capacity_hours=args.capacity_hours
    )

    config.outputs_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    claims_df, worklists_df = load_raw_data(
        config.data_dir
    )

    historical_df = get_historical_claims(
        claims_df,
        worklists_df
    )

    historical_df = add_core_features(
        historical_df
    )

    print("Historical claims:", historical_df.shape[0])
    print("Historical worklists:", historical_df["worklist_id"].nunique())

    # --------------------------------------------------------
    # Generate explainability visuals
    # --------------------------------------------------------

    plot_paths = generate_all_plots(
        historical_df,
        worklists_df,
        config.outputs_dir
    )

    print("\nGenerated plots:")
    for path in plot_paths:
        print(path)

    # --------------------------------------------------------
    # Train-test split
    # --------------------------------------------------------

    train_df, test_df = split_by_worklist(
        historical_df,
        target_col="is_recovered",
        seed=config.seed,
        test_size=config.test_size
    )

    print("\nTrain claims:", train_df.shape[0])
    print("Test claims:", test_df.shape[0])
    print("Train worklists:", train_df["worklist_id"].nunique())
    print("Test worklists:", test_df["worklist_id"].nunique())

    # --------------------------------------------------------
    # Feature engineering after split
    # Adjustment codes are fit on train only
    # --------------------------------------------------------

    train_df, test_df, adjustment_code_columns, mlb = (
        add_adjustment_code_features(
            train_df,
            test_df
        )
    )

    features, categorical_features, numeric_features = (
        get_feature_columns(
            adjustment_code_columns
        )
    )

    print("\nAdjustment code columns:")
    print(adjustment_code_columns)

    # --------------------------------------------------------
    # Stage 1: classification
    # --------------------------------------------------------

    (
        best_stage1_model_name,
        best_stage1_model,
        stage1_results_df
    ) = train_stage1_classifiers(
        train_df=train_df,
        test_df=test_df,
        features=features,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        target_col="is_recovered",
        seed=config.seed
    )

    print("\nStage 1 classification results:")
    print(stage1_results_df)

    print("\nBest Stage 1 model:", best_stage1_model_name)

    # --------------------------------------------------------
    # Stage 2: amount regression on recovered claims only
    # --------------------------------------------------------

    (
        best_stage2_model_name,
        best_stage2_model,
        stage2_results_df
    ) = train_stage2_amount_models(
        train_df=train_df,
        test_df=test_df,
        features=features,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        target_col="recovered_amount_target",
        seed=config.seed
    )

    print("\nStage 2 amount model results:")
    print(stage2_results_df)

    print("\nBest Stage 2 model:", best_stage2_model_name)

    # --------------------------------------------------------
    # Final predictions
    # --------------------------------------------------------

    test_predictions = create_final_predictions(
        test_df=test_df,
        stage1_model=best_stage1_model,
        stage2_model=best_stage2_model,
        features=features,
        threshold=config.stage1_threshold
    )

    # --------------------------------------------------------
    # Approach A: predicted recovery only
    # --------------------------------------------------------

    approach_a_metrics = regression_metrics(
        test_predictions,
        actual_col="recovered_amount_target",
        prediction_col="score_approach_a"
    )

    ranking_a = evaluate_ranking(
        test_predictions,
        score_col="score_approach_a"
    )

    # --------------------------------------------------------
    # Approach B: predicted recovery per hour
    # --------------------------------------------------------

    ranking_b = evaluate_ranking(
        test_predictions,
        score_col="score_approach_b"
    )

    capacity_b = evaluate_8_hour_capacity(
        test_predictions,
        score_col="score_approach_b",
        capacity_hours=config.capacity_hours
    )

    print("\nApproach A regression metrics:")
    print(approach_a_metrics)

    print("\nApproach A average ranking metrics:")
    print(ranking_a.mean(numeric_only=True))

    print("\nApproach B average ranking metrics:")
    print(ranking_b.mean(numeric_only=True))

    print("\nApproach B 8-hour capacity metrics:")
    print(capacity_b.mean(numeric_only=True))

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    stage1_results_df.to_csv(
        config.outputs_dir / "stage1_classification_results.csv",
        index=False
    )

    stage2_results_df.to_csv(
        config.outputs_dir / "stage2_amount_model_results.csv",
        index=False
    )

    ranking_a.to_csv(
        config.outputs_dir / "approach_a_ranking_metrics.csv",
        index=False
    )

    ranking_b.to_csv(
        config.outputs_dir / "approach_b_ranking_metrics.csv",
        index=False
    )

    capacity_b.to_csv(
        config.outputs_dir / "approach_b_capacity_metrics.csv",
        index=False
    )

    test_predictions.to_csv(
        config.outputs_dir / "test_predictions.csv",
        index=False
    )

    joblib.dump(
        best_stage1_model,
        config.outputs_dir / "stage1_model.pkl"
    )

    joblib.dump(
        best_stage2_model,
        config.outputs_dir / "stage2_amount_model.pkl"
    )

    joblib.dump(
        mlb,
        config.outputs_dir / "adjustment_code_encoder.pkl"
    )

    summary_df = pd.DataFrame([
        {
            "metric": "stage1_best_model",
            "value": best_stage1_model_name
        },
        {
            "metric": "stage2_best_model",
            "value": best_stage2_model_name
        },
        {
            "metric": "stage1_threshold",
            "value": config.stage1_threshold
        },
        {
            "metric": "approach_a_mae",
            "value": approach_a_metrics["MAE"]
        },
        {
            "metric": "approach_a_rmse",
            "value": approach_a_metrics["RMSE"]
        },
        {
            "metric": "approach_a_r2",
            "value": approach_a_metrics["R2"]
        },
        {
            "metric": "approach_a_avg_ndcg@5",
            "value": ranking_a["ndcg@5"].mean()
        },
        {
            "metric": "approach_a_avg_recovery_captured@5",
            "value": ranking_a["recovery_captured@5"].mean()
        },
        {
            "metric": "approach_b_avg_ndcg@5",
            "value": ranking_b["ndcg@5"].mean()
        },
        {
            "metric": "approach_b_avg_recovery_captured@5",
            "value": ranking_b["recovery_captured@5"].mean()
        },
        {
            "metric": "approach_b_avg_8hr_recovery_captured",
            "value": capacity_b["actual_recovery_captured_8hrs"].mean()
        }
    ])

    summary_df.to_csv(
        config.outputs_dir / "summary_metrics.csv",
        index=False
    )

    print("\nSaved all outputs to:", config.outputs_dir)


if __name__ == "__main__":
    main()
