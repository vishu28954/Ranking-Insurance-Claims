import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def make_one_hot_encoder():
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def make_preprocessor(categorical_features, numeric_features):
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                make_one_hot_encoder(),
                categorical_features
            ),
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_features
            )
        ]
    )

    return preprocessor


def train_stage1_classifiers(
    train_df,
    test_df,
    features,
    categorical_features,
    numeric_features,
    target_col,
    seed=42
):
    preprocessor = make_preprocessor(
        categorical_features,
        numeric_features
    )

    classifiers = {
        "LogisticRegression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=seed
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=seed,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1
        ),

        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            random_state=seed,
            min_samples_leaf=4,
            class_weight="balanced",
            n_jobs=-1
        ),

        "GradientBoosting": GradientBoostingClassifier(
            random_state=seed,
            n_estimators=200,
            learning_rate=0.03,
            max_depth=2
        )
    }

    results = []
    trained_models = {}

    for model_name, classifier in classifiers.items():
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ]
        )

        model.fit(
            train_df[features],
            train_df[target_col]
        )

        pred = model.predict(
            test_df[features]
        )

        pred_proba = model.predict_proba(
            test_df[features]
        )[:, 1]

        results.append({
            "model": model_name,
            "accuracy": accuracy_score(test_df[target_col], pred),
            "precision": precision_score(
                test_df[target_col],
                pred,
                zero_division=0
            ),
            "recall": recall_score(
                test_df[target_col],
                pred,
                zero_division=0
            ),
            "f1": f1_score(
                test_df[target_col],
                pred,
                zero_division=0
            ),
            "roc_auc": roc_auc_score(
                test_df[target_col],
                pred_proba
            ),
            "avg_precision": average_precision_score(
                test_df[target_col],
                pred_proba
            )
        })

        trained_models[model_name] = model

    results_df = pd.DataFrame(results).sort_values(
        "roc_auc",
        ascending=False
    )

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    return best_model_name, best_model, results_df


def train_stage2_amount_models(
    train_df,
    test_df,
    features,
    categorical_features,
    numeric_features,
    target_col,
    seed=42
):
    positive_train_df = train_df[
        train_df["is_recovered"] == 1
    ].copy()

    positive_test_df = test_df[
        test_df["is_recovered"] == 1
    ].copy()

    preprocessor = make_preprocessor(
        categorical_features,
        numeric_features
    )

    amount_models = {
        "GradientBoosting_log": GradientBoostingRegressor(
            random_state=seed,
            n_estimators=300,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=5
        ),

        "RandomForest_log": RandomForestRegressor(
            random_state=seed,
            n_estimators=500,
            min_samples_leaf=3,
            max_features=0.8,
            n_jobs=-1
        ),

        "ExtraTrees_log": ExtraTreesRegressor(
            random_state=seed,
            n_estimators=500,
            min_samples_leaf=3,
            max_features=0.8,
            bootstrap=True,
            n_jobs=-1
        )
    }

    results = []
    trained_models = {}

    y_train_log = np.log1p(
        positive_train_df[target_col]
    )

    for model_name, regressor in amount_models.items():
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", regressor)
            ]
        )

        model.fit(
            positive_train_df[features],
            y_train_log
        )

        pred_log = model.predict(
            positive_test_df[features]
        )

        pred_amount = np.expm1(pred_log)

        pred_amount = np.clip(
            pred_amount,
            0,
            None
        )

        pred_amount = np.minimum(
            pred_amount,
            positive_test_df["amount_yet_to_recover"]
        )

        mae = mean_absolute_error(
            positive_test_df[target_col],
            pred_amount
        )

        rmse = mean_squared_error(
            positive_test_df[target_col],
            pred_amount
        ) ** 0.5

        r2 = r2_score(
            positive_test_df[target_col],
            pred_amount
        )

        results.append({
            "model": model_name,
            "MAE_on_recovered_claims": mae,
            "RMSE_on_recovered_claims": rmse,
            "R2_on_recovered_claims": r2
        })

        trained_models[model_name] = model

    results_df = pd.DataFrame(results).sort_values(
        "MAE_on_recovered_claims"
    )

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    return best_model_name, best_model, results_df


def create_final_predictions(
    test_df,
    stage1_model,
    stage2_model,
    features,
    threshold=0.30
):
    test_df = test_df.copy()

    test_df["recovery_probability"] = (
        stage1_model
        .predict_proba(test_df[features])[:, 1]
    )

    test_df["predicted_amount_if_recovered"] = np.expm1(
        stage2_model.predict(test_df[features])
    )

    test_df["predicted_amount_if_recovered"] = (
        test_df["predicted_amount_if_recovered"]
        .clip(lower=0)
    )

    test_df["predicted_amount_if_recovered"] = np.minimum(
        test_df["predicted_amount_if_recovered"],
        test_df["amount_yet_to_recover"]
    )

    test_df["final_predicted_recovery"] = np.where(
        test_df["recovery_probability"] >= threshold,
        test_df["predicted_amount_if_recovered"],
        0
    )

    test_df["score_approach_a"] = (
        test_df["final_predicted_recovery"]
    )

    test_df["score_approach_b"] = (
        test_df["final_predicted_recovery"] /
        test_df["handle_time_hrs"].clip(lower=0.25)
    )

    return test_df
