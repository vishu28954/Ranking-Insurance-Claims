import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer


def add_core_features(df):
    df = df.copy()

    df["recovered_amount_target"] = (
        df["recovered_amount"]
        .fillna(0)
    )

    df["is_recovered"] = (
        df["recovered_amount_target"] > 0
    ).astype(int)

    df["amount_yet_to_recover"] = (
        -df["payment_variance"]
    ).clip(lower=0)

    df["Percent_amount_yet_to_recover"] = np.where(
        df["expected_payment"] > 0,
        df["amount_yet_to_recover"] / df["expected_payment"] * 100,
        0
    )

    df["log_total_billed"] = np.log1p(
        df["total_billed"]
    )

    df["log_amount_yet"] = np.log1p(
        df["amount_yet_to_recover"]
    )

    df["payer_contract"] = (
        df["payer_id"].astype(str) + "_" +
        df["contract_version"].astype(str)
    )

    df["payer_visit"] = (
        df["payer_id"].astype(str) + "_" +
        df["visit_type"].astype(str)
    )

    df["payer_type_contract"] = (
        df["payer_type"].astype(str) + "_" +
        df["contract_version"].astype(str)
    )

    return df


def clean_adjustment_codes(value):
    if pd.isna(value):
        return []

    value = str(value).strip()

    if value == "" or value.lower() == "nan":
        return []

    return [
        code.strip().replace("-", "")
        for code in value.split("|")
        if code.strip() != ""
    ]


def add_adjustment_code_features(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    old_train_adj_cols = [
        col for col in train_df.columns
        if col.startswith("adj_")
    ]

    old_test_adj_cols = [
        col for col in test_df.columns
        if col.startswith("adj_")
    ]

    train_df = train_df.drop(
        columns=old_train_adj_cols,
        errors="ignore"
    )

    test_df = test_df.drop(
        columns=old_test_adj_cols,
        errors="ignore"
    )

    train_df["adjustment_code_list"] = (
        train_df["adjustment_codes"]
        .apply(clean_adjustment_codes)
    )

    test_df["adjustment_code_list"] = (
        test_df["adjustment_codes"]
        .apply(clean_adjustment_codes)
    )

    mlb = MultiLabelBinarizer()

    train_matrix = mlb.fit_transform(
        train_df["adjustment_code_list"]
    )

    test_matrix = mlb.transform(
        test_df["adjustment_code_list"]
    )

    adjustment_code_columns = [
        f"adj_{code}"
        for code in mlb.classes_
    ]

    train_adj_df = pd.DataFrame(
        train_matrix,
        columns=adjustment_code_columns,
        index=train_df.index
    )

    test_adj_df = pd.DataFrame(
        test_matrix,
        columns=adjustment_code_columns,
        index=test_df.index
    )

    train_df = pd.concat(
        [train_df, train_adj_df],
        axis=1
    )

    test_df = pd.concat(
        [test_df, test_adj_df],
        axis=1
    )

    return train_df, test_df, adjustment_code_columns, mlb


def get_feature_columns(adjustment_code_columns):
    categorical_features = [
        "payer_id",
        "payer_type",
        "visit_type",
        "contract_version",
        "payer_contract",
        "payer_visit",
        "payer_type_contract"
    ]

    numeric_features = [
        "total_billed",
        "expected_payment",
        "actual_payment",
        "amount_yet_to_recover",
        "Percent_amount_yet_to_recover",
        "days_since_payment",
        "log_total_billed",
        "log_amount_yet"
    ] + adjustment_code_columns

    features = categorical_features + numeric_features

    return features, categorical_features, numeric_features
