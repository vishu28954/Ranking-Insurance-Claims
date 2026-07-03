import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def load_raw_data(data_dir):
    claims_df = pd.read_csv(data_dir / "worklist_claims.csv")
    worklists_df = pd.read_csv(data_dir / "worklists.csv")
    return claims_df, worklists_df


def get_historical_claims(claims_df, worklists_df):
    is_historical_mask = (
        worklists_df["is_historical"]
        .astype(str)
        .str.lower()
        .isin(["1", "true", "yes"])
    )

    historical_worklists = worklists_df.loc[
        is_historical_mask,
        "worklist_id"
    ]

    historical_df = claims_df[
        claims_df["worklist_id"].isin(historical_worklists)
    ].copy()

    return historical_df


def split_by_worklist(df, target_col, seed=42, test_size=0.25):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            df[target_col],
            groups=df["worklist_id"]
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    return train_df, test_df
