import numpy as np
import matplotlib.pyplot as plt


def plot_median_recovery_by_label(df, outputs_dir):
    label_stats = (
        df.groupby("relevance_label")
        .agg(
            median_recovered=("recovered_amount_target", "median")
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        label_stats["relevance_label"].astype(str),
        label_stats["median_recovered"]
    )

    ax.set_title("Median recovered amount by relevance label")
    ax.set_xlabel("Relevance label")
    ax.set_ylabel("Median recovered amount ($)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    path = outputs_dir / "plot_median_recovery_by_relevance_label.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return path


def plot_value_concentration_by_label(df, outputs_dir):
    total_claims = len(df)
    total_recovered = df["recovered_amount_target"].sum()

    label_stats = (
        df.groupby("relevance_label")
        .agg(
            claims=("claim_id", "count"),
            total_recovered=("recovered_amount_target", "sum")
        )
        .reset_index()
    )

    label_stats["pct_claims"] = (
        label_stats["claims"] / total_claims * 100
    )

    label_stats["pct_recovered_dollars"] = (
        label_stats["total_recovered"] / total_recovered * 100
    )

    x = np.arange(len(label_stats))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width / 2,
        label_stats["pct_claims"],
        width,
        label="% of claims"
    )

    ax.bar(
        x + width / 2,
        label_stats["pct_recovered_dollars"],
        width,
        label="% of recovered dollars"
    )

    ax.set_title("Claim count vs recovered-dollar concentration")
    ax.set_xlabel("Relevance label")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(label_stats["relevance_label"].astype(str))
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    path = outputs_dir / "plot_value_concentration_by_label.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return path


def plot_recovery_vs_handle_time(df, outputs_dir):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        df["handle_time_hrs"],
        df["recovered_amount_target"],
        alpha=0.7
    )

    ax.set_title("Recovered amount vs handle time")
    ax.set_xlabel("Handle time (hours)")
    ax.set_ylabel("Actual recovered amount ($)")
    ax.grid(alpha=0.3)

    fig.tight_layout()

    path = outputs_dir / "plot_recovery_vs_handle_time.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return path


def plot_worklist_time_pressure(worklists_df, outputs_dir):
    historical_worklists = worklists_df[
        worklists_df["is_historical"]
        .astype(str)
        .str.lower()
        .isin(["1", "true", "yes"])
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        historical_worklists["worklist_id"],
        historical_worklists["total_handle_time_hrs"]
    )

    ax.axhline(
        8,
        linestyle="--",
        label="8-hour analyst capacity"
    )

    ax.set_title("Total handle time per worklist vs 8-hour capacity")
    ax.set_xlabel("Worklist")
    ax.set_ylabel("Total handle time (hours)")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    path = outputs_dir / "plot_worklist_time_pressure.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return path


def generate_all_plots(historical_df, worklists_df, outputs_dir):
    plot_paths = []

    plot_paths.append(
        plot_median_recovery_by_label(
            historical_df,
            outputs_dir
        )
    )

    plot_paths.append(
        plot_value_concentration_by_label(
            historical_df,
            outputs_dir
        )
    )

    plot_paths.append(
        plot_recovery_vs_handle_time(
            historical_df,
            outputs_dir
        )
    )

    plot_paths.append(
        plot_worklist_time_pressure(
            worklists_df,
            outputs_dir
        )
    )

    return plot_paths
