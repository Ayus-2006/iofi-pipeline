"""
Estimate each lane's (or lane-cluster's) sensitivity to macro drivers:

    rate_change ~ b1*IOFI + b2*war_risk + b3*oil + b4*congestion + ...

Rather than a single hand-picked beta per route archetype, we fit one Ridge
model per cluster (falling back to a single pooled model if a cluster has
too few observations to fit reliably).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def build_lane_training_frame(predictions_full: pd.DataFrame, macro_latest: pd.Series, iofi_latest: float):
    """
    Since only one snapshot of predictions_full exists per run, this builds a
    cross-sectional training frame: rows = lanes, columns = current macro
    state (broadcast) + lane-level pct_change_base_m3 as the target.
    In production with monthly snapshots, this should be stacked over time
    (panel data) for genuine time-series lane betas.
    """
    df = predictions_full.copy()
    for col, val in macro_latest.items():
        df[col] = val
    df["IOFI"] = iofi_latest
    return df


def fit_cluster_sensitivities(df: pd.DataFrame, cluster_col: str, driver_cols, target_col="pct_change_base_m3"):
    """Fit one Ridge regression per cluster; pool clusters with <5 rows."""
    results = {}
    pooled_rows = []
    for cluster_id, sub in df.groupby(cluster_col):
        X = sub[driver_cols].values
        y = sub[target_col].values
        if len(sub) < 5:
            pooled_rows.append(sub)
            continue
        model = Ridge(alpha=2.0)
        model.fit(X, y)
        results[cluster_id] = {
            "n_obs": len(sub),
            "coefficients": dict(zip(driver_cols, model.coef_)),
            "intercept": model.intercept_,
            "r2_train": model.score(X, y),
        }

    if pooled_rows:
        pooled = pd.concat(pooled_rows)
        X = pooled[driver_cols].values
        y = pooled[target_col].values
        model = Ridge(alpha=2.0)
        model.fit(X, y)
        results["pooled_small_clusters"] = {
            "n_obs": len(pooled),
            "coefficients": dict(zip(driver_cols, model.coef_)),
            "intercept": model.intercept_,
            "r2_train": model.score(X, y),
        }
    return results


if __name__ == "__main__":
    from lane_clustering import cluster_lanes
    import os
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preds = pd.read_csv(_BASE_DIR + "/data/predictions_full.csv")
    macro = pd.read_csv(_BASE_DIR + "/data/macro_history.csv")
    iofi = pd.read_csv(_BASE_DIR + "/data/iofi_history.csv")
    clustered, k, score = cluster_lanes(preds)
    df = preds.merge(clustered[["origin", "destination", "cluster"]], on=["origin", "destination"])
    macro_latest = macro.drop(columns=["month"]).iloc[-1]
    iofi_latest = iofi["IOFI"].iloc[-1]
    train_df = build_lane_training_frame(df, macro_latest, iofi_latest)
    driver_cols = list(macro_latest.index) + ["IOFI"]
    sens = fit_cluster_sensitivities(train_df, "cluster", driver_cols)
    for cid, r in sens.items():
        print(cid, r["n_obs"], "r2=", round(r["r2_train"], 3))
