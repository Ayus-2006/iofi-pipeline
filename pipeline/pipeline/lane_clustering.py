"""
Cluster trade lanes automatically instead of treating every lane as unique
or manually grouping by region. Uses KMeans over engineered lane features
(distance proxy, chokepoint exposure, historical volatility, rate level).
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CHOKEPOINT_MAP = {
    "hormuz": 1, "suez": 1, "panama": 1, "malacca": 1, "bab-el-mandeb": 1,
}


def build_lane_features(predictions_full: pd.DataFrame) -> pd.DataFrame:
    df = predictions_full.copy()
    df["chokepoint_flag"] = df["route_exposure"].map(lambda r: CHOKEPOINT_MAP.get(str(r).lower(), 0))
    df["rate_volatility_proxy"] = (
        (df["high_m3_usd_feu"] - df["low_m3_usd_feu"]).abs() / df["current_rate_usd_feu"].replace(0, np.nan)
    )
    df["region_code"] = df["region"].astype("category").cat.codes
    feature_cols = [
        "current_rate_usd_feu", "chokepoint_flag", "rate_volatility_proxy",
        "region_code", "pct_change_base_m3",
    ]
    lane_feats = df[["origin", "destination"] + feature_cols].dropna()
    return lane_feats, feature_cols


def cluster_lanes(predictions_full: pd.DataFrame, k_range=range(2, 8)):
    lane_feats, feature_cols = build_lane_features(predictions_full)
    X = lane_feats[feature_cols].values
    Xs = StandardScaler().fit_transform(X)

    best_k, best_score, best_labels = None, -1, None
    for k in k_range:
        if k >= len(Xs):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(Xs, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    lane_feats = lane_feats.copy()
    lane_feats["cluster"] = best_labels
    return lane_feats, best_k, best_score


if __name__ == "__main__":
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    preds = pd.read_csv(_BASE_DIR + "/data/predictions_full.csv")
    clustered, k, score = cluster_lanes(preds)
    print(f"best_k={k} silhouette={score:.3f}")
    clustered.to_csv(_BASE_DIR + "/outputs/lane_clusters.csv", index=False)
    print(clustered["cluster"].value_counts())
