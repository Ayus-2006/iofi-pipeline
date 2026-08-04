"""
Self-updating monthly pipeline.

Each run:
  1. Loads latest macro + IOFI + lane data (swap load_data() for a live
     downloader when a real data feed is connected).
  2. Rebuilds engineered features (lags, momentum, seasonality, interactions).
  3. Re-learns driver weights (Ridge/ElasticNet/Bayesian/XGB + SHAP).
  4. Forecasts each driver independently (ARIMA/ETS/Kalman).
  5. Reclusters lanes and re-estimates lane sensitivities.
  6. Produces quantile-based uncertainty bands.
  7. Backtests the refreshed model against benchmarks.
  8. Saves the best-performing model + a full run manifest — no manual
     parameter tuning required.
"""
import json
import os
import pickle
from datetime import datetime, timezone

import pandas as pd

from feature_engineering import build_feature_matrix, DRIVER_COLS
from weight_learning import fit_all_weight_models
from driver_forecasting import forecast_all_drivers
from lane_clustering import cluster_lanes
from lane_sensitivity import build_lane_training_frame, fit_cluster_sensitivities
from uncertainty import quantile_regression_forecast
from backtest import rolling_origin_backtest
import data_sources

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../iofi_v2
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")


def update_macro_history(live_fetch: bool = True) -> pd.DataFrame:
    """Grows data/macro_history.csv with the latest month, live-fetched from
    free public sources where possible (see data_sources.py). Idempotent: if
    the current target month is already the last row, it's overwritten in
    place rather than duplicated, so re-running mid-month just refreshes
    that row instead of creating dupes."""
    path = f"{DATA_DIR}/macro_history.csv"
    macro = pd.read_csv(path)
    if not live_fetch:
        return macro

    try:
        new_row = data_sources.get_latest_month_row(macro)
    except Exception as e:  # noqa: BLE001
        print(f"[pipeline_runner] Live data fetch failed entirely ({e}); "
              f"continuing with existing macro_history.csv only.")
        return macro

    new_month = new_row["month"]
    if (macro["month"] == new_month).any():
        macro.loc[macro["month"] == new_month, list(new_row.keys())] = list(new_row.values())
    else:
        macro = pd.concat([macro, pd.DataFrame([new_row])], ignore_index=True)

    macro = macro.sort_values("month").reset_index(drop=True)
    macro.to_csv(path, index=False)
    print(f"[pipeline_runner] macro_history.csv now has {len(macro)} monthly rows "
          f"(through {macro['month'].iloc[-1]}).")
    return macro


def load_data(live_fetch: bool = True):
    macro = update_macro_history(live_fetch=live_fetch)
    iofi = pd.read_csv(f"{DATA_DIR}/iofi_history.csv")
    preds = pd.read_csv(f"{DATA_DIR}/predictions_full.csv")

    latest_macro_month = macro["month"].iloc[-1][:7]
    latest_iofi_month = iofi["month"].iloc[-1]
    if latest_macro_month != latest_iofi_month:
        print(f"[pipeline_runner] NOTE: macro data now goes through {latest_macro_month}, "
              f"but iofi_history.csv's latest observed value is still {latest_iofi_month}. "
              f"IOFI itself is your own proprietary index, not something a public feed can "
              f"supply -- run `python3 record_observed_month.py` to log this month's actual "
              f"IOFI value and lane rates before generating the report, or the model will "
              f"train on the same target data as last run.")
    return macro, iofi, preds


def run_pipeline():
    os.makedirs(OUT_DIR, exist_ok=True)
    macro, iofi, preds = load_data()

    # 1. Features
    feat_df = build_feature_matrix(macro).merge(iofi, on="month", how="inner").dropna().reset_index(drop=True)
    feature_cols = [c for c in feat_df.columns if c not in ("month", "IOFI", "month_num")]
    X, y = feat_df[feature_cols], feat_df["IOFI"]

    # 2. Driver weights
    weight_models, importance_df = fit_all_weight_models(X, y)

    # 3. Per-driver forecasts (replaces manual scenario decay)
    driver_forecasts = forecast_all_drivers(macro, DRIVER_COLS, horizon=3)

    # 4. Lane clustering + sensitivities
    clustered, best_k, sil_score = cluster_lanes(preds)
    merged = preds.merge(clustered[["origin", "destination", "cluster"]], on=["origin", "destination"])
    macro_latest = macro.drop(columns=["month"]).iloc[-1]
    iofi_latest = iofi["IOFI"].iloc[-1]
    lane_train = build_lane_training_frame(merged, macro_latest, iofi_latest)
    driver_cols_for_sens = list(macro_latest.index) + ["IOFI"]
    lane_sensitivities = fit_cluster_sensitivities(lane_train, "cluster", driver_cols_for_sens)

    # 5. Uncertainty bands for next-period IOFI
    X_future = X.tail(1)
    quantiles = quantile_regression_forecast(X, y, X_future)

    # 6. Backtest vs benchmarks
    min_train = max(6, len(X) - 8)
    backtest_res, backtest_metrics = rolling_origin_backtest(X, y, min_train=min_train)

    # 7. Persist best model + manifest
    with open(f"{OUT_DIR}/best_model.pkl", "wb") as f:
        pickle.dump(weight_models, f)

    manifest = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "macro_history_through": macro["month"].iloc[-1],
        "macro_history_n_months": len(macro),
        "iofi_history_through": iofi["month"].iloc[-1],
        "n_training_months": len(X),
        "n_lanes": len(preds),
        "best_lane_cluster_k": int(best_k) if best_k else None,
        "lane_cluster_silhouette": round(float(sil_score), 3) if sil_score else None,
        "top_driver_weights": importance_df.head(5).to_dict(orient="records"),
        "driver_forecast_models": {k: v["model"] for k, v in driver_forecasts.items()},
        "next_period_iofi_quantiles": quantiles.iloc[0].to_dict(),
        "backtest_metrics": backtest_metrics,
    }
    with open(f"{OUT_DIR}/run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    importance_df.to_csv(f"{OUT_DIR}/learned_driver_weights.csv", index=False)
    backtest_res.to_csv(f"{OUT_DIR}/backtest_results.csv", index=False)

    print("Pipeline run complete.")
    print(json.dumps(manifest["backtest_metrics"], indent=2))
    return manifest


if __name__ == "__main__":
    run_pipeline()
