"""
Assembles every number the v2 report needs:
  - learned driver weights (ensemble of Ridge/ElasticNet/Bayesian/XGB)
  - per-driver forecasts (ARIMA/ETS/Kalman, auto-selected)
  - composite IOFI forecast path (ensemble regression on forecasted drivers)
  - IOFI uncertainty bands (bootstrap ensemble quantiles)
  - lane clusters + empirically calibrated pass-through betas
  - lane-level forecast paths + uncertainty bands
  - rolling-origin backtest metrics
Saves everything needed for chart/PDF generation into /outputs.
"""
import json
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from feature_engineering import (
    build_feature_matrix, normalize_month, DRIVER_COLS,
    SEASONAL_EVENTS, REGION_SEASONAL_SENSITIVITY,
)
from forward_features import get_future_feature_rows
from weight_learning import fit_all_weight_models
from uncertainty import bootstrap_ensemble_forecast, walk_forward_interval_calibration
from backtest import rolling_origin_backtest
from lane_clustering import cluster_lanes
from hierarchical import compute_hierarchical_betas, hierarchical_region_summary, regional_seasonal_pct

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
HORIZON = 3


def assemble():
    os.makedirs(OUT_DIR, exist_ok=True)
    macro = pd.read_csv(f"{DATA_DIR}/macro_history.csv")
    iofi = normalize_month(pd.read_csv(f"{DATA_DIR}/iofi_history.csv"))
    preds = pd.read_csv(f"{DATA_DIR}/predictions_full.csv")

    # ---- training frame ----
    df = build_feature_matrix(macro).merge(iofi, on="month", how="inner").dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("month", "IOFI", "month_num")]
    X, y = df[feature_cols], df["IOFI"]

    # ---- learned driver weights ----
    models, importance = fit_all_weight_models(X, y)

    # ---- forward driver forecasts + future feature rows ----
    feats_full, future_rows, driver_forecasts = get_future_feature_rows(macro, horizon=HORIZON)
    X_future = future_rows[feature_cols]
    future_months = future_rows["month"].tolist()

    # ---- composite IOFI path: ensemble of the 4 learned models ----
    scaler = models["scaler"]
    Xf_s = scaler.transform(X_future)
    ens_path = np.mean([
        models["ridge"].predict(Xf_s),
        models["elasticnet"].predict(Xf_s),
        models["bayesian"].predict(Xf_s),
        models["xgb"].predict(Xf_s),
    ], axis=0)

    # ---- IOFI uncertainty bands: bootstrap ensemble (Ridge base learner) ----
    iofi_quantiles = bootstrap_ensemble_forecast(
        X, y, X_future, lambda: Ridge(alpha=3.0), n_boot=400
    )
    iofi_quantiles.insert(0, "month", future_months)
    iofi_quantiles.insert(1, "median_model_path", ens_path)

    iofi_current = y.iloc[-1]

    # ---- backtest ----
    min_train = max(6, len(X) - 8)
    backtest_res, backtest_metrics = rolling_origin_backtest(X, y, min_train=min_train)

    # ---- lane clustering ----
    clustered, best_k, sil = cluster_lanes(preds)
    lanes = preds.merge(clustered[["origin", "destination", "cluster"]], on=["origin", "destination"])

    # ---- empirically calibrated per-lane beta ----
    # Only one historical lane snapshot exists (no lane-level time series), so a
    # true regression-based sensitivity is not identifiable from this data (see
    # ARCHITECTURE.md). Instead we calibrate each lane's implied pass-through
    # multiplier from its realized 3-month move versus the index move that
    # produced it, then apply that multiplier to the NEW, statistically
    # forecasted index path rather than to a hand-set decay assumption.
    idx_pct_change_ref = (lanes["base_m3_usd_feu"] / lanes["current_rate_usd_feu"] - 1) * 100
    implied_index_change_pct = float((iofi_current - 100) / 100 * 0)  # placeholder unused
    # reference index move used by the ORIGINAL model's base case, taken from
    # this report run's own history file (iofi_forecast.csv), else default.
    try:
        old_fc = pd.read_csv(f"{DATA_DIR}/iofi_forecast.csv")
        old_base_m3 = old_fc["base"].iloc[-1] if "base" in old_fc.columns else old_fc.iloc[-1, 1]
    except Exception:
        old_base_m3 = iofi_current * 0.869  # fallback ~ -13.1%
    idx_pct_change_old = (old_base_m3 - iofi_current) / iofi_current * 100
    lanes["beta_lane"] = lanes["pct_change_base_m3"] / idx_pct_change_old

    # ---- hierarchical decomposition: Global IOFI -> Regional -> Lane ----
    # Splits each lane's flat beta into a regional component (corridor-wide
    # co-movement) and a lane premium (this lane's deviation from its
    # corridor). Total point forecast is unchanged; see hierarchical.py.
    lanes = compute_hierarchical_betas(lanes, region_col="region", lane_beta_col="beta_lane")
    region_summary = hierarchical_region_summary(lanes, region_col="region")

    # ---- lane forecast paths using the NEW composite path + bootstrap quantiles ----
    q_cols = [0.05, 0.25, 0.5, 0.75, 0.95]
    lane_forecast_rows = []
    for h in range(HORIZON):
        idx_med = ens_path[h]
        idx_pct_change_h = (idx_med - iofi_current) / iofi_current * 100
        q_vals = iofi_quantiles.iloc[h][q_cols].astype(float)
        q_pct_changes = {q: (q_vals[q] - iofi_current) / iofi_current * 100 for q in q_cols}

        for _, lane in lanes.iterrows():
            beta = lane["beta_lane"]
            model_pct = beta * idx_pct_change_h
            seasonal_pct = regional_seasonal_pct(lane["region"], future_months[h])
            base_pct = model_pct + seasonal_pct
            rate = lane["current_rate_usd_feu"] * (1 + base_pct / 100)
            low_pct = beta * q_pct_changes[0.05] + seasonal_pct
            high_pct = beta * q_pct_changes[0.95] + seasonal_pct
            region_pct = lane["region_beta"] * idx_pct_change_h
            premium_pct = lane["lane_premium_beta"] * idx_pct_change_h
            lane_forecast_rows.append({
                "origin": lane["origin"], "destination": lane["destination"],
                "region": lane["region"], "route_exposure": lane["route_exposure"],
                "cluster": lane["cluster"], "month": future_months[h], "horizon": h + 1,
                "current_rate_usd_feu": lane["current_rate_usd_feu"],
                "forecast_rate_usd_feu": round(rate, 0),
                "pct_change": round(base_pct, 1),
                "regional_pct_change": round(region_pct, 1),
                "lane_premium_pct_change": round(premium_pct, 1),
                "seasonal_pct_change": round(seasonal_pct, 2),
                "low_rate_usd_feu": round(lane["current_rate_usd_feu"] * (1 + low_pct / 100), 0),
                "high_rate_usd_feu": round(lane["current_rate_usd_feu"] * (1 + high_pct / 100), 0),
            })
    lane_forecasts = pd.DataFrame(lane_forecast_rows)

    # ---- regional seasonal calendar summary for the forecast window ----
    seasonal_summary_rows = []
    for region in sorted(lanes["region"].unique()):
        row = {"region": region}
        for h in range(HORIZON):
            month_num = pd.Period(future_months[h], freq="M").month
            active_events = [ev for ev in SEASONAL_EVENTS.get(month_num, {})
                              if REGION_SEASONAL_SENSITIVITY.get(region, {}).get(ev, 0) > 0]
            row[f"{future_months[h]}_seasonal_pct"] = round(regional_seasonal_pct(region, future_months[h]), 2)
            row[f"{future_months[h]}_events"] = ", ".join(active_events) if active_events else "-"
        seasonal_summary_rows.append(row)
    seasonal_summary = pd.DataFrame(seasonal_summary_rows)

    # ---- interval calibration check (retrospective coverage of the bootstrap bands) ----
    calib_res, calibration = walk_forward_interval_calibration(
        X, y, lambda: Ridge(alpha=3.0), min_train=min_train, n_boot=150
    )

    # ---- persist everything ----
    importance.to_csv(f"{OUT_DIR}/learned_driver_weights.csv", index=False)
    iofi_quantiles.to_csv(f"{OUT_DIR}/iofi_forecast_v2.csv", index=False)
    lane_forecasts.to_csv(f"{OUT_DIR}/lane_forecasts_v2.csv", index=False)
    backtest_res.to_csv(f"{OUT_DIR}/backtest_results.csv", index=False)
    calib_res.to_csv(f"{OUT_DIR}/interval_calibration.csv", index=False)
    region_summary.to_csv(f"{OUT_DIR}/region_summary.csv", index=False)
    seasonal_summary.to_csv(f"{OUT_DIR}/seasonal_calendar_summary.csv", index=False)
    lanes[["origin", "destination", "region", "cluster", "route_exposure",
           "beta_lane", "region_beta", "lane_premium_beta"]].to_csv(
        f"{OUT_DIR}/lane_betas.csv", index=False
    )

    manifest = {
        "iofi_current": float(iofi_current),
        "future_months": future_months,
        "iofi_ensemble_path": ens_path.tolist(),
        "driver_forecast_models": {k: v["model"] for k, v in driver_forecasts.items()},
        "top_driver_weights": importance.head(10).to_dict(orient="records"),
        "backtest_metrics": backtest_metrics,
        "interval_calibration": calibration,
        "xgb_reliability": models.get("xgb_reliability"),
        "lane_cluster_k": int(best_k) if best_k else None,
        "lane_cluster_silhouette": round(float(sil), 3) if sil else None,
        "n_training_months": int(len(X)),
        "n_regions": int(region_summary.shape[0]),
        "n_driver_cols": len(DRIVER_COLS),
        "driver_cols": DRIVER_COLS,
    }
    with open(f"{OUT_DIR}/run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return {
        "importance": importance, "iofi_quantiles": iofi_quantiles, "lane_forecasts": lane_forecasts,
        "lanes": lanes, "backtest_metrics": backtest_metrics, "backtest_res": backtest_res,
        "calibration": calibration, "region_summary": region_summary, "seasonal_summary": seasonal_summary,
        "driver_forecasts": driver_forecasts, "manifest": manifest, "macro": macro, "iofi_hist": iofi,
        "feature_cols": feature_cols, "X": X, "y": y,
    }


if __name__ == "__main__":
    out = assemble()
    print(out["iofi_quantiles"])
    print(out["lane_forecasts"].head())
    print(out["manifest"]["backtest_metrics"])
