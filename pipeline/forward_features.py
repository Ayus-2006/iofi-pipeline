"""
Builds forward-looking feature rows (h=1,2,3) by appending per-driver
statistical forecasts (ARIMA/ETS/Kalman, chosen automatically) onto the
macro history, then re-running the same feature engineering pipeline used
for training. This is what lets the composite-index and lane models predict
forward using *forecasted* driver paths instead of a manual decay curve.
"""
import pandas as pd
import numpy as np
from feature_engineering import build_feature_matrix, DRIVER_COLS, normalize_month
from driver_forecasting import forecast_all_drivers


def extend_macro_with_forecasts(macro_df: pd.DataFrame, horizon: int = 3):
    macro_df = normalize_month(macro_df)
    driver_forecasts = forecast_all_drivers(macro_df, DRIVER_COLS, horizon=horizon)

    last_month = pd.Period(macro_df["month"].iloc[-1], freq="M")
    future_rows = []
    for h in range(1, horizon + 1):
        month = (last_month + h).strftime("%Y-%m")
        row = {"month": month}
        for col in DRIVER_COLS:
            if col in driver_forecasts:
                row[col] = driver_forecasts[col]["forecast"][h - 1]
            else:
                row[col] = macro_df[col].iloc[-1]
        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)
    extended = pd.concat([macro_df, future_df], ignore_index=True, sort=False)
    # any macro column not explicitly forecasted (e.g. legacy/reference series)
    # gets forward-filled so it doesn't inject NaNs into engineered features
    extended = extended.ffill()
    return extended, driver_forecasts


def get_future_feature_rows(macro_df: pd.DataFrame, horizon: int = 3):
    """Returns (feature_df_full, future_feature_rows, driver_forecasts) where
    future_feature_rows has `horizon` rows aligned to the same columns used
    for model training."""
    extended, driver_forecasts = extend_macro_with_forecasts(macro_df, horizon)
    feats_full = build_feature_matrix(extended)
    future_rows = feats_full.tail(horizon).reset_index(drop=True)
    return feats_full, future_rows, driver_forecasts


if __name__ == "__main__":
    import os
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    macro = pd.read_csv(os.path.join(_BASE_DIR, "data", "macro_history.csv"))
    feats_full, future_rows, fc = get_future_feature_rows(macro, horizon=3)
    print(future_rows[["month", "brent_usd_bbl", "congestion_index"]])
