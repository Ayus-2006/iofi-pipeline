"""
One-time backfill: pull as much free history as each source actually offers
and extend data/macro_history.csv backwards, instead of waiting years for
the monthly run to accumulate it one row at a time.

Real depth available per source (varies; verify against the live source --
these are the documented starting points as of this writing):
  brent_usd_bbl        FRED DCOILBRENTEU        back to 1987
  usd_index             FRED DTWEXBGS             back to 2006 (broad index)
  inr_usd_rate           FRED DEXINUS               back to 1973
  gpr_index               Caldara/Iacoviello GPR     back to 1985 (benchmark), 1900 (historical)
  war_risk_premium_idx (proxy) same as GPR, wherever country columns exist
  trade_volume_growth_idx CPB World Trade Monitor    back to ~2000
  congestion_index          IMF PortWatch                ~2019 onward (post-COVID launch)
  china_pmi_idx (OECD CLI proxy)                       back to ~2000

  eu_ets_carbon_idx and china_export_container_idx have no free live feed of
  any kind (see data_sources.py's FetchWarnings) -- historical months for
  these two are backfilled with the same REGRESSION_PROXY_SPECS regression
  used by the monthly run, fit on whatever rows in the *existing* CSV
  already have real values, then applied to any newly-backfilled month that
  has all of that proxy's predictor columns.

Columns with no long free history (congestion, china PMI) will simply have
NaN before their real start date -- this is fine: driver_forecasting.py fits
each column's ARIMA/ETS/Kalman model on whatever length that column actually
has (see the .dropna() fix in forecast_driver), so a longer Brent or GPR
series still directly improves that driver's own forecast quality even
while congestion_index only reaches back to 2019.

Run once, then let pipeline_runner.py's monthly live-fetch take over:
    python3 backfill_history.py
"""
import os

import pandas as pd

import data_sources

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def backfill():
    path = f"{DATA_DIR}/macro_history.csv"
    current = pd.read_csv(path)
    current["month"] = pd.to_datetime(current["month"])

    print("Fetching full free history for each real driver (this can take a minute)...")
    long_series = {}

    for name, fn in data_sources.REAL_FETCHERS.items():
        try:
            long_series[name] = fn()
            print(f"  {name}: got {len(long_series[name])} months, "
                  f"back to {long_series[name].index.min().date()}")
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: backfill failed ({e}), keeping existing rows only")

    gpr_df = None
    try:
        gpr_df = data_sources.fetch_gpr()
        gpr_col = "GPR" if "GPR" in gpr_df.columns else [c for c in gpr_df.columns if "GPR" in c.upper()][0]
        long_series["gpr_index"] = gpr_df[gpr_col]
        print(f"  gpr_index: got {len(long_series['gpr_index'])} months, "
              f"back to {long_series['gpr_index'].index.min().date()}")
    except Exception as e:  # noqa: BLE001
        print(f"  gpr_index: backfill failed ({e})")

    trade_vol = None
    try:
        trade_vol = data_sources.fetch_world_trade_volume()
        long_series["trade_volume_growth_idx"] = trade_vol
        print(f"  trade_volume_growth_idx: got {len(trade_vol)} months")
    except Exception as e:  # noqa: BLE001
        print(f"  trade_volume_growth_idx: backfill failed ({e})")

    if gpr_df is not None:
        try:
            long_series["war_risk_premium_idx"] = data_sources.proxy_war_risk_premium(gpr_df)
        except Exception as e:  # noqa: BLE001
            print(f"  war_risk_premium_idx proxy: failed ({e})")
    if trade_vol is not None:
        try:
            long_series["panama_restriction_idx"] = data_sources.proxy_panama_restriction(trade_vol)
        except Exception as e:  # noqa: BLE001
            print(f"  panama_restriction_idx proxy: failed ({e})")

    if not long_series:
        print("No sources reachable from this environment -- nothing to backfill. "
              "This is expected if you're running in a network-restricted sandbox; "
              "run this script instead from a machine with normal internet access.")
        return

    all_months = sorted(set().union(*[set(s.index) for s in long_series.values()]))
    extended = pd.DataFrame({"month": all_months})
    for col in current.columns:
        if col == "month":
            continue
        if col in long_series:
            extended[col] = extended["month"].map(long_series[col])
        else:
            extended[col] = pd.NA

    merged = pd.concat([extended, current]).drop_duplicates(subset="month", keep="last")
    merged = merged.sort_values("month").reset_index(drop=True)

    # eu_ets_carbon_idx / china_export_container_idx have no free live feed
    # at all -- backfill NaN months for these two with the same regression
    # proxy the monthly run falls back to, fit on whatever rows in the
    # *original* CSV already carried real values for both target and
    # predictors.
    for target_col, predictor_cols in data_sources.REGRESSION_PROXY_SPECS.items():
        if target_col not in merged.columns:
            continue
        try:
            model = data_sources.fit_regression_proxy(current, target_col, predictor_cols)
        except Exception as e:  # noqa: BLE001
            print(f"  {target_col} regression proxy: could not fit ({e}), leaving gaps as NaN")
            continue
        if model is None:
            print(f"  {target_col} regression proxy: not enough historical overlap to fit, leaving gaps as NaN")
            continue
        need_fill = merged[target_col].isna()
        filled = 0
        for idx in merged.index[need_fill]:
            predictor_values = {c: merged.at[idx, c] for c in predictor_cols}
            pred = data_sources.predict_regression_proxy(model, predictor_cols, predictor_values)
            if pred is not None:
                merged.at[idx, target_col] = pred
                filled += 1
        if filled:
            print(f"  {target_col}: backfilled {filled} historical months via regression proxy")

    merged["month"] = merged["month"].dt.strftime("%Y-%m-%d")
    merged.to_csv(path, index=False)
    print(f"\nmacro_history.csv extended: {len(current)} -> {len(merged)} monthly rows "
          f"(now spans {merged['month'].iloc[0]} to {merged['month'].iloc[-1]}).")
    print("Columns still short on history (no free long-run feed exists): "
          f"{[c for c in current.columns if c not in long_series and c != 'month']}")


if __name__ == "__main__":
    backfill()
