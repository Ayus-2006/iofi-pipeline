"""
Feature engineering for IOFI v2.
Builds lag features, rolling statistics, momentum, seasonality flags,
and nonlinear interaction terms from the macro driver history.
"""
import pandas as pd
import os
import numpy as np

DRIVER_COLS = [
    "brent_usd_bbl", "war_risk_premium_idx", "gpr_index", "congestion_index",
    "vessel_idle_capacity_pct", "trade_volume_growth_idx", "usd_index",
    "inr_usd_rate", "panama_restriction_idx", "eu_ets_carbon_idx",
    # China-side supply drivers (v2.1) -- the model previously only carried
    # demand-pull seasonality anchored to US import/holiday timing (Black
    # Friday, Christmas, US import season). These two add the China supply
    # side: factory output/confidence (PMI-like) and actual export
    # throughput, both of which move on China's own festival calendar
    # (Chinese New Year factory shutdowns, Golden Week) rather than a US one.
    "china_pmi_idx", "china_export_container_idx",
]

# Festival/holiday calendar used for seasonality features AND for the
# region-specific seasonal premium in hierarchical.py. Intensity is 0-1.
# v2.1 expands this beyond the original US-centric set (Black Friday,
# Christmas, US import season) to add South Asian, Middle Eastern, and a
# second China-specific demand event (Singles Day) alongside the existing
# supply-side China events (Chinese New Year, Golden Week).
SEASONAL_EVENTS = {
    # month -> {event_name: intensity 0-1}, approximate calendar anchors
    1: {"chinese_new_year_prep": 0.6, "us_import_season_tail": 0.3},
    2: {"chinese_new_year": 1.0},
    3: {"post_cny_recovery": 0.5},
    4: {"ramadan": 0.6},
    5: {"ramadan_tail": 0.3, "indian_export_season": 0.4, "eid_al_fitr": 0.7},
    6: {"indian_export_season": 0.6, "eid_al_adha": 0.5},
    7: {"us_import_season_start": 0.5},
    8: {"us_import_season": 0.8, "golden_week_prep": 0.3, "ganesh_chaturthi": 0.4},
    9: {"us_import_season_peak": 1.0, "diwali_prep": 0.4},
    10: {"golden_week": 0.7, "us_import_season_tail": 0.6, "diwali": 0.9, "singles_day_prep": 0.4},
    11: {"black_friday": 1.0, "singles_day": 0.9, "thanksgiving": 0.6},
    12: {"christmas": 1.0, "black_friday_tail": 0.4},
}

# Which festival events actually move demand/supply for a given destination
# region, and how strongly (0-1). Used to build a region-specific seasonal
# premium on top of the flat global seasonality features -- e.g. Christmas
# matters for Europe/US lanes but not Middle East ones; Diwali matters for
# Indian-origin-linked demand surges; Golden Week/Singles Day are China-side
# events that hit Intra-Asia and any lane sharing vessel capacity with China
# export lanes, not just the region China itself sits in.
REGION_SEASONAL_SENSITIVITY = {
    "US East Coast": {"black_friday": 0.9, "christmas": 0.8, "thanksgiving": 0.6, "us_import_season_peak": 0.7},
    "US Gulf / Caribbean / Latin America": {"black_friday": 0.8, "christmas": 0.7, "thanksgiving": 0.5, "us_import_season_peak": 0.6},
    "Pacific (US West / Oceania)": {"black_friday": 0.8, "christmas": 0.6, "us_import_season_peak": 0.7},
    "Europe (North)": {"christmas": 0.9, "black_friday": 0.5},
    "Europe (Mediterranean)": {"christmas": 0.7, "black_friday": 0.4},
    "Middle East": {"ramadan": 0.8, "eid_al_fitr": 0.7, "eid_al_adha": 0.6},
    "Red Sea / East Africa": {"ramadan": 0.7, "eid_al_fitr": 0.6, "eid_al_adha": 0.5},
    "Indian Ocean Rim": {"diwali": 0.7, "indian_export_season": 0.5, "ganesh_chaturthi": 0.3},
    "Intra-Asia": {"chinese_new_year": 0.9, "golden_week": 0.8, "singles_day": 0.6, "chinese_new_year_prep": 0.4},
    "Africa / South America (Cape)": {"christmas": 0.3, "diwali": 0.2},
}


def add_seasonality(df: pd.DataFrame, date_col: str = "month") -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df["month_num"] = dt.dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    all_events = sorted({e for m in SEASONAL_EVENTS.values() for e in m})
    for ev in all_events:
        df[f"seas_{ev}"] = df["month_num"].map(lambda m: SEASONAL_EVENTS.get(m, {}).get(ev, 0.0))
    return df


def add_lags_and_momentum(df: pd.DataFrame, cols=None, lags=(1, 2), roll_windows=(3,)) -> pd.DataFrame:
    df = df.copy()
    cols = cols or DRIVER_COLS
    for c in cols:
        if c not in df.columns:
            continue
        for lag in lags:
            df[f"{c}_lag{lag}"] = df[c].shift(lag)
        for w in roll_windows:
            df[f"{c}_rollmean{w}"] = df[c].rolling(w).mean()
            df[f"{c}_rollstd{w}"] = df[c].rolling(w).std()
        df[f"{c}_mom1"] = df[c].diff(1)
        df[f"{c}_roc1"] = df[c].pct_change(1)
    return df


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    def g(name):
        return df[name] if name in df.columns else 0.0

    df["int_oil_congestion"] = g("brent_usd_bbl") * g("congestion_index")
    df["int_warrisk_panama"] = g("war_risk_premium_idx") * g("panama_restriction_idx")
    df["int_tradevol_idle"] = g("trade_volume_growth_idx") * g("vessel_idle_capacity_pct")
    df["int_usd_oil"] = g("usd_index") * g("brent_usd_bbl")
    df["int_geopolitics_panama"] = g("gpr_index") * g("panama_restriction_idx")
    df["int_gpr_congestion"] = g("gpr_index") * g("congestion_index")
    # China supply-side interactions (v2.1): PMI softening combined with
    # rising congestion signals capacity being absorbed rather than cleared;
    # falling export throughput alongside idle vessel capacity signals a
    # genuine demand-side slowdown rather than a port-side bottleneck.
    df["int_chinapmi_congestion"] = g("china_pmi_idx") * g("congestion_index")
    df["int_chinaexport_idle"] = g("china_export_container_idx") * g("vessel_idle_capacity_pct")
    return df


def normalize_month(df: pd.DataFrame, date_col: str = "month") -> pd.DataFrame:
    """Normalize month column to YYYY-MM so macro/IOFI/lane tables join cleanly
    regardless of whether the source used YYYY-MM or YYYY-MM-DD."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m")
    return df


def build_feature_matrix(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline: seasonality + lags/momentum + interactions."""
    df = normalize_month(macro_df)
    df = add_seasonality(df)
    df = add_lags_and_momentum(df)
    df = add_interactions(df)
    return df


if __name__ == "__main__":
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    macro = pd.read_csv(_BASE_DIR + "/data/macro_history.csv")
    feats = build_feature_matrix(macro)
    feats.to_csv(_BASE_DIR + "/outputs/feature_matrix.csv", index=False)
    print(feats.shape)
