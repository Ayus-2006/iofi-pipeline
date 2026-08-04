"""
Hierarchical rate decomposition: Global index -> Regional index -> Lane.

v1 and v2.0 both went straight from the global IOFI move to a per-lane
multiplier in one step. That collapses two genuinely different sources of
variation into a single number: how much a whole trade corridor (say,
"Europe (North)") is moving together, versus how much one specific lane
inside that corridor deviates from its corridor's average. This module
separates them:

    IOFI move (global, %)
        -> Regional move  = median of that region's lanes' own historical
                             beta, applied to the IOFI move
        -> Lane premium    = (lane's own beta - region beta), applied to the
                             IOFI move on top of the regional move

    lane_forecast_move = region_move + lane_premium_move
                        = region_beta * idx_move + (lane_beta - region_beta) * idx_move
                        = lane_beta * idx_move   [identical total to the flat model]

The total per-lane number is mathematically the same as the flat per-lane
beta approach (so nothing about the point forecast changes) -- what changes
is that the report can now show, and a planner can now reason about, how
much of a given lane's move is "this corridor is repricing" versus "this
specific lane is unusual for its corridor." That decomposition is the
literal structure requested for a future panel-data model: once monthly
lane-level snapshots accumulate, the SAME regional grouping used here can be
re-estimated as genuine fixed/random-effects regional coefficients instead
of a median-of-betas proxy -- only the estimation method for each level
changes, not the hierarchy itself.

Still bound by the same single-snapshot limitation as v2.0 lane betas (see
ARCHITECTURE.md / Methodology & Limitations): the regional level here is a
robust summary statistic (median lane beta per region) of the one snapshot
we have, not a regression fit on regional time series -- that requires
monthly panel data that does not yet exist.
"""
import numpy as np
import pandas as pd
from feature_engineering import SEASONAL_EVENTS, REGION_SEASONAL_SENSITIVITY

# Caps how much of a lane's forecast move the festival calendar alone can
# explain, applied to a region's fully-active-event score of 1.0. Kept
# deliberately modest: this is a transparent, capped seasonal adjustment
# layered on top of the learned model, not a replacement for it -- the
# learned model's own seasonality features (seas_* columns) already do the
# primary work of fitting historical seasonal moves; this adjustment only
# nudges lane-level forecasts by region-specific relevance in the 3-month
# forecast window, which the flat/global seasonality features can't express.
MAX_SEASONAL_PCT = 2.5


def regional_seasonal_score(region: str, month_str: str) -> float:
    """Sum of (region sensitivity x event intensity) over festivals active
    in the given calendar month, for the given destination region. Not yet
    scaled to a percentage -- see regional_seasonal_pct."""
    month_num = pd.Period(month_str, freq="M").month
    events = SEASONAL_EVENTS.get(month_num, {})
    sensitivities = REGION_SEASONAL_SENSITIVITY.get(region, {})
    score = sum(sensitivities.get(ev, 0.0) * intensity for ev, intensity in events.items())
    return float(np.clip(score, 0.0, 1.5))


def regional_seasonal_pct(region: str, month_str: str, max_pct: float = MAX_SEASONAL_PCT) -> float:
    """Region-specific seasonal rate premium for one forecast month, as a
    percent adjustment to that lane's rate (demand-pull events push rates
    up; this module only models the demand-pull direction, not
    capacity-driven softening)."""
    return regional_seasonal_score(region, month_str) * max_pct


def compute_hierarchical_betas(lanes: pd.DataFrame, region_col: str = "region",
                                lane_beta_col: str = "beta_lane") -> pd.DataFrame:
    """
    Given a lanes dataframe that already has a per-lane beta_lane column,
    adds:
      - region_beta: median lane beta within that lane's region (robust to
        single-lane outliers, unlike a mean)
      - lane_premium_beta: lane_beta - region_beta (the lane's own deviation
        from its corridor)
    """
    out = lanes.copy()
    region_betas = out.groupby(region_col)[lane_beta_col].median().rename("region_beta")
    out = out.merge(region_betas, on=region_col, how="left")
    out["lane_premium_beta"] = out[lane_beta_col] - out["region_beta"]
    return out


def hierarchical_region_summary(lanes_with_betas: pd.DataFrame, region_col: str = "region") -> pd.DataFrame:
    """Region-level summary table for the report: how many lanes, region beta,
    and the spread of lane premiums within the region (i.e. how much
    within-corridor heterogeneity the flat model was masking)."""
    g = lanes_with_betas.groupby(region_col)
    summary = g.agg(
        n_lanes=("lane_premium_beta", "size"),
        region_beta=("region_beta", "first"),
        premium_min=("lane_premium_beta", "min"),
        premium_median=("lane_premium_beta", "median"),
        premium_max=("lane_premium_beta", "max"),
    ).reset_index()
    return summary.sort_values("region_beta", ascending=False).reset_index(drop=True)
