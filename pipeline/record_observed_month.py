"""
Log this month's *observed* IOFI value and lane rates.

Macro drivers (oil, GPR, USD index, etc.) can be pulled from free public
feeds automatically -- that's what pipeline_runner.py now does every run.
IOFI itself is your own index (the thing this whole tool predicts), and the
per-lane current_rate_usd_feu figures come from your own carrier/rate data,
so nothing on the public internet can supply them. This script is the one
manual step left each month: it appends whatever you enter to
data/iofi_history.csv (and, optionally, data/lane_rate_history.csv), so the
model always trains against real, growing history rather than a snapshot
that never moves.

Usage:
    python3 record_observed_month.py --month 2026-07 --iofi 104.2
    python3 record_observed_month.py --month 2026-07 --iofi 104.2 \
        --lanes lanes_this_month.csv   # optional: origin,destination,current_rate_usd_feu
"""
import argparse
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def record_iofi(month: str, value: float):
    path = f"{DATA_DIR}/iofi_history.csv"
    df = pd.read_csv(path)
    if (df["month"] == month).any():
        df.loc[df["month"] == month, "IOFI"] = value
        print(f"Updated existing row for {month}.")
    else:
        df = pd.concat([df, pd.DataFrame([{"month": month, "IOFI": value}])], ignore_index=True)
        print(f"Appended new row for {month}.")
    df = df.sort_values("month").reset_index(drop=True)
    df.to_csv(path, index=False)
    print(f"data/iofi_history.csv now has {len(df)} monthly observations.")


def record_lane_rates(month: str, lanes_csv: str):
    """Appends a dated snapshot to data/lane_rate_history.csv (created on
    first use). This turns the single cross-sectional lane snapshot the v2
    package started with into a real panel over time, which is exactly what
    ARCHITECTURE.md flags as needed to upgrade lane_sensitivity.py from a
    cross-sectional proxy to a true time-varying panel model."""
    new_rows = pd.read_csv(lanes_csv)
    new_rows["month"] = month
    hist_path = f"{DATA_DIR}/lane_rate_history.csv"
    if os.path.exists(hist_path):
        hist = pd.read_csv(hist_path)
        hist = hist[hist["month"] != month]  # replace this month if re-run
        hist = pd.concat([hist, new_rows], ignore_index=True)
    else:
        hist = new_rows
    hist.to_csv(hist_path, index=False)
    print(f"data/lane_rate_history.csv now has {hist['month'].nunique()} monthly snapshots.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--month", required=True, help="YYYY-MM, e.g. 2026-07")
    ap.add_argument("--iofi", type=float, required=True, help="This month's observed IOFI value")
    ap.add_argument("--lanes", default=None,
                     help="Optional CSV with columns origin,destination,current_rate_usd_feu "
                          "(and any others) for this month's lane snapshot")
    args = ap.parse_args()

    record_iofi(args.month, args.iofi)
    if args.lanes:
        record_lane_rates(args.month, args.lanes)
