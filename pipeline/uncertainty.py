"""
Replace hand-written High/Base/Low scenarios with statistically estimated
prediction intervals:
    - Quantile Regression (GradientBoostingRegressor with quantile loss) for
      5th / 25th / 50th / 75th / 95th percentiles
    - Bootstrap ensemble as a cross-check / fallback for very small samples
"""
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]


def quantile_regression_forecast(X: pd.DataFrame, y: pd.Series, X_future: pd.DataFrame, quantiles=QUANTILES):
    preds = {}
    for q in quantiles:
        model = GradientBoostingRegressor(
            loss="quantile", alpha=q, n_estimators=150, max_depth=2, learning_rate=0.05
        )
        model.fit(X, y)
        preds[q] = model.predict(X_future)
    return pd.DataFrame(preds, index=X_future.index)


def bootstrap_ensemble_forecast(X: pd.DataFrame, y: pd.Series, X_future: pd.DataFrame,
                                  base_estimator_fn, n_boot: int = 300, quantiles=QUANTILES):
    """
    base_estimator_fn: callable returning an unfitted sklearn-like estimator.
    Refits on resampled data n_boot times, collects predictions, and reports
    empirical quantiles across the bootstrap distribution.
    """
    all_preds = np.zeros((n_boot, len(X_future)))
    for i in range(n_boot):
        Xb, yb = resample(X, y)
        model = base_estimator_fn()
        model.fit(Xb, yb)
        all_preds[i, :] = model.predict(X_future)
    out = {}
    for q in quantiles:
        out[q] = np.quantile(all_preds, q, axis=0)
    return pd.DataFrame(out, index=X_future.index)


def walk_forward_interval_calibration(X: pd.DataFrame, y: pd.Series, base_estimator_fn,
                                       min_train: int, n_boot: int = 150,
                                       intervals=((0.05, 0.95), (0.25, 0.75))):
    """
    Retrospective calibration check for the bootstrap prediction intervals.

    At each rolling-origin step, builds the SAME bootstrap-ensemble interval
    used for the live forecast (trained only on data available at that
    point) and checks whether the actually-realized next-month value fell
    inside it. Reports empirical coverage for each nominal interval, e.g.
    "the 90% interval (P5-P95) contained the true value in 8/13 backtested
    months (62%)".

    This is run on a small sample by construction (same folds as the main
    backtest), so treat the coverage rate itself as approximate -- its value
    is in flagging whether intervals are badly over- or under-confident, not
    in pinning down an exact calibration percentage.
    """
    n = len(X)
    rows = []
    for t in range(min_train, n):
        X_train, y_train = X.iloc[:t], y.iloc[:t]
        X_test = X.iloc[[t]]
        y_true = y.iloc[t]
        q_df = bootstrap_ensemble_forecast(
            X_train, y_train, X_test, base_estimator_fn, n_boot=n_boot,
            quantiles=sorted({q for pair in intervals for q in pair}),
        )
        row = {"t": t, "y_true": y_true}
        for lo, hi in intervals:
            row[f"lower_{lo}"] = q_df.iloc[0][lo]
            row[f"upper_{hi}"] = q_df.iloc[0][hi]
        rows.append(row)

    res = pd.DataFrame(rows)
    coverage = {}
    for lo, hi in intervals:
        nominal_pct = round((hi - lo) * 100)
        inside = (res["y_true"] >= res[f"lower_{lo}"]) & (res["y_true"] <= res[f"upper_{hi}"])
        n_inside = int(inside.sum())
        n_total = int(len(res))
        coverage[f"P{int(lo*100)}-P{int(hi*100)}"] = {
            "nominal_coverage_%": nominal_pct,
            "observed_coverage_%": round(100 * n_inside / n_total, 1) if n_total else None,
            "n_inside": n_inside,
            "n_total_folds": n_total,
        }
    coverage["_meta"] = {
        "note": (
            f"Calibration estimated over {len(res)} walk-forward folds -- small-sample, "
            "so read the observed coverage rate as directional (badly miscalibrated vs. "
            "roughly on-target), not as a precise percentage."
        )
    }
    return res, coverage


if __name__ == "__main__":
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from feature_engineering import build_feature_matrix
    from sklearn.linear_model import Ridge

    macro = pd.read_csv(_BASE_DIR + "/data/macro_history.csv")
    iofi = pd.read_csv(_BASE_DIR + "/data/iofi_history.csv")
    df = build_feature_matrix(macro).merge(iofi, on="month").dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("month", "IOFI", "month_num")]
    X, y = df[feature_cols], df["IOFI"]
    X_future = X.tail(1)

    qr = quantile_regression_forecast(X, y, X_future)
    print("Quantile regression forecast:\n", qr)

    boot = bootstrap_ensemble_forecast(X, y, X_future, lambda: Ridge(alpha=1.0), n_boot=200)
    print("Bootstrap ensemble forecast:\n", boot)
