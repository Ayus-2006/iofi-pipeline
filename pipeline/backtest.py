"""
Rolling-origin backtesting: at each origin point, train on all data up to t,
forecast t+1, and score against the true realized value. Benchmarks the
learned model against naive persistence and a moving average.

v2.1 change (data-honesty fix): with ~20-24 monthly observations, a point
MAPE estimate is fragile and easy to over-read. This module now reports,
alongside the point metrics:
  - n_train_initial / n_folds / n_test_total, so a reader can see exactly
    how much data backed the number
  - a fold-resampling bootstrap 95% CI around MAPE for every benchmark, so
    "4.6% MAPE" is reported as "4.6% [x, y]" rather than a bare point value
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))


def directional_accuracy(y_true, y_pred, y_prev):
    true_dir = np.sign(np.array(y_true) - np.array(y_prev))
    pred_dir = np.sign(np.array(y_pred) - np.array(y_prev))
    return np.mean(true_dir == pred_dir) * 100


def hit_rate(y_true, lower, upper):
    y_true = np.array(y_true)
    inside = (y_true >= np.array(lower)) & (y_true <= np.array(upper))
    return np.mean(inside) * 100


def mape_bootstrap_ci(y_true, y_pred, n_boot: int = 2000, ci: float = 0.95, seed: int = 42):
    """
    Fold-resampling bootstrap CI for MAPE. With very few backtest folds
    (typically <15 here), this CI is itself imprecise - that imprecision is
    the point: it makes clear how much uncertainty surrounds the headline
    number instead of hiding it behind a single decimal.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"lower": None, "upper": None, "n_folds": 0}
    rng = np.random.default_rng(seed)
    ape = np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, np.nan)) * 100
    ape = ape[~np.isnan(ape)]
    if len(ape) == 0:
        return {"lower": None, "upper": None, "n_folds": n}
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(ape, size=len(ape), replace=True)
        boot_means[b] = sample.mean()
    alpha = (1 - ci) / 2
    lower = float(np.quantile(boot_means, alpha))
    upper = float(np.quantile(boot_means, 1 - alpha))
    return {"lower": round(lower, 2), "upper": round(upper, 2), "n_folds": int(len(ape)), "ci_level": ci}


def rolling_origin_backtest(X: pd.DataFrame, y: pd.Series, min_train: int = 10, model_fn=None):
    """
    Walk-forward validation. Returns per-fold predictions plus aggregate
    metrics (each with a bootstrap CI) for: learned model, naive
    persistence, 3-month moving average.
    """
    model_fn = model_fn or (lambda: Ridge(alpha=1.0))
    n = len(X)
    records = []
    for t in range(min_train, n):
        X_train, y_train = X.iloc[:t], y.iloc[:t]
        X_test = X.iloc[[t]]
        y_test = y.iloc[t]

        model = model_fn()
        model.fit(X_train, y_train)
        pred_model = model.predict(X_test)[0]

        pred_naive = y.iloc[t - 1]
        pred_ma = y.iloc[max(0, t - 3):t].mean()

        records.append({
            "t": t, "n_train_obs": t, "y_true": y_test, "y_prev": y.iloc[t - 1],
            "pred_model": pred_model, "pred_naive": pred_naive, "pred_ma": pred_ma,
        })

    res = pd.DataFrame(records)
    n_folds = len(res)
    metrics = {}
    for name, col in [("learned_model", "pred_model"), ("naive_persistence", "pred_naive"), ("moving_avg_3", "pred_ma")]:
        ci = mape_bootstrap_ci(res["y_true"], res[col])
        metrics[name] = {
            "MAPE": round(mape(res["y_true"], res[col]), 3),
            "MAPE_CI95_lower": ci["lower"],
            "MAPE_CI95_upper": ci["upper"],
            "MAE": round(mae(res["y_true"], res[col]), 3),
            "RMSE": round(rmse(res["y_true"], res[col]), 3),
            "DirectionalAccuracy_%": round(directional_accuracy(res["y_true"], res[col], res["y_prev"]), 1),
        }

    metrics["_meta"] = {
        "n_total_obs": int(n),
        "n_train_initial": int(min_train),
        "n_folds": int(n_folds),
        "note": (
            f"Backtest trained on an expanding window starting at {min_train} months and "
            f"evaluated walk-forward over the remaining {n_folds} months. With this few folds, "
            "treat point MAPE/RMSE as directional, not precise -- see MAPE_CI95 for the "
            "bootstrap-estimated range around each headline number."
        ),
    }
    return res, metrics


if __name__ == "__main__":
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from feature_engineering import build_feature_matrix

    macro = pd.read_csv(_BASE_DIR + "/data/macro_history.csv")
    iofi = pd.read_csv(_BASE_DIR + "/data/iofi_history.csv")
    df = build_feature_matrix(macro).merge(iofi, on="month").dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("month", "IOFI", "month_num")]
    X, y = df[feature_cols], df["IOFI"]

    res, metrics = rolling_origin_backtest(X, y, min_train=max(6, len(X) - 6))
    print(pd.DataFrame({k: v for k, v in metrics.items() if k != "_meta"}).T)
    print(metrics["_meta"])
    res.to_csv(_BASE_DIR + "/outputs/backtest_results.csv", index=False)
