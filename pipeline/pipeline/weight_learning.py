"""
Learn driver weights from historical data instead of fixing them by hand.
Fits several estimators on IOFI (target) vs macro drivers (features) and
reconciles them into a single "learned weight" per driver, plus SHAP-based
local attributions for explainability.

v2.1 change (XGBoost justification): with ~20-24 monthly observations and
dozens of engineered features, a boosted tree ensemble is the model most
prone to overfitting in this lineup -- linear/Bayesian shrinkage models
degrade more gracefully at this sample size. XGBoost is kept (its
tree-based structure is the only one of the four that can capture the
nonlinear interaction terms -- Oil x Congestion etc. -- without them being
hand-specified as products), but it is now:
  1. run with intentionally conservative hyperparameters (already shallow,
     subsampled, L2-penalized -- unchanged from v2), and
  2. down-weighted in the ensemble importance average when the sample size
     is small, via a shrinkage factor that approaches 0 below ~15 rows and
     approaches 1 above ~60 rows. The unshrunk XGBoost importances are still
     reported separately so nothing is hidden, and `xgb_reliability` in the
     returned dict states the shrinkage factor actually applied and why.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNetCV, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# Below this many training rows, XGBoost's ensemble vote is shrunk toward 0;
# above this many rows, it counts fully. Linear interpolation in between.
XGB_MIN_RELIABLE_N = 15
XGB_FULL_RELIABLE_N = 60


def xgb_reliability_weight(n_obs: int) -> float:
    if n_obs <= XGB_MIN_RELIABLE_N:
        return 0.15  # never fully zeroed -- it still adds nonlinear signal -- but heavily discounted
    if n_obs >= XGB_FULL_RELIABLE_N:
        return 1.0
    span = XGB_FULL_RELIABLE_N - XGB_MIN_RELIABLE_N
    return 0.15 + 0.85 * (n_obs - XGB_MIN_RELIABLE_N) / span


def fit_all_weight_models(X: pd.DataFrame, y: pd.Series, n_splits: int = 3):
    """
    Fits Ridge, ElasticNet (CV), Bayesian Ridge and XGBoost on the same
    standardized feature set. Returns fitted models + a unified importance
    table (normalized to sum to 1 per model, plus a sample-size-aware
    ensemble average).
    """
    n_obs = len(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_splits = max(2, min(n_splits, len(X) - 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    ridge = Ridge(alpha=1.0)
    ridge.fit(Xs, y)

    enet = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1], cv=tscv, max_iter=20000)
    enet.fit(Xs, y)

    bayes = BayesianRidge()
    bayes.fit(Xs, y)

    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0
    )
    xgb_model.fit(Xs, y)

    def norm_abs(coefs):
        a = np.abs(coefs)
        s = a.sum()
        return a / s if s > 0 else a

    xgb_weight = xgb_reliability_weight(n_obs)

    importance = pd.DataFrame({
        "feature": X.columns,
        "ridge_weight": norm_abs(ridge.coef_),
        "elasticnet_weight": norm_abs(enet.coef_),
        "bayesian_weight": norm_abs(bayes.coef_),
        "xgb_importance_raw": xgb_model.feature_importances_,
    })
    # Sample-size-aware ensemble: linear models always get full weight (1.0
    # each); XGBoost's contribution is scaled by xgb_weight and the whole
    # row renormalized, so with few observations the ensemble mean is
    # dominated by the three regularized linear models.
    weighted_sum = (
        importance["ridge_weight"] + importance["elasticnet_weight"] +
        importance["bayesian_weight"] + xgb_weight * importance["xgb_importance_raw"]
    )
    denom = 3 + xgb_weight
    importance["ensemble_weight"] = weighted_sum / denom
    importance = importance.sort_values("ensemble_weight", ascending=False).reset_index(drop=True)

    shap_values = None
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(Xs)
        except Exception:
            shap_values = None

    xgb_reliability = {
        "n_training_obs": int(n_obs),
        "shrinkage_weight_applied": round(float(xgb_weight), 3),
        "min_reliable_n": XGB_MIN_RELIABLE_N,
        "full_reliable_n": XGB_FULL_RELIABLE_N,
        "rationale": (
            "XGBoost is the only estimator in the ensemble that can capture "
            "nonlinear driver interactions without those interactions being "
            "hand-specified, so it is retained rather than dropped -- but at "
            f"n={n_obs} training rows its ensemble vote is scaled by "
            f"{round(float(xgb_weight), 2)}x (full weight requires n>={XGB_FULL_RELIABLE_N}) "
            "so a handful of boosted-tree splits cannot dominate a ranking "
            "the linear/Bayesian models don't support. Raw (unshrunk) XGBoost "
            "importances are kept in xgb_importance_raw for transparency."
        ),
    }

    models = {
        "scaler": scaler,
        "ridge": ridge,
        "elasticnet": enet,
        "bayesian": bayes,
        "xgb": xgb_model,
        "xgb_reliability": xgb_reliability,
        "shap_values": shap_values,
        "feature_names": list(X.columns),
    }
    return models, importance


def shap_top5_for_row(models, X_row: pd.DataFrame):
    """Return top-5 SHAP-driven contributors for a single observation (row)."""
    if not HAS_SHAP:
        return None
    scaler = models["scaler"]
    xgb_model = models["xgb"]
    Xs = scaler.transform(X_row)
    explainer = shap.TreeExplainer(xgb_model)
    sv = explainer.shap_values(Xs)[0]
    contrib = pd.DataFrame({"feature": models["feature_names"], "shap_value": sv})
    contrib["abs_shap"] = contrib["shap_value"].abs()
    return contrib.sort_values("abs_shap", ascending=False).head(5).drop(columns="abs_shap")


if __name__ == "__main__":
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from feature_engineering import build_feature_matrix
    macro = pd.read_csv(_BASE_DIR + "/data/macro_history.csv")
    iofi = pd.read_csv(_BASE_DIR + "/data/iofi_history.csv")
    df = build_feature_matrix(macro)
    df = df.merge(iofi, on="month", how="inner")
    df = df.dropna().reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("month", "IOFI", "month_num")]
    X, y = df[feature_cols], df["IOFI"]
    models, importance = fit_all_weight_models(X, y)
    importance.to_csv(_BASE_DIR + "/outputs/learned_driver_weights.csv", index=False)
    print(importance.head(10))
    print(models["xgb_reliability"])
