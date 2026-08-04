"""
Forecast each macro driver independently instead of hand-decaying scenarios.
Uses ARIMA and ETS from statsmodels, plus a Kalman-filter local-level model
(via statsmodels UnobservedComponents, which is also the Bayesian Structural
Time Series analogue used here given the short history).
Automatically selects the best-fitting model per driver by AIC / in-sample
rolling error, so no manual decay curve is required.
"""
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.structural import UnobservedComponents

warnings.filterwarnings("ignore")


def _fit_arima(series):
    best_aic, best_fit, best_order = np.inf, None, None
    for p in range(3):
        for d in range(2):
            for q in range(3):
                try:
                    fit = ARIMA(series, order=(p, d, q)).fit()
                    if fit.aic < best_aic:
                        best_aic, best_fit, best_order = fit.aic, fit, (p, d, q)
                except Exception:
                    continue
    return best_fit, best_order, best_aic


def _fit_ets(series):
    try:
        fit = ExponentialSmoothing(series, trend="add", damped_trend=True).fit()
        return fit, fit.aic
    except Exception:
        return None, np.inf


def _fit_kalman_local_level(series):
    """Local level + trend model fit via Kalman filter (state-space UC model).
    This serves as the lightweight Bayesian Structural Time Series analogue."""
    try:
        fit = UnobservedComponents(series, level="local linear trend").fit(disp=False)
        return fit, fit.aic
    except Exception:
        return None, np.inf


def forecast_driver(series: pd.Series, horizon: int = 3):
    """
    Fits ARIMA, ETS, and Kalman/BSTS models to a single driver series,
    picks the lowest-AIC model, and returns point + naive interval forecast.
    """
    series = series.astype(float).dropna()
    candidates = {}

    arima_fit, arima_order, arima_aic = _fit_arima(series)
    if arima_fit is not None:
        candidates["ARIMA" + str(arima_order)] = (arima_fit, arima_aic)

    ets_fit, ets_aic = _fit_ets(series)
    if ets_fit is not None:
        candidates["ETS"] = (ets_fit, ets_aic)

    kalman_fit, kalman_aic = _fit_kalman_local_level(series)
    if kalman_fit is not None:
        candidates["Kalman/BSTS"] = (kalman_fit, kalman_aic)

    if not candidates:
        last = series.iloc[-1]
        return {
            "model": "naive_persistence", "forecast": [last] * horizon,
            "lower": [last] * horizon, "upper": [last] * horizon,
        }

    best_name = min(candidates, key=lambda k: candidates[k][1])
    best_fit, _ = candidates[best_name]

    if best_name.startswith("ARIMA"):
        fc = best_fit.get_forecast(steps=horizon)
        mean = fc.predicted_mean.values
        ci = fc.conf_int(alpha=0.2)
        lower, upper = ci.iloc[:, 0].values, ci.iloc[:, 1].values
    elif best_name == "ETS":
        mean = best_fit.forecast(horizon).values
        resid_std = np.std(best_fit.resid)
        lower, upper = mean - 1.28 * resid_std, mean + 1.28 * resid_std
    else:  # Kalman / BSTS
        fc = best_fit.get_forecast(steps=horizon)
        mean = fc.predicted_mean.values
        ci = fc.conf_int(alpha=0.2)
        lower, upper = ci.iloc[:, 0].values, ci.iloc[:, 1].values

    return {
        "model": best_name,
        "aic_scores": {k: round(v[1], 2) for k, v in candidates.items()},
        "forecast": mean.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
    }


def forecast_all_drivers(macro_df: pd.DataFrame, driver_cols, horizon: int = 3):
    results = {}
    for col in driver_cols:
        if col not in macro_df.columns:
            continue
        results[col] = forecast_driver(macro_df[col], horizon=horizon)
    return results


if __name__ == "__main__":
    from feature_engineering import DRIVER_COLS
    import os
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    macro = pd.read_csv(_BASE_DIR + "/data/macro_history.csv")
    out = forecast_all_drivers(macro, DRIVER_COLS, horizon=3)
    for k, v in out.items():
        print(k, "->", v["model"], v["forecast"])
