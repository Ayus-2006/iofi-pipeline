# IOFI v2 — Data-Driven Architecture

## 1. Architecture comparison & recommendation

| Model | Fit for monthly freight w/ ~24-36 months history |
|---|---|
| XGBoost / LightGBM / CatBoost | **Best fit.** Handle nonlinear interactions, tabular macro features, and small-N well with regularization. |
| Random Forest | Good baseline, weaker on extrapolation beyond training range (common in freight shocks). |
| ARIMA / ETS | Best for **per-driver** univariate forecasting (oil, congestion, etc.), not for the multi-driver rate model itself. |
| Kalman Filter / BSTS | Good for local-level/trend extraction with tiny samples; used as driver-forecast component. |
| LSTM / TFT / N-BEATS | **Not recommended now.** These need hundreds-to-thousands of sequences; with ~24-36 monthly points they will overfit and give unstable, non-explainable output. Revisit once 3-5 years of weekly/monthly data accumulate. |
| Prophet | Reasonable for seasonality-heavy single series, used as a driver-forecast alternative. |

**Recommendation: a Hybrid ML + Time-Series architecture.**
- Per-driver forecasting: ARIMA / ETS / Kalman (auto-selected by AIC) → produces forward paths for each macro driver instead of hand-decayed premiums.
- Driver-weight learning: Ridge + ElasticNet + Bayesian Ridge + XGBoost, ensembled, with SHAP for explainability — robust to a small sample, avoids the instability of deep nets.
- Lane layer: KMeans clustering + per-cluster Ridge sensitivity (rather than one global model or 60 independent tiny models).
- Uncertainty: Quantile Gradient Boosting + bootstrap ensemble in place of manual High/Base/Low.

This hybrid is the standard choice in freight-rate and commodity forecasting when history is short: time-series models supply well-behaved driver paths, tree/linear ensembles supply the nonlinear mapping from drivers to rates, and classical uncertainty methods (quantile regression, bootstrap) avoid the false precision of hand scenarios.

## 2. Mathematical formulation

Target IOFI (or lane rate) at time *t*:

```
IOFI_t = f( D_t, D_{t-1}, D_{t-2}, RollStat(D, w), Seasonal_t, Interactions_t ) + ε_t
```

where `D_t` is the driver vector (oil, war risk, GPR, congestion, idle capacity, trade volume, USD index, INR/USD, Panama restriction, EU ETS carbon), `RollStat` are rolling mean/std/momentum features, `Seasonal_t` are calendar/event encodings, and `Interactions_t` are pairwise products (Oil×Congestion, WarRisk×Panama, etc.).

`f(·)` is estimated 4 ways and ensembled:
```
f_ridge, f_enet, f_bayes  : linear in transformed features, L2/L1 penalized
f_xgb                     : additive tree ensemble, captures nonlinearities/interactions natively
w_i = normalized |coefficient| or gain-based importance
ensemble_weight_i = mean(w_i across the 4 estimators)
```

Per-driver forecast:
```
D_i(t+h) = argmin_model AIC(ARIMA(p,d,q)) vs AIC(ETS) vs AIC(Kalman-local-linear-trend)
```

Lane sensitivity (per cluster *c*):
```
Δrate_{lane∈c,t} = β0_c + β1_c·IOFI_t + β2_c·WarRisk_t + β3_c·Oil_t + β4_c·Congestion_t + ... + η_{lane,t}
```
fit via Ridge per cluster, pooling clusters with <5 lanes.

Uncertainty (quantile regression):
```
Q_τ(IOFI_{t+h} | X_t) = GradientBoosting_quantile-loss(τ), τ ∈ {0.05,0.25,0.5,0.75,0.95}
```

## 3. Data pipeline
`macro_history.csv` + `iofi_history.csv` + `predictions_full.csv` → normalize month keys → merge → engineer features → split by rolling origin. Designed so a live monthly data pull can replace the CSV read in `pipeline_runner.load_data()` without touching downstream code.

## 4. Feature engineering strategy
See `pipeline/feature_engineering.py`: seasonality (month sin/cos + named events), lags (1-2 periods, tuned for small N), rolling mean/std, momentum, rate-of-change, and 6 nonlinear interaction terms.

## 5. Training pipeline
`pipeline/weight_learning.py` (driver weights) + `pipeline/driver_forecasting.py` (per-driver forecasts) + `pipeline/lane_clustering.py` + `pipeline/lane_sensitivity.py` (lane layer) + `pipeline/uncertainty.py` (quantile/bootstrap bands).

## 6. Evaluation pipeline
`pipeline/backtest.py`: rolling-origin validation, MAPE/MAE/RMSE/Directional Accuracy/Hit Rate, benchmarked against naive persistence and moving average.

## 7. Explainability
`pipeline/explainability.py`: top-5 SHAP contributors, contribution table, natural-language narrative, heuristic confidence score.

## 8. Self-updating pipeline
`pipeline/pipeline_runner.py` runs steps 1-7 end to end every month, saves `outputs/best_model.pkl` and `outputs/run_manifest.json` with no manual tuning.

## 9. Implementation roadmap
1. **Now**: run this hybrid pipeline monthly on existing ~24 months of data; treat outputs as directionally informative, not high-precision, given sample size.
2. **3-6 months**: accumulate weekly driver data + more lane snapshots (panel data) to properly fit per-lane time-varying betas (currently cross-sectional proxy).
3. **6-12 months**: once ≥100 monthly observations or weekly granularity exist, evaluate Temporal Fusion Transformer / N-BEATS as a challenger model in the backtest harness — promote only if it beats the hybrid on rolling-origin MAPE/coverage.
4. **Ongoing**: expand SHAP-based governance reporting; add automated data-quality checks before each monthly retrain.

## Known limitation of this build
The uploaded data has ~24-36 monthly macro/IOFI observations and a single lane-level snapshot (not a lane panel over time). Per-driver ARIMA/ETS/Kalman and driver-weight learning use the full time series and are genuinely data-driven. The lane-sensitivity regressions, however, are fit cross-sectionally (one snapshot broadcast against current macro state) because no historical lane-level time series was provided — this is flagged in code comments (`lane_sensitivity.py`) and should be upgraded to a true panel model once monthly lane snapshots accumulate.
