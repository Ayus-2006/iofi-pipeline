# IOFI v2 — Data-Driven Freight Rate Model

Replaces all fixed weights / manual scenario decay / hand-picked lane betas
with learned, statistically estimated equivalents. See `ARCHITECTURE.md` for
full math, pipeline design, and roadmap.

## Run everything
```bash
pip install -r requirements.txt
cd pipeline
python3 pipeline_runner.py
```
Outputs land in `outputs/`: `best_model.pkl`, `run_manifest.json`,
`learned_driver_weights.csv`, `backtest_results.csv`.

## Module map
| File | Requirement covered |
|---|---|
| `pipeline/feature_engineering.py` | Seasonality, lags, momentum, rolling stats, interactions (#5,6,7) |
| `pipeline/weight_learning.py` | Ridge/ElasticNet/Bayesian/XGB/SHAP driver weights (#1) |
| `pipeline/driver_forecasting.py` | ARIMA/ETS/Kalman per-driver forecasts (#2) |
| `pipeline/lane_clustering.py` | KMeans lane clustering (#4) |
| `pipeline/lane_sensitivity.py` | Per-cluster lane betas (#3) |
| `pipeline/uncertainty.py` | Quantile regression + bootstrap intervals (#8) |
| `pipeline/backtest.py` | Rolling-origin validation + benchmarks (#9) |
| `pipeline/pipeline_runner.py` | Self-updating monthly orchestrator (#10) |
| `pipeline/explainability.py` | Top-5, SHAP table, NL explanation, confidence (#11) |
| `ARCHITECTURE.md` | Architecture comparison/recommendation + roadmap (#12) |

## Making this genuinely self-updating

`pipeline_runner.load_data()` now calls `pipeline/data_sources.py`, which
pulls the latest reading for 8 of the 13 macro drivers from free public
feeds (FRED, the Caldara/Iacoviello GPR index, CPB World Trade Monitor,
Ember Climate, IMF PortWatch, OECD) and derives the other 5 as documented
proxies, since no free feed exists for them at all. See `DATA_SOURCES.md`
for exactly which is which — read it before trusting any single driver's
number.

**One-time (do this first):**
```bash
cd pipeline
python3 backfill_history.py
```
Pulls as much free history as each source actually offers (Brent and GPR
both go back decades) and extends `data/macro_history.csv` backwards, so
you don't have to wait years to accumulate a deep history one month at a
time.

**Every month:**
```bash
cd pipeline
python3 record_observed_month.py --month 2026-08 --iofi <your observed value>   # the one thing no public feed can supply
python3 pipeline_runner.py                                                       # live macro fetch + retrain, automatic
python3 build_report_v2.py                                                       # refreshed PDF
```
Run it again in 3 months and it uses whatever new macro data has landed
since, plus whatever you've logged with `record_observed_month.py`, and
produces a genuinely different, better-supported forecast — not the same
static output.

**To not have to run it yourself:** `schedule/monthly_pipeline.yml` is a
GitHub Actions workflow — drop it at `.github/workflows/monthly_pipeline.yml`
in your repo and it runs on the 2nd of every month automatically (GitHub's
runners have normal internet access, unlike some sandboxed dev
environments). It still can't log your own observed IOFI value for you —
that step stays manual, on purpose, since it's your proprietary data.
