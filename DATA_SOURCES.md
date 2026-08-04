# Data sources — what's real, what's a proxy

This is the honest accounting of where each of the 13 macro drivers comes
from once `pipeline/data_sources.py` is wired in. Read this before trusting
a specific number, especially anything in the "synthetic proxy" section.

## Real, free, no-signup feeds (6 of 13 currently live; 2 disabled)

| Driver | Source | Native history | Update cadence |
|---|---|---|---|
| `brent_usd_bbl` | FRED series `DCOILBRENTEU` | back to 1987 | daily |
| `usd_index` | FRED series `DTWEXBGS` | back to 2006 | daily |
| `inr_usd_rate` | FRED series `DEXINUS` | back to 1973 | daily |

**FRED fetching:** if a `FRED_API_KEY` environment variable / CI secret is
set, `fetch_fred_series()` uses FRED's official `api.stlouisfed.org`
observations API, which is built for automated polling. If it's unset (or
the API call fails for some other reason), it falls back to the
unauthenticated `fredgraph.csv` chart-rendering endpoint, which is more prone
to timeouts under automated/CI traffic. Get a free key at
https://fred.stlouisfed.org/docs/api/api_key.html and set it as a repo
secret (`FRED_API_KEY`) for the GitHub Actions workflow to pick up.
| `gpr_index` | Caldara & Iacoviello Geopolitical Risk Index ([matteoiacoviello.com/gpr.htm](https://www.matteoiacoviello.com/gpr.htm)), CC-BY | back to 1985 (benchmark), 1900 (historical) | monthly, ~10th of month |
| `trade_volume_growth_idx` | CPB World Trade Monitor. **Changed:** the underlying .xlsx download's filename isn't predictable, so this now scrapes the CPB report page for the reporting month (URL pattern is predictable: `cpb.nl/en/world-trade-monitor/cpb-world-trade-monitor-{month}-{year}`) and parses the stated **month-on-month** growth %, not the YoY figure the old code computed from the (unreachable) file. | current month only, scraped live | monthly |
| `congestion_index` | IMF PortWatch `Daily_Chokepoints_Data` feature service (`n_total` field, averaged over Suez/Hormuz/Bab-el-Mandeb) | ~2019 onward | weekly |
| `china_pmi_idx` | OECD SDMX Composite Leading Indicator for China, used as a PMI **proxy** (OECD does not publish China's actual PMI print) | back to ~2000 | monthly |

**Disabled direct feed — no verified free source exists, now regression-proxied instead:**
- `eu_ets_carbon_idx` — the original `api.ember-climate.org/v1/carbon-price` endpoint never existed; Ember's real public API (ember-climate.org/data/api) covers electricity data only, not carbon price. No free, no-key, numeric EU ETS price API is currently confirmed anywhere. The direct fetcher fails fast with a clear message instead of hitting a fabricated URL.
- `china_export_container_idx` — the original `OECD.SDD.TPS,DSD_TRADE@DF_TRADE` dataflow doesn't exist (hence the 404). No verified free OECD/IMF dataflow with a stable dimension key for this series was found. Same fail-fast treatment.

Rather than freeze either at whatever value happened to be in the seed CSV forever, `get_latest_month_row()` now falls back to a small Ridge regression (`REGRESSION_PROXY_SPECS` in `data_sources.py`) fitted on historical co-movement against the real drivers above, applied using whichever of those predictors actually refreshed live this run:

| Regression-proxied driver | Predictors |
|---|---|
| `eu_ets_carbon_idx` | `brent_usd_bbl`, `usd_index`, `china_pmi_idx` |
| `china_export_container_idx` | `china_pmi_idx`, `trade_volume_growth_idx`, `congestion_index` |

This only fires when the direct fetch fails and at least one predictor is fresh this run; a fully offline run (no live data at all) still just carries the last known value forward, same as before. It's still an estimate, not a measurement — swap in a real (possibly paid) feed, or a manual monthly entry via `record_observed_month.py`, whenever one becomes available; the regression path then simply stops firing since the direct fetch will succeed first.

## Synthetic proxies — no free numeric feed exists (5 of 13)

These are normally sold as commercial data products (Drewry, Alphaliner,
Freightos, the Panama Canal Authority's paid transit-restriction feed).
Since you chose full automation over manual monthly entry, the pipeline
derives stand-ins from the real series above so it can run unattended. They
move in economically sensible directions but are **not measurements** —
treat any driver-importance ranking involving these five with real
skepticism, and swap in a paid feed the moment one is available (the rest
of the pipeline does not need to change, only `data_sources.py`).

| Driver | How it's derived | Confidence |
|---|---|---|
| `war_risk_premium_idx` | Mean of Caldara/Iacoviello **country-specific** GPR indices for Iran, Yemen, Egypt, Israel, Saudi Arabia — i.e. the countries bordering this book's chokepoints (Hormuz, Bab-el-Mandeb, Suez). Real underlying data, reasonable proxy logic. | Medium |
| `vessel_idle_capacity_pct` | Heuristic: idle capacity assumed higher when congestion is low *and* trade-volume growth is weak, z-scored and rescaled to a plausible 3–10% band. | Low |
| `panama_restriction_idx` | Inverse, smoothed function of global trade-volume momentum, rescaled near the original series' historical band. No real signal about the Panama Canal specifically. | Lowest of the five |
| `fbx_composite_legacy_usd_feu` | Composite z-score of oil price + congestion + trade volume, rescaled near the legacy series' historical level. Approximates *direction*, not actual quoted freight rates. | Low |

## What still can't be automated at all

`iofi_history.csv` (the IOFI target itself) and the per-lane
`current_rate_usd_feu` figures in `predictions_full.csv` are **your own
proprietary data** — the thing this tool predicts, not something a public
feed can supply. Run `python3 pipeline/record_observed_month.py --month
YYYY-MM --iofi <value>` once a month with your own observed number(s) to
keep the training target itself current. Skipping this doesn't break
anything — the macro drivers still update automatically — it just means
the model keeps training against the same target history until you do.

## Maintenance reality

Every fetcher hits a real external site whose URL, schema, or auth
requirements can change without notice (this already happened once during
build — GPR's file is versioned monthly under a slightly different name
each time). `data_sources.py` logs a clear warning and falls back to
carrying the last known value forward whenever a fetch fails, so a broken
upstream never crashes a monthly run — but check the console output
occasionally, since a "carried forward" driver isn't actually current.
