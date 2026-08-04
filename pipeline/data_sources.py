"""
Live data sourcing layer for IOFI v2.

This module is what makes the pipeline *actually* self-updating: it pulls the
freshest available reading for each macro driver from a free, public, no-key
source, then hands a clean row to `pipeline_runner.load_data()`. No manual
CSV editing required for these fields.

Sourcing map (see DATA_SOURCES.md for the full writeup):

  REAL, free, no signup:
    brent_usd_bbl            FRED series DCOILBRENTEU
    usd_index                FRED series DTWEXBGS
    inr_usd_rate              FRED series DEXINUS
    gpr_index                 Caldara & Iacoviello GPR index (matteoiacoviello.com)
    trade_volume_growth_idx   CPB World Trade Monitor
    congestion_index           IMF PortWatch port-calls/congestion API
    china_pmi_idx               OECD Composite Leading Indicator, China (PMI proxy)

  SYNTHETIC PROXIES (no free numeric feed exists at all -- built from the
  real series above so the pipeline can still run unattended; documented as
  proxies, not measurements):
    war_risk_premium_idx      mean of GPR country-indices for chokepoint states
    vessel_idle_capacity_pct   derived from congestion + trade-volume momentum
    panama_restriction_idx      derived from trade-volume momentum (placeholder)
    fbx_composite_legacy_usd_feu derived composite of oil/congestion/trade

  REGRESSION PROXIES (no free live feed of any kind was ever confirmed for
  these two -- the original code hit fabricated URLs. Rather than carry the
  seed value forward forever, they are now estimated each run with a small
  Ridge regression fitted on historical co-movement against whichever real
  drivers above came in live *this run* -- see REGRESSION_PROXY_SPECS. Still
  clearly an estimate, not a measurement; swap in a paid feed the moment one
  is available and this whole path becomes dead code automatically (it only
  fires when the direct fetch fails):
    eu_ets_carbon_idx           regressed on brent_usd_bbl, usd_index, china_pmi_idx
    china_export_container_idx  regressed on china_pmi_idx, trade_volume_growth_idx, congestion_index

Every fetch function is wrapped so a single broken upstream endpoint cannot
crash a monthly run: on failure it logs a warning and the caller falls back
to the regression proxy (for the two drivers above) or to carrying the last
known value forward.
"""
import io
import logging
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="[data_sources] %(message)s")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

TIMEOUT = 20
HEADERS = {"User-Agent": "iofi-v2-pipeline/1.0 (research use)"}

# If set (e.g. as a CI secret), FRED_API_KEY routes fetch_fred_series() through
# FRED's official REST API (api.stlouisfed.org/fred/series/observations)
# instead of the unauthenticated fredgraph.csv chart-rendering endpoint. The
# official API is built for automated polling and is far less prone to the
# timeouts that endpoint produces from hosted CI runners; get a free key at
# https://fred.stlouisfed.org/docs/api/api_key.html if this isn't set.
FRED_API_KEY = os.environ.get("FRED_API_KEY", "").strip()

# Browser-like headers: some upstream hosts (FRED in particular) are slower
# or more selective when hit from hosted CI runners than from a laptop; a
# realistic Accept header plus a longer timeout and a couple of retries
# clears this up without weakening the "never crash the run" fallback below.
RETRY_TIMEOUT = 45
RETRY_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,application/json,text/plain,*/*",
}


class FetchWarning(Exception):
    pass


def _get(url: str, *, retries: int = 3, backoff: float = 2.0, **kwargs) -> requests.Response:
    """requests.get with a longer timeout, browser-like headers, and a few
    retries with exponential backoff -- upstream hosts (FRED especially)
    intermittently time out on the first attempt from hosted CI runners."""
    kwargs.setdefault("headers", RETRY_HEADERS)
    kwargs.setdefault("timeout", RETRY_TIMEOUT)
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(url, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_err


def _month_floor(ts):
    return pd.Timestamp(ts).to_period("M").to_timestamp()


# ---------------------------------------------------------------------------
# FRED (Federal Reserve Economic Data) -- stable, free, no API key needed for
# the CSV export endpoint.
# ---------------------------------------------------------------------------
def _fetch_fred_series_api(series_id: str, cosd: str) -> pd.Series:
    """Official FRED REST API -- requires FRED_API_KEY, built for automated
    polling (unlike the graph-rendering endpoint below), and is what the key
    already sitting in CI secrets is for."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    r = _get(
        url,
        params={
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": cosd,
        },
    )
    obs = r.json().get("observations", [])
    df = pd.DataFrame(obs)
    if df.empty:
        raise FetchWarning(f"FRED API returned no observations for {series_id}")
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
    df = df.dropna(subset=["value"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")["value"].mean()
    monthly.name = series_id
    return monthly


def _fetch_fred_series_csv(series_id: str, cosd: str) -> pd.Series:
    """Unauthenticated fallback: the graph-rendering endpoint, used only when
    no FRED_API_KEY is configured. Meant for interactive chart rendering, not
    automated polling -- prone to exactly the timeouts this replaces."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={cosd}"
    r = _get(url)
    df = pd.read_csv(io.StringIO(r.text))
    date_col, val_col = df.columns[0], df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])
    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month")[val_col].mean()
    monthly.name = series_id
    return monthly


def fetch_fred_series(series_id: str, lookback_days: int = 120) -> pd.Series:
    # Only the most recent value is actually used by get_latest_month_row(),
    # so request a small recent window (cosd = cutoff start date) rather
    # than FRED's full multi-decade daily history.
    cosd = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    if FRED_API_KEY:
        try:
            return _fetch_fred_series_api(series_id, cosd)
        except Exception as e:  # noqa: BLE001
            log.warning(
                f"FRED API fetch failed for {series_id} ({e}); "
                "falling back to unauthenticated graph CSV endpoint."
            )
    return _fetch_fred_series_csv(series_id, cosd)


def fetch_brent() -> pd.Series:
    return fetch_fred_series("DCOILBRENTEU")


def fetch_usd_index() -> pd.Series:
    return fetch_fred_series("DTWEXBGS")


def fetch_inr_usd() -> pd.Series:
    return fetch_fred_series("DEXINUS")


# ---------------------------------------------------------------------------
# Geopolitical Risk Index -- Caldara & Iacoviello, updated monthly, free,
# CC-BY licensed. The exact filename is versioned month to month, so we try
# the stable "current" path first and fall back to known mirrors.
# ---------------------------------------------------------------------------
GPR_CANDIDATE_URLS = [
    "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls",
    "https://www.matteoiacoviello.com/gpr_replication_files/data_paper/data_gpr_export.xls",
]

# Country-specific GPR columns used to build the maritime "war risk" proxy --
# countries bordering the main chokepoints this book of lanes actually
# transits (Hormuz, Red Sea/Bab-el-Mandeb, Suez).
WAR_RISK_COUNTRY_COLS = ["GPRC_IRN", "GPRC_YEM", "GPRC_EGY", "GPRC_ISR", "GPRC_SAU"]


def fetch_gpr() -> pd.DataFrame:
    last_err = None
    for url in GPR_CANDIDATE_URLS:
        try:
            r = _get(url)
            df = pd.read_excel(io.BytesIO(r.content))
            date_col = "month" if "month" in df.columns else df.columns[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
            return df.set_index("month")
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise FetchWarning(f"GPR index: all candidate URLs failed ({last_err}). "
                        f"Check https://www.matteoiacoviello.com/gpr.htm for the current link "
                        f"and update GPR_CANDIDATE_URLS in data_sources.py.")


# ---------------------------------------------------------------------------
# CPB World Trade Monitor -- free monthly trade-volume index, no signup.
# ---------------------------------------------------------------------------
import re

# CPB's underlying .xlsx download filename is not predictable, but each
# monthly report has a stable, predictable landing-page URL of the form
# cpb.nl/en/world-trade-monitor/cpb-world-trade-monitor-{month}-{year}
# (confirmed against several published reports). CPB publishes with roughly
# a two-month lag, so we start there and step backward until one resolves.
_CPB_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
               "august", "september", "october", "november", "december"]

_CPB_GROWTH_RE = re.compile(
    r"world (?:merchandise )?trade volume (increased|decreased) (?:by )?"
    r"(-?\d+(?:\.\d+)?)\s*%\s*(?:in|month-on-month)?\s*(?:([A-Za-z]+)\s+(\d{4}))?",
    flags=re.IGNORECASE,
)


def fetch_world_trade_volume() -> pd.Series:
    last_err = None
    start = pd.Timestamp.today().replace(day=1) - pd.DateOffset(months=2)
    for back in range(6):
        report_month = start - pd.DateOffset(months=back)
        month_name = _CPB_MONTHS[report_month.month - 1]
        url = (f"https://www.cpb.nl/en/world-trade-monitor/"
               f"cpb-world-trade-monitor-{month_name}-{report_month.year}")
        try:
            r = _get(url)
            text = re.sub(r"<[^>]+>", " ", r.text)
            text = re.sub(r"\s+", " ", text)
            m = _CPB_GROWTH_RE.search(text)
            if not m:
                raise FetchWarning("growth figure not found on page")
            direction, pct = m.group(1).lower(), float(m.group(2))
            if direction == "decreased" and pct > 0:
                pct = -pct
            month_ts = _month_floor(report_month)
            s = pd.Series({month_ts: pct})
            s.name = "trade_volume_growth_idx"
            return s
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue
    raise FetchWarning(f"CPB World Trade Monitor fetch failed ({last_err}).")


# ---------------------------------------------------------------------------
# EU ETS carbon price -- Ember Climate open API.
# ---------------------------------------------------------------------------
def fetch_eu_ets_price() -> float:
    # NOTE: this endpoint never existed as written -- Ember's actual public
    # API (ember-climate.org/data/api) only serves electricity generation,
    # demand, and emissions data; there is no free, no-key, numeric EU ETS
    # carbon-price API confirmed anywhere (Ember's Carbon Price Tracker page
    # is a chart, not an API). Honestly disabled rather than hitting a
    # fabricated URL. Needs a real source (e.g. a paid EEX/ICE feed, or
    # manual monthly entry) -- see DATA_SOURCES.md. When this raises,
    # get_latest_month_row() falls back to the regression proxy defined in
    # REGRESSION_PROXY_SPECS below rather than freezing the value forever.
    raise FetchWarning(
        "No free, no-key EU ETS carbon-price API is currently confirmed to exist. "
        "Falling back to regression proxy -- see DATA_SOURCES.md for manual sourcing options."
    )


# ---------------------------------------------------------------------------
# IMF PortWatch -- free public port-congestion data (ArcGIS-hosted feature
# service), updated ~weekly. Used as-is; no free direct "vessel idle
# capacity" equivalent exists, so idle capacity is a derived proxy below.
# ---------------------------------------------------------------------------
PORTWATCH_CHOKEPOINTS = ["chokepoint1", "chokepoint6", "chokepoint7"]  # Suez, Hormuz, Bab-el-Mandeb


def fetch_port_congestion() -> float:
    where = " OR ".join(f"portid='{cp}'" for cp in PORTWATCH_CHOKEPOINTS)
    url = ("https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
           "Daily_Chokepoints_Data/FeatureServer/0/query"
           f"?where={requests.utils.quote(where)}&outFields=*&outSR=4326"
           "&orderByFields=date+DESC&resultRecordCount=90&f=json")
    r = _get(url)
    js = r.json()
    feats = js.get("features") or []
    if not feats:
        raise FetchWarning("IMF PortWatch returned no chokepoint transit records.")
    vals = [f["attributes"].get("n_total") for f in feats if f.get("attributes")]
    vals = [v for v in vals if v is not None]
    if not vals:
        raise FetchWarning("IMF PortWatch response schema changed; 'n_total' field not found.")
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# OECD China proxies (Composite Leading Indicator as PMI proxy; goods-export
# value growth as export-container-throughput proxy). Free SDMX API, no key.
# ---------------------------------------------------------------------------
def fetch_oecd_series(dataset: str, dimensions: str) -> pd.Series:
    start_period = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime("%Y-%m")
    url = (f"https://sdmx.oecd.org/public/rest/data/{dataset}/{dimensions}"
           f"?startPeriod={start_period}&dimensionAtObservation=AllDimensions&format=csvfilewithlabels")
    r = _get(url)
    df = pd.read_csv(io.StringIO(r.text))
    time_col = next(c for c in df.columns if "TIME" in c.upper())
    val_col = next(c for c in df.columns if "OBS_VALUE" in c.upper() or "VALUE" == c.upper())
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df["month"] = df[time_col].dt.to_period("M").dt.to_timestamp()
    s = df.set_index("month")[val_col]
    return s


def fetch_china_pmi_proxy() -> pd.Series:
    # Full 9-slot key: REF_AREA.FREQ.INDICATOR.SUBJECT.MEASURE.ADJUSTMENT.UNIT_MEASURE.PRICE_BASE.TRANSFORMATION
    s = fetch_oecd_series("OECD.SDD.STES,DSD_STES@DF_CLI", "CHN.M.LI...AA.IX..H")
    s.name = "china_pmi_idx"
    return s


def fetch_china_export_proxy() -> pd.Series:
    # NOTE: "OECD.SDD.TPS,DSD_TRADE@DF_TRADE" is not a real OECD dataflow --
    # that's what produced the 404 upstream. No free, no-key dataflow with a
    # verified dimension key for China goods-export values was found while
    # fixing this pipeline; rather than guess at a URL that could silently
    # return the wrong series, this proxy is honestly disabled until a real
    # source is confirmed (see DATA_SOURCES.md). Update
    # china_export_container_idx manually, or wire up a paid feed, in the
    # meantime -- the pipeline will keep carrying the last known value
    # forward without crashing. When this raises, get_latest_month_row()
    # falls back to the regression proxy defined in REGRESSION_PROXY_SPECS
    # below rather than freezing the value forever.
    raise FetchWarning(
        "No verified free OECD/IMF dataflow found for china_export_container_idx "
        "(the old endpoint was invalid and 404'd). Falling back to regression proxy -- "
        "see DATA_SOURCES.md for manual sourcing options."
    )


# ---------------------------------------------------------------------------
# Synthetic proxy composites for the 5 drivers with no free numeric feed at
# all (normally sold by Drewry / Alphaliner / Freightos / the Panama Canal
# Authority's paid data products). These are clearly-labeled stand-ins, not
# real observed series -- swap in a paid feed here the moment one is
# available, the rest of the pipeline does not need to change.
# ---------------------------------------------------------------------------
def proxy_war_risk_premium(gpr_df: pd.DataFrame) -> pd.Series:
    cols = [c for c in WAR_RISK_COUNTRY_COLS if c in gpr_df.columns]
    if not cols:
        raise FetchWarning("None of the expected GPR country columns were found for the war-risk proxy.")
    s = gpr_df[cols].mean(axis=1)
    s.name = "war_risk_premium_idx"
    return s


def proxy_vessel_idle_capacity(congestion: pd.Series, trade_growth: pd.Series) -> pd.Series:
    # Heuristic: idle capacity tends to rise when congestion is low AND trade
    # growth is weak/negative (ships sitting idle rather than queued or
    # sailing full). Normalized to a 0-15% plausible band.
    aligned = pd.concat([congestion, trade_growth], axis=1).dropna()
    if aligned.empty:
        raise FetchWarning("Not enough overlapping congestion/trade data to derive idle-capacity proxy.")
    cong_z = (aligned.iloc[:, 0] - aligned.iloc[:, 0].mean()) / aligned.iloc[:, 0].std(ddof=0)
    trade_z = (aligned.iloc[:, 1] - aligned.iloc[:, 1].mean()) / aligned.iloc[:, 1].std(ddof=0)
    raw = -cong_z - trade_z
    scaled = 3 + 7 * (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)  # ~3-10% band
    scaled.name = "vessel_idle_capacity_pct"
    return scaled


def proxy_panama_restriction(trade_growth: pd.Series) -> pd.Series:
    # Weakest proxy of the five: no free numeric feed on Panama Canal draft
    # restrictions exists. Approximated as an inverse, smoothed function of
    # global trade-volume momentum, rescaled to the original series' band.
    # Flag this to the user in run_manifest.json every run.
    mom = trade_growth.rolling(3, min_periods=1).mean()
    z = (mom - mom.mean()) / (mom.std(ddof=0) + 1e-9)
    scaled = 45 - 8 * z  # centered near historical ~45-50 band, mildly inverse to trade momentum
    scaled.name = "panama_restriction_idx"
    return scaled


def proxy_fbx_composite(brent: pd.Series, congestion: pd.Series, trade_growth: pd.Series) -> pd.Series:
    aligned = pd.concat([brent, congestion, trade_growth], axis=1).dropna()
    if aligned.empty:
        raise FetchWarning("Not enough overlapping data to derive FBX composite proxy.")
    z = aligned.apply(lambda c: (c - c.mean()) / (c.std(ddof=0) + 1e-9))
    composite = 1600 + 250 * z.mean(axis=1)  # centered near the legacy series' historical level
    composite.name = "fbx_composite_legacy_usd_feu"
    return composite


# ---------------------------------------------------------------------------
# Regression proxies -- for the two drivers with no free live feed of any
# kind (eu_ets_carbon_idx, china_export_container_idx: see the FetchWarnings
# raised above), estimate this month's value with a small Ridge regression
# fitted on historical co-movement against whichever real drivers came in
# live this run, rather than freezing the seed value forever. Deliberately
# conservative: only fires when the direct fetch failed AND at least one
# predictor actually refreshed live this run (see get_latest_month_row).
# ---------------------------------------------------------------------------
REGRESSION_PROXY_SPECS = {
    "eu_ets_carbon_idx": ["brent_usd_bbl", "usd_index", "china_pmi_idx"],
    "china_export_container_idx": ["china_pmi_idx", "trade_volume_growth_idx", "congestion_index"],
}


def fit_regression_proxy(history: pd.DataFrame, target_col: str, predictor_cols: list):
    """Fits Ridge(alpha=1.0) on a StandardScaler pipeline, using every
    historical row where both the target and all predictors are present.
    Returns None (caller falls back to carry-forward) if fewer than 6 such
    rows exist -- not enough signal to trust a regression estimate."""
    cols = [target_col] + predictor_cols
    missing = [c for c in cols if c not in history.columns]
    if missing:
        raise FetchWarning(f"Regression proxy for {target_col}: missing columns {missing} in history.")
    sub = history[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 6:
        return None
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    model.fit(sub[predictor_cols].values, sub[target_col].values)
    return model


def predict_regression_proxy(model, predictor_cols: list, predictor_values: dict):
    """Applies a fitted regression proxy model. Returns None if the model is
    None (not enough history) or any predictor value is missing/NaN."""
    if model is None:
        return None
    values = [predictor_values.get(c) for c in predictor_cols]
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in values):
        return None
    return float(model.predict([values])[0])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
REAL_FETCHERS = {
    "brent_usd_bbl": fetch_brent,
    "usd_index": fetch_usd_index,
    "inr_usd_rate": fetch_inr_usd,
}


def fetch_all_real_series():
    """Returns dict[name] -> pd.Series (monthly) for every source that has a
    genuine free feed, plus the raw gpr dataframe (needed for the war-risk
    proxy) and congestion/trade series (needed for the other proxies).
    Anything that fails is omitted, with a warning logged -- callers must
    handle missing keys by carrying the last known value forward."""
    out = {}
    for name, fn in REAL_FETCHERS.items():
        try:
            out[name] = fn()
        except Exception as e:  # noqa: BLE001
            log.warning("Could not fetch %s live (%s). Will carry last known value forward.", name, e)

    gpr_df = None
    try:
        gpr_df = fetch_gpr()
        gpr_col = "GPR" if "GPR" in gpr_df.columns else [c for c in gpr_df.columns if "GPR" in c.upper()][0]
        out["gpr_index"] = gpr_df[gpr_col]
    except Exception as e:  # noqa: BLE001
        log.warning("Could not fetch GPR index live (%s). Will carry last known value forward.", e)

    trade_vol = None
    try:
        trade_vol = fetch_world_trade_volume()
        out["trade_volume_growth_idx"] = trade_vol
    except Exception as e:  # noqa: BLE001
        log.warning("Could not fetch CPB World Trade Monitor live (%s). Will carry last known value forward.", e)

    try:
        out["eu_ets_carbon_idx"] = fetch_eu_ets_price()
    except Exception as e:  # noqa: BLE001
        log.warning("Could not fetch EU ETS price live (%s). Will carry last known value forward.", e)

    congestion = None
    try:
        congestion_val = fetch_port_congestion()
        out["congestion_index"] = congestion_val
    except Exception as e:  # noqa: BLE001
        log.warning("Could not fetch IMF PortWatch congestion live (%s). Will carry last known value forward.", e)

    try:
        out["china_pmi_idx"] = fetch_china_pmi_proxy()
    except Exception as e:  # noqa: BLE001
        log.warning("Could not fetch China PMI proxy live (%s). Will carry last known value forward.", e)

    try:
        out["china_export_container_idx"] = fetch_china_export_proxy()
    except Exception as e:  # noqa: BLE001
        log.warning("Could not fetch China export proxy live (%s). Will carry last known value forward.", e)

    # Synthetic proxies -- only computable where their real inputs succeeded.
    if gpr_df is not None:
        try:
            out["war_risk_premium_idx"] = proxy_war_risk_premium(gpr_df)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not derive war-risk proxy (%s).", e)

    if trade_vol is not None:
        try:
            out["panama_restriction_idx"] = proxy_panama_restriction(trade_vol)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not derive Panama restriction proxy (%s).", e)

    return out


def get_latest_month_row(existing_macro: pd.DataFrame) -> dict:
    """Builds one row (dict of column -> value) for the most recent complete
    calendar month, using live fetches where possible and falling back to
    the last known value in `existing_macro` column-by-column otherwise.
    Never raises -- a fully-offline run just returns the last row unchanged
    plus an updated month label, so pipeline_runner always has something
    to work with."""
    target_month = _month_floor(pd.Timestamp.today().replace(day=1) - pd.Timedelta(days=1))
    existing_macro = existing_macro.copy()
    existing_macro["_month_ts"] = pd.to_datetime(existing_macro["month"])
    exact = existing_macro[existing_macro["_month_ts"] == target_month]
    if not exact.empty:
        baseline = exact.iloc[-1]
    else:
        prior = existing_macro[existing_macro["_month_ts"] < target_month]
        baseline = prior.iloc[-1] if not prior.empty else existing_macro.iloc[-1]
    row = baseline.drop("_month_ts").to_dict()
    row["month"] = target_month.strftime("%Y-%m-%d")

    fetched = fetch_all_real_series()
    used_live = []
    data_cols = [c for c in existing_macro.columns if c not in ("month", "_month_ts")]
    for col in data_cols:
        val = fetched.get(col)
        if isinstance(val, pd.Series) and not val.empty:
            match = val[val.index <= target_month]
            if not match.empty:
                row[col] = float(match.iloc[-1])
                used_live.append(col)
        elif isinstance(val, (int, float)):
            row[col] = float(val)
            used_live.append(col)

    # eu_ets_carbon_idx / china_export_container_idx: no free live feed
    # exists for either (see fetch_eu_ets_price / fetch_china_export_proxy).
    # Rather than freeze them at whatever value happened to be in the seed
    # CSV forever, estimate this month's value with a regression fitted on
    # historical co-movement against the real drivers that *did* refresh
    # live this run. Only fires when the direct fetch failed (skip if
    # already in used_live) and at least one predictor is actually fresh --
    # a fully offline run still just carries the baseline forward.
    for target_col, predictor_cols in REGRESSION_PROXY_SPECS.items():
        if target_col in used_live:
            continue  # direct fetch already worked this run -- don't override real data
        predictor_values = {}
        any_live = False
        for c in predictor_cols:
            live_val = fetched.get(c)
            if isinstance(live_val, pd.Series) and not live_val.empty:
                match = live_val[live_val.index <= target_month]
                if not match.empty:
                    predictor_values[c] = float(match.iloc[-1])
                    any_live = True
                    continue
            elif isinstance(live_val, (int, float)):
                predictor_values[c] = float(live_val)
                any_live = True
                continue
            # predictor didn't refresh live this run -- use this run's
            # carried-forward baseline so the feature vector is still complete
            predictor_values[c] = row.get(c)
        if not any_live:
            continue
        try:
            model = fit_regression_proxy(existing_macro, target_col, predictor_cols)
            pred = predict_regression_proxy(model, predictor_cols, predictor_values)
            if pred is not None:
                row[target_col] = pred
                used_live.append(target_col)
                log.info("Estimated %s via regression proxy on %s -> %.3f",
                         target_col, predictor_cols, pred)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not fit/apply regression proxy for %s (%s). "
                        "Carrying forward last known value.", target_col, e)

    # Vessel idle + FBX composite need other proxies already resolved this run.
    try:
        cong = fetched.get("congestion_index")
        trade = fetched.get("trade_volume_growth_idx")
        brent = fetched.get("brent_usd_bbl")
        if isinstance(cong, (int, float)) and isinstance(trade, pd.Series) and not trade.empty:
            cong_series = pd.Series([cong], index=[target_month])
            vessel_idle = proxy_vessel_idle_capacity(cong_series, trade)
            if not vessel_idle.empty:
                row["vessel_idle_capacity_pct"] = float(vessel_idle.iloc[-1])
                used_live.append("vessel_idle_capacity_pct")
        if isinstance(brent, pd.Series) and isinstance(cong, (int, float)) and isinstance(trade, pd.Series):
            brent_at_month = brent[brent.index <= target_month]
            if not brent_at_month.empty and not trade.empty:
                cong_series = pd.Series([cong], index=[target_month])
                fbx = proxy_fbx_composite(brent_at_month.tail(1), cong_series, trade.tail(1))
                if not fbx.empty:
                    row["fbx_composite_legacy_usd_feu"] = float(fbx.iloc[-1])
                    used_live.append("fbx_composite_legacy_usd_feu")
    except Exception as e:  # noqa: BLE001
        log.warning("Could not derive vessel-idle / FBX composite proxies this run (%s).", e)

    log.info("Live/derived this run: %s", ", ".join(sorted(set(used_live))) or "none (fully offline fallback)")
    stale = [c for c in data_cols if c not in used_live]
    if stale:
        log.info("Carried forward from last known value: %s", ", ".join(stale))
    return row
