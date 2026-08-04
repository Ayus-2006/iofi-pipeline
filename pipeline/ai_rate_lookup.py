"""
ai_rate_lookup.py
------------------
Standalone, model-independent lane rate check.

This does NOT touch feature_engineering / weight_learning / lane_sensitivity /
hierarchical.py or anything else in the statistical pipeline. It is a separate
question asked to an LLM: "what is a realistic current ocean freight rate
(USD/FEU, 40ft container) for this specific India -> destination lane, and
what's a reasonable estimate 3 months out." The statistical model's own
calibrated numbers (predictions_full.csv / lane_forecasts_v2.csv) are left
completely untouched -- this just produces a second, independent file:

    outputs/ai_lane_rates.csv

build_report_v2.py then prefers this file's numbers (when present) for the
figures that actually get printed in the report, and falls back to the
model's own numbers for any lane the AI call didn't cover. This keeps the
two systems decoupled: you can delete outputs/ai_lane_rates.csv at any time
and the report reverts to pure-model output with zero code changes.

WHY THIS EXISTS
The report itself says "Only 2 of 60 destination base rates are anchored to
a real published tariff; the rest remain calibrated planning estimates."
That's the root cause of lanes like Nhava Sheva/ICD Tihi -> Durban reading
low -- the base rate was a guess, not an observed rate. Asking a live LLM
(which has broad exposure to shipping-rate reporting, freight news, and
index commentary) for a plausible current figure is a cheap, free sanity
check on those guesses. It is still not a paid rate-benchmarking
subscription or an actual carrier quote -- treat it as a better-informed
planning estimate, same disclaimer the report already carries.

PROVIDER: Groq (free, no credit card required)
  https://console.groq.com -> free API key, fast, OpenAI-compatible
  /chat/completions endpoint. Groq/Llama has NO live web search -- it only
  recalls whatever was in its training data. To stop it from anchoring to
  stale pre-2024 pricing (the original bug -- USA lanes reading $2-3k
  against a real ~$6-8k market), every prompt includes hard current-market
  calibration ranges per region (see REGION_CALIBRATION below), and any
  answer that still lands far under the calibration floor is automatically
  re-anchored to the calibration midpoint before being saved.

FULL COVERAGE GUARANTEE
Every origin x destination combination in data/origins.csv x
data/destinations.csv gets a row in outputs/ai_lane_rates.csv -- if Groq
skips a destination, returns a malformed row, or a whole chunk fails after
retries, that combination is filled directly from REGION_CALIBRATION
(midpoint of the range) instead of being left out. Nothing falls through.

USAGE
  export GROQ_API_KEY=gsk_...
  cd pipeline
  python3 ai_rate_lookup.py                       # all lanes, chunked by region
  python3 ai_rate_lookup.py --refresh              # ignore cache, re-query everything
  python3 ai_rate_lookup.py --origin "ICD Tihi" --region "Africa / South America (Cape)"
                                                    # just re-check one slice
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
CACHE_PATH = os.path.join(OUT_DIR, "ai_lane_rates.csv")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_DEFAULT = "llama-3.3-70b-versatile"

CHUNK_SIZE = 12          # destinations per LLM call
SLEEP_BETWEEN_CALLS = 2  # seconds, be polite to free-tier rate limits
MAX_RETRIES = 3

# Calibration anchors, USD/FEU, India -> destination region, current elevated
# market (sustained Red Sea/Suez avoidance -> Cape of Good Hope rerouting,
# Panama Canal draft restrictions, peak-season demand). These are the floor/
# ceiling a well-informed analyst should land in -- NOT a hard rule, but Groq
# kept anchoring to stale pre-disruption pricing, especially on the
# long-haul lanes, so this gives it a concrete, current reference point
# instead of whatever's baked into its training data, AND doubles as the
# fill-in value for any lane the model skips entirely. Ranges came from a
# route-exposure benchmark pass -- see anchor_benchmark_rates.py for the
# full standalone (no-API-key) version of this same logic.
REGION_CALIBRATION = {
    "Middle East":                          (850, 1100),
    "Red Sea / East Africa":                (1250, 1950),
    "Europe (Mediterranean)":               (3400, 4300),
    "Europe (North)":                       (4300, 5500),
    "US East Coast":                        (6500, 8200),
    "US Gulf / Caribbean / Latin America":  (5000, 8000),   # wide: US ports run high, non-US (Cartagena/Balboa/Kingston) lower
    "Africa / South America (Cape)":        (3000, 4800),
    "Indian Ocean Rim":                     (550, 1100),
    "Pacific (US West / Oceania)":          (2300, 6500),   # wide: US West Coast high, Australia/NZ/Canada much lower
    "Intra-Asia":                           (450, 1100),
}
DEFAULT_DECLINE_PCT = 8.0  # assumed 3-month softening when we have to fall back to a pure calibration fill


def _prompt(origin_row, dest_rows, current_month_label, region=None):
    origin = origin_row["origin"]
    gateway = origin_row["gateway_port"]
    lines = "\n".join(
        f'- {{"destination": "{r.destination}", "country": "{r.country}"}}'
        for r in dest_rows.itertuples()
    )
    calib = REGION_CALIBRATION.get(region)
    calib_block = ""
    if calib:
        lo, hi = calib
        calib_block = f"""
CALIBRATION CONTEXT (important -- do not ignore): informed market estimates
for India -> {region} on this lane type are currently running roughly
${lo:,}-${hi:,} per FEU, driven by sustained Red Sea/Suez avoidance (ships
diverting via the Cape of Good Hope, adding ~3,000nm and 10-14 days),
Panama Canal draft restrictions, and peak-season demand. If your instinct is
to answer well below this range, that instinct is pulling from stale
pre-2024 pricing baked into your training data -- override it and reason
from the current elevated market instead. Not every destination in this
region will land at the same point in the range (e.g. USA ports run at the
high end, non-US ports on a shared "US Gulf" or "Pacific" grouping route can
run lower) -- use judgment per destination, but do not default to
pre-disruption pricing.
"""
    return f"""You are an ocean freight market analyst. For 40ft container (FEU) spot
market ocean freight rates ORIGINATING FROM INDIA ({origin}, gateway port {gateway}),
give your best realistic current-market estimate, and a separate 3-month-ahead
estimate, for EACH destination listed below. Base this on your knowledge of
recent freight market levels, typical trade-lane pricing patterns, seasonality,
and any known disruptions (Red Sea/Suez rerouting, Panama Canal, Cape of Good
Hope diversions, peak season, etc.) as of {current_month_label}.
{calib_block}
Destinations:
{lines}

Respond with ONLY a JSON array, no prose, no markdown fences, one object per
destination, in this exact shape:
[
  {{
    "destination": "<same string as given>",
    "current_rate_usd_feu": <integer, realistic current spot rate>,
    "forecast_rate_usd_feu_3mo": <integer, your 3-month-ahead estimate>,
    "note": "<max 12 words: key driver, e.g. 'Cape diversion keeping rates elevated'>"
  }}
]
Use plain integers (no $ sign, no commas). If you are not confident, still give
your best reasoned estimate rather than omitting a destination -- every
destination in the list must appear exactly once in your answer. Do not
under-anchor to outdated pre-disruption rates -- use the calibration context
above as your primary reference point when it's provided."""


def _extract_json_array(text):
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in model response: {text[:200]}")
    return json.loads(match.group(0))


def call_groq(prompt, api_key, model):
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def call_llm(prompt, api_key, model):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_groq(prompt, api_key, model)
            return _extract_json_array(raw)
        except Exception as e:
            print(f"    attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                return None
            time.sleep(3 * attempt)


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _calibration_fill_row(origin_row, drow, region, fetched_at, reason):
    """Used whenever Groq skips a destination, returns a malformed row, or a
    whole chunk fails after retries -- guarantees every origin x destination
    combination still gets a row instead of silently dropping out."""
    calib = REGION_CALIBRATION.get(region)
    if calib:
        lo, hi = calib
        cur = (lo + hi) / 2
    else:
        cur = None
    fc3 = round(cur * (1 - DEFAULT_DECLINE_PCT / 100), 0) if cur is not None else None
    return {
        "origin": origin_row["origin"],
        "destination": drow["destination"],
        "country": drow["country"],
        "region": drow["region"],
        "ai_current_rate_usd_feu": round(cur, 0) if cur is not None else None,
        "ai_forecast_rate_usd_feu_3mo": fc3,
        "ai_pct_change": round((fc3 - cur) / cur * 100, 1) if cur else None,
        "ai_note": f"Calibration fill ({reason}) -- Groq did not return a usable value",
        "ai_provider": "calibration_fill",
        "ai_model": "region-calibration",
        "ai_confidence": "calibration_fill",
        "fetched_at": fetched_at,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--refresh", action="store_true", help="ignore cache, re-query every lane")
    ap.add_argument("--origin", default=None, help="only re-check this origin")
    ap.add_argument("--region", default=None, help="only re-check this destination region")
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: set GROQ_API_KEY in your environment first (free key, no card needed).")
        print("  https://console.groq.com  ->  API Keys")
        sys.exit(1)
    model = args.model or GROQ_MODEL_DEFAULT

    origins = pd.read_csv(f"{DATA_DIR}/origins.csv")
    dests = pd.read_csv(f"{DATA_DIR}/destinations.csv")
    dests = dests.rename(columns={"port": "destination"})

    if args.origin:
        origins = origins[origins["origin"] == args.origin]
    if args.region:
        dests = dests[dests["region"] == args.region]

    cache = pd.read_csv(CACHE_PATH) if (os.path.exists(CACHE_PATH) and not args.refresh) else pd.DataFrame(
        columns=["origin", "destination", "country", "region", "ai_current_rate_usd_feu",
                 "ai_forecast_rate_usd_feu_3mo", "ai_pct_change", "ai_note", "ai_provider",
                 "ai_model", "ai_confidence", "fetched_at"]
    )
    already_done = set(zip(cache["origin"], cache["destination"])) if not args.refresh else set()

    current_month_label = datetime.now(timezone.utc).strftime("%B %Y")
    fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_rows = []
    n_from_model, n_from_calib_correction, n_from_calib_fill = 0, 0, 0

    for _, origin_row in origins.iterrows():
        for region, region_df in dests.groupby("region"):
            pending = region_df[~region_df.apply(
                lambda r: (origin_row["origin"], r["destination"]) in already_done, axis=1
            )]
            if pending.empty:
                continue
            for chunk_df in chunks(pending, CHUNK_SIZE):
                print(f"[{origin_row['origin']} -> {region}] querying {len(chunk_df)} destinations via groq...")
                prompt = _prompt(origin_row, chunk_df, current_month_label, region=region)
                results = call_llm(prompt, api_key, model)
                if results is None:
                    print(f"  SKIPPED chunk after retries failed -- filling all {len(chunk_df)} "
                          f"destinations from calibration instead.")
                    for _, drow in chunk_df.iterrows():
                        new_rows.append(_calibration_fill_row(origin_row, drow, region, fetched_at, "chunk failed"))
                        n_from_calib_fill += 1
                    time.sleep(SLEEP_BETWEEN_CALLS)
                    continue

                by_dest = {r.get("destination"): r for r in results if isinstance(r, dict)}
                calib = REGION_CALIBRATION.get(region)
                calib_floor = calib[0] * 0.6 if calib else None    # 40% below floor = stale/underselling
                calib_ceiling = calib[1] * 1.6 if calib else None  # 60% above ceiling = implausible spike/overselling

                for _, drow in chunk_df.iterrows():
                    r = by_dest.get(drow["destination"])
                    if not r:
                        print(f"  '{drow['destination']}' missing from model response -- filling from calibration")
                        new_rows.append(_calibration_fill_row(origin_row, drow, region, fetched_at, "missing from response"))
                        n_from_calib_fill += 1
                        continue
                    try:
                        cur = float(r["current_rate_usd_feu"])
                        fc3 = float(r["forecast_rate_usd_feu_3mo"])
                    except (KeyError, TypeError, ValueError):
                        print(f"  malformed row for '{drow['destination']}' -- filling from calibration")
                        new_rows.append(_calibration_fill_row(origin_row, drow, region, fetched_at, "malformed row"))
                        n_from_calib_fill += 1
                        continue

                    note = r.get("note", "")
                    if calib_floor and cur < calib_floor:
                        print(f"  '{drow['destination']}' came back ${cur:,.0f} -- below the "
                              f"${calib_floor:,.0f} sanity floor for {region}, model likely used "
                              f"stale pricing (underselling). Re-anchoring to calibration midpoint.")
                        lo, hi = calib
                        mid = (lo + hi) / 2
                        scale = mid / cur if cur else 1
                        cur, fc3 = mid, round(fc3 * scale, 0)
                        note = f"{note} [re-anchored: model answer was below calibration floor]".strip()
                        n_from_calib_correction += 1
                    elif calib_ceiling and cur > calib_ceiling:
                        print(f"  '{drow['destination']}' came back ${cur:,.0f} -- above the "
                              f"${calib_ceiling:,.0f} sanity ceiling for {region}, likely an implausible "
                              f"spike (overselling). Re-anchoring to calibration midpoint.")
                        lo, hi = calib
                        mid = (lo + hi) / 2
                        scale = mid / cur if cur else 1
                        cur, fc3 = mid, round(fc3 * scale, 0)
                        note = f"{note} [re-anchored: model answer was above calibration ceiling]".strip()
                        n_from_calib_correction += 1
                    else:
                        n_from_model += 1

                    new_rows.append({
                        "origin": origin_row["origin"],
                        "destination": drow["destination"],
                        "country": drow["country"],
                        "region": drow["region"],
                        "ai_current_rate_usd_feu": round(cur, 0),
                        "ai_forecast_rate_usd_feu_3mo": round(fc3, 0),
                        "ai_pct_change": round((fc3 - cur) / cur * 100, 1) if cur else None,
                        "ai_note": note,
                        "ai_provider": "groq",
                        "ai_model": model,
                        "ai_confidence": "model_verified" if "re-anchored" not in note else "model_corrected",
                        "fetched_at": fetched_at,
                    })
                time.sleep(SLEEP_BETWEEN_CALLS)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([cache, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["origin", "destination"], keep="last")
    else:
        combined = cache

    os.makedirs(OUT_DIR, exist_ok=True)
    combined.to_csv(CACHE_PATH, index=False)
    print(f"\nSaved {len(combined)} lane rates -> {CACHE_PATH}")
    print(f"  verified as-is from Groq (within calibration band): {n_from_model}")
    print(f"  Groq answer corrected (was below floor or above ceiling): {n_from_calib_correction}")
    print(f"  filled straight from calibration (Groq gave nothing usable): {n_from_calib_fill}")
    print("Every origin x destination combination in scope now has a row -- none skipped.")
    print("Check the ai_confidence column: 'model_verified' = Groq's own number passed both the")
    print("floor and ceiling sanity checks; 'model_corrected' = Groq answered but was pulled back")
    print("into the realistic range; 'calibration_fill' = Groq gave nothing usable at all.")
    print("This file is independent of the statistical model. Re-run build_report_v2.py to fold it into the report.")


if __name__ == "__main__":
    main()
