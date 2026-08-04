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

PROVIDERS (both free, no credit card required)
  - Groq   (default): https://console.groq.com -> free API key, fast, an
            OpenAI-compatible /chat/completions endpoint.
  - Gemini (fallback): https://aistudio.google.com/apikey -> free API key,
            generous free-tier quota.

USAGE
  export GROQ_API_KEY=gsk_...
  cd pipeline
  python3 ai_rate_lookup.py                       # all lanes, chunked by region
  python3 ai_rate_lookup.py --provider gemini      # use Gemini instead
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

GEMINI_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
)
GEMINI_MODEL_DEFAULT = "gemini-2.0-flash"

CHUNK_SIZE = 12          # destinations per LLM call
SLEEP_BETWEEN_CALLS = 2  # seconds, be polite to free-tier rate limits
MAX_RETRIES = 3


def _prompt(origin_row, dest_rows, current_month_label):
    origin = origin_row["origin"]
    gateway = origin_row["gateway_port"]
    lines = "\n".join(
        f'- {{"destination": "{r.destination}", "country": "{r.country}"}}'
        for r in dest_rows.itertuples()
    )
    return f"""You are an ocean freight market analyst. For 40ft container (FEU) spot
market ocean freight rates ORIGINATING FROM INDIA ({origin}, gateway port {gateway}),
give your best realistic current-market estimate, and a separate 3-month-ahead
estimate, for EACH destination listed below. Base this on your knowledge of
recent freight market levels, typical trade-lane pricing patterns, seasonality,
and any known disruptions (Red Sea/Suez rerouting, Panama Canal, Cape of Good
Hope diversions, peak season, etc.) as of {current_month_label}.

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
destination in the list must appear exactly once in your answer."""


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


def call_gemini(prompt, api_key, model):
    url = GEMINI_URL_TMPL.format(model=model, key=api_key)
    resp = requests.post(
        url,
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def call_llm(prompt, provider, api_key, model):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if provider == "groq":
                raw = call_groq(prompt, api_key, model)
            elif provider == "gemini":
                raw = call_gemini(prompt, api_key, model)
            else:
                raise ValueError(f"Unknown provider: {provider}")
            return _extract_json_array(raw)
        except Exception as e:
            print(f"    attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(3 * attempt)


def chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["groq", "gemini"], default="groq")
    ap.add_argument("--model", default=None)
    ap.add_argument("--refresh", action="store_true", help="ignore cache, re-query every lane")
    ap.add_argument("--origin", default=None, help="only re-check this origin")
    ap.add_argument("--region", default=None, help="only re-check this destination region")
    args = ap.parse_args()

    api_key = os.environ.get("GROQ_API_KEY") if args.provider == "groq" else os.environ.get("GEMINI_API_KEY")
    if not api_key:
        env_name = "GROQ_API_KEY" if args.provider == "groq" else "GEMINI_API_KEY"
        print(f"ERROR: set {env_name} in your environment first (free key, no card needed).")
        print("  Groq:   https://console.groq.com  ->  API Keys")
        print("  Gemini: https://aistudio.google.com/apikey")
        sys.exit(1)
    model = args.model or (GROQ_MODEL_DEFAULT if args.provider == "groq" else GEMINI_MODEL_DEFAULT)

    origins = pd.read_csv(f"{DATA_DIR}/origins.csv")
    dests = pd.read_csv(f"{DATA_DIR}/destinations.csv")

    if args.origin:
        origins = origins[origins["origin"] == args.origin]
    if args.region:
        dests = dests[dests["region"] == args.region]

    cache = pd.read_csv(CACHE_PATH) if (os.path.exists(CACHE_PATH) and not args.refresh) else pd.DataFrame(
        columns=["origin", "destination", "country", "region", "ai_current_rate_usd_feu",
                 "ai_forecast_rate_usd_feu_3mo", "ai_pct_change", "ai_note", "ai_provider",
                 "ai_model", "fetched_at"]
    )
    already_done = set(zip(cache["origin"], cache["destination"])) if not args.refresh else set()

    current_month_label = datetime.now(timezone.utc).strftime("%B %Y")
    new_rows = []

    for _, origin_row in origins.iterrows():
        for region, region_df in dests.groupby("region"):
            pending = region_df[~region_df.apply(
                lambda r: (origin_row["origin"], r["destination"]) in already_done, axis=1
            )]
            if pending.empty:
                continue
            for chunk_df in chunks(pending, CHUNK_SIZE):
                print(f"[{origin_row['origin']} -> {region}] querying {len(chunk_df)} destinations via {args.provider}...")
                prompt = _prompt(origin_row, chunk_df, current_month_label)
                try:
                    results = call_llm(prompt, args.provider, api_key, model)
                except Exception as e:
                    print(f"  SKIPPED chunk after retries failed: {e}")
                    continue
                by_dest = {r.get("destination"): r for r in results if isinstance(r, dict)}
                for _, drow in chunk_df.iterrows():
                    r = by_dest.get(drow["destination"])
                    if not r:
                        print(f"  missing '{drow['destination']}' in model response, skipping")
                        continue
                    try:
                        cur = float(r["current_rate_usd_feu"])
                        fc3 = float(r["forecast_rate_usd_feu_3mo"])
                    except (KeyError, TypeError, ValueError):
                        print(f"  malformed row for '{drow['destination']}', skipping")
                        continue
                    new_rows.append({
                        "origin": origin_row["origin"],
                        "destination": drow["destination"],
                        "country": drow["country"],
                        "region": drow["region"],
                        "ai_current_rate_usd_feu": round(cur, 0),
                        "ai_forecast_rate_usd_feu_3mo": round(fc3, 0),
                        "ai_pct_change": round((fc3 - cur) / cur * 100, 1) if cur else None,
                        "ai_note": r.get("note", ""),
                        "ai_provider": args.provider,
                        "ai_model": model,
                        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
    print(f"\nSaved {len(combined)} AI-sourced lane rates -> {CACHE_PATH}")
    print("This file is independent of the statistical model. Re-run build_report_v2.py to fold it into the report.")


if __name__ == "__main__":
    main()
