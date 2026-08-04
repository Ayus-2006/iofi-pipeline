import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER

from report_data import assemble, OUT_DIR

CHART_DIR = os.path.join(OUT_DIR, "charts")
NAVY = colors.HexColor("#1a2b4c")
RED = colors.HexColor("#c0392b")
TEAL = colors.HexColor("#1abc9c")
LIGHTGRAY = colors.HexColor("#f2f2f2")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1c", parent=styles["Heading1"], textColor=NAVY, spaceAfter=10))
styles.add(ParagraphStyle(name="H2c", parent=styles["Heading2"], textColor=NAVY, spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], fontSize=9.5, leading=13.5))
styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=8, textColor=colors.gray,
                           alignment=TA_CENTER, spaceAfter=8))
styles.add(ParagraphStyle(name="TitleBig", parent=styles["Title"], textColor=NAVY, fontSize=22))
styles.add(ParagraphStyle(name="SubTitle", parent=styles["Normal"], fontSize=11, textColor=colors.gray,
                           alignment=TA_CENTER))


def img(fname, width=6.6 * inch):
    path = os.path.join(CHART_DIR, fname)
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        w, h = im.size
    return Image(path, width=width, height=width * h / w)


AI_RATES_PATH = os.path.join(OUT_DIR, "ai_lane_rates.csv")


def apply_ai_overrides(h3):
    """
    Independent cross-check step. If pipeline/ai_rate_lookup.py has been run,
    outputs/ai_lane_rates.csv exists with LLM-sourced current + 3-month rates
    for some or all lanes. Where available, those figures REPLACE the model's
    calibrated current_rate_usd_feu / forecast_rate_usd_feu / pct_change for
    what actually gets printed in the report. Lanes with no AI figure fall
    back to the model's own number, unchanged. Does not touch lane_fc,
    lane_forecasts_v2.csv, or anything upstream -- purely a display-time swap.
    """
    h3 = h3.copy()
    h3["rate_source"] = "model"
    if not os.path.exists(AI_RATES_PATH):
        return h3
    ai = pd.read_csv(AI_RATES_PATH)
    if ai.empty:
        return h3
    merged = h3.merge(
        ai[["origin", "destination", "ai_current_rate_usd_feu",
            "ai_forecast_rate_usd_feu_3mo", "ai_pct_change"]],
        on=["origin", "destination"], how="left"
    )
    has_ai = merged["ai_current_rate_usd_feu"].notna()
    merged.loc[has_ai, "current_rate_usd_feu"] = merged.loc[has_ai, "ai_current_rate_usd_feu"]
    merged.loc[has_ai, "forecast_rate_usd_feu"] = merged.loc[has_ai, "ai_forecast_rate_usd_feu_3mo"]
    merged.loc[has_ai, "pct_change"] = merged.loc[has_ai, "ai_pct_change"]
    merged.loc[has_ai, "rate_source"] = "ai"
    n_ai = int(has_ai.sum())
    if n_ai:
        print(f"Applied AI-sourced rates to {n_ai}/{len(merged)} lanes (see outputs/ai_lane_rates.csv).")
    return merged.drop(columns=["ai_current_rate_usd_feu", "ai_forecast_rate_usd_feu_3mo", "ai_pct_change"])


def build_lane_table(df, title, cols, headers):
    data = [headers] + df[cols].values.tolist()
    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHTGRAY]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t


def build():
    out = assemble()
    manifest = out["manifest"]
    lanes = out["lanes"]
    lane_fc = out["lane_forecasts"]
    importance = out["importance"]
    iofi_q = out["iofi_quantiles"]
    metrics = out["backtest_metrics"]

    iofi_current = manifest["iofi_current"]
    med_h3 = manifest["iofi_ensemble_path"][-1]
    pct_h3 = (med_h3 - iofi_current) / iofi_current * 100
    p05_h3 = iofi_q.iloc[-1]["0.05"] if "0.05" in iofi_q.columns else iofi_q.iloc[-1][0.05]
    p95_h3 = iofi_q.iloc[-1]["0.95"] if "0.95" in iofi_q.columns else iofi_q.iloc[-1][0.95]

    h3 = lane_fc[lane_fc["horizon"] == 3].copy()
    h3 = apply_ai_overrides(h3)
    biggest_decline = h3.sort_values("pct_change").iloc[0]
    biggest_increase = h3.sort_values("pct_change").iloc[-1]

    doc_path = os.path.join(OUT_DIR, "IOFI_Report_2026_07_v2.pdf")
    doc = SimpleDocTemplate(doc_path, pagesize=letter,
                             topMargin=0.7 * inch, bottomMargin=0.6 * inch,
                             leftMargin=0.7 * inch, rightMargin=0.7 * inch)
    story = []

    # ---------------- Title ----------------
    story.append(Paragraph("India Ocean Freight Index", styles["TitleBig"]))
    story.append(Paragraph("IOFI Monthly Report & 3-Month Lane Forecast — v2 (Data-Driven Model)", styles["SubTitle"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "July 2026 | Nhava Sheva, ICD Tihi, ICD Dhannad → 60 destination ports | 180 lanes",
        styles["SubTitle"]))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", color=NAVY, thickness=1))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"Current IOFI: <b>{iofi_current:.1f}</b> &nbsp;&nbsp; "
        f"3-mo model median: <b>{med_h3:.1f}</b> ({'+' if pct_h3>=0 else ''}{pct_h3:.1f}%) &nbsp;&nbsp; "
        f"5th–95th pct range: {p05_h3:.1f} – {p95_h3:.1f}",
        styles["Body"]))
    story.append(Spacer(1, 10))

    # ---------------- Executive summary ----------------
    story.append(Paragraph("Executive Summary", styles["H1c"]))
    top_driver = importance.iloc[0]["feature"].replace("_", " ")
    story.append(Paragraph(
        f"The IOFI stands at {iofi_current:.1f} this month. This release replaces the fixed-weight, "
        f"hand-decayed version of the model with a fully data-driven pipeline: driver weights are learned "
        f"(Ridge, ElasticNet, Bayesian Ridge, XGBoost, ensembled and cross-checked with SHAP), each macro "
        f"driver is forecast independently with an auto-selected time-series model (ARIMA / ETS / Kalman "
        f"filter), lanes are grouped by an automatic clustering step, and uncertainty is expressed as "
        f"statistically estimated percentile bands rather than hand-set High/Base/Low scenarios.",
        styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"The single largest learned contributor to the current index level is <b>{top_driver}</b>. "
        f"Under the model's median forecast path, the index reaches <b>{med_h3:.1f}</b> "
        f"({'+' if pct_h3>=0 else ''}{pct_h3:.1f}%) within three months, with a statistically estimated "
        f"5th–95th percentile range of {p05_h3:.1f} to {p95_h3:.1f}. Unlike the prior version, this path is "
        f"not assumed to decay — it is the output of the driver-level forecasts feeding back through the "
        f"learned index model, so it can just as easily continue rising as it can fall, depending on what "
        f"the underlying time-series models project for each driver.",
        styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Across the 180 India-origin lanes, the model-median case implies moves ranging from "
        f"{h3['pct_change'].min():.1f}% to +{h3['pct_change'].max():.1f}%. "
        f"{biggest_decline['destination']} ({biggest_decline['origin']}) shows the largest projected "
        f"decline ({biggest_decline['pct_change']:.1f}%), while {biggest_increase['destination']} "
        f"({biggest_increase['origin']}) shows the largest projected increase "
        f"(+{biggest_increase['pct_change']:.1f}%). In backtesting over "
        f"{metrics['_meta']['n_folds']} walk-forward months, the learned model achieved a "
        f"{metrics['learned_model']['MAPE']:.1f}% MAPE (bootstrap 95% CI: "
        f"{metrics['learned_model']['MAPE_CI95_lower']:.1f}%\u2013{metrics['learned_model']['MAPE_CI95_upper']:.1f}%) "
        f"and {metrics['learned_model']['DirectionalAccuracy_%']:.0f}% directional accuracy, versus "
        f"{metrics['naive_persistence']['MAPE']:.1f}% MAPE for naive persistence. With this few backtest "
        f"folds the CI is itself wide -- treat the point MAPE as directional, not precise.",
        styles["Body"]))
    story.append(PageBreak())

    # ---------------- Section 1: History & forecast ----------------
    story.append(Paragraph("1. IOFI: History &amp; Statistically Learned 3-Month Forecast", styles["H2c"]))
    story.append(Paragraph(
        f"The IOFI is a composite of {manifest['n_driver_cols']} macro/structural drivers of India-origin "
        "container freight rates -- ten carried over from v2.0 plus two China-side supply drivers added in "
        "v2.1 (Section 3). Base = 100 at the start of the history window (Aug 2024). Instead of a hand-picked "
        "scenario decay, each driver is forecast independently (ARIMA/ETS/Kalman filter, auto-selected by "
        "AIC), the forecasted driver paths are fed through the learned index model (Ridge/ElasticNet/Bayesian "
        "Ridge/XGBoost ensemble) to get a median path, and a bootstrap ensemble produces 5th/25th/50th/75th/95th "
        "percentile bands.", styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(img("iofi_forecast_v2.png"))
    story.append(Paragraph("Figure 1. IOFI history and 3-month statistically forecasted percentile band.", styles["Caption"]))

    q_table_df = iofi_q.copy()
    q_table_df.columns = [str(c) for c in q_table_df.columns]
    disp = q_table_df[["month", "0.05", "0.25", "0.5", "0.75", "0.95"]].round(1)
    disp.columns = ["Month", "P5", "P25", "Median (P50)", "P75", "P95"]
    story.append(Spacer(1, 6))
    story.append(build_lane_table(disp, "", list(disp.columns), list(disp.columns)))
    story.append(PageBreak())

    # ---------------- Section 2: Driver attribution ----------------
    story.append(Paragraph("2. Driver Attribution — Learned, Not Assigned", styles["H2c"]))
    story.append(Paragraph(
        "Each driver's contribution is estimated, not hand-weighted. Ridge, ElasticNet, Bayesian Ridge and "
        "XGBoost are each fit on the engineered feature set (lags, rolling stats, momentum, seasonality, "
        "nonlinear interactions) and their normalized importances are ensembled. XGBoost's contribution is "
        "cross-checked with SHAP values so the ranking below reflects genuine learned sensitivity rather than "
        "an analyst's prior.", styles["Body"]))
    story.append(Spacer(1, 6))
    xgb_rel = manifest.get("xgb_reliability") or {}
    if xgb_rel:
        story.append(Paragraph(
            f"<b>Sample-size guard on XGBoost:</b> at n={xgb_rel.get('n_training_obs')} training months, "
            f"XGBoost's vote in the ensemble average is scaled by "
            f"{xgb_rel.get('shrinkage_weight_applied')}x rather than counted at full weight (full weight "
            f"requires n\u2265{xgb_rel.get('full_reliable_n')} months) -- boosted trees are the estimator most "
            "prone to overfitting at this sample size, so it is discounted rather than dropped: it remains "
            "the only estimator here that captures nonlinear driver interactions without those interactions "
            "being hand-specified. Raw, unshrunk XGBoost importances are retained in "
            "learned_driver_weights.csv for full transparency.", styles["Body"]))
        story.append(Spacer(1, 6))
    story.append(img("driver_attribution_v2.png"))
    story.append(Paragraph("Figure 2. Top 10 learned features by ensemble-normalized importance (sample-size-adjusted).", styles["Caption"]))
    story.append(PageBreak())

    # ---------------- Section 3: Driver deep dives ----------------
    story.append(Paragraph("3. Driver Deep-Dives", styles["H2c"]))
    story.append(Paragraph(
        f"v2.1 adds two China-side supply drivers to the {manifest['n_driver_cols'] - 2} carried in v2.0 -- "
        "a manufacturing-PMI-like index and an export container throughput index -- so the model captures "
        "China's own festival calendar (Chinese New Year factory shutdowns, Golden Week) and export capacity "
        "directly, rather than only inferring China-side effects indirectly through congestion and idle "
        "capacity. See Figure 6b.", styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(img("oil_vs_gpr_v2.png"))
    story.append(Paragraph("Figure 3. Oil price vs. geopolitical risk index — both learned as top contributors.", styles["Caption"]))
    story.append(Spacer(1, 8))
    story.append(img("congestion_vs_idle_v2.png"))
    story.append(Paragraph("Figure 4. Port congestion vs. vessel idle capacity (idle capacity carries a learned inverse relationship).", styles["Caption"]))
    story.append(PageBreak())
    story.append(img("warrisk_vs_panama_v2.png"))
    story.append(Paragraph("Figure 5. War-risk insurance premium vs. Panama Canal draft restriction — two structurally distinct drivers.", styles["Caption"]))
    story.append(Spacer(1, 8))
    story.append(img("tradevol_vs_inr_v2.png"))
    story.append(Paragraph("Figure 6. Global trade volume growth vs. INR/USD.", styles["Caption"]))
    story.append(PageBreak())
    story.append(img("china_pmi_vs_exports_v2.png"))
    story.append(Paragraph(
        "Figure 6b. China manufacturing PMI-like index vs. export container throughput — new in v2.1. Both "
        "dip sharply every February (Chinese New Year factory shutdowns) and, to a lesser extent, every "
        "October (Golden Week); throughput growth stalls through the recent congestion spike as capacity "
        "gets absorbed rather than cleared.", styles["Caption"]))
    story.append(PageBreak())

    # ---------------- Section 4: Lane-level forecasts ----------------
    story.append(Paragraph("4. Lane-Level Rate Forecasts — Hierarchical Global \u2192 Regional \u2192 Seasonal \u2192 Lane", styles["H2c"]))
    story.append(Paragraph(
        f"Lane translation is now structured in three learned levels plus one calendar-driven adjustment, "
        f"instead of one flat step. The global IOFI move feeds a <b>regional</b> move (the median historical "
        f"pass-through of all lanes within a destination region — {manifest.get('n_regions', 'n/a')} regions "
        f"in total), and each lane's own historical pass-through is expressed as a <b>lane premium</b> on top "
        f"of its region: the amount that lane moves beyond what its corridor alone would explain. On top of "
        f"both, a <b>regional seasonal premium</b> adds the festival/holiday calendar effect relevant to that "
        f"specific destination region and forecast month — Diwali for Indian Ocean Rim, Golden Week and "
        f"Chinese New Year for Intra-Asia (the China-side counterpart to the existing US import-season/"
        f"Black Friday/Christmas seasonality), Ramadan/Eid for Middle East and Red Sea/East Africa, and US "
        f"import-season peak for the US-bound regions — capped at \u00b1{2.5}% per month so it nudges the "
        f"learned forecast rather than overriding it. The regional, lane-premium, and seasonal components sum "
        f"back to the total forecast (see lane_forecasts_v2.csv columns regional_pct_change / "
        f"lane_premium_pct_change / seasonal_pct_change), letting a planner see whether a lane's move is "
        f"\u201cthis corridor is repricing,\u201d \u201cthis specific lane is unusual for its corridor,\u201d or "
        f"\u201cthis month's festival calendar for this region.\u201d For reference, lanes are also grouped "
        f"into {manifest['lane_cluster_k']} automatically discovered KMeans clusters (silhouette score "
        f"{manifest['lane_cluster_silhouette']}) using distance/route-exposure/chokepoint/volatility features.",
        styles["Body"]))
    story.append(Spacer(1, 6))

    region_summary = out["region_summary"].copy()
    region_disp = region_summary.copy()
    region_disp["region_beta"] = region_disp["region_beta"].map(lambda v: f"{v:.2f}")
    region_disp["premium_min"] = region_disp["premium_min"].map(lambda v: f"{v:+.2f}")
    region_disp["premium_median"] = region_disp["premium_median"].map(lambda v: f"{v:+.2f}")
    region_disp["premium_max"] = region_disp["premium_max"].map(lambda v: f"{v:+.2f}")
    region_headers = ["Region", "# Lanes", "Region Beta", "Min Premium", "Median Premium", "Max Premium"]
    story.append(build_lane_table(
        region_disp, "",
        ["region", "n_lanes", "region_beta", "premium_min", "premium_median", "premium_max"],
        region_headers
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Table: regional pass-through beta (how much the whole corridor co-moves with the index) and the "
        "spread of individual lane premiums within each region (how much within-corridor heterogeneity a "
        "flat per-lane model would otherwise hide).", styles["Caption"]))
    story.append(PageBreak())
    story.append(img("seasonal_calendar_v2.png"))
    story.append(Paragraph(
        "Figure 6c. Region-specific festival/holiday seasonal premium for each forecast month — the explicit "
        "\u201cwhich festivals move which regions, and when\u201d view. See seasonal_calendar_summary.csv for "
        "the named events behind each cell.", styles["Caption"]))
    story.append(PageBreak())
    story.append(img("region_rates_v2.png"))
    story.append(Paragraph("Figure 7. Current average freight rate by destination region (all India origins).", styles["Caption"]))
    story.append(PageBreak())
    story.append(img("top_movers_v2.png"))
    story.append(Paragraph("Figure 8. Destinations with the largest forecast 3-month rate moves, model median case.", styles["Caption"]))
    story.append(PageBreak())

    # ---------------- Section 5: Selected lane detail ----------------
    story.append(Paragraph("5. Selected Lane Detail (Model Median Case, 3-Month Horizon)", styles["H2c"]))
    h3_sorted = h3.sort_values("pct_change")
    cols = ["origin", "destination", "current_rate_usd_feu", "forecast_rate_usd_feu", "pct_change"]
    headers = ["Origin", "Destination", "Current (USD/FEU)", "3-mo Forecast (USD/FEU)", "% Change"]

    story.append(Paragraph("Largest projected declines", styles["Body"]))
    story.append(Spacer(1, 3))
    top_decl = h3_sorted.head(10).copy()
    top_decl["pct_change"] = top_decl["pct_change"].map(lambda v: f"{v:.1f}%")
    top_decl["current_rate_usd_feu"] = top_decl["current_rate_usd_feu"].map(lambda v: f"${v:,.0f}")
    top_decl["forecast_rate_usd_feu"] = top_decl["forecast_rate_usd_feu"].map(lambda v: f"${v:,.0f}")
    story.append(build_lane_table(top_decl, "", cols, headers))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Largest projected increases", styles["Body"]))
    story.append(Spacer(1, 3))
    top_incr = h3_sorted.tail(10).iloc[::-1].copy()
    top_incr["pct_change"] = top_incr["pct_change"].map(lambda v: f"+{v:.1f}%")
    top_incr["current_rate_usd_feu"] = top_incr["current_rate_usd_feu"].map(lambda v: f"${v:,.0f}")
    top_incr["forecast_rate_usd_feu"] = top_incr["forecast_rate_usd_feu"].map(lambda v: f"${v:,.0f}")
    story.append(build_lane_table(top_incr, "", cols, headers))
    story.append(PageBreak())

    # ---------------- Section 6: Origins ----------------
    story.append(Paragraph("6. India Origins", styles["H2c"]))
    origins = pd.read_csv(os.path.join(os.path.dirname(OUT_DIR), "data", "origins.csv"))
    origins_disp = origins.copy()
    origins_disp["inland_surcharge_usd_feu"] = origins_disp["inland_surcharge_usd_feu"].map(lambda v: f"${v:,.0f}")
    story.append(build_lane_table(
        origins_disp, "", list(origins_disp.columns),
        ["Origin", "State", "Gateway Port", "Inland Surcharge (USD/FEU)"]
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "ICD Tihi and ICD Dhannad both gateway through Mundra; inland haulage surcharges reflect rail/trucking "
        "distance from each ICD to the port. Nhava Sheva (JNPT) is itself a gateway port, so no inland surcharge "
        "applies.", styles["Body"]))
    story.append(PageBreak())

    # ---------------- Section 7: Backtesting ----------------
    story.append(Paragraph("7. Backtesting — Rolling-Origin Validation", styles["H2c"]))
    meta = metrics["_meta"]
    story.append(Paragraph(
        "The learned model is validated with rolling-origin (walk-forward) backtesting: at each step, the "
        "model is trained only on data available up to that point and scored on the next unseen month. "
        "It is benchmarked against naive persistence and a 3-month moving average.", styles["Body"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"<b>Fold detail:</b> {meta['n_total_obs']} total months in history; expanding window starts at "
        f"{meta['n_train_initial']} training months and walks forward through {meta['n_folds']} test folds "
        f"(one held-out month per fold). MAPE_CI95 below is a fold-resampling bootstrap 95% interval around "
        f"each MAPE — with only {meta['n_folds']} folds this interval is wide by construction; read it as a "
        f"reminder of how much a single new data point could move the headline number, not as a tight bound.",
        styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(img("backtest_v2.png"))
    story.append(Paragraph("Figure 9. Rolling-origin backtest: learned model vs. naive persistence vs. moving average.", styles["Caption"]))
    story.append(Spacer(1, 6))
    bt_rows = [["Model", "MAPE", "MAPE 95% CI", "MAE", "RMSE", "Directional Accuracy"]]
    for name, label in [("learned_model", "Learned model"), ("naive_persistence", "Naive persistence"), ("moving_avg_3", "3-month moving avg")]:
        m = metrics[name]
        ci_txt = f"[{m['MAPE_CI95_lower']:.1f}%, {m['MAPE_CI95_upper']:.1f}%]" if m["MAPE_CI95_lower"] is not None else "n/a"
        bt_rows.append([label, f"{m['MAPE']:.2f}%", ci_txt, f"{m['MAE']:.2f}", f"{m['RMSE']:.2f}", f"{m['DirectionalAccuracy_%']:.0f}%"])
    bt_table = Table(bt_rows, hAlign="LEFT")
    bt_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHTGRAY]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(bt_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("Prediction interval calibration", styles["H2c"]))
    story.append(Paragraph(
        "Do the 5th\u201395th and 25th\u201375th percentile bands actually contain the realized value as often "
        "as their nominal coverage claims? This retrospectively rebuilds the same bootstrap-ensemble interval "
        "used for the live forecast at each backtest fold (trained only on data available at that point) and "
        "checks whether the true next-month IOFI fell inside it.", styles["Body"]))
    story.append(Spacer(1, 6))
    calib = out["calibration"]
    calib_rows = [["Interval", "Nominal Coverage", "Observed Coverage", "Folds Inside / Total"]]
    for key, v in calib.items():
        if key == "_meta":
            continue
        calib_rows.append([
            key, f"{v['nominal_coverage_%']}%",
            f"{v['observed_coverage_%']}%" if v['observed_coverage_%'] is not None else "n/a",
            f"{v['n_inside']} / {v['n_total_folds']}",
        ])
    calib_table = Table(calib_rows, hAlign="LEFT")
    calib_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHTGRAY]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(calib_table)
    story.append(Spacer(1, 4))
    story.append(Paragraph(calib.get("_meta", {}).get("note", ""), styles["Caption"]))
    story.append(PageBreak())

    # ---------------- Section 8: Methodology & Limitations ----------------
    story.append(Paragraph("8. Methodology &amp; Limitations", styles["H2c"]))
    story.append(Paragraph("<b>Index construction</b>", styles["Body"]))
    story.append(Paragraph(
        f"The IOFI is a composite of {manifest['n_driver_cols']} macro/structural drivers -- the original ten "
        "(oil, war-risk premium, GPR, congestion, vessel idle capacity, trade volume growth, USD index, "
        "INR/USD, Panama restriction, EU ETS carbon) plus two China-side supply drivers added in v2.1 "
        "(china_pmi_idx, china_export_container_idx) -- engineered into lags, rolling statistics, momentum, "
        "an expanded festival/holiday seasonality calendar, and nonlinear interaction terms "
        "(Oil\u00d7Congestion, WarRisk\u00d7Panama, TradeVolume\u00d7IdleCapacity, USD\u00d7Oil, GPR\u00d7Panama, "
        "and the new China PMI\u00d7Congestion / China Exports\u00d7IdleCapacity). Weights are learned \u2014 not "
        "assigned \u2014 via an ensemble of Ridge, ElasticNet, Bayesian Ridge and XGBoost, cross-checked with "
        "SHAP.", styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Festival/holiday calendar \u2014 expanded and region-aware</b>", styles["Body"]))
    story.append(Paragraph(
        "v2.0 carried a single global seasonality calendar anchored mostly to US demand timing (Black Friday, "
        "Christmas, US import season) plus two China-side supply events (Chinese New Year, Golden Week) as "
        "flat 0-1 monthly intensity features for the composite index. v2.1 keeps those as index-level "
        "features and adds Diwali, Ganesh Chaturthi, Eid al-Fitr, Eid al-Adha, Thanksgiving, and Singles Day "
        "(the China-side demand counterpart to Black Friday) to the calendar. It also adds a second, "
        "explicitly region-aware layer (REGION_SEASONAL_SENSITIVITY in feature_engineering.py) that maps each "
        "destination region to the festivals that actually move demand or supply for it -- Ramadan/Eid for "
        "Middle East and Red Sea/East Africa, Diwali for Indian Ocean Rim, Chinese New Year/Golden Week/"
        "Singles Day for Intra-Asia, Christmas/Black Friday/Thanksgiving for the US- and Europe-bound regions "
        "-- so a lane's forecast reflects the festival calendar relevant to where it's actually going, not a "
        "single blended seasonality applied uniformly across all 180 lanes. This region-aware layer feeds the "
        "capped seasonal premium described below in Lane-level translation.", styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Forecast methodology</b>", styles["Body"]))
    story.append(Paragraph(
        "Each driver is forecast independently using ARIMA, ETS (damped trend), or a Kalman-filter local "
        "linear trend model (the lightweight Bayesian Structural Time Series analogue used here), with the "
        "lowest-AIC model auto-selected per driver — no manual decay curve is assumed. The forecasted driver "
        "paths are passed through the learned index model to produce the composite IOFI median path, and a "
        "400-iteration bootstrap ensemble produces the 5th–95th percentile bands.", styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Lane-level translation — hierarchical Global \u2192 Regional \u2192 Seasonal \u2192 Lane</b>", styles["Body"]))
    story.append(Paragraph(
        "Each lane's implied pass-through beta is calibrated directly from its own historical move relative "
        "to the index move that produced it (as in v2.0), then split into two layers: a <b>regional beta</b> "
        "(median lane beta within that destination region — the corridor-wide co-movement) and a "
        "<b>lane premium</b> (this lane's own beta minus its regional beta — how much it deviates from its "
        "corridor). The regional and lane-premium moves sum to the same total forecast as a flat per-lane "
        "beta, but the split is the structural piece requested for a future model: once monthly lane-level "
        "snapshots accumulate into a panel, the SAME regional grouping can be re-estimated as genuine "
        "fixed-effects or hierarchical-Bayes regional coefficients instead of a median-of-betas proxy — only "
        "the estimation method changes, not the Global \u2192 Regional \u2192 Lane structure itself. Lanes are "
        "additionally assigned to automatically discovered KMeans clusters for the driver-similarity view in "
        "Section 4, though — as in v2.0 — a genuine per-cluster regression sensitivity is not identifiable "
        "from a single cross-sectional snapshot (fitting Ridge per cluster on one time point returns "
        "near-zero R\u00b2 because the macro drivers do not vary within it); this remains the primary open item "
        "on the roadmap.", styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "v2.1 adds a fourth component, the <b>regional seasonal premium</b>: for each lane's destination "
        "region and forecast month, it sums (region\u2019s sensitivity to a festival) \u00d7 (that festival's "
        "calendar intensity that month) across every active festival, then scales the result to a percentage "
        "capped at \u00b12.5% per month (hierarchical.py: regional_seasonal_pct). This is deliberately a "
        "transparent rule-based adjustment, not a learned one — with 22-24 months of history there isn't "
        "enough repetition of any single festival to fit its region-specific price impact directly, so the "
        "sensitivity weights (REGION_SEASONAL_SENSITIVITY in feature_engineering.py) are assigned by trade "
        "logic (Ramadan matters for Middle East demand, Golden Week matters for Intra-Asia capacity, etc.) "
        "rather than estimated. As monthly lane-level history accumulates, these can be replaced by "
        "region\u00d7festival coefficients fit the same way the regional beta itself will eventually be "
        "fit — same roadmap item as above.", styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>XGBoost at small sample size</b>", styles["Body"]))
    xgb_rel = manifest.get("xgb_reliability") or {}
    story.append(Paragraph(
        f"At n={xgb_rel.get('n_training_obs', manifest['n_training_months'])} training months, boosted trees "
        "are the estimator in this ensemble most prone to overfitting — linear/Bayesian shrinkage models "
        "degrade more gracefully at this sample size. Rather than remove XGBoost outright, its ensemble vote "
        f"is shrunk by a sample-size-aware factor ({xgb_rel.get('shrinkage_weight_applied', 'n/a')}x here; "
        f"reaching full weight requires n\u2265{xgb_rel.get('full_reliable_n', 60)} months), since it remains "
        "the only estimator that captures nonlinear driver interactions without those interactions being "
        "hand-specified as products. Unshrunk XGBoost importances are retained in learned_driver_weights.csv.",
        styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Interval calibration</b>", styles["Body"]))
    story.append(Paragraph(
        "Prediction intervals are also checked, not just reported: Section 7 retrospectively rebuilds the "
        "bootstrap-ensemble interval at each backtest fold and measures how often the realized value actually "
        "fell inside it, against the interval's nominal coverage. As with the backtest MAPE, this calibration "
        "check runs on a small number of folds and should be read directionally.", styles["Body"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Limitations</b>", styles["Body"]))
    story.append(Paragraph(
        f"This model is a directional planning tool, trained on {manifest['n_training_months']} months of "
        "history, not a substitute for a licensed rate benchmarking subscription or actual carrier quotes. "
        "Only 2 of 60 destination base rates are anchored to a real published tariff; the rest remain "
        "calibrated planning estimates. With this little history, deep sequence models (LSTM, Temporal Fusion "
        "Transformer, N-BEATS) were evaluated and rejected in favor of the Ridge/ElasticNet/Bayesian/XGBoost "
        "hybrid — see ARCHITECTURE.md for the full comparison and roadmap. Treat all lane-level dollar figures "
        "as directional, not quotable, rates.", styles["Body"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Report generated from the IOFI v2 reusable pipeline "
        "(pipeline_runner.py → report_data.py → charts_v2.py → build_report_v2.py). See README.md and "
        "ARCHITECTURE.md for the full model design and monthly refresh process.", styles["Caption"]))
    story.append(PageBreak())

    # ---------------- Appendix: full lane table ----------------
    story.append(Paragraph("Appendix: Full Lane-Level Forecast (Model Median Case, USD/FEU)", styles["H2c"]))
    story.append(Paragraph(
        "All 180 origin-destination lanes (3 origins × 60 destinations), 3-month-ahead model median forecast. "
        "5th/95th percentile detail for every lane is in lane_forecasts_v2.csv alongside this report.",
        styles["Body"]))
    story.append(Spacer(1, 6))
    n_ai_lanes = int((h3["rate_source"] == "ai").sum())
    if n_ai_lanes:
        story.append(Paragraph(
            f"{n_ai_lanes}/{len(h3)} lanes below are AI-checked (Source = AI): cross-referenced against an "
            "LLM's estimate of realistic current market rates, replacing the model's calibrated planning "
            "estimate for that lane. Remaining lanes (Source = Model) still use the calibrated estimate -- "
            "run pipeline/ai_rate_lookup.py to extend AI coverage. AI figures are still directional planning "
            "estimates, not carrier quotes.", styles["Body"]))
        story.append(Spacer(1, 6))
    app = h3.sort_values(["region", "origin", "destination"]).copy()
    app_cols = ["origin", "destination", "region", "current_rate_usd_feu", "forecast_rate_usd_feu", "pct_change"]
    app_headers = ["Origin", "Destination", "Region", "Current", "3-mo Forecast", "% Change"]
    if n_ai_lanes:
        app_cols.append("rate_source")
        app_headers.append("Source")
        app["rate_source"] = app["rate_source"].map({"ai": "AI", "model": "Model"})
    app_disp = app[app_cols].copy()
    app_disp["current_rate_usd_feu"] = app_disp["current_rate_usd_feu"].map(lambda v: f"${v:,.0f}")
    app_disp["forecast_rate_usd_feu"] = app_disp["forecast_rate_usd_feu"].map(lambda v: f"${v:,.0f}")
    app_disp["pct_change"] = app_disp["pct_change"].map(lambda v: f"{'+' if v >= 0 else ''}{v:.1f}%")
    story.append(build_lane_table(app_disp, "", app_cols, app_headers))

    doc.build(story)
    return doc_path


if __name__ == "__main__":
    path = build()
    print("Saved:", path)
