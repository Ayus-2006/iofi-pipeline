import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from report_data import assemble, OUT_DIR

CHART_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

NAVY = "#1a2b4c"
RED = "#c0392b"
TEAL = "#1abc9c"
GOLD = "#d4a017"
GREEN_FILL = "#d5f0ea"

plt.rcParams["font.size"] = 10


def chart_history_forecast(iofi_hist, iofi_q):
    hist_months = pd.to_datetime(iofi_hist["month"])
    hist_vals = iofi_hist["IOFI"]
    fc_months = pd.to_datetime(iofi_q["month"])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(hist_months, hist_vals, color=NAVY, linewidth=2, label="IOFI (history)")

    connect_m = pd.concat([hist_months.tail(1), fc_months])
    connect_med = pd.concat([hist_vals.tail(1), iofi_q["median_model_path"]])
    connect_p05 = pd.concat([hist_vals.tail(1), iofi_q["0.05"] if "0.05" in iofi_q else iofi_q[0.05]])
    connect_p95 = pd.concat([hist_vals.tail(1), iofi_q["0.95"] if "0.95" in iofi_q else iofi_q[0.95]])

    ax.plot(connect_m, connect_med, color=TEAL, linewidth=2, linestyle="--", label="Learned model — median forecast")
    ax.fill_between(connect_m, connect_p05, connect_p95, color=GREEN_FILL, alpha=0.7, label="5th–95th percentile band")
    ax.plot(connect_m, connect_p95, color=RED, linewidth=1, linestyle=":", label="95th percentile")
    ax.plot(connect_m, connect_p05, color="gray", linewidth=1, linestyle=":", label="5th percentile")
    ax.axvline(hist_months.iloc[-1], color="gray", linewidth=0.8)

    ax.set_title("India Ocean Freight Index (IOFI) — History & Statistically Learned 3-Month Forecast")
    ax.set_ylabel("Index (base 100)")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/iofi_forecast_v2.png", dpi=150)
    plt.close(fig)


def chart_driver_attribution(importance):
    top = importance.head(10).copy()
    top["label"] = top["feature"].str.replace("_", " ").str.title()
    top = top.sort_values("ensemble_weight")
    colors = [RED if w >= 0 else TEAL for w in top["ensemble_weight"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top["label"], top["ensemble_weight"], color=colors)
    for b, w in zip(bars, top["ensemble_weight"]):
        ax.text(b.get_width() + 0.002, b.get_y() + b.get_height() / 2, f"{w:.3f}",
                va="center", fontsize=8)
    ax.set_title("Learned Driver Attribution — Top 10 Features (Ridge/ElasticNet/Bayesian/XGBoost ensemble)")
    ax.set_xlabel("Ensemble-normalized importance")
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/driver_attribution_v2.png", dpi=150)
    plt.close(fig)


def chart_driver_pair(macro, col_a, col_b, label_a, label_b, color_a, color_b, fname, title):
    months = pd.to_datetime(macro["month"])
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    ax1.plot(months, macro[col_a], color=color_a, linewidth=2, label=label_a)
    ax2.plot(months, macro[col_b], color=color_b, linewidth=2, label=label_b)
    ax1.set_ylabel(label_a, color=color_a)
    ax2.set_ylabel(label_b, color=color_b)
    ax1.tick_params(axis="y", labelcolor=color_a)
    ax2.tick_params(axis="y", labelcolor=color_b)
    ax1.set_title(title)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b'%y"))
    fig.autofmt_xdate(rotation=45)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/{fname}.png", dpi=150)
    plt.close(fig)


def chart_region_avg_rates(lanes):
    region_avg = lanes.groupby("region")["current_rate_usd_feu"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(region_avg.index, region_avg.values, color=NAVY)
    for b, v in zip(bars, region_avg.values):
        ax.text(v + 20, b.get_y() + b.get_height() / 2, f"${v:,.0f}", va="center", fontsize=8)
    ax.set_title("Current Average Freight Rate by Destination Region")
    ax.set_xlabel("Avg. current rate, USD/FEU (all India origins)")
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/region_rates_v2.png", dpi=150)
    plt.close(fig)


def chart_top_movers(lane_forecasts):
    h3 = lane_forecasts[lane_forecasts["horizon"] == 3].copy()
    by_dest = h3.groupby("destination")["pct_change"].mean().sort_values()
    biggest = by_dest.head(14)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(biggest.index, biggest.values, color=TEAL)
    ax.set_title("Destinations With the Largest Forecast 3-Month Rate Moves (Model Median Case)")
    ax.set_xlabel("Forecast 3-month rate change, % (median case)")
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/top_movers_v2.png", dpi=150)
    plt.close(fig)


def chart_backtest(backtest_res, backtest_metrics):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(backtest_res["t"], backtest_res["y_true"], color=NAVY, linewidth=2, marker="o", markersize=3, label="Actual IOFI")
    ax.plot(backtest_res["t"], backtest_res["pred_model"], color=TEAL, linewidth=2, marker="o", markersize=3, label="Learned model (rolling-origin)")
    ax.plot(backtest_res["t"], backtest_res["pred_naive"], color="gray", linestyle="--", linewidth=1.3, label="Naive persistence")
    ax.plot(backtest_res["t"], backtest_res["pred_ma"], color=GOLD, linestyle=":", linewidth=1.3, label="3-month moving average")
    ax.set_title("Rolling-Origin Backtest — Learned Model vs. Benchmarks")
    ax.set_xlabel("Time step (month index)")
    ax.set_ylabel("IOFI")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/backtest_v2.png", dpi=150)
    plt.close(fig)


def chart_seasonal_calendar(seasonal_summary, future_months):
    regions = seasonal_summary["region"].tolist()
    vals = seasonal_summary[[f"{m}_seasonal_pct" for m in future_months]].values
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(vals, cmap="YlOrRd", aspect="auto", vmin=0)
    ax.set_xticks(range(len(future_months)))
    ax.set_xticklabels(future_months)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=8)
    for i in range(len(regions)):
        for j in range(len(future_months)):
            v = vals[i, j]
            ax.text(j, i, f"{v:.1f}%" if v > 0 else "-", ha="center", va="center",
                    fontsize=8, color="white" if v > 1.0 else "black")
    ax.set_title("Region-Specific Festival/Holiday Seasonal Premium by Forecast Month")
    fig.colorbar(im, ax=ax, label="Seasonal rate premium, %")
    fig.tight_layout()
    fig.savefig(f"{CHART_DIR}/seasonal_calendar_v2.png", dpi=150)
    plt.close(fig)


def generate_all(out):
    chart_history_forecast(out["iofi_hist"], out["iofi_quantiles"])
    chart_driver_attribution(out["importance"])
    chart_driver_pair(out["macro"], "brent_usd_bbl", "gpr_index", "Brent crude, USD/bbl", "GPR index",
                       GOLD, RED, "oil_vs_gpr_v2", "Oil Price vs. Geopolitical Risk Index")
    chart_driver_pair(out["macro"], "congestion_index", "vessel_idle_capacity_pct", "Congestion index", "Vessel idle capacity, %",
                       TEAL, "#8e44ad", "congestion_vs_idle_v2", "Port Congestion vs. Vessel Idle Capacity")
    chart_driver_pair(out["macro"], "war_risk_premium_idx", "panama_restriction_idx", "War-risk premium index", "Panama restriction index",
                       RED, GOLD, "warrisk_vs_panama_v2", "War-Risk Premium vs. Panama Draft Restriction")
    chart_driver_pair(out["macro"], "trade_volume_growth_idx", "inr_usd_rate", "Trade volume growth, %", "INR/USD rate",
                       TEAL, "#5d4037", "tradevol_vs_inr_v2", "Global Trade Volume Growth vs. INR/USD")
    chart_driver_pair(out["macro"], "china_pmi_idx", "china_export_container_idx", "China PMI-like index", "China export container throughput idx",
                       "#c0392b", "#2980b9", "china_pmi_vs_exports_v2", "China Manufacturing PMI vs. Export Container Throughput")
    chart_region_avg_rates(out["lanes"])
    chart_top_movers(out["lane_forecasts"])
    chart_backtest(out["backtest_res"], out["backtest_metrics"])
    chart_seasonal_calendar(out["seasonal_summary"], out["manifest"]["future_months"])
    print("All charts generated in", CHART_DIR)


if __name__ == "__main__":
    out = assemble()
    generate_all(out)
