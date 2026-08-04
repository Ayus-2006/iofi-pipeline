"""
Every prediction should ship with: top-5 contributing variables, a SHAP-based
feature contribution table, a natural-language explanation, and a confidence
score. This module assembles that bundle from a fitted model + a data row.
"""
import numpy as np
import pandas as pd


def confidence_score(interval_width: float, historical_avg_width: float) -> float:
    """
    Simple, bounded confidence heuristic: narrower-than-average intervals
    imply higher confidence. Returned as a 0-100 score.
    """
    if historical_avg_width <= 0:
        return 50.0
    ratio = interval_width / historical_avg_width
    score = 100 * np.exp(-0.7 * max(0, ratio - 1))
    return float(np.clip(score, 5, 99))


def build_explanation_bundle(top5_df: pd.DataFrame, point_forecast: float,
                              lower: float, upper: float, conf_score: float,
                              target_name: str = "IOFI"):
    lines = [f"The forecast for {target_name} is {point_forecast:.1f} "
             f"(90% range: {lower:.1f} to {upper:.1f})."]
    if top5_df is not None and len(top5_df) > 0:
        lines.append("Key contributors:")
        for _, row in top5_df.iterrows():
            direction = "pushing it up" if row.get("shap_value", row.iloc[1]) > 0 else "pulling it down"
            lines.append(f"  - {row['feature']}: {direction}")
    lines.append(f"Confidence score: {conf_score:.0f}/100.")
    narrative = "\n".join(lines)

    return {
        "point_forecast": point_forecast,
        "interval": (lower, upper),
        "confidence_score": conf_score,
        "top5_contributors": top5_df.to_dict(orient="records") if top5_df is not None else [],
        "narrative": narrative,
    }


if __name__ == "__main__":
    demo_top5 = pd.DataFrame({
        "feature": ["brent_usd_bbl_lag1", "congestion_index", "war_risk_premium_idx",
                    "int_oil_congestion", "usd_index"],
        "shap_value": [1.8, 1.2, -0.9, 0.7, -0.4],
    })
    bundle = build_explanation_bundle(demo_top5, 104.2, 98.1, 110.6, conf_score=72)
    print(bundle["narrative"])
