"""Risk scoring module — converts raw anomaly scores to human-readable risk bands.

Transforms Isolation Forest decision_function outputs into a 0–100 risk score
using min-max scaling (inverted, since lower raw scores = higher risk), then
assigns categorical risk bands for analyst triage.
"""

import numpy as np
import pandas as pd


# Risk band thresholds
RISK_BANDS = {
    "Low": (0, 39),
    "Medium": (40, 69),
    "High": (70, 100),
}


def compute_risk_scores(anomaly_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw anomaly scores to a 0–100 risk score with risk bands.

    Applies inverted min-max scaling: the most negative anomaly_score
    (most anomalous) maps to 100, the most positive maps to 0.

    Args:
        anomaly_df: DataFrame from score_anomalies() with columns
            'account_id' and 'anomaly_score'.

    Returns:
        DataFrame with columns: account_id, anomaly_score, risk_score, risk_band.
    """
    scores = anomaly_df["anomaly_score"].values.copy()

    # Invert: more negative raw score → higher risk
    min_score = scores.min()
    max_score = scores.max()

    if max_score == min_score:
        # Edge case: all scores identical
        risk_scores = np.full_like(scores, 50.0)
    else:
        # Invert so most anomalous (most negative) → 100
        risk_scores = (1 - (scores - min_score) / (max_score - min_score)) * 100

    risk_scores = np.round(risk_scores, 1)

    result = anomaly_df.copy()
    result["risk_score"] = risk_scores
    result["risk_band"] = result["risk_score"].apply(_assign_band)

    return result[["account_id", "anomaly_score", "risk_score", "risk_band"]]


def _assign_band(score: float) -> str:
    """Assign a risk band label based on the numeric risk score.

    Args:
        score: Risk score between 0 and 100.

    Returns:
        One of 'Low', 'Medium', 'High'.
    """
    for band, (low, high) in RISK_BANDS.items():
        if low <= score <= high:
            return band
    return "High"  # Scores above 100 edge case
