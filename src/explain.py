"""Plain-language explanation generator for flagged accounts.

For each high-risk account (risk_score >= 70), generates a human-readable
explanation by identifying which engineered features are statistical
outliers (z-score > 2) relative to the full population.

This is rule-based logic — no ML-based explainability (SHAP, LIME)
is used. This keeps the explanations transparent and auditable,
which is critical for AML compliance reporting.
"""

from typing import Optional

import numpy as np
import pandas as pd


# Human-readable descriptions for each feature
FEATURE_DESCRIPTIONS = {
    "transaction_count": "transaction volume {pct:.0f}% above average",
    "avg_amount": "average transaction amount {pct:.0f}% above normal",
    "std_amount": "highly variable transaction amounts (std dev {pct:.0f}% above normal)",
    "max_daily_spend": "single-day spend peak {pct:.0f}% above average",
    "night_ratio": "{value:.0%} of transactions occur at night (00:00–06:00)",
    "cross_border_ratio": "{value:.0%} of transactions are cross-border",
    "merchant_diversity": "{value:.0f} unique merchants used (unusually {direction})",
    "country_diversity": "{value:.0f} countries transacted in",
    "rolling_7d_avg": "recent 7-day spending average {pct:.0f}% above baseline",
    "account_age_days": "account is only {value:.0f} days old",
}


def compute_zscores(features_df: pd.DataFrame) -> pd.DataFrame:
    """Compute z-scores for each feature across all accounts.

    Args:
        features_df: Feature matrix indexed by account_id.

    Returns:
        DataFrame of z-scores with the same shape and index.
    """
    means = features_df.mean()
    stds = features_df.std()
    # Avoid division by zero
    stds = stds.replace(0, np.nan)
    return (features_df - means) / stds


def explain_account(
    account_id: str,
    features_df: pd.DataFrame,
    zscores_df: pd.DataFrame,
    z_threshold: float = 2.0,
) -> str:
    """Generate a plain-language explanation for a single flagged account.

    Checks which features have z-scores exceeding the threshold and
    constructs a human-readable sentence for each.

    Args:
        account_id: The account to explain.
        features_df: Full feature matrix.
        zscores_df: Pre-computed z-scores.
        z_threshold: Z-score cutoff for 'outlier' status.

    Returns:
        Plain-language explanation string.
    """
    if account_id not in features_df.index:
        return f"Account {account_id}: no feature data available."

    row = features_df.loc[account_id]
    z_row = zscores_df.loc[account_id]
    means = features_df.mean()

    reasons = []

    for feature, z_val in z_row.items():
        if pd.isna(z_val):
            continue
        # Check for high outliers (z > threshold) or low outliers for account_age_days
        if feature == "account_age_days":
            # Young accounts are suspicious → check for negative z-score
            if z_val < -z_threshold:
                desc = FEATURE_DESCRIPTIONS.get(feature, f"{feature} is anomalous")
                reasons.append(desc.format(
                    value=row[feature], pct=0, direction="low"
                ))
        elif abs(z_val) > z_threshold:
            value = row[feature]
            mean_val = means[feature]
            if mean_val > 0:
                pct = ((value - mean_val) / mean_val) * 100
            else:
                pct = 0

            desc = FEATURE_DESCRIPTIONS.get(feature, f"{feature} is anomalous")
            direction = "high" if z_val > 0 else "low"
            try:
                reasons.append(desc.format(
                    value=value, pct=abs(pct), direction=direction
                ))
            except (KeyError, IndexError):
                reasons.append(f"{feature}: z-score = {z_val:.1f}")

    if not reasons:
        return (
            f"Account {account_id[:12]}...: flagged by model ensemble; "
            f"no single feature is a strong outlier — review holistic pattern."
        )

    reason_text = "; ".join(reasons)
    return f"Flagged: {reason_text}."


def generate_explanations(
    risk_df: pd.DataFrame,
    features_df: pd.DataFrame,
    min_risk_score: float = 70.0,
    z_threshold: float = 2.0,
) -> pd.DataFrame:
    """Generate explanations for all high-risk accounts.

    Args:
        risk_df: DataFrame with 'account_id', 'risk_score', 'risk_band'.
        features_df: Feature matrix indexed by account_id.
        min_risk_score: Only explain accounts at or above this score.
        z_threshold: Z-score cutoff for outlier detection.

    Returns:
        DataFrame with columns: account_id, risk_score, risk_band, explanation.
    """
    high_risk = risk_df[risk_df["risk_score"] >= min_risk_score].copy()

    if high_risk.empty:
        print("No accounts meet the risk score threshold.")
        return pd.DataFrame(columns=["account_id", "risk_score", "risk_band", "explanation"])

    zscores_df = compute_zscores(features_df)

    explanations = []
    for _, row in high_risk.iterrows():
        explanation = explain_account(
            row["account_id"], features_df, zscores_df, z_threshold
        )
        explanations.append(explanation)

    high_risk["explanation"] = explanations
    return high_risk[["account_id", "risk_score", "risk_band", "explanation"]]
