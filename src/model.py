"""Anomaly detection model for the Financial Crime Detection Engine.

Uses scikit-learn's Isolation Forest for unsupervised anomaly detection.
"""

# -----------------------------------------------------------------------
# Why Isolation Forest?
# -----------------------------------------------------------------------
# 1. **Unsupervised**: We have no labelled fraud data in production — only
#    synthetically injected flags.  Isolation Forest doesn't need labels.
# 2. **Efficient on tabular data**: Scales well to our ~700-account feature
#    matrix and would handle much larger datasets without issue.
# 3. **Interpretable enough**: Feature importances can be approximated via
#    average path length contributions, making it explainable at a level
#    appropriate for a junior-analyst portfolio project.
# 4. **Robust to contamination**: The contamination parameter lets us hint
#    at the expected fraud rate (~3 %) without hard labels.
# -----------------------------------------------------------------------

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def train_model(
    features_df: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42,
    n_estimators: int = 200,
) -> tuple[IsolationForest, StandardScaler]:
    """Train an Isolation Forest on the engineered feature matrix.

    Features are standardised before training so that all dimensions
    contribute equally to the isolation splits.

    Args:
        features_df: Feature matrix from build_features() (indexed by account_id).
        contamination: Expected proportion of anomalies (matches our ~3 % injection rate).
        random_state: Random seed for reproducibility.
        n_estimators: Number of isolation trees.

    Returns:
        Tuple of (trained IsolationForest model, fitted StandardScaler).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_df)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    return model, scaler


def score_anomalies(
    model: IsolationForest,
    scaler: StandardScaler,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Score each account using the trained Isolation Forest.

    Returns raw anomaly scores from decision_function() (more negative =
    more anomalous) alongside the binary prediction.

    Args:
        model: Trained IsolationForest.
        scaler: Fitted StandardScaler used during training.
        features_df: Feature matrix (same columns as training data).

    Returns:
        DataFrame with columns: account_id, anomaly_score, is_anomaly.
        anomaly_score is the raw decision_function output (lower = more anomalous).
        is_anomaly is 1 for predicted anomalies, 0 otherwise.
    """
    X_scaled = scaler.transform(features_df)

    raw_scores = model.decision_function(X_scaled)
    predictions = model.predict(X_scaled)  # 1 = normal, -1 = anomaly

    results = pd.DataFrame({
        "account_id": features_df.index,
        "anomaly_score": raw_scores,
        "is_anomaly": (predictions == -1).astype(int),
    })

    return results
