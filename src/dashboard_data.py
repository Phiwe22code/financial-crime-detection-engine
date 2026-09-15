"""Shared data preparation for the dashboard interfaces.

The dashboard is deliberately kept separate from the analytics modules.  This
loader executes the existing pipeline once and exposes the resulting objects to
Dash callbacks and the live simulator.
"""

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.db import query
from src.explain import generate_explanations
from src.features import build_features
from src.model import score_anomalies, train_model
from src.scoring import compute_risk_scores


@dataclass(frozen=True)
class DashboardData:
    """All immutable baseline data needed by the dashboard."""

    transactions: pd.DataFrame
    accounts: pd.DataFrame
    customers: pd.DataFrame
    merchants: pd.DataFrame
    features: pd.DataFrame
    anomaly_scores: pd.DataFrame
    risk: pd.DataFrame
    explanations: pd.DataFrame
    model: IsolationForest
    scaler: StandardScaler


@lru_cache(maxsize=1)
def load_dashboard_data() -> DashboardData:
    """Run the existing analytics pipeline once and cache its outputs."""
    transactions = query("SELECT * FROM transactions")
    accounts = query("SELECT * FROM accounts")
    customers = query("SELECT * FROM customers")
    merchants = query("SELECT * FROM merchants")

    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
    features = build_features(transactions, accounts)
    model, scaler = train_model(features)
    anomaly_scores = score_anomalies(model, scaler, features)
    risk = compute_risk_scores(anomaly_scores).merge(
        accounts[["account_id", "customer_id", "status", "is_dormant"]],
        on="account_id",
        how="left",
    )
    explanations = generate_explanations(risk, features)

    return DashboardData(
        transactions=transactions,
        accounts=accounts,
        customers=customers,
        merchants=merchants,
        features=features,
        anomaly_scores=anomaly_scores,
        risk=risk,
        explanations=explanations,
        model=model,
        scaler=scaler,
    )


def alert_queue(data: DashboardData) -> pd.DataFrame:
    """Return the high-risk queue enriched with total exposure."""
    spend = (
        data.transactions.groupby("account_id")["amount"]
        .sum()
        .rename("total_exposure")
    )
    queue = data.explanations.merge(spend, on="account_id", how="left")
    queue["account"] = queue["account_id"].str[:8] + "..."
    return queue.sort_values("risk_score", ascending=False).reset_index(drop=True)
