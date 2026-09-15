"""Automatic South African transaction-stream simulator.

Simulation events are held in browser session state by the Dash application.
They never write to the project's SQLite database.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.dashboard_data import DashboardData
from src.explain import compute_zscores, explain_account
from src.features import build_features
from src.hotspots import city_locations, province_hotspots


SCENARIO_LABELS = {
    "normal": "Routine activity",
    "structuring": "Possible structuring",
    "cross_border": "Rapid cross-border activity",
    "volume_spike": "Transaction velocity spike",
    "dormant_reactivation": "Dormant account reactivation",
}

RISK_COLORS = {"Low": "#2dd4bf", "Medium": "#fbbf24", "High": "#fb4b64"}


def _risk_band(score: float) -> str:
    if score < 40:
        return "Low"
    if score < 70:
        return "Medium"
    return "High"


def _pick_location(rng: np.random.Generator, suspicious: bool) -> pd.Series:
    provinces = province_hotspots()
    cities = city_locations()
    if suspicious:
        weights = provinces.set_index("province")["credit_share"]
        weights = weights / weights.sum()
        province = rng.choice(weights.index.to_numpy(), p=weights.to_numpy())
    else:
        # Broad national activity distribution used only by the simulator.
        weights = np.array([0.25, 0.16, 0.17, 0.09, 0.08, 0.07, 0.07, 0.08, 0.03])
        province = rng.choice(provinces["province"].to_numpy(), p=weights)
    candidates = cities[cities["province"] == province]
    return candidates.iloc[int(rng.integers(0, len(candidates)))]


def _choose_account(
    data: DashboardData,
    rng: np.random.Generator,
    scenario: str,
) -> pd.Series:
    accounts = data.accounts[data.accounts["currency"] == "ZAR"]
    if scenario == "dormant_reactivation":
        dormant = accounts[accounts["is_dormant"] == 1]
        if not dormant.empty:
            accounts = dormant
    return accounts.iloc[int(rng.integers(0, len(accounts)))]


def _transaction_values(
    rng: np.random.Generator,
    scenario: str,
    tick: int,
) -> dict[str, Any]:
    now = datetime.now().replace(microsecond=0) + timedelta(seconds=tick)
    values: dict[str, Any] = {
        "amount": round(float(np.clip(rng.lognormal(4.2, 1.0), 15, 18_000)), 2),
        "timestamp": now,
        "transaction_type": rng.choice(["purchase", "transfer", "withdrawal", "deposit"]),
        "channel": rng.choice(["online", "ATM", "branch", "mobile"]),
        "is_international": 0,
        "country": "ZA",
    }
    if scenario == "structuring":
        values.update(
            amount=round(float(rng.uniform(9_050, 9_990)), 2),
            transaction_type="transfer",
            channel="mobile",
        )
    elif scenario == "cross_border":
        values.update(
            amount=round(float(rng.uniform(16_000, 42_000)), 2),
            transaction_type="transfer",
            channel="online",
            is_international=1,
            country=rng.choice(["US", "UK", "NG", "KE", "AE"]),
        )
    elif scenario == "volume_spike":
        values.update(
            amount=round(float(rng.uniform(2_500, 8_500)), 2),
            transaction_type="purchase",
            channel="online",
        )
    elif scenario == "dormant_reactivation":
        values.update(
            amount=round(float(rng.uniform(8_000, 30_000)), 2),
            timestamp=now.replace(hour=int(rng.integers(0, 5))),
            transaction_type="transfer",
            channel="mobile",
        )
    return values


def _rule_score(
    event: dict[str, Any],
    account: pd.Series,
    prior_events: list[dict[str, Any]],
) -> tuple[float, list[str]]:
    score = 12.0
    reasons: list[str] = []
    amount = float(event["amount"])
    hour = pd.Timestamp(event["timestamp"]).hour

    if 9_000 <= amount < 10_000:
        score += 58
        reasons.append("amount falls just below the R10,000 monitoring threshold")
    if amount >= 15_000:
        score += 24
        reasons.append("unusually large transaction amount")
    if int(event["is_international"]) == 1:
        score += 28
        reasons.append("high-value cross-border transfer")
    if hour < 6:
        score += 24
        reasons.append("transaction occurred between midnight and 06:00")
    if int(account.get("is_dormant", 0)) == 1:
        score += 42
        reasons.append("activity originated from a dormant account")

    recent = [
        item for item in prior_events
        if item.get("account_id") == event["account_id"]
        and abs(
            (pd.Timestamp(event["timestamp"]) - pd.Timestamp(item["timestamp"]))
            .total_seconds()
        ) <= 3600
    ]
    if len(recent) >= 2 or event["scenario"] == "volume_spike":
        score += 38
        reasons.append("multiple transactions occurred inside a short time window")

    return min(score, 100.0), reasons


def _model_score(
    data: DashboardData,
    account_id: str,
    event: dict[str, Any],
    prior_events: list[dict[str, Any]],
) -> tuple[float, str]:
    base_txns = data.transactions[data.transactions["account_id"] == account_id].copy()
    account_events = [item for item in prior_events if item["account_id"] == account_id]
    account_events.append(event)
    simulated = pd.DataFrame(account_events)
    required = list(base_txns.columns)
    for column in required:
        if column not in simulated:
            simulated[column] = None
    simulated = simulated[required]
    simulated["timestamp"] = pd.to_datetime(simulated["timestamp"])
    combined = pd.concat([base_txns, simulated], ignore_index=True)
    account_row = data.accounts[data.accounts["account_id"] == account_id]
    updated_features = build_features(combined, account_row)

    scaled = data.scaler.transform(updated_features[data.features.columns])
    raw = float(data.model.decision_function(scaled)[0])
    low = float(data.anomaly_scores["anomaly_score"].min())
    high = float(data.anomaly_scores["anomaly_score"].max())
    model_risk = float(np.clip((1 - (raw - low) / (high - low)) * 100, 0, 100))

    comparison = data.features.copy()
    comparison.loc[account_id] = updated_features.loc[account_id]
    explanation = explain_account(
        account_id,
        comparison,
        compute_zscores(comparison),
        z_threshold=2.0,
    )
    return round(model_risk, 1), explanation


def generate_event(
    data: DashboardData,
    tick: int,
    prior_events: list[dict[str, Any]] | None = None,
    seed: int = 2026,
) -> dict[str, Any]:
    """Generate and score one event for the live fraud lab."""
    prior_events = prior_events or []
    rng = np.random.default_rng(seed + tick)
    scenario = rng.choice(
        ["normal", "structuring", "cross_border", "volume_spike", "dormant_reactivation"],
        p=[0.82, 0.06, 0.05, 0.04, 0.03],
    )
    account = _choose_account(data, rng, scenario)
    account_id = str(account["account_id"])
    location = _pick_location(rng, suspicious=scenario != "normal")
    merchant = data.merchants.iloc[int(rng.integers(0, len(data.merchants)))]
    values = _transaction_values(rng, scenario, tick)
    event: dict[str, Any] = {
        "transaction_id": f"SIM-{tick:06d}-{int(rng.integers(1000, 9999))}",
        "account_id": account_id,
        "merchant_id": str(merchant["merchant_id"]),
        "currency": "ZAR",
        "province": str(location["province"]),
        "city": str(location["city"]),
        "lat": float(location["lat"]),
        "lon": float(location["lon"]),
        "scenario": str(scenario),
        "is_flagged": int(scenario != "normal"),
        **values,
    }

    model_risk, model_explanation = _model_score(
        data, account_id, event, prior_events
    )
    rule_risk, reasons = _rule_score(event, account, prior_events)
    risk_score = round(max(model_risk, rule_risk), 1)
    band = _risk_band(risk_score)
    baseline = float(
        data.risk.loc[data.risk["account_id"] == account_id, "risk_score"].iloc[0]
    )

    if reasons:
        explanation = "Flagged: " + "; ".join(reasons) + "."
    else:
        explanation = model_explanation

    event.update(
        timestamp=pd.Timestamp(event["timestamp"]).isoformat(),
        scenario_label=SCENARIO_LABELS[scenario],
        baseline_risk=round(baseline, 1),
        model_risk=model_risk,
        rule_risk=round(rule_risk, 1),
        risk_score=risk_score,
        risk_delta=round(risk_score - baseline, 1),
        risk_band=band,
        risk_color=RISK_COLORS[band],
        explanation=explanation,
        is_alert=band == "High",
    )
    return event
