"""Tests for live simulation rules and project-data integration."""

from pathlib import Path

import pandas as pd
import pytest

from src.dashboard_data import load_dashboard_data
from src.db import DB_PATH, query
from src.simulator import _risk_band, _rule_score, generate_event


def test_risk_band_boundaries():
    assert _risk_band(39.9) == "Low"
    assert _risk_band(40.0) == "Medium"
    assert _risk_band(69.9) == "Medium"
    assert _risk_band(70.0) == "High"


def test_structuring_rule_creates_high_risk_signal():
    event = {
        "account_id": "A",
        "amount": 9_750.0,
        "timestamp": "2026-09-15T12:00:00",
        "is_international": 0,
        "scenario": "structuring",
    }
    score, reasons = _rule_score(event, pd.Series({"is_dormant": 0}), [])
    assert score >= 70
    assert any("threshold" in reason for reason in reasons)


@pytest.mark.skipif(not Path(DB_PATH).exists(), reason="generated project database is not present")
def test_generated_events_are_temporary_and_south_african():
    load_dashboard_data.cache_clear()
    data = load_dashboard_data()
    before = int(query("SELECT COUNT(*) AS count FROM transactions")["count"].iloc[0])
    events = []
    for tick in range(1, 16):
        events.append(generate_event(data, tick=tick, prior_events=events))
    after = int(query("SELECT COUNT(*) AS count FROM transactions")["count"].iloc[0])

    supported_provinces = {
        "Eastern Cape", "Free State", "Gauteng", "KwaZulu-Natal", "Limpopo",
        "Mpumalanga", "North West", "Northern Cape", "Western Cape",
    }
    assert before == after
    assert all(event["province"] in supported_provinces for event in events)
    assert all(0 <= event["risk_score"] <= 100 for event in events)
    assert any(event["is_alert"] for event in events)
