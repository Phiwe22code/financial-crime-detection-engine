"""Unit tests for feature engineering functions.

Tests use small hand-crafted DataFrames with known expected outputs
to verify correctness of individual feature computations.
"""

import pandas as pd
import numpy as np
import pytest

from src.features import (
    transaction_count,
    avg_amount,
    std_amount,
    night_ratio,
    cross_border_ratio,
    merchant_diversity,
    max_daily_spend,
    build_features,
)


# --- Fixtures ---

@pytest.fixture
def sample_transactions():
    """Small hand-crafted transaction DataFrame."""
    return pd.DataFrame({
        "transaction_id": ["t1", "t2", "t3", "t4", "t5", "t6"],
        "account_id": ["A", "A", "A", "B", "B", "B"],
        "merchant_id": ["m1", "m2", "m1", "m1", "m1", "m3"],
        "amount": [100.0, 200.0, 300.0, 50.0, 50.0, 50.0],
        "timestamp": [
            "2025-06-01 02:00:00",  # night
            "2025-06-01 14:00:00",  # day
            "2025-06-02 03:00:00",  # night
            "2025-06-01 10:00:00",  # day
            "2025-06-01 23:00:00",  # day (23:00 is not 00-06)
            "2025-06-02 01:00:00",  # night
        ],
        "is_international": [1, 0, 1, 0, 0, 0],
        "country": ["US", "US", "UK", "ZA", "ZA", "ZA"],
    })


# --- Tests ---

class TestTransactionCount:
    def test_counts_per_account(self, sample_transactions):
        result = transaction_count(sample_transactions)
        assert result["A"] == 3
        assert result["B"] == 3

    def test_returns_series(self, sample_transactions):
        result = transaction_count(sample_transactions)
        assert isinstance(result, pd.Series)
        assert result.name == "transaction_count"


class TestAvgAmount:
    def test_mean_per_account(self, sample_transactions):
        result = avg_amount(sample_transactions)
        assert result["A"] == pytest.approx(200.0)  # (100+200+300)/3
        assert result["B"] == pytest.approx(50.0)   # (50+50+50)/3


class TestStdAmount:
    def test_std_per_account(self, sample_transactions):
        result = std_amount(sample_transactions)
        assert result["A"] == pytest.approx(100.0)  # std of [100,200,300]
        assert result["B"] == pytest.approx(0.0)    # std of [50,50,50]


class TestNightRatio:
    def test_night_fraction(self, sample_transactions):
        result = night_ratio(sample_transactions)
        # Account A: 2 night txns out of 3 → 0.667
        assert result["A"] == pytest.approx(2 / 3, rel=1e-2)
        # Account B: 1 night txn out of 3 → 0.333
        assert result["B"] == pytest.approx(1 / 3, rel=1e-2)


class TestCrossBorderRatio:
    def test_international_fraction(self, sample_transactions):
        result = cross_border_ratio(sample_transactions)
        assert result["A"] == pytest.approx(2 / 3, rel=1e-2)
        assert result["B"] == pytest.approx(0.0)


class TestMerchantDiversity:
    def test_unique_merchants(self, sample_transactions):
        result = merchant_diversity(sample_transactions)
        assert result["A"] == 2  # m1, m2
        assert result["B"] == 2  # m1, m3


class TestMaxDailySpend:
    def test_max_single_day_total(self, sample_transactions):
        result = max_daily_spend(sample_transactions)
        # Account A: day1 = 100+200=300, day2 = 300 → max = 300
        assert result["A"] == pytest.approx(300.0)
        # Account B: day1 = 50+50=100, day2 = 50 → max = 100
        assert result["B"] == pytest.approx(100.0)


class TestBuildFeatures:
    def test_returns_all_columns_without_accounts(self, sample_transactions):
        result = build_features(sample_transactions)
        expected_cols = {
            "transaction_count", "avg_amount", "std_amount",
            "max_daily_spend", "night_ratio", "cross_border_ratio",
            "merchant_diversity", "country_diversity", "rolling_7d_avg",
        }
        assert expected_cols == set(result.columns)

    def test_no_nans(self, sample_transactions):
        result = build_features(sample_transactions)
        assert not result.isna().any().any()
