"""Feature engineering for fraud/AML detection.

Computes per-account behavioural features from raw transaction data.
Each feature targets a known fraud/AML signal:

- **transaction_count**: High volume may indicate mule accounts or automated fraud.
- **avg_amount**: Unusual average spend compared to peer accounts.
- **std_amount**: High variance suggests mixing legitimate and illicit transactions.
- **max_daily_spend**: Single-day spikes can indicate account takeover or bust-out fraud.
- **night_ratio**: Transactions between 00:00–06:00 correlate with card-not-present fraud.
- **cross_border_ratio**: Rapid international activity is a top AML red flag.
- **merchant_diversity**: Very high or very low diversity can both be suspicious.
- **country_diversity**: Multiple countries in a short period signals layering.
- **rolling_7d_avg**: Captures recent behavioural changes vs. baseline.
- **account_age_days**: New accounts are higher risk ("bust-out" fraud window).
"""

from typing import Optional

import numpy as np
import pandas as pd


def transaction_count(df: pd.DataFrame) -> pd.Series:
    """Count total transactions per account.

    Args:
        df: Transaction DataFrame with 'account_id' column.

    Returns:
        Series indexed by account_id with transaction counts.
    """
    return df.groupby("account_id")["transaction_id"].count().rename("transaction_count")


def avg_amount(df: pd.DataFrame) -> pd.Series:
    """Mean transaction amount per account.

    Args:
        df: Transaction DataFrame with 'account_id' and 'amount' columns.

    Returns:
        Series indexed by account_id.
    """
    return df.groupby("account_id")["amount"].mean().rename("avg_amount")


def std_amount(df: pd.DataFrame) -> pd.Series:
    """Standard deviation of transaction amounts per account.

    Args:
        df: Transaction DataFrame with 'account_id' and 'amount' columns.

    Returns:
        Series indexed by account_id.
    """
    return df.groupby("account_id")["amount"].std().fillna(0).rename("std_amount")


def max_daily_spend(df: pd.DataFrame) -> pd.Series:
    """Maximum single-day total spend per account.

    Args:
        df: Transaction DataFrame with 'account_id', 'amount', and 'timestamp'.

    Returns:
        Series indexed by account_id.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily = df.groupby(["account_id", "date"])["amount"].sum()
    return daily.groupby("account_id").max().rename("max_daily_spend")


def night_ratio(df: pd.DataFrame) -> pd.Series:
    """Fraction of transactions occurring between 00:00 and 06:00.

    Args:
        df: Transaction DataFrame with 'account_id' and 'timestamp'.

    Returns:
        Series indexed by account_id (0.0 to 1.0).
    """
    df = df.copy()
    hours = pd.to_datetime(df["timestamp"]).dt.hour
    df["is_night"] = ((hours >= 0) & (hours < 6)).astype(int)
    night_counts = df.groupby("account_id")["is_night"].sum()
    total_counts = df.groupby("account_id")["is_night"].count()
    return (night_counts / total_counts).fillna(0).rename("night_ratio")


def cross_border_ratio(df: pd.DataFrame) -> pd.Series:
    """Fraction of transactions flagged as international.

    Args:
        df: Transaction DataFrame with 'account_id' and 'is_international'.

    Returns:
        Series indexed by account_id (0.0 to 1.0).
    """
    return (
        df.groupby("account_id")["is_international"]
        .mean()
        .fillna(0)
        .rename("cross_border_ratio")
    )


def merchant_diversity(df: pd.DataFrame) -> pd.Series:
    """Number of unique merchants per account.

    Args:
        df: Transaction DataFrame with 'account_id' and 'merchant_id'.

    Returns:
        Series indexed by account_id.
    """
    return (
        df.groupby("account_id")["merchant_id"]
        .nunique()
        .rename("merchant_diversity")
    )


def country_diversity(df: pd.DataFrame) -> pd.Series:
    """Number of unique transaction countries per account.

    Args:
        df: Transaction DataFrame with 'account_id' and 'country'.

    Returns:
        Series indexed by account_id.
    """
    return (
        df.groupby("account_id")["country"]
        .nunique()
        .rename("country_diversity")
    )


def rolling_7d_avg(df: pd.DataFrame) -> pd.Series:
    """Average of the rolling 7-day mean transaction amount per account.

    Captures recent spending velocity. We compute each account's daily
    average amount, apply a 7-day rolling mean, then take the overall
    average of that rolling series.

    Args:
        df: Transaction DataFrame with 'account_id', 'amount', 'timestamp'.

    Returns:
        Series indexed by account_id.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    daily_avg = df.groupby(["account_id", "date"])["amount"].mean().reset_index()
    daily_avg["date"] = pd.to_datetime(daily_avg["date"])
    daily_avg = daily_avg.sort_values(["account_id", "date"])

    result = {}
    for aid, group in daily_avg.groupby("account_id"):
        group = group.set_index("date").sort_index()
        rolling = group["amount"].rolling(window=7, min_periods=1).mean()
        result[aid] = rolling.mean()

    return pd.Series(result, name="rolling_7d_avg")


def account_age_days(
    df: pd.DataFrame, accounts_df: pd.DataFrame
) -> pd.Series:
    """Age of each account in days (from opened_date to most recent transaction).

    Args:
        df: Transaction DataFrame with 'account_id' and 'timestamp'.
        accounts_df: Accounts DataFrame with 'account_id' and 'opened_date'.

    Returns:
        Series indexed by account_id.
    """
    latest_txn = (
        pd.to_datetime(df.groupby("account_id")["timestamp"].max())
    )
    opened = accounts_df.set_index("account_id")["opened_date"]
    opened = pd.to_datetime(opened)
    age = (latest_txn - opened).dt.days
    return age.fillna(0).rename("account_age_days")


def build_features(
    df: pd.DataFrame,
    accounts_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute all per-account features from raw transactions.

    This is the main entry point for feature engineering. It composes
    all individual feature functions into a single feature matrix.

    Args:
        df: Raw transaction DataFrame.
        accounts_df: Accounts DataFrame (needed for account_age_days).
            If None, account_age_days is excluded.

    Returns:
        DataFrame indexed by account_id with one column per feature.
    """
    features = [
        transaction_count(df),
        avg_amount(df),
        std_amount(df),
        max_daily_spend(df),
        night_ratio(df),
        cross_border_ratio(df),
        merchant_diversity(df),
        country_diversity(df),
        rolling_7d_avg(df),
    ]

    if accounts_df is not None:
        features.append(account_age_days(df, accounts_df))

    feature_matrix = pd.concat(features, axis=1)
    # Fill any remaining NaN with 0 (e.g. accounts with a single transaction)
    feature_matrix = feature_matrix.fillna(0)

    return feature_matrix
