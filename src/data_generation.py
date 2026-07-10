"""Synthetic data generation for the Financial Crime Detection Engine.

Generates realistic banking data (customers, accounts, merchants, transactions)
with a configurable percentage of deliberately suspicious patterns injected
to simulate fraud/AML scenarios:

- **Sudden volume spikes**: dormant accounts suddenly making many transactions
- **Rapid cross-border activity**: multiple countries in short time windows
- **Structuring**: round-number transactions just under reporting thresholds
- **Dormant reactivation**: accounts with no activity suddenly becoming active

Usage:
    python -m src.data_generation
"""

import random
import uuid
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

# --- Constants ---
COUNTRIES = ["US", "UK", "DE", "FR", "NG", "ZA", "AE", "SG", "BR", "IN", "CN", "JP"]
CURRENCIES = {"US": "USD", "UK": "GBP", "DE": "EUR", "FR": "EUR", "NG": "NGN",
              "ZA": "ZAR", "AE": "AED", "SG": "SGD", "BR": "BRL", "IN": "INR",
              "CN": "CNY", "JP": "JPY"}
ACCOUNT_TYPES = ["checking", "savings", "business"]
CHANNELS = ["online", "ATM", "branch", "mobile"]
MERCHANT_CATEGORIES = ["retail", "travel", "crypto", "gambling", "electronics",
                        "groceries", "dining", "utilities", "healthcare", "entertainment"]
TRANSACTION_TYPES = ["purchase", "transfer", "withdrawal", "deposit"]

# Reporting threshold — structuring transactions cluster just below this
REPORTING_THRESHOLD = 10_000.0


def generate_customers(n: int = 500) -> pd.DataFrame:
    """Generate synthetic customer records.

    Args:
        n: Number of customers to generate.

    Returns:
        DataFrame with columns: customer_id, first_name, last_name,
        date_of_birth, country, registration_date, is_pep.
    """
    records = []
    for _ in range(n):
        country = random.choice(COUNTRIES)
        records.append({
            "customer_id": str(uuid.uuid4()),
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "date_of_birth": fake.date_of_birth(minimum_age=18, maximum_age=80).isoformat(),
            "country": country,
            "registration_date": fake.date_between(
                start_date="-5y", end_date="-6m"
            ).isoformat(),
            "is_pep": int(random.random() < 0.03),  # ~3 % are PEPs
        })
    return pd.DataFrame(records)


def generate_accounts(
    customers_df: pd.DataFrame, n: int = 700
) -> pd.DataFrame:
    """Generate synthetic bank accounts linked to customers.

    Some accounts are marked dormant to enable dormant-reactivation scenarios.

    Args:
        customers_df: DataFrame of customers (needs customer_id, country).
        n: Number of accounts to generate.

    Returns:
        DataFrame with columns: account_id, customer_id, account_type,
        currency, opened_date, is_dormant, status.
    """
    customer_ids = customers_df["customer_id"].tolist()
    customer_countries = dict(
        zip(customers_df["customer_id"], customers_df["country"])
    )
    records = []
    for _ in range(n):
        cid = random.choice(customer_ids)
        country = customer_countries[cid]
        records.append({
            "account_id": str(uuid.uuid4()),
            "customer_id": cid,
            "account_type": random.choice(ACCOUNT_TYPES),
            "currency": CURRENCIES.get(country, "USD"),
            "opened_date": fake.date_between(
                start_date="-4y", end_date="-3m"
            ).isoformat(),
            "is_dormant": int(random.random() < 0.10),  # ~10 % start dormant
            "status": "active",
        })
    return pd.DataFrame(records)


def generate_merchants(n: int = 50) -> pd.DataFrame:
    """Generate synthetic merchant records.

    Args:
        n: Number of merchants to generate.

    Returns:
        DataFrame with columns: merchant_id, merchant_name, category,
        country, risk_level.
    """
    records = []
    for _ in range(n):
        category = random.choice(MERCHANT_CATEGORIES)
        # Higher risk for crypto / gambling merchants
        if category in ("crypto", "gambling"):
            risk = random.choice(["medium", "high"])
        else:
            risk = random.choices(["low", "medium"], weights=[0.85, 0.15])[0]
        records.append({
            "merchant_id": str(uuid.uuid4()),
            "merchant_name": fake.company(),
            "category": category,
            "country": random.choice(COUNTRIES),
            "risk_level": risk,
        })
    return pd.DataFrame(records)


def _random_timestamp(
    start: datetime, end: datetime
) -> datetime:
    """Return a random datetime between start and end."""
    delta = (end - start).total_seconds()
    offset = random.random() * delta
    return start + timedelta(seconds=offset)


def generate_normal_transactions(
    accounts_df: pd.DataFrame,
    merchants_df: pd.DataFrame,
    n: int = 48_500,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Generate normal (non-suspicious) transactions.

    Transaction amounts follow a log-normal distribution to mimic real
    spending patterns (many small purchases, few large ones).

    Args:
        accounts_df: DataFrame of accounts.
        merchants_df: DataFrame of merchants.
        n: Number of normal transactions.
        start_date: Period start.
        end_date: Period end.

    Returns:
        DataFrame of normal transactions.
    """
    if start_date is None:
        start_date = datetime(2025, 1, 1)
    if end_date is None:
        end_date = datetime(2025, 12, 31, 23, 59, 59)

    account_ids = accounts_df["account_id"].tolist()
    merchant_ids = merchants_df["merchant_id"].tolist()
    merchant_countries = dict(
        zip(merchants_df["merchant_id"], merchants_df["country"])
    )
    account_currencies = dict(
        zip(accounts_df["account_id"], accounts_df["currency"])
    )

    records = []
    for _ in range(n):
        aid = random.choice(account_ids)
        mid = random.choice(merchant_ids)
        m_country = merchant_countries[mid]
        a_currency = account_currencies[aid]
        # Log-normal: median ~$50, occasional larger amounts
        amount = round(float(np.random.lognormal(mean=3.9, sigma=1.2)), 2)
        amount = min(amount, 25_000.0)  # cap extreme outliers
        ts = _random_timestamp(start_date, end_date)

        records.append({
            "transaction_id": str(uuid.uuid4()),
            "account_id": aid,
            "merchant_id": mid,
            "amount": amount,
            "currency": a_currency,
            "timestamp": ts.isoformat(),
            "transaction_type": random.choice(TRANSACTION_TYPES),
            "channel": random.choice(CHANNELS),
            "is_international": int(random.random() < 0.08),
            "country": m_country if random.random() < 0.85 else random.choice(COUNTRIES),
            "is_flagged": 0,
        })
    return pd.DataFrame(records)


def generate_suspicious_transactions(
    accounts_df: pd.DataFrame,
    merchants_df: pd.DataFrame,
    n: int = 1_500,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """Generate deliberately suspicious transactions for model training.

    Injects four fraud/AML typologies:
    1. **Structuring** — amounts just under the $10,000 reporting threshold
       (e.g., $9,500–$9,999).
    2. **Rapid cross-border** — multiple countries within short time windows.
    3. **Volume spikes** — bursts of many transactions in a single day.
    4. **Dormant reactivation** — transactions on accounts marked dormant.

    Args:
        accounts_df: DataFrame of accounts.
        merchants_df: DataFrame of merchants.
        n: Number of suspicious transactions to inject.
        start_date: Period start.
        end_date: Period end.

    Returns:
        DataFrame of suspicious transactions (is_flagged=1).
    """
    if start_date is None:
        start_date = datetime(2025, 1, 1)
    if end_date is None:
        end_date = datetime(2025, 12, 31, 23, 59, 59)

    account_ids = accounts_df["account_id"].tolist()
    dormant_ids = accounts_df.loc[
        accounts_df["is_dormant"] == 1, "account_id"
    ].tolist()
    merchant_ids = merchants_df["merchant_id"].tolist()
    account_currencies = dict(
        zip(accounts_df["account_id"], accounts_df["currency"])
    )

    records = []
    per_type = n // 4

    # --- Typology 1: Structuring (just-under-threshold amounts) ---
    for _ in range(per_type):
        aid = random.choice(account_ids)
        mid = random.choice(merchant_ids)
        # Cluster between 9,000 and 9,999
        amount = round(random.uniform(9_000, 9_999), 2)
        ts = _random_timestamp(start_date, end_date)
        records.append({
            "transaction_id": str(uuid.uuid4()),
            "account_id": aid,
            "merchant_id": mid,
            "amount": amount,
            "currency": account_currencies[aid],
            "timestamp": ts.isoformat(),
            "transaction_type": "transfer",
            "channel": random.choice(["online", "mobile"]),
            "is_international": 0,
            "country": random.choice(COUNTRIES),
            "is_flagged": 1,
        })

    # --- Typology 2: Rapid cross-border activity ---
    for _ in range(per_type):
        aid = random.choice(account_ids)
        mid = random.choice(merchant_ids)
        amount = round(float(np.random.lognormal(mean=5.5, sigma=0.8)), 2)
        # Cluster timestamps in tight windows (same day)
        base_ts = _random_timestamp(start_date, end_date)
        ts = base_ts + timedelta(minutes=random.randint(0, 120))
        if ts > end_date:
            ts = end_date
        country = random.choice(COUNTRIES)
        records.append({
            "transaction_id": str(uuid.uuid4()),
            "account_id": aid,
            "merchant_id": mid,
            "amount": amount,
            "currency": account_currencies[aid],
            "timestamp": ts.isoformat(),
            "transaction_type": random.choice(TRANSACTION_TYPES),
            "channel": "online",
            "is_international": 1,
            "country": country,
            "is_flagged": 1,
        })

    # --- Typology 3: Volume spikes (many txns in one day) ---
    spike_accounts = random.sample(account_ids, min(15, len(account_ids)))
    txns_per_account = max(1, per_type // len(spike_accounts))
    for aid in spike_accounts:
        spike_date = _random_timestamp(start_date, end_date)
        for _ in range(txns_per_account):
            mid = random.choice(merchant_ids)
            amount = round(float(np.random.lognormal(mean=4.5, sigma=1.0)), 2)
            ts = spike_date + timedelta(minutes=random.randint(0, 1440))
            if ts > end_date:
                ts = end_date
            records.append({
                "transaction_id": str(uuid.uuid4()),
                "account_id": aid,
                "merchant_id": mid,
                "amount": amount,
                "currency": account_currencies[aid],
                "timestamp": ts.isoformat(),
                "transaction_type": "purchase",
                "channel": random.choice(CHANNELS),
                "is_international": int(random.random() < 0.3),
                "country": random.choice(COUNTRIES),
                "is_flagged": 1,
            })

    # --- Typology 4: Dormant account reactivation ---
    if dormant_ids:
        dormant_per = per_type // len(dormant_ids) + 1
        for aid in dormant_ids:
            # Burst of activity in a short period
            reactivation_date = _random_timestamp(
                start_date + timedelta(days=180), end_date
            )
            for _ in range(min(dormant_per, 20)):
                mid = random.choice(merchant_ids)
                amount = round(float(np.random.lognormal(mean=5.0, sigma=1.5)), 2)
                ts = reactivation_date + timedelta(
                    hours=random.randint(0, 72)
                )
                if ts > end_date:
                    ts = end_date
                records.append({
                    "transaction_id": str(uuid.uuid4()),
                    "account_id": aid,
                    "merchant_id": mid,
                    "amount": amount,
                    "currency": account_currencies[aid],
                    "timestamp": ts.isoformat(),
                    "transaction_type": random.choice(TRANSACTION_TYPES),
                    "channel": random.choice(["online", "mobile"]),
                    "is_international": int(random.random() < 0.4),
                    "country": random.choice(COUNTRIES),
                    "is_flagged": 1,
                })

    return pd.DataFrame(records[:n])  # trim to requested count


def generate_all_data(
    n_customers: int = 500,
    n_accounts: int = 700,
    n_merchants: int = 50,
    n_transactions: int = 50_000,
    suspicious_pct: float = 0.03,
) -> dict[str, pd.DataFrame]:
    """Generate the full synthetic dataset.

    Args:
        n_customers: Number of customer records.
        n_accounts: Number of account records.
        n_merchants: Number of merchant records.
        n_transactions: Total number of transactions.
        suspicious_pct: Fraction of transactions that are suspicious.

    Returns:
        Dictionary with keys 'customers', 'accounts', 'merchants',
        'transactions' mapping to their respective DataFrames.
    """
    print("Generating customers...")
    customers = generate_customers(n_customers)

    print("Generating accounts...")
    accounts = generate_accounts(customers, n_accounts)

    print("Generating merchants...")
    merchants = generate_merchants(n_merchants)

    n_suspicious = int(n_transactions * suspicious_pct)
    n_normal = n_transactions - n_suspicious

    print(f"Generating {n_normal:,} normal transactions...")
    normal_txns = generate_normal_transactions(accounts, merchants, n_normal)

    print(f"Generating {n_suspicious:,} suspicious transactions...")
    suspicious_txns = generate_suspicious_transactions(
        accounts, merchants, n_suspicious
    )

    transactions = pd.concat(
        [normal_txns, suspicious_txns], ignore_index=True
    ).sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

    print(f"Total transactions: {len(transactions):,}")
    print(f"Flagged (suspicious): {transactions['is_flagged'].sum():,} "
          f"({transactions['is_flagged'].mean():.1%})")

    return {
        "customers": customers,
        "accounts": accounts,
        "merchants": merchants,
        "transactions": transactions,
    }


def load_data_to_db(data: dict[str, pd.DataFrame]) -> None:
    """Load generated DataFrames into the SQLite database.

    Initialises the schema (dropping existing tables) and inserts all data.

    Args:
        data: Dictionary from generate_all_data().
    """
    from src.db import init_db, insert_dataframe, DB_PATH

    # Remove existing DB for a clean start
    if DB_PATH.exists():
        DB_PATH.unlink()

    print(f"\nInitialising database at {DB_PATH}...")
    init_db()

    for table_name, df in data.items():
        count = insert_dataframe(df, table_name)
        print(f"  Inserted {count:,} rows into '{table_name}'")

    print("Database populated successfully.")


if __name__ == "__main__":
    data = generate_all_data()
    load_data_to_db(data)
    print("\nDone! Data generation complete.")
