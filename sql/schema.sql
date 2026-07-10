-- ============================================================
-- Financial Crime Detection Engine — Database Schema
-- ============================================================

-- Core customer identity table for KYC reference.
CREATE TABLE IF NOT EXISTS customers (
    customer_id       TEXT PRIMARY KEY,
    first_name        TEXT NOT NULL,
    last_name         TEXT NOT NULL,
    date_of_birth     DATE,
    country           TEXT,
    registration_date DATE,
    is_pep            BOOLEAN DEFAULT 0   -- Politically Exposed Person flag
);

-- Bank accounts linked to customers; tracks dormancy and status.
CREATE TABLE IF NOT EXISTS accounts (
    account_id   TEXT PRIMARY KEY,
    customer_id  TEXT NOT NULL,
    account_type TEXT NOT NULL,            -- e.g. checking, savings, business
    currency     TEXT NOT NULL,
    opened_date  DATE,
    is_dormant   BOOLEAN DEFAULT 0,
    status       TEXT DEFAULT 'active',
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

-- Merchant reference table; category and risk_level inform transaction risk.
CREATE TABLE IF NOT EXISTS merchants (
    merchant_id   TEXT PRIMARY KEY,
    merchant_name TEXT NOT NULL,
    category      TEXT NOT NULL,           -- e.g. retail, travel, crypto, gambling, electronics
    country       TEXT,
    risk_level    TEXT DEFAULT 'low'       -- low, medium, high
);

-- Core transaction ledger.
-- is_flagged marks injected suspicious activity for model evaluation.
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id   TEXT PRIMARY KEY,
    account_id       TEXT NOT NULL,
    merchant_id      TEXT NOT NULL,
    amount           REAL NOT NULL,
    currency         TEXT NOT NULL,
    timestamp        DATETIME NOT NULL,
    transaction_type TEXT NOT NULL,        -- purchase, transfer, withdrawal, deposit
    channel          TEXT NOT NULL,        -- online, ATM, branch, mobile
    is_international BOOLEAN DEFAULT 0,
    country          TEXT,
    is_flagged       BOOLEAN DEFAULT 0,
    FOREIGN KEY (account_id)  REFERENCES accounts  (account_id),
    FOREIGN KEY (merchant_id) REFERENCES merchants (merchant_id)
);

-- ============================================================
-- Indexes for common query patterns
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_transactions_account_id  ON transactions (account_id);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp   ON transactions (timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant_id ON transactions (merchant_id);
CREATE INDEX IF NOT EXISTS idx_accounts_customer_id     ON accounts (customer_id);
