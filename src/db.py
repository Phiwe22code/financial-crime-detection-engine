"""Database utilities for loading data into and querying from SQLite.

Provides functions to initialise the database schema, bulk-insert
generated data, and run read queries that return pandas DataFrames.
"""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

# Default database path
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "financial_crime.db"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a connection to the SQLite database.

    Args:
        db_path: Path to the database file. Defaults to data/financial_crime.db.

    Returns:
        sqlite3.Connection: Active database connection.
    """
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Create database tables from sql/schema.sql.

    Args:
        db_path: Path to the database file. Defaults to data/financial_crime.db.
    """
    schema_path = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    conn = get_connection(db_path)
    try:
        conn.executescript(schema_sql)
        conn.commit()
    finally:
        conn.close()


def insert_dataframe(
    df: pd.DataFrame,
    table_name: str,
    db_path: Optional[Path] = None,
    if_exists: str = "append",
) -> int:
    """Insert a pandas DataFrame into a SQLite table.

    Args:
        df: DataFrame to insert.
        table_name: Target table name.
        db_path: Path to the database file.
        if_exists: How to behave if table exists ('append', 'replace', 'fail').

    Returns:
        int: Number of rows inserted.
    """
    conn = get_connection(db_path)
    try:
        rows = df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        conn.commit()
        return rows if rows is not None else len(df)
    finally:
        conn.close()


def query(
    sql: str,
    db_path: Optional[Path] = None,
    params: Optional[tuple] = None,
) -> pd.DataFrame:
    """Execute a SQL query and return results as a DataFrame.

    Args:
        sql: SQL query string.
        db_path: Path to the database file.
        params: Optional query parameters.

    Returns:
        pd.DataFrame: Query results.
    """
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()
