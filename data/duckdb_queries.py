"""
data/duckdb_queries.py
======================
DuckDB analytical query layer.

DuckDB can query Parquet files directly without loading them into memory,
enabling fast SQL analytics over the full 336K+ transaction dataset.

Usage
-----
    from data.duckdb_queries import AMLQueryEngine
    engine = AMLQueryEngine()
    df = engine.top_suspicious_customers(n=20)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

log = logging.getLogger(__name__)

try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False
    log.warning("duckdb not installed. Run: pip install duckdb")

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")


class AMLQueryEngine:
    """
    High-performance analytical query engine using DuckDB over Parquet files.

    All queries are pure SQL executed in-process; no database server needed.
    """

    def __init__(self) -> None:
        if not _HAS_DUCKDB:
            raise ImportError("Install duckdb: pip install duckdb")

        self.conn = duckdb.connect(database=":memory:")

        # Register Parquet views
        txn_path  = str(DATA_DIR / "transactions.parquet")
        cust_path = str(DATA_DIR / "customers.parquet")

        if Path(txn_path).exists():
            self.conn.execute(f"CREATE VIEW transactions AS SELECT * FROM read_parquet('{txn_path}')")
        if Path(cust_path).exists():
            self.conn.execute(f"CREATE VIEW customers AS SELECT * FROM read_parquet('{cust_path}')")

        risk_path = str(REPORTS_DIR / "customer_risk_scores.parquet")
        if Path(risk_path).exists():
            self.conn.execute(f"CREATE VIEW risk_scores AS SELECT * FROM read_parquet('{risk_path}')")

        log.info("DuckDB engine initialised.")

    # ------------------------------------------------------------------
    def sql(self, query: str) -> pd.DataFrame:
        """Execute raw SQL and return a DataFrame."""
        return self.conn.execute(query).df()

    # ------------------------------------------------------------------
    def transaction_summary(self) -> pd.DataFrame:
        """High-level transaction statistics."""
        return self.sql("""
            SELECT
                COUNT(*)                                        AS total_transactions,
                SUM(is_suspicious::INT)                         AS suspicious_count,
                ROUND(AVG(is_suspicious::FLOAT) * 100, 4)       AS suspicious_pct,
                ROUND(SUM(amount_usd), 2)                       AS total_volume_usd,
                ROUND(AVG(amount_usd), 2)                       AS avg_amount_usd,
                MIN(timestamp)                                  AS earliest_txn,
                MAX(timestamp)                                  AS latest_txn
            FROM transactions
        """)

    # ------------------------------------------------------------------
    def top_suspicious_customers(self, n: int = 20) -> pd.DataFrame:
        """Top N customers by suspicious transaction count."""
        return self.sql(f"""
            SELECT
                sender_id                                       AS customer_id,
                COUNT(*)                                        AS suspicious_txn_count,
                ROUND(SUM(amount_usd), 2)                       AS total_suspicious_usd,
                ROUND(AVG(amount_usd), 2)                       AS avg_suspicious_usd,
                COUNT(DISTINCT receiver_id)                     AS unique_counterparties,
                LIST(DISTINCT label)[1:3]                       AS typologies
            FROM transactions
            WHERE is_suspicious = TRUE
            GROUP BY sender_id
            ORDER BY suspicious_txn_count DESC
            LIMIT {n}
        """)

    # ------------------------------------------------------------------
    def daily_alert_trend(self, days: int = 90) -> pd.DataFrame:
        """Daily suspicious alert counts and volumes for the last N days."""
        return self.sql(f"""
            SELECT
                CAST(timestamp AS DATE)                         AS date,
                COUNT(*)                                        AS alerts,
                SUM(is_suspicious::INT)                         AS suspicious,
                ROUND(SUM(CASE WHEN is_suspicious THEN amount_usd ELSE 0 END), 2)
                                                                AS flagged_usd
            FROM transactions
            WHERE timestamp >= CURRENT_DATE - INTERVAL '{days} days'
            GROUP BY CAST(timestamp AS DATE)
            ORDER BY date
        """)

    # ------------------------------------------------------------------
    def typology_breakdown(self) -> pd.DataFrame:
        """Suspicious transaction count and volume by typology label."""
        return self.sql("""
            SELECT
                label                                           AS typology,
                COUNT(*)                                        AS count,
                ROUND(SUM(amount_usd), 2)                       AS total_usd,
                ROUND(AVG(amount_usd), 2)                       AS avg_usd
            FROM transactions
            WHERE is_suspicious = TRUE
            GROUP BY label
            ORDER BY count DESC
        """)

    # ------------------------------------------------------------------
    def cross_border_analysis(self) -> pd.DataFrame:
        """Suspicious transaction breakdown by country corridor."""
        return self.sql("""
            SELECT
                country_origin,
                country_dest,
                COUNT(*)                                        AS count,
                ROUND(SUM(amount_usd), 2)                       AS total_usd
            FROM transactions
            WHERE is_suspicious = TRUE
              AND country_origin != country_dest
            GROUP BY country_origin, country_dest
            ORDER BY count DESC
            LIMIT 30
        """)

    # ------------------------------------------------------------------
    def structuring_alerts(self, threshold_usd: float = 10_000.0) -> pd.DataFrame:
        """Identify structuring patterns: multiple sub-threshold transactions per sender per day."""
        return self.sql(f"""
            SELECT
                sender_id,
                CAST(timestamp AS DATE)                         AS date,
                COUNT(*)                                        AS txn_count,
                ROUND(SUM(amount_usd), 2)                       AS total_daily_usd,
                ROUND(MAX(amount_usd), 2)                       AS max_single_txn
            FROM transactions
            WHERE amount_usd < {threshold_usd}
              AND structuring_flag = TRUE
            GROUP BY sender_id, CAST(timestamp AS DATE)
            HAVING COUNT(*) >= 3
            ORDER BY txn_count DESC
            LIMIT 100
        """)

    # ------------------------------------------------------------------
    def high_risk_network_customers(self) -> pd.DataFrame:
        """
        Join transaction data with GNN risk scores to surface
        high-risk customers with large transaction volumes.
        """
        return self.sql("""
            SELECT
                t.sender_id                                     AS customer_id,
                ROUND(AVG(r.gnn_risk_score), 4)                 AS gnn_risk_score,
                COUNT(t.transaction_id)                         AS txn_count,
                ROUND(SUM(t.amount_usd), 2)                     AS total_usd,
                SUM(t.is_suspicious::INT)                       AS suspicious_count
            FROM transactions t
            LEFT JOIN risk_scores r ON t.sender_id = r.customer_id
            GROUP BY t.sender_id
            HAVING AVG(r.gnn_risk_score) > 0.6
            ORDER BY gnn_risk_score DESC
            LIMIT 50
        """)

    # ------------------------------------------------------------------
    def hourly_velocity_anomalies(self) -> pd.DataFrame:
        """Detect unusually high transaction velocity (count per customer per hour)."""
        return self.sql("""
            SELECT
                sender_id,
                DATE_TRUNC('hour', timestamp)                   AS hour_bucket,
                COUNT(*)                                        AS txn_per_hour,
                ROUND(SUM(amount_usd), 2)                       AS hourly_usd
            FROM transactions
            GROUP BY sender_id, DATE_TRUNC('hour', timestamp)
            HAVING COUNT(*) >= 5
            ORDER BY txn_per_hour DESC
            LIMIT 100
        """)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the DuckDB connection."""
        self.conn.close()

    # ------------------------------------------------------------------
    def __enter__(self) -> "AMLQueryEngine":
        return self

    def __exit__(self, *args) -> None:
        self.close()
