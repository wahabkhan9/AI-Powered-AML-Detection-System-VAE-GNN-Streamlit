"""
data/duckdb_queries.py
======================
DuckDB analytical query layer – FULLY WIRED into the AML pipeline.

DuckDB queries Parquet files directly without loading them into pandas,
giving 10-100x speedup over pandas for aggregation queries at scale.

Usage in pipeline
-----------------
    from data.duckdb_queries import AMLQueryEngine
    with AMLQueryEngine() as engine:
        engine.run_all_and_save()          # ← pipeline call

Usage in dashboard
------------------
    engine = AMLQueryEngine()
    df = engine.top_suspicious_customers(20)

Usage with raw SQL
------------------
    engine.sql("SELECT COUNT(*) FROM transactions WHERE is_suspicious = TRUE")
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

try:
    import duckdb
    _HAS_DUCKDB = True
except ImportError:
    _HAS_DUCKDB = False
    log.warning("duckdb not installed – AMLQueryEngine unavailable. pip install duckdb")

DATA_DIR    = Path("data")
REPORTS_DIR = Path("reports")


class AMLQueryEngine:
    """
    High-performance DuckDB-based analytics over Parquet files.

    All methods return pandas DataFrames.
    The engine uses an in-memory DuckDB database with Parquet views –
    no database server or disk database file is required.
    """

    def __init__(self) -> None:
        if not _HAS_DUCKDB:
            raise ImportError("Install duckdb: pip install duckdb")

        self.conn = duckdb.connect(database=":memory:")
        self._registered_views: List[str] = []
        self._register_views()

    # ------------------------------------------------------------------
    # View registration
    # ------------------------------------------------------------------
    def _register_views(self) -> None:
        view_map = {
            "transactions": DATA_DIR / "transactions.parquet",
            "customers":    DATA_DIR / "customers.parquet",
            "risk_scores":  REPORTS_DIR / "customer_risk_scores.parquet",
            "vae_alerts":   REPORTS_DIR / "vae_alerts.parquet",
        }
        for view_name, path in view_map.items():
            if path.exists():
                self.conn.execute(
                    f"CREATE OR REPLACE VIEW {view_name} AS "
                    f"SELECT * FROM read_parquet('{path}')"
                )
                self._registered_views.append(view_name)
            else:
                log.debug("Parquet not found – view '%s' skipped: %s", view_name, path)

        log.info("DuckDB views registered: %s", self._registered_views)

    # ------------------------------------------------------------------
    # Raw SQL
    # ------------------------------------------------------------------
    def sql(self, query: str) -> pd.DataFrame:
        """Execute arbitrary SQL and return a DataFrame."""
        return self.conn.execute(query).df()

    # ------------------------------------------------------------------
    # Preset analytical queries
    # ------------------------------------------------------------------
    def transaction_summary(self) -> pd.DataFrame:
        """Dataset-level KPIs."""
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

    def top_suspicious_customers(self, n: int = 20) -> pd.DataFrame:
        """Top N senders by suspicious transaction count."""
        return self.sql(f"""
            SELECT
                sender_id                                       AS customer_id,
                COUNT(*)                                        AS suspicious_txn_count,
                ROUND(SUM(amount_usd), 2)                       AS total_suspicious_usd,
                ROUND(AVG(amount_usd), 2)                       AS avg_amount_usd,
                COUNT(DISTINCT receiver_id)                     AS unique_counterparties
            FROM transactions
            WHERE is_suspicious = TRUE
            GROUP BY sender_id
            ORDER BY suspicious_txn_count DESC
            LIMIT {n}
        """)

    def daily_alert_trend(self, days: int = 90) -> pd.DataFrame:
        """Daily alert counts and flagged volume."""
        return self.sql(f"""
            SELECT
                CAST(timestamp AS DATE)                         AS date,
                COUNT(*)                                        AS total_txns,
                SUM(is_suspicious::INT)                         AS suspicious,
                ROUND(SUM(CASE WHEN is_suspicious THEN amount_usd ELSE 0 END), 2)
                                                                AS flagged_usd
            FROM transactions
            WHERE timestamp >= CURRENT_DATE - INTERVAL '{days} days'
            GROUP BY CAST(timestamp AS DATE)
            ORDER BY date
        """)

    def typology_breakdown(self) -> pd.DataFrame:
        """Suspicious count + volume by money-laundering typology."""
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

    def cross_border_analysis(self) -> pd.DataFrame:
        """Top cross-border corridors for suspicious activity."""
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

    def structuring_alerts(self, threshold_usd: float = 10_000.0) -> pd.DataFrame:
        """Structuring pattern: multiple sub-threshold transactions per sender per day."""
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

    def hourly_velocity_anomalies(self) -> pd.DataFrame:
        """Customers with unusually high transaction velocity per hour."""
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

    def high_risk_network_customers(self) -> pd.DataFrame:
        """
        Join transaction stats with GNN risk scores.
        Only available after the GNN training step.
        """
        if "risk_scores" not in self._registered_views:
            log.warning("risk_scores view not available – run train_gnn first.")
            return pd.DataFrame()
        return self.sql("""
            SELECT
                t.sender_id                                     AS customer_id,
                ROUND(AVG(r.gnn_risk_score), 4)                 AS gnn_risk_score,
                COUNT(t.transaction_id)                         AS txn_count,
                ROUND(SUM(t.amount_usd), 2)                     AS total_usd,
                SUM(t.is_suspicious::INT)                       AS suspicious_count
            FROM transactions t
            LEFT JOIN risk_scores r ON t.sender_id = r.customer_id
            WHERE r.gnn_risk_score > 0.5
            GROUP BY t.sender_id
            ORDER BY gnn_risk_score DESC
            LIMIT 50
        """)

    def vae_flagged_customers(self) -> pd.DataFrame:
        """
        Customers flagged by the VAE+GAN combined score.
        Only available after train_vae step.
        """
        if "vae_alerts" not in self._registered_views:
            log.warning("vae_alerts view not available – run train_vae first.")
            return pd.DataFrame()
        return self.sql("""
            SELECT
                customer_id,
                ROUND(AVG(vae_score), 6)                        AS avg_vae_score,
                ROUND(AVG(combined_score), 6)                   AS avg_combined_score,
                MAX(is_flagged::INT)                            AS is_flagged
            FROM vae_alerts
            WHERE is_flagged = TRUE
            GROUP BY customer_id
            ORDER BY avg_combined_score DESC
            LIMIT 100
        """)

    # ------------------------------------------------------------------
    # Pipeline entry-point: run ALL queries and save to CSV
    # ------------------------------------------------------------------
    def run_all_and_save(self, output_dir: Optional[Path] = None) -> Dict[str, str]:
        """
        Execute every preset query and write results to CSV files.
        Called directly by the pipeline orchestrator.

        Returns a dict of {query_name: output_path}.
        """
        out = Path(output_dir or REPORTS_DIR)
        out.mkdir(parents=True, exist_ok=True)

        queries = {
            "transaction_summary":       self.transaction_summary,
            "top_suspicious_customers":  lambda: self.top_suspicious_customers(50),
            "daily_alert_trend":         lambda: self.daily_alert_trend(90),
            "typology_breakdown":        self.typology_breakdown,
            "cross_border_analysis":     self.cross_border_analysis,
            "structuring_alerts":        self.structuring_alerts,
            "hourly_velocity_anomalies": self.hourly_velocity_anomalies,
        }

        # Add optional queries (only if dependent views exist)
        if "risk_scores" in self._registered_views:
            queries["high_risk_network"] = self.high_risk_network_customers
        if "vae_alerts" in self._registered_views:
            queries["vae_flagged"]       = self.vae_flagged_customers

        results = {}
        manifest = {"run_at": datetime.utcnow().isoformat(), "queries": []}

        for name, fn in queries.items():
            try:
                df = fn()
                path = out / f"duckdb_{name}.csv"
                df.to_csv(path, index=False)
                results[name] = str(path)
                manifest["queries"].append({
                    "name": name, "rows": len(df), "file": str(path)
                })
                log.info("DuckDB %-35s → %d rows", name, len(df))
            except Exception as exc:
                log.warning("DuckDB query '%s' failed: %s", name, exc)
                manifest["queries"].append({"name": name, "error": str(exc)})

        (out / "duckdb_manifest.json").write_text(json.dumps(manifest, indent=2))
        log.info(
            "DuckDB analytics complete. %d / %d queries succeeded.",
            sum(1 for q in manifest["queries"] if "error" not in q),
            len(manifest["queries"]),
        )
        return results

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "AMLQueryEngine":
        return self

    def __exit__(self, *args) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CLI: python -m data.duckdb_queries
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    with AMLQueryEngine() as engine:
        results = engine.run_all_and_save()
        print(f"\nSaved {len(results)} analytics files:")
        for name, path in results.items():
            print(f"  {name:40s} → {path}")