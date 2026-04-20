"""
agents/report_writer.py
=======================
SAR Report Writer Agent – generates Suspicious Activity Reports (SARs) for
the highest-risk alerts, using a template-driven narrative engine.

Each SAR follows FinCEN Form 111 structure:
  Part I   – Filer information (institution)
  Part II  – Suspect information
  Part III – Suspicious activity description
  Part IV  – Supporting documentation reference

Output: reports/sar_reports.json  (machine-readable)
        reports/sar_summary.csv   (compliance dashboard feed)
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
MODEL_DIR = Path("models")

INSTITUTION_NAME = "Acme Financial Services Inc."
INSTITUTION_EIN = "12-3456789"
INSTITUTION_CONTACT = "AML Compliance Dept. | aml@acmefs.com | +1-800-555-0100"

TOP_N_ALERTS = 200   # generate SARs for top-N highest risk alerts


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class SARFilerInfo:
    institution_name: str = INSTITUTION_NAME
    ein: str = INSTITUTION_EIN
    contact: str = INSTITUTION_CONTACT
    report_date: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"))
    filing_type: str = "INITIAL"


@dataclass
class SARSuspectInfo:
    customer_id: str
    customer_name: str
    country: str
    jurisdiction_risk: str
    account_type: str
    annual_income_usd: float
    pep: bool
    risk_score: float
    gnn_risk_score: float


@dataclass
class SARActivityInfo:
    typologies_detected: List[str]
    total_flagged_amount_usd: float
    flagged_transaction_count: int
    date_range_start: str
    date_range_end: str
    anomaly_score_vae: float
    structuring_detected: bool
    cross_border: bool
    activity_description: str


@dataclass
class SARReport:
    sar_id: str = field(default_factory=lambda: "SAR-" + uuid.uuid4().hex[:8].upper())
    generated_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat(timespec="seconds") + "Z"
    )
    priority: str = "HIGH"
    filer: SARFilerInfo = field(default_factory=SARFilerInfo)
    suspect: Optional[SARSuspectInfo] = None
    activity: Optional[SARActivityInfo] = None
    status: str = "DRAFT"


# ---------------------------------------------------------------------------
# Narrative templates
# ---------------------------------------------------------------------------
def _narrative(
    suspect: SARSuspectInfo,
    activity: SARActivityInfo,
) -> str:
    typology_str = ", ".join(activity.typologies_detected) if activity.typologies_detected else "anomalous transaction patterns"
    cross_border_str = (
        "Cross-border transfers were identified as part of this activity, "
        "involving high-risk jurisdictions. "
        if activity.cross_border
        else ""
    )
    structuring_str = (
        "Multiple transactions were structured below the $10,000 reporting threshold "
        "in a pattern consistent with smurfing / structuring. "
        if activity.structuring_detected
        else ""
    )
    pep_str = (
        "The subject is identified as a Politically Exposed Person (PEP), "
        "warranting enhanced due diligence. "
        if suspect.pep
        else ""
    )
    return (
        f"The filer is reporting suspicious activity associated with customer "
        f"{suspect.customer_id} ({suspect.account_type} account) based in {suspect.country}. "
        f"Between {activity.date_range_start} and {activity.date_range_end}, "
        f"the subject conducted {activity.flagged_transaction_count} transactions "
        f"totalling USD {activity.total_flagged_amount_usd:,.2f}. "
        f"AI-based anomaly detection (VAE reconstruction score: {activity.anomaly_score_vae:.4f}) "
        f"and graph network analysis (GNN risk score: {suspect.gnn_risk_score:.4f}) flagged this "
        f"account as high-risk. "
        f"Detected typologies: {typology_str}. "
        f"{structuring_str}"
        f"{cross_border_str}"
        f"{pep_str}"
        f"The institution's customer risk score is {suspect.risk_score:.4f} "
        f"(jurisdiction risk: {suspect.jurisdiction_risk}). "
        f"This SAR is filed in accordance with 31 U.S.C. § 5318(g) and related FinCEN guidance."
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class ReportWriterAgent:
    """Generate SAR reports for high-risk customers."""

    def run(self) -> None:
        log.info("=== ReportWriterAgent starting ===")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Load customer data
        cust_df = pd.read_parquet(DATA_DIR / "customers.parquet")
        txn_df = pd.read_parquet(DATA_DIR / "transactions.parquet")

        # Load GNN risk scores (may not exist on first run)
        risk_path = REPORTS_DIR / "customer_risk_scores.parquet"
        if risk_path.exists():
            risk_df = pd.read_parquet(risk_path)
        else:
            log.warning("GNN risk scores not found; using customer risk_score only.")
            risk_df = cust_df[["customer_id", "risk_score"]].copy()
            risk_df["gnn_risk_score"] = risk_df["risk_score"]

        # Load VAE alert scores if available
        alert_path = REPORTS_DIR / "vae_alerts.parquet"
        if alert_path.exists():
            alert_df = pd.read_parquet(alert_path)
        else:
            # Derive from transaction data directly
            suspicious_txn = txn_df[txn_df["is_suspicious"]].groupby("sender_id").agg(
                vae_score=("amount_usd", lambda x: float(x.mean()) / 1e5),
            )
            alert_df = suspicious_txn.reset_index().rename(
                columns={"sender_id": "customer_id"}
            )

        # Merge all info
        merged = cust_df.merge(risk_df, on="customer_id", how="left", suffixes=("", "_gnn"))
        merged = merged.merge(alert_df, on="customer_id", how="left")
        merged["gnn_risk_score"] = merged.get("gnn_risk_score", merged["risk_score"]).fillna(merged["risk_score"])
        merged["vae_score"] = merged.get("vae_score", pd.Series(0.0, index=merged.index)).fillna(0.0)
        merged["combined_score"] = (
            0.5 * merged["risk_score"] + 0.5 * merged["gnn_risk_score"]
        )

        top_suspects = merged.nlargest(TOP_N_ALERTS, "combined_score")

        # Transaction stats per customer
        txn_stats = txn_df.groupby("sender_id").agg(
            total_amount=("amount_usd", "sum"),
            txn_count=("transaction_id", "count"),
            min_date=("timestamp", "min"),
            max_date=("timestamp", "max"),
            structuring_detected=("structuring_flag", "any"),
            cross_border=("is_cross_border", "any") if "is_cross_border" in txn_df.columns else ("country_origin", lambda x: False),
            typologies=("label", lambda x: list(x[x != "normal"].unique())),
        ).reset_index().rename(columns={"sender_id": "customer_id"})

        top_suspects = top_suspects.merge(txn_stats, on="customer_id", how="left")
        top_suspects["total_amount"] = top_suspects["total_amount"].fillna(0.0)
        top_suspects["txn_count"] = top_suspects["txn_count"].fillna(0).astype(int)

        # Generate SARs
        sars: List[SARReport] = []
        for _, row in top_suspects.iterrows():
            suspect = SARSuspectInfo(
                customer_id=row["customer_id"],
                customer_name=row.get("name", "Unknown"),
                country=row.get("country", "N/A"),
                jurisdiction_risk=str(row.get("jurisdiction_risk", "medium")),
                account_type=str(row.get("account_type", "retail")),
                annual_income_usd=float(row.get("annual_income_usd", 0.0)),
                pep=bool(row.get("pep", False)),
                risk_score=float(row.get("risk_score", 0.0)),
                gnn_risk_score=float(row.get("gnn_risk_score", 0.0)),
            )
            min_d = row.get("min_date")
            max_d = row.get("max_date")
            activity = SARActivityInfo(
                typologies_detected=row.get("typologies") or [],
                total_flagged_amount_usd=float(row["total_amount"]),
                flagged_transaction_count=int(row["txn_count"]),
                date_range_start=str(min_d)[:10] if pd.notna(min_d) else "N/A",
                date_range_end=str(max_d)[:10] if pd.notna(max_d) else "N/A",
                anomaly_score_vae=float(row.get("vae_score", 0.0)),
                structuring_detected=bool(row.get("structuring_detected", False)),
                cross_border=bool(row.get("cross_border", False)),
                activity_description="",
            )
            activity.activity_description = _narrative(suspect, activity)

            priority = "CRITICAL" if row["combined_score"] > 0.85 else "HIGH" if row["combined_score"] > 0.65 else "MEDIUM"
            sar = SARReport(suspect=suspect, activity=activity, priority=priority)
            sars.append(sar)

        # Persist
        sar_dicts = [asdict(s) for s in sars]
        (REPORTS_DIR / "sar_reports.json").write_text(json.dumps(sar_dicts, indent=2, default=str))

        summary_rows = [
            {
                "sar_id": s.sar_id,
                "customer_id": s.suspect.customer_id,
                "priority": s.priority,
                "total_flagged_amount_usd": s.activity.total_flagged_amount_usd,
                "flagged_txn_count": s.activity.flagged_transaction_count,
                "gnn_risk_score": s.suspect.gnn_risk_score,
                "vae_score": s.activity.anomaly_score_vae,
                "typologies": ", ".join(s.activity.typologies_detected),
                "status": s.status,
                "generated_at": s.generated_at,
            }
            for s in sars
        ]
        pd.DataFrame(summary_rows).to_csv(REPORTS_DIR / "sar_summary.csv", index=False)
        log.info("Generated %d SAR reports. Saved to %s", len(sars), REPORTS_DIR)
        log.info("=== ReportWriterAgent finished ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ReportWriterAgent().run()
