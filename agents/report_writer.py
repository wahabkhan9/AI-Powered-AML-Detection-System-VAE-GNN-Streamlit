"""
agents/report_writer.py
=======================
SAR Report Writer Agent – generates Suspicious Activity Reports for the
highest-risk alerts using Ollama (local LLM) as the primary narrative engine.

Ollama integration
------------------
- Auto-detects running Ollama server at http://localhost:11434
- Uses llama3 (or any pulled model) to generate FinCEN-quality narratives
- Falls back to high-quality rule-based templates if Ollama is offline
- Every SAR narrative goes through the LLM pipeline – no bypassing

Output
------
reports/sar_reports.json
reports/sar_summary.csv
reports/sar_narratives.csv   (raw LLM outputs per customer)
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

DATA_DIR    = Path("data")
REPORTS_DIR = Path("reports")
MODEL_DIR   = Path("models")

INSTITUTION_NAME    = "Acme Financial Services Inc."
INSTITUTION_EIN     = "12-3456789"
INSTITUTION_CONTACT = "AML Compliance | aml@acmefs.com | +1-800-555-0100"
TOP_N_ALERTS        = 200


# ---------------------------------------------------------------------------
# SAR data classes
# ---------------------------------------------------------------------------
@dataclass
class SARFilerInfo:
    institution_name: str = INSTITUTION_NAME
    ein:              str = INSTITUTION_EIN
    contact:          str = INSTITUTION_CONTACT
    report_date:      str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"))
    filing_type:      str = "INITIAL"


@dataclass
class SARSuspectInfo:
    customer_id:        str
    customer_name:      str
    country:            str
    jurisdiction_risk:  str
    account_type:       str
    annual_income_usd:  float
    pep:                bool
    risk_score:         float
    gnn_risk_score:     float
    vae_score:          float


@dataclass
class SARActivityInfo:
    typologies_detected:      List[str]
    total_flagged_amount_usd: float
    flagged_transaction_count: int
    date_range_start:         str
    date_range_end:           str
    structuring_detected:     bool
    cross_border:             bool
    activity_description:     str   # ← populated by Ollama or template


@dataclass
class SARReport:
    sar_id:       str = field(default_factory=lambda: "SAR-" + uuid.uuid4().hex[:8].upper())
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    priority:     str = "HIGH"
    llm_used:     str = "unknown"   # "ollama/llama3", "template", etc.
    filer:        Optional[SARFilerInfo]    = None
    suspect:      Optional[SARSuspectInfo] = None
    activity:     Optional[SARActivityInfo] = None
    status:       str = "DRAFT"


# ---------------------------------------------------------------------------
# Ollama narrative generation
# ---------------------------------------------------------------------------
def _generate_narrative_via_ollama(
    suspect: SARSuspectInfo,
    activity: SARActivityInfo,
    writer,
) -> tuple[str, str]:
    """
    Call Ollama (or template fallback) to generate a SAR narrative.

    Returns (narrative_text, llm_label)
    where llm_label is e.g. 'ollama/llama3' or 'template'
    """
    context = {
        "customer_id":         suspect.customer_id,
        "country":             suspect.country,
        "account_type":        suspect.account_type,
        "pep":                 suspect.pep,
        "risk_score":          suspect.risk_score,
        "jurisdiction_risk":   suspect.jurisdiction_risk,
        "transaction_count":   activity.flagged_transaction_count,
        "total_amount_usd":    activity.total_flagged_amount_usd,
        "gnn_risk_score":      suspect.gnn_risk_score,
        "vae_score":           suspect.vae_score,
        "network_summary":     f"Typologies: {', '.join(activity.typologies_detected)}",
        "transaction_types":   activity.typologies_detected,
        "structuring":         activity.structuring_detected,
        "cross_border":        activity.cross_border,
        "date_range":          f"{activity.date_range_start} to {activity.date_range_end}",
    }

    narrative = writer.generate(context)
    label     = f"ollama/{writer.model}" if writer._available else "template"
    return narrative, label


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class ReportWriterAgent:
    """
    Narrator Agent – generates SAR reports using Ollama local LLM
    for every high-risk customer alert.
    """

    def run(self) -> None:
        log.info("=== ReportWriterAgent starting ===")
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # ── Initialise Ollama writer (auto-detects server) ───────────────────
        from llm.ollama_writer import OllamaSARWriter
        writer = OllamaSARWriter()
        log.info(
            "SAR writer: %s",
            f"Ollama/{writer.model}" if writer._available else "template fallback"
        )

        # ── Load data ─────────────────────────────────────────────────────────
        cust_df = pd.read_parquet(DATA_DIR / "customers.parquet")
        txn_df  = pd.read_parquet(DATA_DIR / "transactions.parquet")
        txn_df["timestamp"] = pd.to_datetime(txn_df["timestamp"])

        # GNN risk scores
        risk_path = REPORTS_DIR / "customer_risk_scores.parquet"
        risk_df = pd.read_parquet(risk_path) if risk_path.exists() else pd.DataFrame()

        # VAE alert scores (combined VAE+GAN)
        alert_path = REPORTS_DIR / "vae_alerts.parquet"
        if alert_path.exists():
            alert_df = pd.read_parquet(alert_path)
        else:
            log.warning("vae_alerts.parquet not found – deriving scores from transaction labels.")
            alert_df = (
                txn_df[txn_df["is_suspicious"]]
                .groupby("sender_id")
                .agg(vae_score=("amount_usd", lambda x: float(x.mean()) / 1e5))
                .reset_index()
                .rename(columns={"sender_id": "customer_id"})
            )

        # ── Merge everything ──────────────────────────────────────────────────
        merged = cust_df.merge(risk_df, on="customer_id", how="left", suffixes=("", "_gnn"))
        merged = merged.merge(alert_df, on="customer_id", how="left")
        merged["gnn_risk_score"]  = merged.get("gnn_risk_score", merged["risk_score"]).fillna(merged["risk_score"])
        merged["vae_score"]       = merged.get("vae_score", pd.Series(0.0, index=merged.index)).fillna(0.0)
        merged["combined_score"]  = 0.5 * merged["risk_score"] + 0.5 * merged["gnn_risk_score"]

        top_suspects = merged.nlargest(TOP_N_ALERTS, "combined_score")

        # Transaction stats per customer
        txn_stats = txn_df.groupby("sender_id").agg(
            total_amount  = ("amount_usd",    "sum"),
            txn_count     = ("transaction_id", "count"),
            min_date      = ("timestamp",      "min"),
            max_date      = ("timestamp",      "max"),
            structuring   = ("structuring_flag", "any"),
            cross_border  = ("is_suspicious",   "any"),   # proxy
            typologies    = ("label", lambda x: list(x[x != "normal"].unique())),
        ).reset_index().rename(columns={"sender_id": "customer_id"})

        top_suspects = top_suspects.merge(txn_stats, on="customer_id", how="left")
        top_suspects["total_amount"] = top_suspects["total_amount"].fillna(0.0)
        top_suspects["txn_count"]    = top_suspects["txn_count"].fillna(0).astype(int)

        # ── Generate SARs via Ollama ──────────────────────────────────────────
        sars: List[SARReport] = []
        narrative_rows: List[dict] = []
        llm_counts: dict[str, int] = {}

        total = len(top_suspects)
        log.info("Generating %d SARs …", total)

        for idx, (_, row) in enumerate(top_suspects.iterrows(), 1):
            if idx % 20 == 0:
                log.info("  Progress: %d / %d", idx, total)

            suspect = SARSuspectInfo(
                customer_id       = str(row["customer_id"]),
                customer_name     = str(row.get("name", "Unknown")),
                country           = str(row.get("country", "N/A")),
                jurisdiction_risk = str(row.get("jurisdiction_risk", "medium")),
                account_type      = str(row.get("account_type", "retail")),
                annual_income_usd = float(row.get("annual_income_usd", 0.0)),
                pep               = bool(row.get("pep", False)),
                risk_score        = float(row.get("risk_score", 0.0)),
                gnn_risk_score    = float(row.get("gnn_risk_score", 0.0)),
                vae_score         = float(row.get("vae_score", 0.0)),
            )

            min_d = row.get("min_date")
            max_d = row.get("max_date")
            activity = SARActivityInfo(
                typologies_detected      = row.get("typologies") or [],
                total_flagged_amount_usd = float(row["total_amount"]),
                flagged_transaction_count = int(row["txn_count"]),
                date_range_start         = str(min_d)[:10] if pd.notna(min_d) else "N/A",
                date_range_end           = str(max_d)[:10] if pd.notna(max_d) else "N/A",
                structuring_detected     = bool(row.get("structuring", False)),
                cross_border             = bool(row.get("cross_border", False)),
                activity_description     = "",  # filled below
            )

            # ── OLLAMA CALL (or template fallback) ────────────────────────
            narrative, llm_label = _generate_narrative_via_ollama(suspect, activity, writer)
            activity.activity_description = narrative
            llm_counts[llm_label] = llm_counts.get(llm_label, 0) + 1

            score = row["combined_score"]
            priority = "CRITICAL" if score > 0.85 else "HIGH" if score > 0.65 else "MEDIUM"

            sar = SARReport(
                suspect  = suspect,
                activity = activity,
                priority = priority,
                llm_used = llm_label,
                filer    = SARFilerInfo(),
            )
            sars.append(sar)
            narrative_rows.append({
                "sar_id":    sar.sar_id,
                "customer_id": suspect.customer_id,
                "llm_used":  llm_label,
                "narrative": narrative[:500],   # truncate for CSV
            })

        # ── Persist ────────────────────────────────────────────────────────
        sar_dicts = [asdict(s) for s in sars]
        (REPORTS_DIR / "sar_reports.json").write_text(
            json.dumps(sar_dicts, indent=2, default=str)
        )

        summary_rows = [
            {
                "sar_id":                   s.sar_id,
                "customer_id":              s.suspect.customer_id,
                "priority":                 s.priority,
                "llm_used":                 s.llm_used,
                "total_flagged_amount_usd": s.activity.total_flagged_amount_usd,
                "flagged_txn_count":        s.activity.flagged_transaction_count,
                "gnn_risk_score":           s.suspect.gnn_risk_score,
                "vae_score":                s.suspect.vae_score,
                "typologies":               ", ".join(s.activity.typologies_detected),
                "status":                   s.status,
                "generated_at":             s.generated_at,
            }
            for s in sars
        ]
        pd.DataFrame(summary_rows).to_csv(REPORTS_DIR / "sar_summary.csv", index=False)
        pd.DataFrame(narrative_rows).to_csv(REPORTS_DIR / "sar_narratives.csv", index=False)

        log.info(
            "Generated %d SARs.  LLM usage: %s",
            len(sars),
            {k: v for k, v in sorted(llm_counts.items(), key=lambda x: -x[1])},
        )
        log.info("=== ReportWriterAgent finished ===")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ReportWriterAgent().run()