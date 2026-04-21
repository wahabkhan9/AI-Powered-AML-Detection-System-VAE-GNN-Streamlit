"""
pipeline/orchestrator.py
========================
End-to-end AML pipeline orchestrator.

Steps (in order)
----------------
  1. generate_data    – synthetic transaction dataset
  2. train_vae        – VAE anomaly detector + GAN fusion + SHAP
  3. train_gan        – GAN for synthetic suspicious augmentation
  4. train_gnn        – GNN network investigator
  5. write_reports    – SAR report generation via Ollama / templates
  6. run_duckdb       – DuckDB analytical queries → CSV reports (FULLY WIRED)
  7. run_eda          – Matplotlib EDA figures

CLI: python -m pipeline.orchestrator [--steps step1 step2 ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)

REPORTS_DIR      = Path("reports")
DATA_DIR         = Path("data")
PIPELINE_MANIFEST = REPORTS_DIR / "pipeline_manifest.json"

ALL_STEPS = [
    "generate_data",
    "train_vae",
    "train_gan",
    "train_gnn",
    "write_reports",
    "run_duckdb",
    "run_eda",
]


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    name:        str
    status:      str  = "pending"   # pending / success / failed / skipped
    duration_s:  float = 0.0
    error:       Optional[str] = None
    started_at:  str = ""
    finished_at: str = ""
    artifacts:   List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DuckDB analytics step – FULLY WIRED
# ---------------------------------------------------------------------------
def _step_run_duckdb() -> None:
    """
    Run DuckDB analytical queries over the generated Parquet files.
    Outputs CSV reports to reports/ for dashboard consumption.
    """
    try:
        from data.duckdb_queries import AMLQueryEngine
    except ImportError:
        log.warning("duckdb not installed – skipping analytics. pip install duckdb")
        return

    txn_path = DATA_DIR / "transactions.parquet"
    if not txn_path.exists():
        log.warning("transactions.parquet not found – run generate_data first.")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with AMLQueryEngine() as engine:
        queries = {
            "transaction_summary":        engine.transaction_summary,
            "top_suspicious_customers":   lambda: engine.top_suspicious_customers(50),
            "daily_alert_trend":          lambda: engine.daily_alert_trend(90),
            "typology_breakdown":         engine.typology_breakdown,
            "cross_border_analysis":      engine.cross_border_analysis,
            "structuring_alerts":         engine.structuring_alerts,
            "hourly_velocity_anomalies":  engine.hourly_velocity_anomalies,
        }

        manifest_rows = []
        for name, fn in queries.items():
            try:
                df = fn()
                out_path = REPORTS_DIR / f"duckdb_{name}.csv"
                df.to_csv(out_path, index=False)
                log.info("DuckDB %-35s → %d rows → %s", name, len(df), out_path)
                manifest_rows.append({"query": name, "rows": len(df), "file": str(out_path)})
            except Exception as exc:
                log.warning("DuckDB query '%s' failed: %s", name, exc)
                manifest_rows.append({"query": name, "rows": 0, "error": str(exc)})

        # Additionally run high-risk network join if risk scores exist
        risk_path = REPORTS_DIR / "customer_risk_scores.parquet"
        if risk_path.exists():
            try:
                df = engine.high_risk_network_customers()
                df.to_csv(REPORTS_DIR / "duckdb_high_risk_network.csv", index=False)
                log.info("DuckDB high_risk_network → %d rows", len(df))
                manifest_rows.append({
                    "query": "high_risk_network",
                    "rows": len(df),
                    "file": str(REPORTS_DIR / "duckdb_high_risk_network.csv"),
                })
            except Exception as exc:
                log.warning("high_risk_network query failed: %s", exc)

    # Write DuckDB run manifest
    (REPORTS_DIR / "duckdb_manifest.json").write_text(
        json.dumps({"run_at": datetime.utcnow().isoformat(), "queries": manifest_rows}, indent=2)
    )
    log.info("DuckDB analytics complete. %d query outputs saved to reports/", len(manifest_rows))


def _step_run_eda() -> None:
    import subprocess, sys
    eda_path = Path("notebooks/01_eda.py")
    if not eda_path.exists():
        log.warning("EDA notebook not found.")
        return
    subprocess.run([sys.executable, str(eda_path)], check=False)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class PipelineOrchestrator:
    """
    Runs the AML pipeline step-by-step with timing, error isolation,
    and a JSON manifest of results.
    """

    def __init__(self, steps: Optional[List[str]] = None) -> None:
        self.steps_to_run = set(steps or ALL_STEPS)
        self.results: List[StepResult] = []
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def run(self) -> bool:
        log.info("=" * 60)
        log.info("AML Pipeline  |  steps: %s", sorted(self.steps_to_run))
        log.info("=" * 60)

        registry: Dict[str, Callable] = {
            "generate_data": self._step_generate_data,
            "train_vae":     self._step_train_vae,
            "train_gan":     self._step_train_gan,
            "train_gnn":     self._step_train_gnn,
            "write_reports": self._step_write_reports,
            "run_duckdb":    _step_run_duckdb,      # ← DuckDB fully wired
            "run_eda":       _step_run_eda,
        }

        for step_name in ALL_STEPS:
            if step_name not in self.steps_to_run:
                self.results.append(StepResult(name=step_name, status="skipped"))
                log.info("[SKIP] %-20s", step_name)
                continue

            result = StepResult(
                name=step_name,
                started_at=datetime.utcnow().isoformat(),
            )
            t0 = time.perf_counter()
            try:
                log.info("[RUN ] %-20s …", step_name)
                registry[step_name]()
                result.status = "success"
                # Record key output artefacts
                result.artifacts = self._scan_artifacts(step_name)
            except Exception as exc:
                result.status = "failed"
                result.error  = traceback.format_exc()
                log.error("[FAIL] %s: %s", step_name, exc)
            finally:
                result.duration_s   = round(time.perf_counter() - t0, 3)
                result.finished_at  = datetime.utcnow().isoformat()
                self.results.append(result)
                log.info("[%-7s] %-20s  %.1fs", result.status.upper(), step_name, result.duration_s)

            if result.status == "failed":
                log.error("Pipeline halted at failed step: %s", step_name)
                break

        self._write_manifest()
        success = all(r.status in ("success", "skipped") for r in self.results)
        log.info("Pipeline %s.", "COMPLETED ✓" if success else "FAILED ✗")
        return success

    # ------------------------------------------------------------------
    @staticmethod
    def _scan_artifacts(step_name: str) -> List[str]:
        """Return list of key output files produced by this step."""
        artifact_map = {
            "generate_data": ["data/transactions.parquet", "data/customers.parquet"],
            "train_vae":     ["models/vae_model.pth", "models/scaler.pkl",
                              "reports/vae_alerts.parquet",
                              "reports/shap_feature_importance.csv"],
            "train_gan":     ["models/gan_generator.pth", "models/gan_discriminator.pth"],
            "train_gnn":     ["models/gnn_model.pth", "reports/customer_risk_scores.parquet"],
            "write_reports": ["reports/sar_reports.json", "reports/sar_summary.csv"],
            "run_duckdb":    [str(p) for p in REPORTS_DIR.glob("duckdb_*.csv")],
            "run_eda":       [str(p) for p in REPORTS_DIR.glob("eda_*.png")],
        }
        return [p for p in artifact_map.get(step_name, []) if Path(p).exists()]

    # ------------------------------------------------------------------
    def _write_manifest(self) -> None:
        manifest = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "steps": [asdict(r) for r in self.results],
        }
        PIPELINE_MANIFEST.write_text(json.dumps(manifest, indent=2))
        log.info("Manifest → %s", PIPELINE_MANIFEST)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        icons = {"success": "✓", "failed": "✗", "skipped": "–", "pending": "?"}
        lines = ["\n── Pipeline Summary ─────────────────────────"]
        for r in self.results:
            arts = f"  [{len(r.artifacts)} files]" if r.artifacts else ""
            lines.append(f"  {icons.get(r.status,'?')} {r.name:<20} {r.status:<8} {r.duration_s:>6.1f}s{arts}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Step callables
    # ------------------------------------------------------------------
    @staticmethod
    def _step_generate_data() -> None:
        from generate_data import generate_dataset
        generate_dataset()

    @staticmethod
    def _step_train_vae() -> None:
        from agents.anomaly_detector import AnomalyDetectorAgent
        AnomalyDetectorAgent().run()

    @staticmethod
    def _step_train_gan() -> None:
        from agents.gan_trainer import GANTrainerAgent
        GANTrainerAgent().run()

    @staticmethod
    def _step_train_gnn() -> None:
        from agents.network_investigator import NetworkInvestigatorAgent
        NetworkInvestigatorAgent().run()

    @staticmethod
    def _step_write_reports() -> None:
        from agents.report_writer import ReportWriterAgent
        ReportWriterAgent().run()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AML pipeline orchestrator")
    parser.add_argument("--steps", nargs="+", choices=ALL_STEPS,
                        help="Subset of steps to run (default: all)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )
    orch = PipelineOrchestrator(steps=args.steps)
    ok   = orch.run()
    print(orch.summary())
    raise SystemExit(0 if ok else 1)