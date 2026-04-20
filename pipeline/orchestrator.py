"""
pipeline/orchestrator.py
========================
End-to-end AML pipeline orchestrator.

Executes the full pipeline in order:
  1. Data generation
  2. Feature engineering validation
  3. VAE anomaly detection training
  4. GNN network investigation training
  5. SAR report writing

Supports:
  - Full run (all steps)
  - Partial run (--steps flag)
  - Step-level timing and error isolation
  - JSON execution manifest
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
PIPELINE_MANIFEST = REPORTS_DIR / "pipeline_manifest.json"

ALL_STEPS = ["generate_data", "train_vae", "train_gan", "train_gnn", "write_reports"]


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------
@dataclass
class StepResult:
    name: str
    status: str = "pending"   # pending / success / failed / skipped
    duration_s: float = 0.0
    error: Optional[str] = None
    started_at: str = ""
    finished_at: str = ""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class PipelineOrchestrator:
    """
    Runs the full AML detection pipeline sequentially.

    Parameters
    ----------
    steps : list of step names to execute (None = all)
    """

    def __init__(self, steps: Optional[List[str]] = None) -> None:
        self.steps_to_run = set(steps or ALL_STEPS)
        self.results: List[StepResult] = []
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def run(self) -> bool:
        """Execute pipeline; returns True if all steps succeeded."""
        log.info("=" * 60)
        log.info("AML Pipeline starting | steps: %s", sorted(self.steps_to_run))
        log.info("=" * 60)

        step_registry: Dict[str, Callable] = {
            "generate_data": self._step_generate_data,
            "train_vae":     self._step_train_vae,
            "train_gan":     self._step_train_gan,
            "train_gnn":     self._step_train_gnn,
            "write_reports": self._step_write_reports,
        }

        for step_name in ALL_STEPS:
            if step_name not in self.steps_to_run:
                self.results.append(StepResult(name=step_name, status="skipped"))
                log.info("[SKIP] %s", step_name)
                continue

            result = StepResult(
                name=step_name,
                started_at=datetime.utcnow().isoformat(),
            )
            t0 = time.perf_counter()
            try:
                log.info("[RUN ] %s …", step_name)
                step_registry[step_name]()
                result.status = "success"
            except Exception as exc:
                result.status = "failed"
                result.error = traceback.format_exc()
                log.error("[FAIL] %s: %s", step_name, exc)
            finally:
                result.duration_s = round(time.perf_counter() - t0, 3)
                result.finished_at = datetime.utcnow().isoformat()
                self.results.append(result)
                log.info(
                    "[%-7s] %s  (%.1fs)",
                    result.status.upper(), step_name, result.duration_s,
                )

            if result.status == "failed":
                log.error("Pipeline halted after failed step: %s", step_name)
                break

        self._write_manifest()
        success = all(r.status in ("success", "skipped") for r in self.results)
        log.info("Pipeline %s.", "COMPLETED" if success else "FAILED")
        return success

    # ------------------------------------------------------------------
    # Steps
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

    # ------------------------------------------------------------------
    def _write_manifest(self) -> None:
        manifest = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "steps": [asdict(r) for r in self.results],
        }
        PIPELINE_MANIFEST.write_text(json.dumps(manifest, indent=2))
        log.info("Pipeline manifest written to %s", PIPELINE_MANIFEST)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        lines = ["\n── Pipeline Summary ──────────────────"]
        for r in self.results:
            icon = {"success": "✓", "failed": "✗", "skipped": "–", "pending": "?"}.get(r.status, "?")
            lines.append(f"  {icon} {r.name:<20} {r.status:<8}  {r.duration_s:.1f}s")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AML detection pipeline orchestrator"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=None,
        help="Subset of pipeline steps to run (default: all)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    )
    orchestrator = PipelineOrchestrator(steps=args.steps)
    success = orchestrator.run()
    print(orchestrator.summary())
    raise SystemExit(0 if success else 1)