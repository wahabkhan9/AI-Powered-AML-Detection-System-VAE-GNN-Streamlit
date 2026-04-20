"""
airflow/dags/aml_pipeline_dag.py
==================================
Apache Airflow DAG – Scheduled AML Detection Pipeline.

Runs the full pipeline on a daily schedule:
  1. generate_data      (or ingest real data)
  2. train_vae          (retrain or fine-tune anomaly detector)
  3. train_gan          (retrain/update GAN)
  4. train_gnn          (retrain network investigator)
  5. write_reports      (generate SARs)
  6. run_eda            (refresh analytics)
  7. notify             (email / Slack alert with summary)

Usage
-----
1. Install Airflow:  pip install apache-airflow
2. Copy this file to $AIRFLOW_HOME/dags/
3. Set AIRFLOW_HOME and start: airflow standalone
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Guard: only import Airflow when it's actually installed
# ---------------------------------------------------------------------------
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    from airflow.utils.dates import days_ago
    from airflow.models import Variable
    _HAS_AIRFLOW = True
except ImportError:
    _HAS_AIRFLOW = False

if not _HAS_AIRFLOW:
    raise ImportError(
        "apache-airflow is required to use this DAG. "
        "Install with: pip install apache-airflow"
    )

# Project root on the Python path
PROJECT_ROOT = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Default args
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "aml-team",
    "depends_on_past": False,
    "email": [os.getenv("AML_ALERT_EMAIL", "aml@example.com")],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=6),
}


# ---------------------------------------------------------------------------
# Callable task functions
# ---------------------------------------------------------------------------
def task_generate_data(**context) -> None:
    from generate_data import generate_dataset
    generate_dataset()


def task_train_vae(**context) -> None:
    from agents.anomaly_detector import AnomalyDetectorAgent
    AnomalyDetectorAgent().run()


def task_train_gan(**context) -> None:
    from agents.gan_trainer import GANTrainerAgent
    GANTrainerAgent().run()


def task_train_gnn(**context) -> None:
    from agents.network_investigator import NetworkInvestigatorAgent
    NetworkInvestigatorAgent().run()


def task_write_reports(**context) -> None:
    from agents.report_writer import ReportWriterAgent
    ReportWriterAgent().run()


def task_run_eda(**context) -> None:
    import subprocess
    subprocess.run(
        [sys.executable, "notebooks/01_eda.py"],
        cwd=PROJECT_ROOT,
        check=True,
    )


def task_notify_summary(**context) -> None:
    """
    Push a summary notification (logs it; extend with Slack/email as needed).
    """
    import json
    from pathlib import Path

    manifest_path = Path(PROJECT_ROOT) / "reports" / "pipeline_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        steps = manifest.get("steps", [])
        summary = "\n".join(
            f"  {'✓' if s['status']=='success' else '✗'} {s['name']} ({s['duration_s']:.1f}s)"
            for s in steps
        )
        print(f"\nAML Pipeline completed at {manifest['generated_at']}\n{summary}")
    else:
        print("Pipeline manifest not found – skipping notification.")


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="aml_detection_pipeline",
    description="Daily AML detection pipeline: data → VAE → GAN → GNN → SARs",
    schedule_interval="0 2 * * *",   # 02:00 UTC daily
    start_date=days_ago(1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    max_active_runs=1,
    tags=["aml", "ml", "finance", "compliance"],
) as dag:

    dag.doc_md = """
    # AML Detection Pipeline

    Runs the full AI-powered AML detection system daily at 02:00 UTC.

    **Steps:**
    1. Generate / ingest transaction data
    2. Train VAE anomaly detector
    3. Train GAN for synthetic augmentation
    4. Train GNN network investigator
    5. Write SAR reports
    6. Refresh EDA analytics
    7. Send pipeline summary notification
    """

    t_generate = PythonOperator(
        task_id="generate_data",
        python_callable=task_generate_data,
        doc_md="Generate synthetic transaction dataset.",
    )

    t_train_vae = PythonOperator(
        task_id="train_vae",
        python_callable=task_train_vae,
        doc_md="Train VAE anomaly detector on normal transactions.",
    )

    t_train_gan = PythonOperator(
        task_id="train_gan",
        python_callable=task_train_gan,
        doc_md="Train GAN for synthetic suspicious sample generation.",
    )

    t_train_gnn = PythonOperator(
        task_id="train_gnn",
        python_callable=task_train_gnn,
        doc_md="Train GNN network investigator on transaction graph.",
    )

    t_reports = PythonOperator(
        task_id="write_reports",
        python_callable=task_write_reports,
        doc_md="Generate SAR reports for high-risk alerts.",
    )

    t_eda = PythonOperator(
        task_id="run_eda",
        python_callable=task_run_eda,
        doc_md="Refresh EDA figures and analytics.",
    )

    t_notify = PythonOperator(
        task_id="notify_summary",
        python_callable=task_notify_summary,
        trigger_rule="all_done",   # run even if upstream fails
        doc_md="Send pipeline completion notification.",
    )

    # ── Task dependencies ────────────────────────────────────────────────────
    t_generate >> [t_train_vae, t_train_gan]
    t_train_vae >> t_train_gnn
    t_train_gan >> t_train_gnn
    t_train_gnn >> t_reports
    t_reports >> t_eda
    t_eda >> t_notify