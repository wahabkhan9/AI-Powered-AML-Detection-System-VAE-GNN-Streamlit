"""
main.py
=======
Unified entry-point for the Autonomous Agentic AI for Financial Crime Defense.

Modes
-----
  python main.py pipeline          Run full training pipeline
  python main.py dashboard         Launch Streamlit dashboard
  python main.py api               Start FastAPI inference server
  python main.py score <json_file> Score transactions from a JSON file
  python main.py demo              Run a quick end-to-end demo with synthetic data

Architecture Overview
---------------------
  generate_data  →  [VAE Analyst]  ─┐
                    [GAN Trainer]   ├──►  [GNN Detective]  ──►  [LLM Narrator]
                                   ─┘         │                       │
                                         LangGraph                Commander
                                        Orchestrator              Assessment
                                              │
                                    Streamlit Dashboard
                                    FastAPI REST API
                                    DuckDB Analytics
                                    Causal What-If
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)-28s – %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------
def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full training + report pipeline."""
    from pipeline.orchestrator import PipelineOrchestrator
    steps = args.steps if hasattr(args, "steps") and args.steps else None
    orch = PipelineOrchestrator(steps=steps)
    success = orch.run()
    print(orch.summary())
    sys.exit(0 if success else 1)


def run_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit compliance dashboard."""
    port = getattr(args, "port", 8501)
    log.info("Starting Streamlit dashboard on port %d …", port)
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "dashboard/app.py",
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ], check=True)


def run_api(args: argparse.Namespace) -> None:
    """Start the FastAPI inference server."""
    port = getattr(args, "port", 8000)
    log.info("Starting FastAPI inference API on port %d …", port)
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--reload",
    ], check=True)


def run_score(args: argparse.Namespace) -> None:
    """Score transactions from a JSON file using the full agent pipeline."""
    json_path = Path(args.json_file)
    if not json_path.exists():
        log.error("File not found: %s", json_path)
        sys.exit(1)

    transactions = json.loads(json_path.read_text())
    if isinstance(transactions, dict):
        transactions = [transactions]

    log.info("Scoring %d transactions …", len(transactions))
    from agents.orchestrator_agent import OrchestratorAgent
    agent = OrchestratorAgent()
    result = agent.process(transactions)

    print("\n" + "=" * 60)
    print(f"  RISK LEVEL : {result['final_risk_level']}")
    print(f"  RISK SCORE : {result['final_risk_score']:.4f}")
    print(f"  FLAGGED    : {len(result['flagged_transaction_ids'])} / {len(transactions)}")
    print(f"  ACTION     : {result['action_recommendation']}")
    print("=" * 60)

    output_path = Path("reports/score_result.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps({
        "risk_level": result["final_risk_level"],
        "risk_score": result["final_risk_score"],
        "flagged_ids": result["flagged_transaction_ids"],
        "action": result["action_recommendation"],
        "network_summary": result["network_summary"],
        "sar_ids": result["sar_ids"],
    }, indent=2))
    log.info("Results saved → %s", output_path)


def run_demo(args: argparse.Namespace) -> None:
    """
    End-to-end demo: generates data, runs all agents, prints summary.
    No training – uses stub models if real ones aren't present.
    """
    import random
    log.info("Running end-to-end DEMO …")

    # Generate 100 synthetic transactions on the fly
    transactions = [
        {
            "transaction_id": f"DEMO_{i:04d}",
            "timestamp": "2024-06-15T10:30:00",
            "sender_id": f"CUST_{random.randint(0, 30):04d}",
            "receiver_id": f"CUST_{random.randint(0, 30):04d}",
            "amount_usd": random.choice([9_500, 9_800, 9_900, random.uniform(100, 50_000)]),
            "transaction_type": random.choice(["WIRE_TRANSFER", "ACH", "CASH_DEPOSIT"]),
            "country_origin": random.choice(["US", "PA", "KY", "BZ"]),
            "country_dest": random.choice(["US", "PA", "KY", "DE"]),
            "round_amount": random.choice([True, False]),
            "rapid_movement": random.choice([True, False]),
            "structuring_flag": random.random() < 0.3,
            "is_suspicious": random.random() < 0.15,
            "label": "normal",
        }
        for i in range(100)
    ]

    from agents.orchestrator_agent import OrchestratorAgent
    agent = OrchestratorAgent()
    result = agent.process(transactions)

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║  AML DEMO – Multi-Agent Pipeline Result" + " " * 18 + "║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Transactions Processed : {len(transactions):<31}║")
    print(f"║  Flagged Alerts         : {len(result['flagged_transaction_ids']):<31}║")
    print(f"║  Customers Investigated : {len(result['customer_ids']):<31}║")
    print(f"║  SARs Generated         : {len(result['sar_ids']):<31}║")
    print(f"║  Final Risk Level       : {result['final_risk_level']:<31}║")
    print(f"║  Final Risk Score       : {result['final_risk_score']:<31.4f}║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Network: {result['network_summary'][:47]:<47}║")
    print("╠" + "═" * 58 + "╣")
    print(f"║  Action: {result['action_recommendation'][:49]:<49}║")
    print("╚" + "═" * 58 + "╝\n")

    if result["sar_narratives"]:
        print("Sample SAR Narrative:")
        print("-" * 60)
        print(result["sar_narratives"][0][:500] + " …")
        print("-" * 60)


def run_whatif(args: argparse.Namespace) -> None:
    """Run causal what-if analysis on the current dataset."""
    txn_path = Path("data/transactions.parquet")
    if not txn_path.exists():
        log.error("No transaction data found. Run: python main.py pipeline")
        sys.exit(1)

    import pandas as pd
    from causal.causal_inference import AMLCausalAnalyzer

    txn_df = pd.read_parquet(txn_path)
    analyzer = AMLCausalAnalyzer(txn_df)

    # What-if: lower threshold by 20%
    # Load current threshold
    meta_path = Path("models/vae_meta.json")
    current_threshold = json.loads(meta_path.read_text())["threshold"] if meta_path.exists() else 0.01
    new_threshold = current_threshold * 0.80

    result = analyzer.what_if_threshold(current_threshold, new_threshold)
    print("\n── What-If Analysis: Threshold Change ──────────────────────")
    print(f"  Scenario        : {result.scenario}")
    print(f"  Baseline Alerts : {result.baseline_alerts:,}")
    print(f"  New Alerts      : {result.counterfactual_alerts:,}  ({result.delta_pct:+.1f}%)")
    print(f"  Recommendation  : {result.recommendation}")

    # What-if: new rule
    rule_result = analyzer.what_if_rule(
        rule_name="Flag all PA/KY/BZ → US wire transfers > $5K",
        affected_fraction=0.12,
        expected_reduction_pct=20.0,
    )
    print("\n── What-If Analysis: New Rule ───────────────────────────────")
    print(f"  Scenario        : {rule_result.scenario}")
    print(f"  New Alerts      : +{rule_result.delta_alerts:,}  ({rule_result.delta_pct:+.1f}%)")
    print(f"  Recommendation  : {rule_result.recommendation}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Autonomous Agentic AI for Financial Crime Defense",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py pipeline                      Run full pipeline
  python main.py pipeline --steps train_vae    Run only VAE training
  python main.py dashboard                     Launch Streamlit UI
  python main.py api --port 9000               Start API on port 9000
  python main.py score transactions.json       Score a JSON file
  python main.py demo                          Quick end-to-end demo
  python main.py whatif                        What-if threshold analysis
        """,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # pipeline
    p_pipe = sub.add_parser("pipeline", help="Run training pipeline")
    p_pipe.add_argument("--steps", nargs="+", choices=[
        "generate_data", "train_vae", "train_gan", "train_gnn", "write_reports"
    ])

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch Streamlit dashboard")
    p_dash.add_argument("--port", type=int, default=8501)

    # api
    p_api = sub.add_parser("api", help="Start FastAPI server")
    p_api.add_argument("--port", type=int, default=8000)

    # score
    p_score = sub.add_parser("score", help="Score transactions from JSON file")
    p_score.add_argument("json_file", help="Path to JSON file with transaction list")

    # demo
    sub.add_parser("demo", help="Run quick end-to-end demo")

    # whatif
    sub.add_parser("whatif", help="Run causal what-if analysis")

    return parser


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    commands = {
        "pipeline":  run_pipeline,
        "dashboard": run_dashboard,
        "api":       run_api,
        "score":     run_score,
        "demo":      run_demo,
        "whatif":    run_whatif,
    }
    commands[args.command](args)
