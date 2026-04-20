"""
gnn/train_gnn.py
================
Standalone entry-point for GNN training (delegates to NetworkInvestigatorAgent).
Can also be run directly:  python -m gnn.train_gnn
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.network_investigator import NetworkInvestigatorAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)

if __name__ == "__main__":
    NetworkInvestigatorAgent().run()
