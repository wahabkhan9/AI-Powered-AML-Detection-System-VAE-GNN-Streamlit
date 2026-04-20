"""
tests/conftest.py
=================
Shared pytest fixtures and configuration.
"""
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch


@pytest.fixture(autouse=True)
def set_seeds():
    """Fix all random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    yield
