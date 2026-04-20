"""
explainability/shap_explainer.py
=================================
SHAP-based explainability for the VAE anomaly detector.

Provides:
  - Global feature importance (mean |SHAP|)
  - Per-transaction waterfall explanations
  - Background dataset sampling utilities
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    log.warning("shap not installed; explainability features unavailable.")

import torch
from models.vae import VAE


# ---------------------------------------------------------------------------
# Wrapper: VAE reconstruction error as a callable for SHAP
# ---------------------------------------------------------------------------
class VAEAnomalyWrapper:
    """
    Thin wrapper that takes a numpy array, runs the VAE,
    and returns reconstruction error per sample.
    """

    def __init__(self, model: VAE, device: torch.device) -> None:
        self.model = model
        self.device = device
        self.model.eval()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            scores = self.model.anomaly_score(tensor)
        return scores.cpu().numpy()


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------
class VAESHAPExplainer:
    """
    Compute SHAP values for the VAE reconstruction-error scorer.

    Parameters
    ----------
    model        : trained VAE instance
    feature_cols : list of feature names
    device       : torch device
    """

    def __init__(
        self,
        model: VAE,
        feature_cols: List[str],
        device: Optional[torch.device] = None,
    ) -> None:
        if not _HAS_SHAP:
            raise ImportError("Install shap: pip install shap")
        self.device = device or torch.device("cpu")
        self.wrapper = VAEAnomalyWrapper(model, self.device)
        self.feature_cols = feature_cols
        self._explainer: Optional[shap.KernelExplainer] = None
        self._background: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def fit(
        self,
        X_background: np.ndarray,
        n_background: int = 200,
    ) -> "VAESHAPExplainer":
        """
        Fit KernelExplainer on a background dataset sample.

        Parameters
        ----------
        X_background : full training set (scaled)
        n_background : number of background samples (k-means medoids)
        """
        self._background = shap.kmeans(X_background, n_background).data
        self._explainer = shap.KernelExplainer(
            self.wrapper, self._background, link="identity"
        )
        log.info(
            "SHAP KernelExplainer fitted on %d background samples.",
            len(self._background),
        )
        return self

    # ------------------------------------------------------------------
    def explain(
        self,
        X: np.ndarray,
        nsamples: int = 100,
    ) -> "shap.Explanation":
        """
        Compute SHAP values for a batch of samples.

        Returns shap.Explanation object (values, base_values, data).
        """
        if self._explainer is None:
            raise RuntimeError("Call .fit() before .explain()")
        shap_values = self._explainer.shap_values(X, nsamples=nsamples)
        base = self._explainer.expected_value
        return shap.Explanation(
            values=shap_values,
            base_values=base,
            data=X,
            feature_names=self.feature_cols,
        )

    # ------------------------------------------------------------------
    def global_importance(self, shap_values: np.ndarray) -> pd.DataFrame:
        """
        Return a DataFrame with mean absolute SHAP values per feature.
        """
        mean_abs = np.abs(shap_values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": mean_abs})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    def plot_summary(self, X: np.ndarray, shap_values: np.ndarray) -> None:
        """Render a SHAP summary dot plot (requires matplotlib)."""
        shap.summary_plot(shap_values, X, feature_names=self.feature_cols)

    # ------------------------------------------------------------------
    def plot_waterfall(
        self,
        explanation: "shap.Explanation",
        idx: int = 0,
    ) -> None:
        """Render a waterfall plot for a single sample."""
        shap.plots.waterfall(explanation[idx])
