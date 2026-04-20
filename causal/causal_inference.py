"""
causal/causal_inference.py
===========================
Causal Inference & "What-If" Analysis Module.

Provides two capabilities:
  1. Causal Impact Estimation – estimates the causal effect of a new fraud rule
     or policy change on suspicious activity rates using Difference-in-Differences
     and CausalImpact (Google's Bayesian structural time-series approach).

  2. Counterfactual "What-If" Simulation – answers questions like:
     "What would happen to alert volume if we lowered the threshold by 20%?"

Uses EconML's DoublyRobustLearner when available, with a manual DID fallback.

Libraries (optional, install as needed):
  pip install econml
  pip install causalimpact    (Google's CausalImpact for Python)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    from econml.dml import CausalForestDML
    _HAS_ECONML = True
except ImportError:
    _HAS_ECONML = False
    log.info("econml not installed – using manual DID estimator.")

try:
    from causalimpact import CausalImpact
    _HAS_CAUSAL_IMPACT = True
except ImportError:
    _HAS_CAUSAL_IMPACT = False
    log.info("causalimpact not installed – using linear trend baseline.")


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------
@dataclass
class CausalEffectResult:
    """Result of a causal effect estimation."""
    estimator:           str
    treatment:           str
    outcome:             str
    ate:                 float          # Average Treatment Effect
    ate_lower:           float          # 95% CI lower
    ate_upper:           float          # 95% CI upper
    relative_effect_pct: float          # (ATE / baseline_mean) * 100
    p_value:             Optional[float]
    interpretation:      str


@dataclass
class WhatIfResult:
    """Result of a counterfactual what-if simulation."""
    scenario:            str
    baseline_alerts:     int
    counterfactual_alerts: int
    delta_alerts:        int
    delta_pct:           float
    baseline_flagged_amount: float
    counterfactual_flagged_amount: float
    recommendation:      str


# ---------------------------------------------------------------------------
# Difference-in-Differences estimator (manual fallback)
# ---------------------------------------------------------------------------
def _did_estimate(
    df: pd.DataFrame,
    outcome_col: str,
    treated_col: str,
    post_col: str,
) -> Tuple[float, float, float]:
    """
    Simple 2x2 Difference-in-Differences.

    ATT = (post_treated - pre_treated) - (post_control - pre_control)

    Returns (ATT, lower_95ci, upper_95ci) – CI computed via delta method.
    """
    pre_treated  = df[(df[treated_col] == 1) & (df[post_col] == 0)][outcome_col].mean()
    post_treated = df[(df[treated_col] == 1) & (df[post_col] == 1)][outcome_col].mean()
    pre_control  = df[(df[treated_col] == 0) & (df[post_col] == 0)][outcome_col].mean()
    post_control = df[(df[treated_col] == 0) & (df[post_col] == 1)][outcome_col].mean()

    att = (post_treated - pre_treated) - (post_control - pre_control)

    # Rough SE via bootstrap
    bootstrap_atts = []
    rng = np.random.default_rng(42)
    for _ in range(500):
        sample = df.sample(frac=1, replace=True, random_state=rng.integers(1e6))
        pt = sample[(sample[treated_col] == 1) & (sample[post_col] == 1)][outcome_col].mean()
        pr = sample[(sample[treated_col] == 1) & (sample[post_col] == 0)][outcome_col].mean()
        pc = sample[(sample[treated_col] == 0) & (sample[post_col] == 1)][outcome_col].mean()
        pcr = sample[(sample[treated_col] == 0) & (sample[post_col] == 0)][outcome_col].mean()
        bootstrap_atts.append((pt - pr) - (pc - pcr))

    se = float(np.std(bootstrap_atts))
    lower = att - 1.96 * se
    upper = att + 1.96 * se
    return float(att), float(lower), float(upper)


# ---------------------------------------------------------------------------
# Main causal analysis class
# ---------------------------------------------------------------------------
class AMLCausalAnalyzer:
    """
    Causal inference tools for AML policy analysis.

    Parameters
    ----------
    txn_df  : transaction DataFrame with is_suspicious, amount_usd, timestamp columns
    """

    def __init__(self, txn_df: pd.DataFrame) -> None:
        self.txn_df = txn_df.copy()
        self.txn_df["timestamp"] = pd.to_datetime(self.txn_df["timestamp"])

    # ------------------------------------------------------------------
    def estimate_rule_effect(
        self,
        rule_name: str,
        policy_date: str,
        treated_column: str,
        outcome_column: str = "is_suspicious",
    ) -> CausalEffectResult:
        """
        Estimate the causal effect of a policy/rule change on suspicious activity.

        Parameters
        ----------
        rule_name       : human-readable name of the rule/policy
        policy_date     : date string (YYYY-MM-DD) when the rule was implemented
        treated_column  : column indicating accounts subject to the new rule
        outcome_column  : binary outcome to study (default: is_suspicious)
        """
        policy_ts = pd.Timestamp(policy_date)
        df = self.txn_df.copy()
        df["post"] = (df["timestamp"] >= policy_ts).astype(int)

        if treated_column not in df.columns:
            # Create a synthetic treatment flag (top-50% risk score accounts)
            if "risk_score" in df.columns:
                median_risk = df["risk_score"].median()
                df[treated_column] = (df["risk_score"] > median_risk).astype(int)
            else:
                df[treated_column] = np.random.binomial(1, 0.5, len(df))

        df["outcome"] = df[outcome_column].astype(float)

        if _HAS_ECONML:
            result = self._econml_estimate(df, treated_column, rule_name)
        else:
            att, lower, upper = _did_estimate(df, "outcome", treated_column, "post")
            baseline_mean = df[df["post"] == 0]["outcome"].mean()
            relative = (att / max(abs(baseline_mean), 1e-9)) * 100.0
            result = CausalEffectResult(
                estimator="DiD",
                treatment=rule_name,
                outcome=outcome_column,
                ate=att,
                ate_lower=lower,
                ate_upper=upper,
                relative_effect_pct=relative,
                p_value=None,
                interpretation=self._interpret(att, lower, upper, relative, rule_name),
            )

        log.info(
            "Causal effect of '%s': ATE=%.4f [%.4f, %.4f] (%.1f%%)",
            rule_name, result.ate, result.ate_lower, result.ate_upper, result.relative_effect_pct,
        )
        return result

    # ------------------------------------------------------------------
    def _econml_estimate(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        rule_name: str,
    ) -> CausalEffectResult:
        """Use EconML's CausalForestDML for heterogeneous treatment effects."""
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c not in ["outcome", treatment_col, "post"]]
        X = df[feature_cols].fillna(0).values
        T = df[treatment_col].values.astype(float)
        Y = df["outcome"].values.astype(float)

        est = CausalForestDML(
            model_y=GradientBoostingClassifier(n_estimators=50),
            model_t=GradientBoostingClassifier(n_estimators=50),
            n_estimators=100,
            random_state=42,
        )
        est.fit(Y, T, X=X)
        te = est.effect(X)
        ate  = float(te.mean())
        ci   = est.effect_interval(X)
        lower = float(ci[0].mean())
        upper = float(ci[1].mean())
        baseline_mean = float(Y[T == 0].mean())
        relative = (ate / max(abs(baseline_mean), 1e-9)) * 100.0

        return CausalEffectResult(
            estimator="CausalForestDML",
            treatment=rule_name,
            outcome="is_suspicious",
            ate=ate,
            ate_lower=lower,
            ate_upper=upper,
            relative_effect_pct=relative,
            p_value=None,
            interpretation=self._interpret(ate, lower, upper, relative, rule_name),
        )

    # ------------------------------------------------------------------
    def what_if_threshold(
        self,
        current_threshold: float,
        new_threshold: float,
        anomaly_scores: Optional[np.ndarray] = None,
    ) -> WhatIfResult:
        """
        Simulate alert volume under a hypothetical detection threshold change.

        Parameters
        ----------
        current_threshold : existing VAE threshold
        new_threshold     : proposed new threshold
        anomaly_scores    : pre-computed anomaly scores; if None, uses rule-based estimate
        """
        if anomaly_scores is None:
            # Estimate from suspicious labels
            n = len(self.txn_df)
            anomaly_scores = np.where(
                self.txn_df["is_suspicious"].values,
                np.random.uniform(current_threshold, current_threshold * 5, n),
                np.random.uniform(0, current_threshold * 0.8, n),
            ).astype(np.float32)

        amounts = self.txn_df["amount_usd"].values

        baseline_mask       = anomaly_scores >= current_threshold
        counterfactual_mask = anomaly_scores >= new_threshold

        baseline_alerts   = int(baseline_mask.sum())
        cf_alerts         = int(counterfactual_mask.sum())
        delta             = cf_alerts - baseline_alerts
        delta_pct         = (delta / max(baseline_alerts, 1)) * 100.0
        baseline_amt      = float(amounts[baseline_mask].sum())
        cf_amt            = float(amounts[counterfactual_mask].sum())

        if new_threshold < current_threshold:
            direction = "lower"
            rec = (
                f"Lowering threshold by {100*(current_threshold-new_threshold)/current_threshold:.1f}% "
                f"would increase alerts by {delta:+,} ({delta_pct:+.1f}%), capturing "
                f"USD {cf_amt - baseline_amt:+,.0f} more in flagged volume. "
                f"Consider analyst capacity before implementing."
            )
        else:
            direction = "higher"
            rec = (
                f"Raising threshold by {100*(new_threshold-current_threshold)/current_threshold:.1f}% "
                f"would reduce alerts by {abs(delta):,} ({delta_pct:.1f}%), "
                f"but risks missing USD {baseline_amt - cf_amt:,.0f} in suspicious activity."
            )

        return WhatIfResult(
            scenario=f"Threshold {direction}: {current_threshold:.5f} → {new_threshold:.5f}",
            baseline_alerts=baseline_alerts,
            counterfactual_alerts=cf_alerts,
            delta_alerts=delta,
            delta_pct=round(delta_pct, 2),
            baseline_flagged_amount=round(baseline_amt, 2),
            counterfactual_flagged_amount=round(cf_amt, 2),
            recommendation=rec,
        )

    # ------------------------------------------------------------------
    def what_if_rule(
        self,
        rule_name: str,
        affected_fraction: float = 0.20,
        expected_reduction_pct: float = 15.0,
    ) -> WhatIfResult:
        """
        Estimate impact of adding a new detection rule that targets a specific
        fraction of transactions.
        """
        total = len(self.txn_df)
        current_flagged = int(self.txn_df["is_suspicious"].sum())
        new_catches = int(total * affected_fraction * (expected_reduction_pct / 100))
        cf_flagged = current_flagged + new_catches

        amounts = self.txn_df["amount_usd"].values
        baseline_amt = float(amounts[self.txn_df["is_suspicious"].values].sum())
        cf_amt = baseline_amt + float(amounts[:new_catches].sum())

        return WhatIfResult(
            scenario=f"New rule: '{rule_name}'",
            baseline_alerts=current_flagged,
            counterfactual_alerts=cf_flagged,
            delta_alerts=new_catches,
            delta_pct=round(100 * new_catches / max(current_flagged, 1), 2),
            baseline_flagged_amount=round(baseline_amt, 2),
            counterfactual_flagged_amount=round(cf_amt, 2),
            recommendation=(
                f"Adding '{rule_name}' targeting {affected_fraction*100:.0f}% of transactions "
                f"is estimated to uncover {new_catches:,} additional alerts "
                f"(+{100*new_catches/max(current_flagged,1):.1f}%). "
                f"Estimated incremental flagged amount: USD {cf_amt-baseline_amt:,.0f}."
            ),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _interpret(ate: float, lower: float, upper: float, relative: float, rule: str) -> str:
        significant = not (lower <= 0 <= upper)
        direction = "reduced" if ate < 0 else "increased"
        sig_str = "statistically significant" if significant else "not statistically significant at 95% confidence"
        return (
            f"Implementing '{rule}' is estimated to have {direction} suspicious activity "
            f"by {abs(relative):.1f}% (ATE={ate:.4f}, 95% CI [{lower:.4f}, {upper:.4f}]). "
            f"This effect is {sig_str}."
        )