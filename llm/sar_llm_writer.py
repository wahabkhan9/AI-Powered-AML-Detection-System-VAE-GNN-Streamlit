"""
llm/sar_llm_writer.py
=====================
LLM-powered SAR narrative enhancement.

Uses an LLM (OpenAI-compatible) to enrich the structured SAR data
into publication-quality compliance narratives.  Falls back to the
rule-based template engine if no API key is available.

Environment variables
---------------------
OPENAI_API_KEY  : OpenAI API key (or compatible endpoint key)
LLM_BASE_URL    : optional override for base URL (e.g. local vLLM)
LLM_MODEL       : model identifier (default: gpt-4o-mini)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

_LLM_AVAILABLE = False
try:
    from openai import OpenAI
    _LLM_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    pass

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL")

SYSTEM_PROMPT = """You are a senior AML compliance analyst at a US financial institution.
Your task is to write a concise, professional Suspicious Activity Report (SAR) narrative
that would be accepted by FinCEN under 31 U.S.C. § 5318(g).

Rules:
- Write in plain, factual language.
- Do not speculate beyond the data provided.
- Structure as: (1) Who, (2) What, (3) When/Where, (4) How, (5) Why suspicious.
- Use FinCEN-approved terminology.
- Keep narrative between 150-250 words.
"""


@dataclass
class SARNarrativeRequest:
    customer_id: str
    country: str
    account_type: str
    total_amount_usd: float
    txn_count: int
    typologies: list[str]
    structuring: bool
    cross_border: bool
    vae_score: float
    gnn_score: float
    risk_score: float
    date_range: str


class LLMSARWriter:
    """
    Wraps an LLM to produce enriched SAR narratives.
    Falls back gracefully to template-based generation.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model
        self._client: Optional[OpenAI] = None
        if _LLM_AVAILABLE:
            kwargs = {"api_key": os.getenv("OPENAI_API_KEY")}
            if DEFAULT_BASE_URL:
                kwargs["base_url"] = DEFAULT_BASE_URL
            self._client = OpenAI(**kwargs)
            log.info("LLM writer initialised with model: %s", model)
        else:
            log.info("No LLM API key found; using template fallback.")

    # ------------------------------------------------------------------
    def generate(self, req: SARNarrativeRequest) -> str:
        """Generate a SAR narrative string for the given alert."""
        if self._client is not None:
            return self._llm_generate(req)
        return self._template_generate(req)

    # ------------------------------------------------------------------
    def _llm_generate(self, req: SARNarrativeRequest) -> str:
        user_content = (
            f"Generate a SAR narrative for the following alert:\n"
            f"{json.dumps(req.__dict__, indent=2, default=str)}"
        )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=350,
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            log.warning("LLM call failed (%s); using template fallback.", exc)
            return self._template_generate(req)

    # ------------------------------------------------------------------
    @staticmethod
    def _template_generate(req: SARNarrativeRequest) -> str:
        typologies = ", ".join(req.typologies) if req.typologies else "anomalous patterns"
        return (
            f"The filer reports suspicious activity by customer {req.customer_id} "
            f"({req.account_type}, {req.country}). "
            f"During {req.date_range}, the subject executed {req.txn_count} transactions "
            f"totalling USD {req.total_amount_usd:,.2f}. "
            f"Detected typologies: {typologies}. "
            f"{'Structuring below $10,000 threshold observed. ' if req.structuring else ''}"
            f"{'Cross-border transactions to high-risk jurisdictions identified. ' if req.cross_border else ''}"
            f"AI anomaly score: {req.vae_score:.4f} (VAE), "
            f"Graph risk score: {req.gnn_score:.4f} (GNN), "
            f"Customer risk rating: {req.risk_score:.4f}. "
            f"Filing made under 31 U.S.C. § 5318(g)."
        )
