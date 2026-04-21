"""
llm/ollama_writer.py
====================
Ollama-backed SAR narrative writer using local LLMs (Llama 3, Mistral, etc.).

Ollama runs entirely on-premise – no API key, no data leaving the machine.
Falls back to rule-based template generation when Ollama is not running.

Setup
-----
1. Install Ollama:  https://ollama.com/download
2. Pull a model:    ollama pull llama3
3. Start server:    ollama serve          (default: http://localhost:11434)
4. This module auto-detects the running server.

Environment variables
---------------------
OLLAMA_BASE_URL  : Ollama server URL  (default: http://localhost:11434)
OLLAMA_MODEL     : model tag          (default: llama3)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

import requests

log = logging.getLogger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3")
TIMEOUT_S       = 120   # generation timeout

SYSTEM_PROMPT = """\
You are a senior AML (Anti-Money Laundering) compliance analyst at a US financial institution.
Your task is to draft a Suspicious Activity Report (SAR) narrative for submission to FinCEN
under 31 U.S.C. § 5318(g).

Instructions:
- Write in formal, factual compliance language.
- Structure: (1) Subject identification, (2) Suspicious activity description,
  (3) Supporting indicators, (4) Recommended action.
- Do NOT include personal opinions or speculation.
- Length: 150-250 words.
- Use past tense for observed activity.
"""


def _ollama_available() -> bool:
    """Check if Ollama server is reachable."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _models_available() -> list[str]:
    """Return list of pulled model tags."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        data = r.json()
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


class OllamaSARWriter:
    """
    SAR narrative writer backed by a local Ollama LLM.

    Falls back gracefully to rule-based templates when:
    - Ollama is not running
    - The specified model is not pulled
    """

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self._available = _ollama_available()

        if self._available:
            models = _models_available()
            # Check if our model is available; pick first available if not
            if model not in models and models:
                fallback = models[0]
                log.warning("Model '%s' not found in Ollama. Using '%s' instead.", model, fallback)
                self.model = fallback
            elif not models:
                log.warning("No models pulled in Ollama. Run: ollama pull llama3")
                self._available = False
            else:
                log.info("Ollama online – using model '%s'", self.model)
        else:
            log.info("Ollama not running – using template-based SAR writer.")

    # ------------------------------------------------------------------
    def generate(self, context: Dict[str, Any]) -> str:
        """
        Generate a SAR narrative from a context dict.

        Parameters
        ----------
        context : dict with keys:
            customer_id, transaction_count, total_amount_usd,
            gnn_risk_score, vae_score, network_summary, transaction_types
        """
        if self._available:
            return self._generate_ollama(context)
        return self._generate_template(context)

    # ------------------------------------------------------------------
    def _generate_ollama(self, context: Dict[str, Any]) -> str:
        user_prompt = (
            f"Draft a SAR narrative for the following alert context:\n"
            f"{json.dumps(context, indent=2, default=str)}\n\n"
            f"Remember: factual, 150-250 words, FinCEN compliant."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "num_predict": 400,
            },
        }

        try:
            r = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=TIMEOUT_S,
            )
            r.raise_for_status()
            data = r.json()
            content = data.get("message", {}).get("content", "").strip()
            if content:
                return content
        except Exception as exc:
            log.warning("Ollama generation failed (%s) – using template.", exc)

        return self._generate_template(context)

    # ------------------------------------------------------------------
    @staticmethod
    def _generate_template(context: Dict[str, Any]) -> str:
        cid           = context.get("customer_id", "UNKNOWN")
        txn_count     = context.get("transaction_count", 0)
        total_amt     = context.get("total_amount_usd", 0.0)
        gnn_score     = context.get("gnn_risk_score", 0.0)
        vae_score     = context.get("vae_score", 0.0)
        net_summary   = context.get("network_summary", "")
        txn_types     = context.get("transaction_types", [])
        types_str     = ", ".join(txn_types) if txn_types else "various transaction types"

        return (
            f"The filer is reporting suspicious financial activity associated with account "
            f"{cid}. Between the dates referenced herein, this account conducted "
            f"{txn_count} transactions totalling USD {total_amt:,.2f} via {types_str}. "
            f"AI-based anomaly detection flagged this account with a Variational Autoencoder "
            f"reconstruction score of {vae_score:.4f} and a Graph Neural Network risk score "
            f"of {gnn_score:.4f}, both indicating materially elevated risk. "
            f"Network analysis findings: {net_summary} "
            f"The pattern of activity is consistent with layering and/or structuring under "
            f"the Bank Secrecy Act. The filer recommends enhanced due diligence and has "
            f"preserved all relevant transaction records. This SAR is filed voluntarily "
            f"and in good faith pursuant to 31 U.S.C. § 5318(g) and FinCEN guidance. "
            f"The filer requests confidentiality as provided by law."
        )

    # ------------------------------------------------------------------
    def pull_model(self, model_tag: str = "llama3") -> bool:
        """
        Pull a model via Ollama API (blocking until complete).
        Returns True on success.
        """
        if not self._available:
            log.error("Ollama is not running. Start it with: ollama serve")
            return False
        try:
            log.info("Pulling model '%s' from Ollama registry …", model_tag)
            r = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_tag},
                timeout=600,  # large model download
                stream=True,
            )
            for line in r.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "status" in chunk:
                        log.debug("[pull] %s", chunk["status"])
            log.info("Model '%s' pulled successfully.", model_tag)
            self.model = model_tag
            return True
        except Exception as exc:
            log.error("Failed to pull model: %s", exc)
            return False
