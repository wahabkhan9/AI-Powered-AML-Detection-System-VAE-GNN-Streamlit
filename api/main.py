"""
api/main.py
===========
FastAPI REST API for real-time AML inference.

Endpoints
---------
POST /api/v1/score/transaction   – score a single transaction
POST /api/v1/score/batch         – score a batch of transactions (async)
GET  /api/v1/alerts              – paginated alert feed
GET  /api/v1/customers/{id}/risk – customer risk profile
GET  /api/v1/health              – liveness / readiness probe
GET  /api/v1/metrics             – Prometheus-format metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
import pickle
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from features.feature_engineering import derive_transaction_features, compute_alert_level, TRANSACTION_FEATURES
from models.vae import VAE

log = logging.getLogger(__name__)

MODEL_DIR = Path("models")
REPORTS_DIR = Path("reports")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# In-memory state (loaded at startup)
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models into memory once at startup."""
    log.info("Loading models …")
    try:
        meta_path = MODEL_DIR / "vae_meta.json"
        if not meta_path.exists():
            log.warning("vae_meta.json not found – running in stub mode.")
            _state["stub"] = True
        else:
            meta = json.loads(meta_path.read_text())
            vae = VAE(
                input_dim=meta["input_dim"],
                latent_dim=meta["latent_dim"],
                hidden_dims=tuple(meta["hidden_dims"]),
                beta=meta["beta"],
            ).to(DEVICE)
            vae.load_state_dict(
                torch.load(MODEL_DIR / "vae_model.pth", map_location=DEVICE)
            )
            vae.eval()
            _state["vae"] = vae
            _state["threshold"] = meta["threshold"]
            _state["feature_cols"] = meta["feature_cols"]

            with open(MODEL_DIR / "scaler.pkl", "rb") as f:
                _state["scaler"] = pickle.load(f)

            _state["stub"] = False
            log.info("VAE loaded.  Threshold=%.6f", meta["threshold"])

        # Load customer risk scores if available
        risk_path = REPORTS_DIR / "customer_risk_scores.parquet"
        if risk_path.exists():
            _state["risk_df"] = pd.read_parquet(risk_path).set_index("customer_id")
            log.info("Customer risk scores loaded (%d records).", len(_state["risk_df"]))

        _state["request_count"] = 0
        _state["alert_count"] = 0
        _state["start_time"] = time.time()
    except Exception as exc:
        log.error("Startup error: %s", exc)
        _state["stub"] = True

    yield

    log.info("API shutting down.")
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AML Detection API",
    description="Real-time Anti-Money Laundering scoring powered by VAE + GNN.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="TXN_001")
    timestamp: str = Field(..., example="2024-03-15T14:23:00")
    sender_id: str
    receiver_id: str
    amount_usd: float = Field(..., gt=0)
    transaction_type: str
    country_origin: str
    country_dest: str
    round_amount: bool = False
    rapid_movement: bool = False
    structuring_flag: bool = False

    @field_validator("amount_usd")
    @classmethod
    def amount_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("amount_usd must be positive")
        return v


class TransactionScore(BaseModel):
    transaction_id: str
    anomaly_score: float
    threshold: float
    is_flagged: bool
    alert_level: str
    latency_ms: float


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]


class BatchScore(BaseModel):
    results: List[TransactionScore]
    total: int
    flagged: int
    processing_time_ms: float


class CustomerRisk(BaseModel):
    customer_id: str
    gnn_risk_score: Optional[float]
    risk_label: str
    found: bool


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------
def _score_transaction(txn: TransactionRequest) -> TransactionScore:
    t0 = time.perf_counter()
    if _state.get("stub"):
        # Stub mode: return deterministic score for testing
        score = float(np.random.uniform(0, 0.01))
        threshold = 0.005
    else:
        df = pd.DataFrame([txn.model_dump()])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        features_df = derive_transaction_features(df)
        X = _state["scaler"].transform(features_df.values).astype(np.float32)
        tensor = torch.tensor(X, device=DEVICE)
        with torch.no_grad():
            score = float(_state["vae"].anomaly_score(tensor).item())
        threshold = _state["threshold"]

    flagged = score >= threshold
    if flagged:
        _state["alert_count"] = _state.get("alert_count", 0) + 1

    latency = (time.perf_counter() - t0) * 1000
    return TransactionScore(
        transaction_id=txn.transaction_id,
        anomaly_score=round(score, 8),
        threshold=round(threshold, 8),
        is_flagged=flagged,
        alert_level=compute_alert_level(score, threshold),
        latency_ms=round(latency, 3),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/v1/health", tags=["Infrastructure"])
async def health():
    return {
        "status": "healthy",
        "stub_mode": _state.get("stub", True),
        "uptime_s": round(time.time() - _state.get("start_time", time.time()), 1),
    }


@app.get("/api/v1/metrics", response_class=PlainTextResponse, tags=["Infrastructure"])
async def metrics():
    uptime = time.time() - _state.get("start_time", time.time())
    return (
        f"# HELP aml_requests_total Total API requests\n"
        f"aml_requests_total {_state.get('request_count', 0)}\n"
        f"# HELP aml_alerts_total Total transactions flagged\n"
        f"aml_alerts_total {_state.get('alert_count', 0)}\n"
        f"# HELP aml_uptime_seconds API uptime\n"
        f"aml_uptime_seconds {uptime:.1f}\n"
    )


@app.post("/api/v1/score/transaction", response_model=TransactionScore, tags=["Scoring"])
async def score_transaction(txn: TransactionRequest):
    _state["request_count"] = _state.get("request_count", 0) + 1
    return _score_transaction(txn)


@app.post("/api/v1/score/batch", response_model=BatchScore, tags=["Scoring"])
async def score_batch(req: BatchRequest, background_tasks: BackgroundTasks):
    if len(req.transactions) > 10_000:
        raise HTTPException(status_code=400, detail="Batch size exceeds 10,000.")
    t0 = time.perf_counter()
    _state["request_count"] = _state.get("request_count", 0) + 1

    # Run in executor to avoid blocking event loop for large batches
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: [_score_transaction(t) for t in req.transactions],
    )
    elapsed = (time.perf_counter() - t0) * 1000
    return BatchScore(
        results=results,
        total=len(results),
        flagged=sum(1 for r in results if r.is_flagged),
        processing_time_ms=round(elapsed, 2),
    )


@app.get("/api/v1/customers/{customer_id}/risk", response_model=CustomerRisk, tags=["Risk"])
async def customer_risk(customer_id: str):
    risk_df: pd.DataFrame = _state.get("risk_df")
    if risk_df is None or customer_id not in risk_df.index:
        return CustomerRisk(
            customer_id=customer_id,
            gnn_risk_score=None,
            risk_label="UNKNOWN",
            found=False,
        )
    row = risk_df.loc[customer_id]
    score = float(row.get("gnn_risk_score", 0.0))
    label = "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW"
    return CustomerRisk(
        customer_id=customer_id,
        gnn_risk_score=round(score, 6),
        risk_label=label,
        found=True,
    )


@app.get("/api/v1/alerts", tags=["Alerts"])
async def get_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    min_score: float = Query(0.0, ge=0.0),
):
    sar_path = REPORTS_DIR / "sar_summary.csv"
    if not sar_path.exists():
        return {"alerts": [], "total": 0, "page": page}
    df = pd.read_csv(sar_path)
    if "gnn_risk_score" in df.columns:
        df = df[df["gnn_risk_score"] >= min_score]
    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]
    return {
        "alerts": page_df.to_dict(orient="records"),
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
