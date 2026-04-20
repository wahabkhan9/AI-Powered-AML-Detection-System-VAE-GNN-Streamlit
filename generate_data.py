"""
generate_data.py
================
Synthetic AML transaction dataset generator.

Produces a realistic dataset of financial transactions incorporating known
money-laundering typologies:
  - Structuring (smurfing)
  - Layering / round-tripping
  - Trade-Based Money Laundering (TBML)
  - Rapid fan-out / fan-in

Output
------
data/transactions.parquet
data/customers.parquet
data/transaction_edges.parquet  (source_id, dest_id for GNN graph)
"""

from __future__ import annotations

import hashlib
import logging
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from faker import Faker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

TOTAL_CUSTOMERS = 5_000
TOTAL_TRANSACTIONS = 336_254
SUSPICIOUS_CUSTOMER_RATE = 0.04          # 4 % of customers are launderers
SAR_THRESHOLD_USD = 10_000               # US reporting threshold
OUTPUT_DIR = Path("data")

JURISDICTIONS = {
    "low":    ["US", "UK", "DE", "JP", "AU"],
    "medium": ["MX", "BR", "ZA", "IN", "TH"],
    "high":   ["PA", "KY", "BZ", "VU", "LB"],
}

TRANSACTION_TYPES = [
    "WIRE_TRANSFER", "ACH", "CASH_DEPOSIT", "CASH_WITHDRAWAL",
    "TRADE_PAYMENT", "INTERNAL_TRANSFER", "CRYPTO_EXCHANGE", "CHECK",
]

fake = Faker()
Faker.seed(SEED)


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------
@dataclass
class Customer:
    customer_id: str
    name: str
    country: str
    jurisdiction_risk: str         # low / medium / high
    account_type: str              # retail / corporate / correspondent
    onboard_date: datetime
    risk_score: float              # 0-1, higher = riskier
    is_suspicious: bool
    annual_income_usd: float
    pep: bool                      # Politically Exposed Person


@dataclass
class Transaction:
    transaction_id: str
    timestamp: datetime
    sender_id: str
    receiver_id: str
    amount_usd: float
    transaction_type: str
    country_origin: str
    country_dest: str
    is_suspicious: bool
    label: str                     # "normal" / typology name
    structuring_flag: bool
    round_amount: bool
    rapid_movement: bool


# ---------------------------------------------------------------------------
# Customer factory
# ---------------------------------------------------------------------------
def _jurisdiction_risk(country: str) -> str:
    for risk, countries in JURISDICTIONS.items():
        if country in countries:
            return risk
    return "medium"


def generate_customers(n: int = TOTAL_CUSTOMERS) -> List[Customer]:
    customers: List[Customer] = []
    all_countries = (
        JURISDICTIONS["low"] + JURISDICTIONS["medium"] + JURISDICTIONS["high"]
    )
    weights = [0.60 / 5] * 5 + [0.28 / 5] * 5 + [0.12 / 5] * 5

    n_suspicious = int(n * SUSPICIOUS_CUSTOMER_RATE)
    suspicious_ids = set(random.sample(range(n), n_suspicious))

    for i in range(n):
        is_susp = i in suspicious_ids
        country = random.choices(all_countries, weights=weights, k=1)[0]
        jur = _jurisdiction_risk(country)
        # Suspicious customers skew toward high-risk jurisdictions
        if is_susp and random.random() < 0.6:
            country = random.choice(JURISDICTIONS["high"])
            jur = "high"

        base_risk = {"low": 0.1, "medium": 0.4, "high": 0.7}[jur]
        risk_score = min(1.0, np.random.beta(2, 5) + base_risk * 0.5)
        if is_susp:
            risk_score = min(1.0, risk_score + np.random.uniform(0.3, 0.5))

        cust = Customer(
            customer_id=f"CUST_{i:06d}",
            name=fake.name(),
            country=country,
            jurisdiction_risk=jur,
            account_type=random.choices(
                ["retail", "corporate", "correspondent"],
                weights=[0.70, 0.25, 0.05],
            )[0],
            onboard_date=fake.date_time_between("-5y", "-6m"),
            risk_score=round(risk_score, 4),
            is_suspicious=is_susp,
            annual_income_usd=round(
                np.random.lognormal(mean=10.8, sigma=0.9), 2
            ),
            pep=(random.random() < 0.02),
        )
        customers.append(cust)

    log.info("Generated %d customers (%d suspicious)", n, n_suspicious)
    return customers


# ---------------------------------------------------------------------------
# Transaction generators
# ---------------------------------------------------------------------------
def _make_txn_id() -> str:
    return "TXN_" + uuid.uuid4().hex[:12].upper()


def _normal_transaction(
    customers: List[Customer],
    ts: datetime,
) -> Transaction:
    sender, receiver = random.sample(customers, 2)
    amount = round(np.random.lognormal(mean=7.5, sigma=1.4), 2)
    return Transaction(
        transaction_id=_make_txn_id(),
        timestamp=ts,
        sender_id=sender.customer_id,
        receiver_id=receiver.customer_id,
        amount_usd=amount,
        transaction_type=random.choice(TRANSACTION_TYPES),
        country_origin=sender.country,
        country_dest=receiver.country,
        is_suspicious=False,
        label="normal",
        structuring_flag=False,
        round_amount=(amount % 100 == 0),
        rapid_movement=False,
    )


def _structuring_transactions(
    launderer: Customer,
    normal_customers: List[Customer],
    base_ts: datetime,
    n: int = 10,
) -> List[Transaction]:
    """Break a large amount into sub-threshold chunks (smurfing)."""
    total = round(random.uniform(50_000, 500_000), 2)
    chunks = np.random.dirichlet(np.ones(n)) * total
    txns = []
    for i, chunk in enumerate(chunks):
        chunk = round(min(chunk, SAR_THRESHOLD_USD - random.uniform(10, 499)), 2)
        ts = base_ts + timedelta(hours=i * random.uniform(0.5, 4))
        recv = random.choice(normal_customers)
        txns.append(Transaction(
            transaction_id=_make_txn_id(),
            timestamp=ts,
            sender_id=launderer.customer_id,
            receiver_id=recv.customer_id,
            amount_usd=chunk,
            transaction_type=random.choice(["CASH_DEPOSIT", "WIRE_TRANSFER", "ACH"]),
            country_origin=launderer.country,
            country_dest=recv.country,
            is_suspicious=True,
            label="structuring",
            structuring_flag=True,
            round_amount=False,
            rapid_movement=False,
        ))
    return txns


def _layering_transactions(
    launderer: Customer,
    accomplices: List[Customer],
    base_ts: datetime,
) -> List[Transaction]:
    """Round-trip large amounts through a chain of accounts."""
    amount = round(random.uniform(100_000, 2_000_000), 2)
    chain = [launderer] + accomplices[:4] + [launderer]
    txns = []
    ts = base_ts
    for i in range(len(chain) - 1):
        decay = random.uniform(0.90, 0.99)  # fee / commission
        ts += timedelta(hours=random.uniform(1, 24))
        txns.append(Transaction(
            transaction_id=_make_txn_id(),
            timestamp=ts,
            sender_id=chain[i].customer_id,
            receiver_id=chain[i + 1].customer_id,
            amount_usd=round(amount * (decay ** i), 2),
            transaction_type=random.choice(["WIRE_TRANSFER", "CRYPTO_EXCHANGE"]),
            country_origin=chain[i].country,
            country_dest=chain[i + 1].country,
            is_suspicious=True,
            label="layering",
            structuring_flag=False,
            round_amount=True,
            rapid_movement=(i < 2),
        ))
    return txns


def _tbml_transactions(
    launderer: Customer,
    counter_party: Customer,
    base_ts: datetime,
) -> List[Transaction]:
    """Inflated trade invoices to move value across borders."""
    txns = []
    for _ in range(random.randint(2, 6)):
        base_ts += timedelta(days=random.randint(1, 14))
        amount = round(random.uniform(20_000, 300_000), 2)
        txns.append(Transaction(
            transaction_id=_make_txn_id(),
            timestamp=base_ts,
            sender_id=launderer.customer_id,
            receiver_id=counter_party.customer_id,
            amount_usd=amount,
            transaction_type="TRADE_PAYMENT",
            country_origin=launderer.country,
            country_dest=counter_party.country,
            is_suspicious=True,
            label="tbml",
            structuring_flag=False,
            round_amount=True,
            rapid_movement=False,
        ))
    return txns


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate_dataset() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    customers = generate_customers()
    suspicious_custs = [c for c in customers if c.is_suspicious]
    normal_custs = [c for c in customers if not c.is_suspicious]

    transactions: List[Transaction] = []

    # ── normal transactions ──────────────────────────────────────────────────
    log.info("Generating normal transactions …")
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2024, 6, 30)
    date_range_seconds = int((end_date - start_date).total_seconds())

    n_normal = int(TOTAL_TRANSACTIONS * 0.95)
    for _ in range(n_normal):
        ts = start_date + timedelta(
            seconds=random.randint(0, date_range_seconds)
        )
        transactions.append(_normal_transaction(customers, ts))

    # ── suspicious typologies ────────────────────────────────────────────────
    log.info("Generating suspicious transaction typologies …")
    for launderer in suspicious_custs:
        ts = start_date + timedelta(
            seconds=random.randint(0, date_range_seconds)
        )
        typology = random.choice(["structuring", "layering", "tbml"])
        if typology == "structuring":
            transactions += _structuring_transactions(
                launderer, normal_custs, ts,
                n=random.randint(5, 15),
            )
        elif typology == "layering":
            accomplices = random.sample(suspicious_custs + normal_custs, 4)
            transactions += _layering_transactions(launderer, accomplices, ts)
        else:
            counter = random.choice(normal_custs)
            transactions += _tbml_transactions(launderer, counter, ts)

    log.info("Total transactions: %d", len(transactions))

    # ── serialize ─────────────────────────────────────────────────────────────
    txn_df = pd.DataFrame([t.__dict__ for t in transactions])
    txn_df["timestamp"] = pd.to_datetime(txn_df["timestamp"])
    txn_df = txn_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    cust_df = pd.DataFrame([c.__dict__ for c in customers])
    cust_df["onboard_date"] = pd.to_datetime(cust_df["onboard_date"])

    # Transaction edges for GNN
    edge_df = txn_df[["sender_id", "receiver_id", "amount_usd", "is_suspicious"]].copy()
    edge_df.columns = ["source_id", "dest_id", "amount_usd", "is_suspicious"]

    txn_path = OUTPUT_DIR / "transactions.parquet"
    cust_path = OUTPUT_DIR / "customers.parquet"
    edge_path = OUTPUT_DIR / "transaction_edges.parquet"

    txn_df.to_parquet(txn_path, index=False, compression="snappy")
    cust_df.to_parquet(cust_path, index=False, compression="snappy")
    edge_df.to_parquet(edge_path, index=False, compression="snappy")

    # Compute and log dataset statistics
    n_susp = txn_df["is_suspicious"].sum()
    log.info(
        "Saved %d transactions | %d suspicious (%.2f%%)",
        len(txn_df), n_susp, 100 * n_susp / len(txn_df),
    )
    log.info("Files written → %s, %s, %s", txn_path, cust_path, edge_path)

    # Checksum manifest
    manifest = {}
    for p in [txn_path, cust_path, edge_path]:
        sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
        manifest[p.name] = {"rows": None, "sha256": sha256}
    manifest["transactions.parquet"]["rows"] = len(txn_df)
    manifest["customers.parquet"]["rows"] = len(cust_df)
    manifest["transaction_edges.parquet"]["rows"] = len(edge_df)

    import json
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written to data/manifest.json")


if __name__ == "__main__":
    generate_dataset()
