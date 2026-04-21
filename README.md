# 🕵️ APEX AML Intelligence Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Overview

**APEX AML** is a production-grade, multi-agent Anti-Money Laundering detection system that combines **deep learning anomaly detection (VAE)**, **graph neural networks (GNN)**, **generative AI (Ollama)**, and **causal inference** to identify and report financial crime. The system is designed as an autonomous agentic workflow – a team of AI agents that collaborate to detect, investigate, and explain suspicious transactions.

**What this system actually does:**
- Generates synthetic financial transactions with realistic money laundering patterns (structuring, layering, smurfing)
- Detects anomalies using a **Variational Autoencoder (VAE)** trained on normal transaction patterns
- Builds transaction graphs and applies a **Graph Neural Network (GNN)** to identify high-risk customer networks
- Automatically generates **Suspicious Activity Reports (SAR)** with two modes: template-based or **LLM‑powered narratives** (via Ollama)
- Provides a **7‑page Streamlit dashboard** with real‑time KPIs, SHAP explainability, network visualization, and causal “what‑if” policy simulation
- Orchestrates the entire pipeline using a **LangGraph‑based multi‑agent state machine** (The Commander)
- Supports **DuckDB** for high‑performance SQL analytics over Parquet data
- Includes **Apache Airflow DAGs** for production scheduling
- Exposes a **FastAPI** REST API for programmatic access

---

## 🏗️ Architecture (Multi‑Agent System)
┌─────────────────────────────────────────────────────────────────────────┐
│ LANGGRAPH ORCHESTRATOR │
│ (The Commander – state machine) │
└─────────────────────────────────────────────────────────────────────────┘
│ │ │ │
▼ ▼ ▼ ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ VAE Agent │ │ GNN Agent │ │ SAR Agent │ │ LLM Agent │
│ (Analyst) │ │ (Detective) │ │ (Narrator) │ │ (Explainer) │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
│ │ │ │
└────────────────┴────────────────┴────────────────┘
│
▼
┌─────────────────────────────┐
│ Streamlit Dashboard │
│ + FastAPI + DuckDB │
└─────────────────────────────┘

text

**Agent Roles:**
- **VAE Agent** – Unsupervised anomaly detection using reconstruction error.
- **GNN Agent** – Graph neural network on transaction networks → risk scores.
- **SAR Agent** – Generates structured Suspicious Activity Reports.
- **LLM Agent** – (Optional) Uses Ollama (Llama 3) to write human‑readable SAR narratives.
- **Commander** – LangGraph state machine that routes data between agents.

---

## ✨ Implemented Features

| Component | Status | Notes |
|-----------|--------|-------|
| Synthetic data generator | ✅ Fully implemented | Creates `transactions.parquet`, `customers.parquet` |
| VAE anomaly detection | ✅ Fully implemented | Trained on normal transactions, threshold‑based |
| GNN network investigator | ✅ Fully implemented | PyTorch Geometric GNN on customer transaction graph |
| SAR report writer (template) | ✅ Fully implemented | JSON + CSV reports |
| Ollama local LLM integration | ✅ Implemented | Uses `llama3.2:3b` via Ollama (optional, fallback to template) |
| LangGraph orchestrator | ✅ Implemented | Multi‑agent state machine in `langgraph/orchestrator.py` |
| Streamlit dashboard | ✅ 7 pages | Command Center, Threat Analysis, Network Intel, SAR Ops, Model Registry, Policy Simulator, Data Intelligence |
| SHAP explainability | ✅ Implemented | Feature importance and beeswarm plots |
| Causal inference (what‑if) | ✅ Implemented | `causal/causal_inference.py` – threshold simulation, DiD |
| DuckDB analytics | ✅ Implemented | `data/duckdb_queries.py` – high‑performance SQL |
| Apache Airflow DAG | ✅ Implemented | `airflow/dags/aml_pipeline_dag.py` |
| FastAPI REST API | ✅ Implemented | `api/main.py` – endpoints for predictions and reports |
| GAN fusion (auxiliary) | ✅ Implemented | `agents/gan_trainer.py` – GAN for synthetic anomaly generation |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) (optional – for LLM‑powered SARs)
- Docker (optional – for containerised deployment)

### Installation

```bash
git clone https://github.com/wahabkhan9/AI-Powered-AML-Detection-System-VAE-GNN-Streamlit.git
cd AI-Powered-AML-Detection-System-VAE-GNN-Streamlit

python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt
Generate synthetic data
bash
python generate_data.py
Train all models and run full pipeline
bash
python main.py pipeline
This single command runs:

Data generation (if missing)

VAE training

GNN training

SAR report generation (template mode)

Causal inference data preparation

DuckDB analytics pre‑computation

Launch the Streamlit dashboard
bash
streamlit run dashboard/app.py
Navigate to http://localhost:8501

(Optional) Enable LLM‑powered SARs
bash
ollama pull llama3.2:3b
export USE_LLM=true   # or set in .env file
python main.py pipeline --steps write_reports
Run the FastAPI server
bash
uvicorn api.main:app --reload
Run Airflow DAG (production scheduling)
bash
cd airflow
docker-compose up -d
# Then access http://localhost:8080
📁 Project Structure
text
├── agents/                  # Core AI agents
│   ├── anomaly_detector.py  # VAE model
│   ├── network_investigator.py # GNN model
│   ├── sar_report_writer.py # SAR generation
│   └── gan_trainer.py       # GAN for augmentation
├── dashboard/               # Streamlit frontend (7 pages)
│   └── app.py
├── api/                     # FastAPI REST endpoints
│   ├── main.py
│   └── routes.py
├── causal/                  # Causal inference (what‑if simulator)
│   └── causal_inference.py
├── data/                    # Data layer
│   ├── generator.py
│   ├── duckdb_queries.py    # DuckDB analytics
│   └── schemas.py
├── langgraph/               # Multi‑agent orchestrator
│   └── orchestrator.py
├── airflow/                 # Airflow DAGs for scheduling
│   └── dags/aml_pipeline_dag.py
├── models/                  # Saved model checkpoints
├── reports/                 # Generated SAR reports and metrics
├── tests/                   # Unit tests (pytest)
├── .env.example             # Environment variables template
├── requirements.txt
├── main.py                  # Entry point for full pipeline
└── README.md
⚙️ Configuration
Create a .env file (copy from .env.example):

env
# Data generation
N_TRANSACTIONS=100000
ANOMALY_RATIO=0.01

# VAE training
VAE_EPOCHS=50
VAE_BATCH_SIZE=256
VAE_ANOMALY_PERCENTILE=95.0

# GNN training
GNN_HIDDEN_CHANNELS=64
GNN_NUM_LAYERS=3

# Ollama LLM
USE_LLM=false
OLLAMA_MODEL=llama3.2:3b

# API
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
🧪 Testing
bash
pytest tests/ -v --cov=agents
📊 Dashboard Pages
Page	Description
Command Center	Live KPIs, weekly anomaly trend, typology distribution, recent alerts table
Threat Analysis	VAE score distributions, SHAP feature importance, alert deep‑dive
Network Intelligence	Interactive GNN‑colored transaction graph, risk score histogram
SAR Operations	View, filter, and export Suspicious Activity Reports (JSON/CSV)
Model Registry	Training loss curves, confusion matrix, precision/recall
Policy Simulator	Causal what‑if: threshold changes, new rule impact, DiD estimation
Data Intelligence	DuckDB SQL editor and preset analytical queries
🛠️ Production Deployment
Docker
bash
docker build -t apex-aml .
docker run -p 8501:8501 -p 8000:8000 apex-aml
Docker Compose (full stack)
bash
docker-compose up -d
Spins up: Streamlit, FastAPI, Airflow, and Postgres metadata DB.

🤝 Contributing
Contributions are welcome. Please follow the existing code style (black, isort) and write tests for new features.

Fork the repo

Create a feature branch (git checkout -b feature/improvement)

Commit changes (git commit -m 'Add improvement')

Push to branch (git push origin feature/improvement)

Open a Pull Request

📄 License
MIT License – see LICENSE file.

📧 Contact
Project maintainer: Wahab Khan

🙏 Acknowledgments
PyTorch & PyTorch Geometric – GNN implementation

LangGraph – Multi‑agent orchestration

Ollama – Local LLM inference

DuckDB – Embedded analytical SQL

Streamlit – Dashboard framework

SHAP – Model explainability

EconML – Causal inference

Apache Airflow – Workflow scheduling
