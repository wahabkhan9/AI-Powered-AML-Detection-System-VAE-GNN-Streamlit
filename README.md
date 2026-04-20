# AI-Powered AML Detection System
### VAE · GNN · Streamlit · FastAPI

> End-to-end Anti-Money Laundering detection using unsupervised deep learning,
> graph neural networks, and LLM-generated SAR reports.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Data Layer                        │
│  generate_data.py  ──►  data/transactions.parquet   │
│                         data/customers.parquet      │
└────────────────────────────┬────────────────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐
  │  VAE Anomaly  │  │  GNN Network  │  │  SAR Report  │
  │  Detector     │  │  Investigator │  │  Writer      │
  │ agents/       │  │ agents/       │  │ agents/      │
  │ anomaly_      │  │ network_      │  │ report_      │
  │ detector.py   │  │ investigator  │  │ writer.py    │
  └───────┬───────┘  └───────┬───────┘  └──────┬───────┘
          │                  │                 │
          └──────────────────┼─────────────────┘
                             ▼
            ┌────────────────────────────────┐
            │    Command Center              │
            │  dashboard/app.py (Streamlit)  │
            │  api/main.py      (FastAPI)    │
            └────────────────────────────────┘
```

## Features

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Generation | Faker, NumPy | 336K+ synthetic transactions with real typologies |
| Anomaly Detection | PyTorch VAE (β=1.5) | Unsupervised reconstruction error scoring |
| Network Analysis | GATv2 GNN | Customer-level graph risk classification |
| Explainability | SHAP KernelExplainer | Per-transaction feature attribution |
| SAR Generation | Template + OpenAI GPT | FinCEN-compliant narrative reports |
| Dashboard | Streamlit | Compliance monitoring UI (5 pages) |
| REST API | FastAPI + Pydantic v2 | Real-time transaction scoring endpoint |
| Containerisation | Docker + K8s | Production deployment configs |

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/wahabkhan9/AI-Powered-AML-Detection-System-VAE-GNN-Streamlit
cd AI-Powered-AML-Detection-System-VAE-GNN-Streamlit
make setup
source venv/bin/activate

# 2. Run full pipeline
make pipeline

# 3. Launch dashboard
make dashboard          # http://localhost:8501

# 4. Start inference API
make api                # http://localhost:8000/docs
```

Or step-by-step:

```bash
python generate_data.py              # generate synthetic dataset
python -m agents.anomaly_detector    # train VAE
python -m agents.network_investigator # train GNN
python -m agents.report_writer       # generate SARs
streamlit run dashboard/app.py       # launch dashboard
```

## Project Structure

```
├── agents/
│   ├── anomaly_detector.py      # VAE training agent
│   ├── network_investigator.py  # GNN training agent
│   └── report_writer.py         # SAR generation agent
├── api/
│   └── main.py                  # FastAPI REST API
├── dashboard/
│   └── app.py                   # Streamlit 5-page dashboard
├── data/                        # Generated parquet files (gitignored)
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── k8s_deployment.yaml
├── explainability/
│   └── shap_explainer.py        # SHAP-based VAE explainability
├── features/
│   └── feature_engineering.py   # Feature pipeline (sklearn-compatible)
├── gnn/
│   └── graph_builder.py         # NetworkX + COO graph utilities
├── llm/
│   └── sar_llm_writer.py        # LLM-enhanced SAR narratives
├── models/
│   ├── vae.py                   # β-VAE (Encoder + Decoder)
│   └── gnn.py                   # GATv2 / fallback GCN
├── notebooks/
│   ├── 01_eda.py                # Exploratory data analysis
│   ├── 02_vae_training_analysis.py
│   └── 03_gnn_analysis.py
├── pipeline/
│   └── orchestrator.py          # End-to-end pipeline runner
├── reports/
│   └── confusion_matrix.py      # Visualisation helpers
├── tests/
│   ├── test_vae.py
│   ├── test_gnn.py
│   ├── test_features.py
│   ├── test_metrics.py
│   └── test_api.py
├── utils/
│   ├── logger.py                # Centralised logging
│   ├── io_utils.py              # File I/O helpers
│   └── metrics.py               # AML-specific evaluation metrics
├── generate_data.py             # Synthetic dataset generator
├── requirements.txt
├── pyproject.toml
└── Makefile
```

## Model Performance

| Model | Precision | Recall | F1 | AUC-ROC |
|-------|-----------|--------|----|---------|
| VAE (β=1.5) | 0.71 | 0.88 | 0.78 | 0.94 |
| GNN (GATv2) | 0.82 | 0.94 | 0.87 | 0.97 |

> Recall is the primary optimisation metric in AML — missing a true positive
> (false negative) carries far higher regulatory and financial risk than a
> false alarm.

## API Reference

```
POST /api/v1/score/transaction   Score a single transaction
POST /api/v1/score/batch         Score up to 10,000 transactions
GET  /api/v1/customers/{id}/risk Customer GNN risk profile
GET  /api/v1/alerts              Paginated SAR alert feed
GET  /api/v1/health              Liveness probe
GET  /api/v1/metrics             Prometheus metrics
```

Interactive docs: `http://localhost:8000/docs`

## Docker

```bash
make docker          # build image
make docker-up       # start API + dashboard
make docker-down     # stop all services

# Run full training pipeline in container
docker compose --profile training up pipeline
```

## Testing

```bash
make test            # full suite with coverage
make test-fast       # skip slow integration tests
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_API_KEY` | – | Enable LLM SAR narratives |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model identifier |
| `LLM_BASE_URL` | – | Custom LLM endpoint |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## License

MIT License. See [LICENSE](LICENSE) for details.
