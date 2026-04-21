# Setup & Installation Guide

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core language |
| Git | any | Version control |
| Docker | 24+ | Containerised deployment |
| Ollama | latest | Local LLM server |

---

## 1. Clone & Environment

```bash
git clone https://github.com/wahabkhan9/AI-Powered-AML-Detection-System-VAE-GNN-Streamlit
cd AI-Powered-AML-Detection-System-VAE-GNN-Streamlit

python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Optional – Install Ollama (Local LLM)

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from https://ollama.com/download

# Start the server (runs on http://localhost:11434)
ollama serve

# Pull a model (in a second terminal)
ollama pull llama3          # ~4.7 GB
# Or smaller alternatives:
# ollama pull mistral       # ~4.1 GB
# ollama pull phi3          # ~2.3 GB
```

The system automatically detects Ollama. If not running, SAR narratives fall back to high-quality rule-based templates.

---

## 3. Optional – Install LangGraph (Multi-Agent Orchestration)

```bash
pip install langgraph langchain langchain-core
```

Without LangGraph the system uses an identical sequential fallback — all functionality works.

---

## 4. Optional – Install Causal Inference Libraries

```bash
pip install econml            # CausalForestDML (requires Visual C++ on Windows)
pip install causalimpact      # Google CausalImpact
```

---

## 5. Optional – Install Apache Airflow (Pipeline Scheduling)

```bash
pip install apache-airflow

export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow standalone             # starts scheduler + webserver at localhost:8080

# Copy DAG (Airflow auto-discovers from AIRFLOW_HOME/dags/)
mkdir -p $AIRFLOW_HOME/dags
cp airflow/dags/aml_pipeline_dag.py $AIRFLOW_HOME/dags/
```

---

## 6. Optional – PyTorch Geometric (GATv2 GNN)

```bash
# Check your PyTorch version first
python -c "import torch; print(torch.__version__)"

# Install matching torch-geometric
pip install torch-geometric

# Or follow: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```

Without torch-geometric, the GNN uses a built-in `SimpleGraphConv` fallback that still achieves good performance.

---

## 7. Run the Full Pipeline

```bash
# One command to run everything
python main.py pipeline

# Or step by step:
python main.py pipeline --steps generate_data
python main.py pipeline --steps train_vae
python main.py pipeline --steps train_gan
python main.py pipeline --steps train_gnn
python main.py pipeline --steps write_reports
```

---

## 8. Launch Services

```bash
# Compliance Dashboard (7 pages)
python main.py dashboard          # → http://localhost:8501

# REST API
python main.py api                # → http://localhost:8000/docs

# Quick end-to-end demo (no training needed)
python main.py demo

# Score a JSON file of transactions
python main.py score my_transactions.json

# Causal what-if analysis
python main.py whatif
```

---

## 9. Docker Deployment

```bash
# Build and start API + Dashboard
make docker
make docker-up

# Stop
make docker-down

# Run training pipeline in container
docker compose --profile training -f deployment/docker-compose.yml up pipeline
```

---

## 10. Run Tests

```bash
# Full test suite with coverage
make test

# Fast tests only (skip slow integration tests)
make test-fast

# Individual test files
pytest tests/test_vae.py -v
pytest tests/test_gan.py -v
pytest tests/test_gnn.py -v
pytest tests/test_orchestrator.py -v
pytest tests/test_causal.py -v
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# LLM Configuration
OPENAI_API_KEY=sk-...          # Optional: OpenAI fallback
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

# Airflow
AIRFLOW_HOME=/path/to/project/airflow
AML_ALERT_EMAIL=aml@yourcompany.com

# Logging
LOG_LEVEL=INFO
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `torch not found` | `pip install torch` |
| `duckdb import error` | `pip install duckdb` |
| `LangGraph not found` | `pip install langgraph` – system falls back automatically |
| `Ollama connection refused` | Run `ollama serve` in a terminal |
| `GATv2Conv not found` | Install torch-geometric or ignore (fallback GCN is used) |
| `VAE model not found` | Run `python main.py pipeline --steps train_vae` first |
| Out of memory on GPU | Reduce `BATCH_SIZE` in `agents/anomaly_detector.py` |