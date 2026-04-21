# ============================================================
# Makefile – AML Detection System
# Usage: make <target>
# ============================================================

.PHONY: help setup data train-vae train-gnn reports dashboard api test lint format clean docker

PYTHON  := python
VENV    := venv
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
STREAMLIT := $(VENV)/bin/streamlit
UVICORN   := $(VENV)/bin/uvicorn

# ── Help ─────────────────────────────────────────────────────
help:
	@echo ""
	@echo "AML Detection System – Developer Tasks"
	@echo "======================================="
	@echo "  make setup       Create venv and install deps"
	@echo "  make data        Generate synthetic transaction dataset"
	@echo "  make train-vae   Train VAE anomaly detector"
	@echo "  make train-gnn   Train GNN network investigator"
	@echo "  make reports     Generate SAR reports"
	@echo "  make pipeline    Run full pipeline (data→vae→gnn→reports)"
	@echo "  make dashboard   Launch Streamlit dashboard"
	@echo "  make api         Start FastAPI server"
	@echo "  make test        Run test suite"
	@echo "  make lint        Lint code with ruff"
	@echo "  make format      Format code with black"
	@echo "  make clean       Remove generated files"
	@echo "  make docker      Build Docker image"
	@echo ""

# ── Setup ────────────────────────────────────────────────────
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "✓ Environment ready. Activate with: source $(VENV)/bin/activate"

# ── Pipeline steps ───────────────────────────────────────────
data:
	$(PYTHON) generate_data.py

train-vae:
	$(PYTHON) -m agents.anomaly_detector

train-gan:
	$(PYTHON) -m agents.gan_trainer

train-gnn:
	$(PYTHON) -m agents.network_investigator

reports:
	$(PYTHON) -m agents.report_writer

pipeline:
	$(PYTHON) main.py pipeline

demo:
	$(PYTHON) main.py demo

whatif:
	$(PYTHON) main.py whatif

# ── Services ─────────────────────────────────────────────────
dashboard:
	$(STREAMLIT) run dashboard/app.py \
		--server.port 8501 \
		--browser.gatherUsageStats false

api:
	$(UVICORN) api.main:app --host 0.0.0.0 --port 8000 --reload

# ── Analysis notebooks ───────────────────────────────────────
eda:
	$(PYTHON) notebooks/01_eda.py

analysis-vae:
	$(PYTHON) notebooks/02_vae_training_analysis.py

analysis-gnn:
	$(PYTHON) notebooks/03_gnn_analysis.py

# ── Quality ──────────────────────────────────────────────────
test:
	$(PYTEST) tests/ -v --cov=. --cov-report=term-missing

test-fast:
	$(PYTEST) tests/ -v -k "not test_anomalous_samples"

lint:
	$(VENV)/bin/ruff check .

format:
	$(VENV)/bin/black .

type-check:
	$(VENV)/bin/mypy agents/ models/ features/ utils/ api/ --ignore-missing-imports

# ── Docker ───────────────────────────────────────────────────
docker:
	docker build -f deployment/Dockerfile -t aml-detection:latest .

docker-up:
	docker compose -f deployment/docker-compose.yml up -d

docker-down:
	docker compose -f deployment/docker-compose.yml down

# ── Clean ────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov dist build *.egg-info
	@echo "✓ Cleaned"

clean-all: clean
	rm -rf $(VENV) data/*.parquet models/*.pth models/*.pkl models/*.json \
	       reports/*.png reports/*.csv reports/*.json logs/
	@echo "✓ Full clean complete"
