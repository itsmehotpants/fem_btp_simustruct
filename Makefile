.PHONY: data train app test docker clean help install

# ============================================================
# SimuStruct AI V3 — Build Commands
# ============================================================

help: ## Show this help message
	@echo "SimuStruct AI V3 — Available Commands:"
	@echo "  make install  — Install Python dependencies"
	@echo "  make data     — Generate training data (500 samples)"
	@echo "  make data-full— Generate full dataset (5000 samples)"
	@echo "  make train    — Train AI model"
	@echo "  make app      — Run Streamlit frontend"
	@echo "  make api      — Run FastAPI compute engine"
	@echo "  make test     — Run all tests"
	@echo "  make docker   — Build and run with Docker Compose"
	@echo "  make clean    — Remove generated data and models"

install: ## Install Python dependencies
	pip install -r requirements_compute.txt
	pip install -r requirements_frontend.txt
	pip install pytest

data: ## Generate training data (500 samples — quick)
	python 1_generate_data.py --n_samples 500

data-full: ## Generate full dataset (5000 samples)
	python 1_generate_data.py --n_samples 5000

train: ## Train AI model
	python 2_train_ai.py

train-fast: ## Train with fewer epochs (for testing)
	python 2_train_ai.py --epochs 50 --batch_size 2048

app: ## Run Streamlit frontend
	streamlit run app.py --server.port 8501

api: ## Run FastAPI compute engine
	uvicorn compute_engine.main:app --host 0.0.0.0 --port 8000 --reload

test: ## Run all tests
	python -m pytest tests/ -v --tb=short

test-materials: ## Run material database tests only
	python -m pytest tests/test_materials.py -v

test-fem: ## Run FEM solver tests only
	python -m pytest tests/test_fem.py -v

test-ai: ## Run AI model tests only
	python -m pytest tests/test_inference.py -v

docker: ## Build and run with Docker Compose
	docker-compose up --build

docker-down: ## Stop Docker Compose
	docker-compose down

clean: ## Remove generated data, models, and logs
	rm -rf data/hdf5/*.npz data/hdf5/*.h5
	rm -rf models/simustruct_surrogate.pth models/best_checkpoint.pth models/scaler.pkl
	rm -rf logs/*.json
	rm -rf __pycache__ src/__pycache__ models/__pycache__ tests/__pycache__
