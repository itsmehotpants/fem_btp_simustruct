# ⚙️ SimuStruct AI — V3

<div align="center">

**Real-Time Structural Analysis via Deep Learning Surrogate Models**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Spring Boot](https://img.shields.io/badge/Spring_Boot-3.2-6DB33F?style=for-the-badge&logo=spring&logoColor=white)](https://spring.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

*B.Tech Project (BTP) • LNMIIT Jaipur • 2023–2027 Batch*

</div>

---

## 🌟 Overview

SimuStruct AI is a professional-grade engineering platform that uses **deep learning surrogate models** to predict structural stress and displacement fields in **real-time** (< 10ms), replacing traditional FEM solvers that take minutes to hours.

### Key Highlights

| Feature | Description |
|---------|-------------|
| 🧪 **22 Engineering Materials** | Complete database from structural steel to CFRP composites |
| 🧠 **Fourier-Encoded MLP** | 8-frequency positional encoding for sharp stress gradients |
| ⚛️ **Physics-Informed Loss** | PINN: enforces Navier-Cauchy equilibrium (div(σ)=0) |
| 📊 **Graph Neural Network** | GATConv on mesh topology for arbitrary geometries |
| 🔬 **FEM Validation** | Side-by-side comparison with analytical Kirsch solution |
| 🔩 **Fatigue Analysis** | Basquin S-N curves + Goodman mean-stress correction |
| 📐 **SCF Library** | Peterson/Neuber/Pilkey theoretical validation |
| 🎯 **Design Optimizer** | Differential evolution with AI surrogate objective |
| 📈 **Uncertainty Quantification** | MC Dropout ±2σ confidence intervals |
| 📄 **PDF Reports** | Professional ReportLab engineering reports |
| 🐳 **Docker Deployment** | 3-service containerized architecture |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      CLIENT LAYER                                │
│   Streamlit Frontend (app.py)                                    │
│   Design Studio | Simulation Hub | Analytics | Export & Credits  │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP REST
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│               CONTROLLER LAYER (Java Spring Boot)                │
│   /api/v1/simulate  |  /api/v1/validate  |  /api/v1/health     │
│   OOD Gatekeeper  |  Input Validation  |  Rate Limiting         │
└────────────────────────────┬─────────────────────────────────────┘
                             │ HTTP Internal
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│              COMPUTE ENGINE (Python FastAPI)                      │
│   /infer  (PyTorch) | /fem  (Analytical/FEniCSx) | /health      │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip or conda
- (Optional) JDK 17+ for Spring Boot
- (Optional) Docker for containerized deployment

### 1. Install Dependencies

```bash
# Using pip
pip install -r requirements_compute.txt
pip install -r requirements_frontend.txt

# OR using conda
conda env create -f environment.yml
conda activate simu_ai
```

### 2. Generate Training Data

```bash
# Quick test (500 samples, ~2 minutes)
python 1_generate_data.py --n_samples 500

# Full dataset (5000 samples)
python 1_generate_data.py --n_samples 5000
```

### 3. Train AI Model

```bash
# Full training (300 epochs, cosine annealing)
python 2_train_ai.py

# Quick training for testing
python 2_train_ai.py --epochs 50 --batch_size 2048
```

### 4. Launch the Platform

```bash
# Option A: Streamlit only (simplest)
streamlit run app.py

# Option B: FastAPI + Streamlit
uvicorn compute_engine.main:app --port 8000 &
streamlit run app.py

# Option C: Full stack with Docker
docker-compose up --build
```

---

## 📁 Project Structure

```
simustruct_v3/
├── 1_generate_data.py          # FEM data generation pipeline
├── 2_train_ai.py               # Model training pipeline
├── app.py                      # Streamlit frontend
├── config.yaml                 # Centralized configuration
├── Makefile                    # One-command operations
│
├── src/                        # Shared Python library
│   ├── materials.py            # 22-material database
│   ├── data_utils.py           # LHS sampling, scaler, HDF5 I/O
│   ├── fem_solver.py           # FEniCSx wrapper + Kirsch fallback
│   ├── inference.py            # Model loading + forward pass
│   ├── fatigue.py              # S-N curve, Basquin equation
│   ├── scf_library.py          # Peterson/Neuber SCF formulas
│   ├── optimizer.py            # Differential evolution optimizer
│   └── report.py               # PDF report generation
│
├── models/                     # AI model architectures
│   ├── enhanced_mlp.py         # Fourier-encoded MLP
│   ├── gnn.py                  # Graph Attention Network
│   ├── pinn_loss.py            # Physics-informed loss
│   └── mc_dropout.py           # MC Dropout uncertainty
│
├── compute_engine/             # FastAPI microservice
│   └── main.py                 # /infer, /fem, /health endpoints
│
├── backend/                    # Java Spring Boot controller
│   ├── pom.xml
│   └── src/main/java/com/simustruct/
│       ├── controller/SimulationController.java
│       ├── service/OODValidationService.java
│       ├── service/ComputeEngineClient.java
│       └── model/{Request,Result,Validation}.java
│
├── assets/style.css            # Dark mode CSS theme
├── tests/                      # Pytest test suite
├── docker-compose.yml          # 3-service orchestration
└── Dockerfile.*                # Container definitions
```

---

## 🧪 Material Database

22 engineering materials spanning 9 categories:

| Category | Materials | E Range (GPa) |
|----------|-----------|---------------|
| Steel | ASTM A36, A572, SS304, SS316 | 193–200 |
| Cast Iron | Gray A48, Ductile A536 | 120–169 |
| Aluminum | 6061-T6, 7075-T6, 2024-T3 | 68.9–73.1 |
| Titanium | Ti-6Al-4V, Grade 2 CP | 103–113.8 |
| Copper Alloys | C11000, Brass C26000 | 110–117 |
| Nickel Alloys | Inconel 718 | 200 |
| Composites | CFRP, GFRP | 25–135 |
| Polymers | PC, ABS, HDPE | 0.8–2.4 |
| Concrete & Elastomers | Normal concrete, Natural rubber | 0.001–30 |

---

## 📊 Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| AI R² (Von Mises) | > 0.95 | Coefficient of determination on test set |
| Max Relative Error | < 5% | Mean |AI - FEM| / FEM × 100 |
| Inference Time | < 10 ms | Wall-clock time for single prediction |
| FEM Speedup | > 100× | FEM time / AI inference time |
| SCF Accuracy | < 8% | Deviation from Peterson's formula |

---

## 🧑‍💻 Make Commands

```bash
make help           # Show all available commands
make install        # Install dependencies
make data           # Generate 500 training samples
make data-full      # Generate 5000 samples
make train          # Train AI model (300 epochs)
make train-fast     # Quick training (50 epochs)
make app            # Launch Streamlit UI
make api            # Start FastAPI engine
make test           # Run all pytest tests
make docker         # Docker Compose build & run
make clean          # Remove generated files
```

---

## 🧪 Running Tests

```bash
# All tests
make test

# Individual test suites
python -m pytest tests/test_materials.py -v    # Material database
python -m pytest tests/test_fem.py -v          # FEM solver + SCF
python -m pytest tests/test_inference.py -v    # AI models + fatigue
```

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Services:
# - Compute Engine: http://localhost:8000
# - Spring Boot:    http://localhost:8080
# - Streamlit UI:   http://localhost:8501
```

---

## 👨‍💻 Developer Credits

<div align="center">

### 🎓 Project Team

| Role | Name | Contribution |
|------|------|-------------|
| **Lead Developer & AI/ML Engineer** | **Naman** | Architecture, AI models, frontend, deployment |

### 🏛️ Institution

**The LNM Institute of Information Technology (LNMIIT), Jaipur**
B.Tech Project (BTP) — 2023–2027 Batch

### 🔧 Technology Stack

| Layer | Technology |
|-------|-----------|
| AI/ML | PyTorch, Fourier MLP, GNN (GATConv), PINN |
| Physics | FEniCSx, Gmsh, Kirsch Analytical Solution |
| Backend | Python FastAPI, Java Spring Boot 3.2 |
| Frontend | Streamlit, Matplotlib, Custom Dark Mode CSS |
| Data | HDF5, Latin Hypercube Sampling, NumPy/SciPy |
| DevOps | Docker, Docker Compose, Makefile |
| Testing | Pytest, JUnit |

</div>

---

## 📜 License

This project is developed as part of a B.Tech Project at LNMIIT Jaipur.
For academic use only.

---

<div align="center">

**Built with ❤️ at LNMIIT Jaipur**

*SimuStruct AI V3.0.0 — Real-Time Structural Analysis via Deep Learning*

</div>
