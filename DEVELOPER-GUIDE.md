# Maveric Developer Guide

> A comprehensive guide for setting up and working with the Maveric RIC Algorithm Development Platform (RADP).

## Table of Contents

1. **[Prerequisites](#prerequisites)**
2. **[Quick Start](#quick-start)**
3. **[Step-by-Step Development Guideline](#step-by-step-development-guideline)**
4. **[Environment Setup](#environment-setup)**
5. **[Core RADP Setup](#core-radp-setup)**
6. **[Agentic Stack Setup](#agentic-stack-setup)**
7. **[Services Setup](#services-setup)**
8. **[Application-Specific Setup](#application-specific-setup)**
   - [MRO Application Setup](#mro-application-setup)
   - [Energy Savings Application Setup](#energy-savings-application-setup)
   - [Load Balancing Application Setup](#load-balancing-application-setup)
   - [Coverage & Capacity Optimization Setup](#coverage--capacity-optimization-setup)
9. **[Notebooks Setup](#notebooks-setup)**
10. **[Testing Setup](#testing-setup)**
11. **[Development Workflow](#development-workflow)**
12. **[Troubleshooting](#troubleshooting)**
13. **[Quick Reference](#quick-reference)**

---
# **Prerequisites**

## System Requirements

| Requirement | Specification |
|-------------|---------------|
| **Python** | 3.10.x (3.9 fails: `matplotlib==3.10.7` and the pinned agentic stack need ≥3.10; 3.11+ is untested) |
| **Docker** | Latest version with Docker Compose |
| **OS** | Linux, macOS, or Windows with WSL2 |
| **Memory** | 8GB RAM minimum (16GB recommended) |
| **Storage** | 10GB+ free space |

If `python3.10 --version` fails, install 3.10 first, e.g. `pyenv install 3.10 && pyenv local 3.10`,
`uv python install 3.10`, `brew install python@3.10`, or the python.org installer.

## Optional Requirements

| Component | Purpose |
|-----------|----------|
| **CUDA** | GPU acceleration (requires NVIDIA drivers) |
| **ffmpeg** | Video generation in notebooks |

---

# **Quick Start**

For developers who want to get up and running quickly:

## Prerequisites 
```md
# Install Docker (if not already installed)
- Visit https://docs.docker.com/get-docker/ for installation instructions

# Install unzip utility (for extracting sample data)
- Ubuntu/Debian: sudo apt-get install unzip
- macOS: brew install unzip
- Windows: unzip is built-in

# Install ffmpeg (for notebook video generation)
- Ubuntu/Debian: sudo apt-get install ffmpeg
- macOS: brew install ffmpeg
- Windows: Download from https://ffmpeg.org/download.html
```

## Install Requirements
```bash
# 1. Clone and navigate to the repository
git clone https://github.com/lf-connectivity/maveric.git
cd maveric

# 2. Set up Python environment (Python 3.10 — see System Requirements)
# macOS/Linux:
python3.10 -m venv .venv
source .venv/bin/activate
pip3 install --upgrade pip

# Windows:
py -3.10 -m venv .venv
.venv\Scripts\activate
pip3 install --upgrade pip

# 3. Set environment variables
export PYTHONPATH="$(pwd)":$PYTHONPATH # macOS or Linux
$env:PYTHONPATH="$(pwd);$env:PYTHONPATH" # Windows (Powershell)

# 4. Install dependencies
pip3 install -r requirements-dev.txt
pip3 install -r radp/client/requirements.txt
pip3 install -r radp/common/requirements.txt
pip3 install -r radp/digital_twin/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip3 install -r radp/utility/requirements.txt
pip3 install -r apps/requirements.txt
pip3 install -r notebooks/requirements.txt
pip3 install -r tests/requirements.txt
pip3 install -r services/requirements-dev.txt
pip3 install -r services/api_manager/requirements.txt
pip3 install -r services/orchestration/requirements.txt
pip3 install -r services/rf_prediction/requirements.txt
pip3 install -r services/training/requirements.txt
pip3 install -r services/ue_tracks_generation/requirements.txt
pip3 install -r requirements-agentic.txt # pinned agentic stack (Agentic MRO + Agentic Mobility)

# 5. Install GPU Enabled Pytorch (Optional)
IMPORTANT: Make sure to uninstall CPU based PyTorch if already installed  
Visit https://pytorch.org/get-started/locally/ for download instuctions 

# 6. Start RADP services (first build takes ~8 min)
# For CPU support
docker build -t radp radp
docker compose -f dc.yml -f dc-prod.yml up -d --build

# For GPU support (if you have NVIDIA GPU)
docker build -f radp/Dockerfile-cuda -t radp radp
docker compose -f dc.yml -f dc-prod.yml -f dc-cuda.yml up -d --build

# 7. Point the Python client at the stack you started
cp .env-prod .env   # prod stack, API on 8080 (Windows: copy .env-prod .env)
# or, if you started the dev stack (dc-dev.yml): cp .env-dev .env   (API on 8081)

# 8. Verify setup
docker compose -f dc.yml -f dc-prod.yml ps   # all services Up, kafka (healthy)
```

> **Which notebooks need what?** See [Notebook Prerequisites](#notebook-prerequisites). In short: `mro.ipynb`
> needs no Docker; `energy_savings.ipynb` and `load_balancing.ipynb` need a running stack and a matching `.env`.

---

---

# **Step-by-Step Development Guideline**

Someone looking for a detailed walkthrough of the set up instead of the quick run, this section provides a comprehensive, step-by-step approach to setting up your development environment, from basic prerequisites to running your first application. Follow these steps in order for a smooth setup experience.

## **Phase 1: System Preparation**
- Install system dependencies (Docker, Python, utilities)
- Set up virtual environment and environment variables
- Configure development tools

## **Phase 2: Core Platform Setup**
- Install RADP core libraries and dependencies
- Build and start Docker services
- Verify platform functionality

## **Phase 3: Application Development**
- Install application-specific dependencies
- Set up notebooks and sample data
- Run example applications

## **Phase 4: Testing and Validation**
- Install testing frameworks
- Run test suites
- Validate end-to-end functionality

---

# **Environment Setup**

## 1. Python Virtual Environment

```bash
# Create virtual environment (Python 3.10)
python3.10 -m venv .venv # macOS/Linux
py -3.10 -m venv .venv # Windows

# Activate virtual environment
source .venv/bin/activate # macOS/Linux
.venv\Scripts\activate  # Windows

# Upgrade pip
pip3 install --upgrade pip
```

## 2. Python Path Configuration

***Critical**: Set the PYTHONPATH to include the project root*:

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export PYTHONPATH="$(pwd)":$PYTHONPATH # macOS/Linux
$env:PYTHONPATH="$(pwd);$env:PYTHONPATH" # Windows

# Or run before each development session
export PYTHONPATH="/path/to/maveric":$PYTHONPATH # macOS/Linux
$env:PYTHONPATH="C:\path\to\maveric;$env:PYTHONPATH" # Windows

```

---

# **Core RADP Setup**

The RADP (RIC Algorithm Development Platform) is the core library providing RF simulation and digital twin capabilities.

## Installation Order

Install RADP dependencies in this specific order:

```bash
# 1. Core development dependencies
pip3 install -r requirements-dev.txt

# 2. RADP client (API communication)
pip3 install -r radp/client/requirements.txt

# 3. RADP common utilities
pip3 install -r radp/common/requirements.txt

# 4. RADP digital twin (with PyTorch CPU support)
pip3 install -r radp/digital_twin/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 5. PyTorch with GPU support (Optional)
IMPORTANT: Make sure to uninstall CPU based PyTorch if already installed  
https://pytorch.org/get-started/locally/

# 6. RADP utility functions
pip3 install -r radp/utility/requirements.txt

# 7. RADP testing
pip3 install -r tests/requirements.txt
```

## Dependencies Overview

| Component | Purpose | Key Dependencies |
|-----------|---------|------------------|
| `radp/client` | API client for RADP services | pandas, requests, pyarrow |
| `radp/common` | Shared utilities and constants | - |
| `radp/digital_twin` | RF simulation and ML models | torch, gpytorch, scikit-learn |
| `radp/utility` | Helper utilities | pandas, numpy |

## Docker Setup for RADP

```bash
# Build RADP Docker image
docker build -t radp radp

# For GPU support (if you have NVIDIA GPU)
docker build -f radp/Dockerfile-cuda -t radp radp
```

---

# **Agentic Stack Setup**

The agentic applications — Agentic MRO and Agentic Mobility — are built on LangGraph/LangChain and share a common LLM provider layer. Because LangGraph and LangChain ship breaking releases frequently, their dependencies are pinned in a single file at the repo root so a stray `pip install` cannot silently break the notebooks or pipelines.

## Install the Pinned Agentic Stack

After completing the Core RADP installs above, run:

```bash
pip3 install -r requirements-agentic.txt
```

This installs the locked set of agentic-stack packages:

```
langgraph==1.2.0
langgraph-checkpoint==4.1.0
langchain-core==1.4.0
langchain-groq==1.1.2
langchain-aws==1.5.0
groq==0.37.1
boto3==1.43.11
botocore==1.43.11
pydantic==2.13.4
tenacity==9.1.4
geopy==2.4.1
```

> **Note:** This file supersedes the older, unpinned `apps/mobility_robustness_optimization/agentic_mro/requirements.txt` — you no longer need to install that one.

### Optional: OpenAI or Ollama Backend

The shared LLM provider also supports OpenAI and a local Ollama backend, but neither package is included in `requirements-agentic.txt` (so users who only need Groq or Bedrock don't pay the cost). Install them only when you need them:

```bash
# OpenAI backend
pip3 install openai

# Local model backend (Ollama via langchain-community)
pip3 install langchain-community
# And install/run Ollama from https://ollama.com
```

## LLM Provider Configuration

Both Agentic MRO and Agentic Mobility use the shared LLM provider module at `radp/agentic/llm/provider.py` — a single source of truth for Groq / Bedrock / OpenAI / local-Ollama selection. It picks its backend from either a config dict or the `LLM_PROVIDER` environment variable.

Add to your `.env` file at `radp/digital_twin/agentic_mobility/.env` (copy from `.env.example` if you haven't already):

```bash
# Pick one provider
LLM_PROVIDER=groq    # one of: groq | bedrock | openai | local

# Provider-specific keys (set only what you use)
GROQ_API_KEY=your_groq_key_here

AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
BEDROCK_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_REGION=us-east-1

OPENAI_API_KEY=your_openai_key

# Local Ollama (no key needed)
OLLAMA_HOST=http://localhost:11434
```

For **Agentic MRO**, the same set of providers is selectable via `apps/mobility_robustness_optimization/agentic_mro/config.yaml`. Flip `provider:` between `groq` and `bedrock` (etc.) to swap; no code changes needed.

## Mobility Scenario Presets

Three reproducible mobility scenario presets ship in `radp/digital_twin/agentic_mobility/defaults/` — `urban.py`, `suburban.py`, and `rural.py` — selectable via the `--preset` flag on the end-to-end example:

```bash
# Needs a GROQ_API_KEY (or Bedrock credentials) in the .env file
python -m radp.digital_twin.agentic_mobility.examples.end_to_end_example --preset urban
python -m radp.digital_twin.agentic_mobility.examples.end_to_end_example --preset suburban
python -m radp.digital_twin.agentic_mobility.examples.end_to_end_example --preset rural
```

Each call prints a UE distribution summary and writes a distinct CSV under `radp/digital_twin/agentic_mobility/examples/generated_ues/`.

## Verifying the Agentic Setup

```bash
# Activate venv and set PYTHONPATH (existing steps)
source .venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Confirm the pinned install is correct
python -c "import langgraph, langchain_core, langchain_groq, langchain_aws, groq, boto3, geopy, pydantic, tenacity; print('agentic deps OK')"
```

For the full agentic test suites, see [Agentic Test Suites](#agentic-test-suites) in the Testing Setup section.

---

# **Services Setup**

The services layer provides backend functionality for training, orchestration, and data processing.

## Service Dependencies

Install all service requirements:

```bash
# Development tools for services
pip3 install -r services/requirements-dev.txt

# Individual service dependencies
pip3 install -r services/api_manager/requirements.txt
pip3 install -r services/orchestration/requirements.txt
pip3 install -r services/rf_prediction/requirements.txt
pip3 install -r services/training/requirements.txt
pip3 install -r services/ue_tracks_generation/requirements.txt
```

## Service Architecture

| Service | Purpose | Port | Dependencies |
|---------|---------|------|--------------|
| `api_manager` | REST API gateway | 8080 (prod) / 8081 (dev) | Flask, Kafka |
| `orchestration` | Job orchestration | - | Kafka |
| `training` | ML model training | - | PyTorch, GPyTorch |
| `rf_prediction` | RF simulation | - | NumPy, SciPy |
| `ue_tracks_generation` | UE mobility simulation | - | NumPy, Pandas |

## Starting Services

```bash
# Development mode
docker compose -f dc.yml -f dc-dev.yml up -d --build

# Production mode
docker compose -f dc.yml -f dc-prod.yml up -d --build

# With GPU support
docker compose -f dc.yml -f dc-dev.yml -f dc-cuda.yml up -d --build
docker compose -f dc.yml -f dc-prod.yml -f dc-cuda.yml up -d --build

```

# **Application-Specific Setup**

The applications layer contains example implementations and use cases for different RIC optimization scenarios. Each application demonstrates specific algorithms and techniques for cellular network optimization.

**Quick Navigation:**
- [MRO Application Setup](#mro-application-setup) - Mobility Robustness Optimization with handover algorithms
- [Energy Savings Application Setup](#energy-savings-application-setup) - Dynamic power management and energy efficiency
- [Load Balancing Application Setup](#load-balancing-application-setup) - Traffic distribution and load optimization
- [Coverage & Capacity Optimization Setup](#coverage--capacity-optimization-setup) - Coverage and capacity enhancement algorithms

*If you want to get started with a specific application, jump directly to its setup section below. Each section includes all necessary dependencies and configuration steps.*

## Applications Overview

| Application | Purpose | Key Features | Primary Algorithms |
|-------------|---------|--------------|--------------------|
| `coverage_capacity_optimization` | CCO algorithms | RL-based optimization | Reinforcement Learning, Antenna Tilt Control |
| `energy_savings` | Energy efficiency | Dynamic power management | RL-based Power Control, Cell Sleep Mode |
| `load_balancing` | Load distribution | Traffic-aware optimization | RL-based Load Distribution |
| `mobility_robustness_optimization` | MRO algorithms | Handover optimization | ML-based Handover Parameter Tuning |

---

## MRO Application Setup

For detailed MRO (Mobility Robustness Optimization) usage, see [README-MRO.md](README-MRO.md).

### MRO-Specific Dependencies

None beyond the Quick Start: `notebooks/requirements.txt` already pins scikit-learn, matplotlib,
stable-baselines3 2.0.0 (with gymnasium) and xgboost.

> **Do not** `pip install gymnasium stable-baselines3[extra]` unpinned on top: the `[extra]` set pulls
> `opencv-python` 5.x, which upgrades numpy to 2.x, after which pandas 1.5.3 and scipy 1.10.1 fail to import
> (`ValueError: numpy.dtype size changed`). The same applies to every application below.

### Agentic MRO-Specific Dependencies

Agentic MRO uses the pinned agentic stack — install it from the repo root:

```bash
pip3 install -r requirements-agentic.txt
```

See [Agentic Stack Setup](#agentic-stack-setup) for the full package list, optional OpenAI/Ollama backends, and LLM provider configuration (Agentic MRO selects its provider via `apps/mobility_robustness_optimization/agentic_mro/config.yaml`).

---

## Energy Savings Application Setup

For detailed usage, see [README-Energy-Savings.md](README-Energy-Savings.md).

### Energy Savings-Specific Dependencies

None beyond the Quick Start (`notebooks/requirements.txt` pins torch, stable-baselines3, gymnasium, matplotlib
and plotly). Running it does need the RADP Docker stack — see [Notebook Prerequisites](#notebook-prerequisites).

---

## Load Balancing Application Setup

For detailed usage, see [README-Load-Balancing.md](README-Load-Balancing.md).

### Load Balancing-Specific Dependencies

None beyond the Quick Start (`notebooks/requirements.txt` and `radp/digital_twin/requirements.txt` pin
stable-baselines3, gymnasium, matplotlib and scipy). Running it does need the RADP Docker stack — see
[Notebook Prerequisites](#notebook-prerequisites).

---

## Coverage & Capacity Optimization Setup

### CCO-Specific Dependencies

None beyond the Quick Start (torch, stable-baselines3, gymnasium, scipy and matplotlib are already pinned).


---

# **Notebooks Setup**

Jupyter notebooks provide interactive examples and data analysis capabilities.

### Installation

The Quick Start installs everything the notebooks need, including Jupyter (Notebook 7 and JupyterLab 4).
If you skipped it, at minimum run:

```bash
# Install notebook dependencies
pip3 install -r notebooks/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Optional: PyTorch with GPU
# https://pytorch.org/get-started/locally/
```

> Notebook 7 has no classic nbextensions, so the old `jupyter nbextension enable codefolding/main` step is gone.

## Notebook Prerequisites

Run notebooks from the `notebooks/` directory (paths inside them are relative to it), with the Quick Start venv
as the kernel.

| Notebook | Docker stack? | Sample data (`notebooks/data/`) | Extra setup | Runtime (laptop CPU) |
|----------|---------------|---------------------------------|-------------|----------------------|
| `mro.ipynb` | No | `mro_data.zip` | — | ~10 min |
| `energy_savings.ipynb` | **Yes** — dev or prod | `energy_saving_data.zip` | Repo-root `.env` matching the stack (`cp .env-dev .env` or `cp .env-prod .env`) | ~3 h at the default `TOTAL_TIMESTEPS = 25000` (~2 RL steps/s); ~15 min with `2048` |
| `load_balancing.ipynb` | **Yes** — dev or prod | `load_balancing_data.zip` | Same as energy savings | ~4 h at the default `30000`; ~15 min with `2048` |
| `coo_with_radp_digital_twin.ipynb` | No | `sim_data.zip` | — | not measured |
| `mobility_model.ipynb` | No | None (generates UE tracks) | — | not measured |
| `traffic_demand_demonstration.ipynb` | No | None (generates traffic) | — | not measured |

The energy-savings and load-balancing notebooks train their Bayesian Digital Twin through the RADP API, then copy
the model out of the training container. They read `RADP_SERVICE_PORT` from `.env` and pick the matching container
(`radp_dev-training-1` for 8081, `radp_prod-training-1` for 8080). Re-running them reuses the model already
trained under the same `BDT_MODEL_ID`.

### Sample Data Setup

```bash
# Install unzip if needed
# Ubuntu/Debian: sudo apt-get install unzip
# macOS: brew install unzip
# Windows: unzip is built-in

# Extract sample data
cd notebooks/data
unzip sim_data.zip
unzip mro_data.zip
unzip energy_saving_data.zip
unzip load_balancing_data.zip
```

## Starting Jupyter

```bash
# Start Jupyter notebook server
jupyter notebook

# Or start JupyterLab
jupyter lab
```

---

# **Testing Setup**

## Test Dependencies

```bash
# Test dependencies are included in common installations
# See Quick Start section for complete installation

# For CUDA testing (if applicable)
pip3 install -r tests/cuda/requirements.txt
```

## Running Tests

```bash
# Unit tests
pytest

# Component tests
python3 tests/run_component_tests.py

# End-to-end tests
python3 tests/run_end_to_end_tests.py

# Validation tests
python3 tests/run_validation_tests.py
```

## Agentic Test Suites

The agentic stack ships its own offline-safe test suites. Run them in order — all four should report green:

```bash
# Activate venv and set PYTHONPATH first
source .venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"

# 1. MRO validation suite
pytest apps/mobility_robustness_optimization/agentic_mro/tests/test_validation.py -v
# Expected: 42 passed

# 2. Shared LLM provider swap tests
pytest tests/agentic/test_provider_swap.py -v
# Expected: 8 passed

# 3. Mobility pipeline tests
pytest tests/agentic_mobility/test_pipeline.py -v
# Expected: 8 passed

# 4. Scenario preset tests
pytest tests/agentic_mobility/test_presets.py -v
# Expected: green
```

If any of the above fail with `ModuleNotFoundError`, your `PYTHONPATH` is not set — re-run the `export` line.

### Smoke Test the Preset CLI

```bash
# Needs a GROQ_API_KEY (or Bedrock credentials) in the .env file
python -m radp.digital_twin.agentic_mobility.examples.end_to_end_example --preset urban
python -m radp.digital_twin.agentic_mobility.examples.end_to_end_example --preset suburban
python -m radp.digital_twin.agentic_mobility.examples.end_to_end_example --preset rural
```

Each call should print a UE distribution summary and write a distinct CSV under `radp/digital_twin/agentic_mobility/examples/generated_ues/`.

---

# **Development Workflow**

## 1. Development Branch Setup

```bash
# Pull latest main branch
git checkout main
git pull origin main

# Create development branch
git checkout -b feature/your-feature-name
```

## 2. Code Changes and Testing

```bash
# Make your changes
# ... edit files ...

# Run tests
pytest

# Run component tests
python3 tests/run_component_tests.py

# Run pre-commit checks
pre-commit install
python3 -m pre_commit run --all-files
```

## 3. Local Service Testing

```bash
# Start services in development mode
docker compose -f dc.yml -f dc-dev.yml up -d --build

# Monitor individual services (container names use the compose service names, with underscores)
docker logs -f radp_dev-api_manager-1
docker logs -f radp_dev-training-1
docker logs -f radp_dev-orchestration-1
docker logs -f radp_dev-rf_prediction-1
docker logs -f radp_dev-ue_tracks_generation-1

# Monitor all services simultaneously
docker compose -f dc.yml -f dc-dev.yml logs -f

# Check service health and status
docker compose -f dc.yml -f dc-dev.yml ps
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Test API connectivity (the API has no /health route; an unknown model id returning
# a JSON 404 "model ... not found" means the API is up)
curl -i http://localhost:8081/model/connectivity-check
# API reference: services/api_manager/openapi.yaml

# Monitor resource usage
docker stats

# Check Kafka topics (if needed; expect "jobs" and "outputs")
docker exec -it radp_dev-kafka-1 kafka-topics --bootstrap-server localhost:9092 --list
```

## 4. Commit and Push

```bash
# Commit with descriptive message
git add .
git commit -m "Add feature: brief description"

# Push to remote
git push origin feature/your-feature-name

# Create pull request
```

---

# **Troubleshooting**

## Common Issues

#### 1. PYTHONPATH Not Set
**Error**: `ModuleNotFoundError: No module named 'radp'` (or `No module named 'radp.agentic'`)
**Solution**: 
```bash
# Confirm PYTHONPATH includes the repo root
echo $PYTHONPATH
# If empty or missing:
export PYTHONPATH="$(pwd)":$PYTHONPATH
```

#### 2. Docker Permission Issues
**Error**: `Permission denied` when running Docker commands
**Solution**:
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER
# Log out and back in

# On macOS/Windows, ensure Docker Desktop is running
```

#### 3. Port Conflicts
**Error**: `Port already in use`
**Solution**:
```bash
# Check what's using the port
lsof -i :8081  # macOS/Linux
netstat -ano | findstr :8081  # Windows

# Stop the conflicting service, or change the host port in dc-dev.yml / dc-prod.yml
# (prod: 8080, 9094; dev: 8081, 9095, 5678-5682). If you change the API port,
# set RADP_SERVICE_PORT in .env to match — .env only tells the Python client where to connect.
```
Other local stacks are a common cause: Docker Desktop restarts containers that have a restart policy when it launches, so a stack you ran last week may already hold 8080/8081.

#### 4. CUDA/PyTorch Issues
**Error**: CUDA-related errors
**Solution**:
```bash
# Use CPU-only PyTorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or ensure CUDA compatibility
nvidia-smi  # Check CUDA installation
```

#### 5. Memory Issues
**Error**: Out of memory during training
**Solution**:
```bash
# Reduce batch size in training scripts
# Use smaller datasets for testing
# Ensure sufficient RAM (16GB+ recommended)
```
#### 6. Kafka Docker Issue
**Error**: Dependency failed to start: container radp_prod-kafka-1 exited
**Solution**:
```bash
# Save `services\kafka\entrypoint.sh` with LF EOL sequence
```
Simply save `services\kafka\entrypoint.sh` with LF EOL sequence.

#### 7. Groq SDK Not Installed
**Error**: `ImportError: Groq SDK not installed`
**Solution**: You skipped the pinned agentic install. Run it:
```bash
pip3 install -r requirements-agentic.txt
```

#### 8. Ollama Backend Not Installed
**Error**: `ImportError: Ollama support not installed`
**Solution**: You selected `LLM_PROVIDER=local` but did not install the optional Ollama backend.
```bash
pip3 install langchain-community
```
Then make sure the Ollama service itself is running locally (default host `http://localhost:11434`).

#### 9. Missing or Unknown LLM Provider
**Error**: `ValueError: Groq API key required` / `Unsupported LLM provider`
**Solution**: Either the env var is unset or the value is unknown. Open `radp/digital_twin/agentic_mobility/.env` and confirm:
- `LLM_PROVIDER` is one of `groq`, `bedrock`, `openai`, or `local`.
- The matching API key for that provider is set (Groq: `GROQ_API_KEY`, OpenAI: `OPENAI_API_KEY`, AWS: `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`).

#### 10. Pydantic Deprecation Warnings
**Error**: Tests pass but emit `PydanticDeprecatedSince20` warnings
**Solution**: Cosmetic only — the codebase still uses Pydantic v1 idioms (`.dict()`, `.copy(update=...)`) in a few places. Safe to ignore.

#### 11. A Stray `pip install` Broke the Agentic Stack
**Error**: Notebooks/pipelines ran fine yesterday but `pip install some-package` broke them today
**Solution**: Re-pin to the known-good snapshot:
```bash
pip3 install -r requirements-agentic.txt --force-reinstall
```

#### 12. `--preset` Flag Not Recognized
**Error**: `end_to_end_example.py` does not accept `--preset`
**Solution**: You're on an older copy of `end_to_end_example.py`. Pull the latest `dev` branch.

#### 13. `No matching distribution found for matplotlib==3.10.7`
**Error**: The first `pip3 install -r requirements-dev.txt` fails
**Solution**: Your venv is Python 3.9. Recreate it with Python 3.10 (`python3.10 -m venv .venv`).

#### 14. `ValueError: numpy.dtype size changed, may indicate binary incompatibility`
**Error**: `import pandas` or `import scipy` fails after installing extra packages
**Solution**: Something upgraded numpy to 2.x (typically an unpinned `pip install stable-baselines3[extra]`, which pulls `opencv-python` 5.x). Restore the pins:
```bash
pip3 install -r notebooks/requirements.txt   # re-pins numpy 1.24.x
```

#### 15. Energy Savings / Load Balancing Notebook: `Connection refused` on port 8080 or 8081
**Error**: `HTTPConnectionPool(host='127.0.0.1', port=8080): Max retries exceeded ... Connection refused`
**Solution**: The notebook connects to `RADP_SERVICE_PORT` from the repo-root `.env`, defaulting to the dev port 8081. Make `.env` match the stack you started (`cp .env-dev .env` for `dc-dev.yml`, `cp .env-prod .env` for `dc-prod.yml`), check `docker compose ... ps`, then restart the kernel.

#### 16. Energy Savings / Load Balancing Notebook: `'docker cp' from 'radp_..._training-1' failed: No such container`
**Error**: The BDT cell raises after training finished
**Solution**: The container is derived from `RADP_SERVICE_PORT` (8081 → `radp_dev-training-1`, 8080 → `radp_prod-training-1`). Make `.env` match the stack you started; `docker ps` shows the real names.

#### 17. `RADP rejected the training request ... validation_errors`
**Error**: The BDT cell raises with a 400 from `/train`
**Solution**: Training data must have `avg_rsrp` in −170…0 dBm and `cell_el_deg` in 0…20°. The `validation_errors` list in the message names each offending column and how many rows fail.


## Getting Help

1. **Check logs**: `docker logs <container-name>`
2. **Verify services**: `docker compose -f dc.yml -f dc-dev.yml ps` (or `dc-prod.yml`)
3. **Test connectivity**: `curl -i http://localhost:8081/model/connectivity-check` — a JSON 404 means the API is up
4. **Review documentation**: Check specific README files in each module

## Useful Commands

```bash
# Clean up Docker resources
docker system prune -a

# Reset virtual environment
rm -rf .venv
python3.10 -m venv .venv
source .venv/bin/activate

# Check service status
docker compose -f dc.yml -f dc-dev.yml ps

# View service logs
docker compose -f dc.yml -f dc-dev.yml logs -f
```

---

# **Quick Reference**


## Directory Structure

```
maveric/
├── radp/                    # Core RADP library
│   ├── agentic/             # Shared agentic infrastructure
│   │   └── llm/provider.py  # Shared LLM provider (Groq/Bedrock/OpenAI/local)
│   └── digital_twin/
│       └── agentic_mobility/
│           └── defaults/    # Mobility scenario presets (urban/suburban/rural)
├── services/                # Backend services
├── apps/                    # Example applications
│   └── mobility_robustness_optimization/
│       └── agentic_mro/
│           └── tests/       # MRO validation tests + golden fixtures
├── notebooks/               # Jupyter notebooks
├── tests/                   # Test suites
│   ├── agentic/             # LLM provider swap tests
│   └── agentic_mobility/    # Mobility pipeline + preset tests
├── requirements-dev.txt     # Development dependencies
├── requirements-agentic.txt # Pinned agentic stack (LangGraph/LangChain)
├── .env-dev                 # Development environment
├── .env-prod                # Production environment
├── dc.yml                   # Docker Compose base
├── dc-dev.yml               # Docker Compose development
└── dc-prod.yml              # Docker Compose production
```

---

---

# **Application-Specific Guides**

- **MRO (Mobility Robustness Optimization)**: See [README-MRO.md](README-MRO.md) for detailed setup and usage
- **Agentic CLI Walkthroughs**: See [README-AGENTIC.md](README-AGENTIC.md) for end-to-end agentic workflows
- **Coverage & Capacity Optimization**: See [apps/coverage_capacity_optimization/](apps/coverage_capacity_optimization/)
- **Energy Savings**: See [apps/energy_savings/](apps/energy_savings/)
- **Load Balancing**: See [apps/load_balancing/](apps/load_balancing/)

---

# **License**

See [LICENSE](LICENSE) file for details.
