# Nirnaya Decision Engine Server (Production Microservice)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Multi--Platform-2496ED.svg)](https://www.docker.com/)
[![SQLite](https://img.shields.io/badge/SQLite-WAL--Mode-003B57.svg)](https://www.sqlite.org/)
[![Contract](https://img.shields.io/badge/API-TypeSafe--Jev--Compatible-success.svg)](https://api.typesafe.ai)

**Nirnaya Server** is a production-grade, local-first **System-One typed probabilistic decision microservice** implementing exact API parity with **TypeSafe AI Jev** (`POST /v1/systemone`). It eliminates auto-regressive decoding loops, stop tokens, and JSON validation errors by restricting evaluation strictly to the single next-token readout over categorical, continuous rubric, and binary decisions.

---

## 1. Key Features

- **Exact TypeSafe AI Jev Contract (`/v1/systemone`):** Drop-in cloud API replacement accepting `state` and typed `questions` with `criteria` or `options`/`levels`.
- **Model Engine Agnostic:** JSON-driven model dispatcher supporting lightweight CPU models (`Qwen3-0.6B-GGUF`), balanced high-accuracy models (`Qwen3.5-4B-GGUF`), 1-bit quantized ternary LLMs (`Bonsai-27B`), and dense HuggingFace models (`Qwen3-4B`).
- **High-Performance SQLite with WAL Mode:** Persistent API key verification and audit logging with zero database locking under heavy concurrent traffic (`PRAGMA journal_mode = WAL;`).
- **Authentication & Security:** Constant-time SHA-256 Bearer Token and `X-API-Key` authentication with role-based access control (`client` vs `admin`).
- **Multi-Platform Deployment Matrix:**
  - 🐧 **Linux CPU:** Debian Bookworm + OpenBLAS AVX2/AVX-512 acceleration.
  - 🚀 **Linux NVIDIA GPU:** CUDA 12.1+ container with complete GPU layer offloading.
  - 🍏 **Apple Silicon (Mac Metal):** Native Metal GPU acceleration script (`launch_metal.sh`) + ARM64 container.

---

## 2. Directory Structure

```
nirnaya_server/
├── docker-compose.cpu.yml       # Production Compose for Linux / Windows CPU
├── docker-compose.gpu.yml       # Production Compose for Linux NVIDIA GPU (CUDA 12.1+)
├── docker-compose.metal.yml     # Production Compose for Apple Silicon (ARM64 / Metal)
├── app/
│   ├── main.py                  # FastAPI application entrypoint & lifespan
│   ├── config.py                # Environment configuration (.env loader)
│   ├── auth.py                  # Bearer token & X-API-Key security
│   ├── db.py                    # SQLite connection pool with WAL mode
│   ├── models/
│   │   ├── api_keys.py          # API key hashing, storage, and revocation
│   │   └── audit.py             # Telemetry & request token audit logs
│   ├── api/
│   │   ├── v1/
│   │   │   ├── systemone.py     # POST /v1/systemone (Jev Contract)
│   │   │   ├── models.py        # GET /v1/models (Registered models)
│   │   │   └── admin.py         # API key creation & audit logs
│   │   └── health.py            # /health, /ready probes
│   ├── services/
│   │   └── engine_service.py    # Thread-safe model cache and inference executor
│   └── core/                    # Nirnaya decision engine core logic
├── docker/
│   ├── Dockerfile.cpu           # Linux CPU production image
│   ├── Dockerfile.cuda          # Linux NVIDIA CUDA GPU image
│   ├── Dockerfile.metal         # Apple Silicon ARM64 container
│   └── entrypoint.sh            # Dual-service orchestrator (FastAPI + Streamlit)
├── scripts/
│   ├── test_client.py           # End-to-end integration test client
│   ├── launch_metal.sh          # Native macOS Apple Silicon Metal launcher
│   ├── run_streamlit.ps1        # Streamlit runner (PowerShell)
│   └── run_streamlit.bat        # Streamlit runner (CMD)
├── streamlit_app/               # Streamlit Decision Studio UI
├── tests/                       # Unit & integration test suites
├── requirements.txt             # Backend dependencies
├── requirements-ui.txt          # Frontend dependencies
└── .env.example                 # Configuration template
```

---

## 3. Configuration (`.env`)

Configure the active model and server settings in `nirnaya_server/.env`:

```ini
# Network Settings
NIRNAYA_HOST=0.0.0.0
NIRNAYA_PORT=8000
NIRNAYA_ENV=production

# Model Engine Selection
# Options: "qwen3-0.6b-gguf", "qwen3.5-4b-gguf", "bonsai-27b", "qwen3-4b"
NIRNAYA_DEFAULT_MODEL=qwen3-0.6b-gguf
NIRNAYA_MODELS_CONFIG=app/core/models_config.json

# Database & Security
NIRNAYA_DB_PATH=data/nirnaya.db
INITIAL_ADMIN_KEY=nir_live_root_secret_key_change_me
REQUIRE_AUTH=true

# Performance
WARMUP_ON_STARTUP=false
```

---

## 4. Docker Deployment Instructions

Each environment has a dedicated, self-contained Docker Compose file located at the repository root (`nirnaya_server/`). All Compose setups:
- Run both the **FastAPI Decision Backend (`:8000`)** and **Streamlit Decision Studio UI (`:8501`)** concurrently via `entrypoint.sh`.
- Persist SQLite database and audit logs in `./data`.
- Mount local model weights from `./models` so weights are not redownloaded across restarts.
- Have `restart: unless-stopped` enabled for high availability and automatic reboot recovery.

### Quick Command Matrix

| Target Environment | Compose Command | Image Target | Default Model |
| :--- | :--- | :--- | :--- |
| 🍏 **Apple Silicon (Mac Metal / ARM64)** | `docker compose -f docker-compose.metal.yml up --build -d` | `nirnaya-platform:metal` | `qwen3-0.6b-gguf` |
| 🐧 **Linux / Windows CPU** | `docker compose -f docker-compose.cpu.yml up --build -d` | `nirnaya-platform:cpu` | `qwen3-0.6b-gguf` |
| 🚀 **Linux NVIDIA GPU (CUDA)** | `docker compose -f docker-compose.gpu.yml up --build -d` | `nirnaya-platform:cuda` | `qwen3.5-4b-gguf` |

---

### A. Apple Silicon Mac Deployment (Tested & Verified)

#### 1. Containerized Metal Build (Docker Compose)
Optimized for Apple Silicon ARM64 chips (M1/M2/M3/M4) with OpenBLAS and ARM NEON vector instructions:

```bash
# 1. Ensure write permissions for SQLite database persistence
chmod -R 777 data/

# 2. Build and launch container in detached mode
docker compose -f docker-compose.metal.yml up --build -d

# 3. Verify container status and healthcheck probe
docker compose -f docker-compose.metal.yml ps

# 4. View unified logs (FastAPI backend + Streamlit UI)
docker compose -f docker-compose.metal.yml logs -f
```

> [!NOTE]
> Docker Desktop on macOS runs containers within an ARM64 Linux VM where the macOS hypervisor does not expose the proprietary macOS Metal GPU framework to Linux guests. `Dockerfile.metal` maximizes performance through ARM64 NEON matrix kernels and OpenBLAS.

#### 2. Native macOS Metal GPU Launcher (Bare-Metal 100% Metal GPU)
For 100% native Apple Metal GPU acceleration directly utilizing Apple Silicon Unified Memory (MPS / Metal framework), use the native launcher script:

```bash
# Run bare-metal macOS server with Metal GPU compilation
bash scripts/launch_metal.sh
```
This script creates a `.venv_metal` virtualenv, compiles `llama-cpp-python` with `CMAKE_ARGS="-DGGML_METAL=on"`, and starts Uvicorn with direct Apple Metal GPU offloading.

---

### B. Linux / Windows CPU Deployment
Optimized for x86_64 CPU systems with OpenBLAS and AVX2/AVX-512 acceleration:

```bash
# Build and run CPU container
docker compose -f docker-compose.cpu.yml up --build -d

# Verify container status
docker compose -f docker-compose.cpu.yml ps
```

---

### C. Linux NVIDIA GPU Deployment (CUDA 12.1+)
Configured with discrete NVIDIA GPU passthrough and CUDA 12.1+ runtime:

```bash
# Prerequisites: NVIDIA Container Toolkit installed
docker compose -f docker-compose.gpu.yml up --build -d

# Verify GPU container status
docker compose -f docker-compose.gpu.yml ps
```

---

### D. Verifying the Deployment

Run the included verification client to test health, readiness, model registration, and an active decision query:

```bash
# From within the running container:
docker exec -it nirnaya-platform-metal python /app/nirnaya_server/scripts/test_client.py

# Or directly from the host:
python scripts/test_client.py
```

Expected output:
```
[1/4] Health Check:    HTTP 200 -> {'status': 'ok', 'service': 'nirnaya-server'}
[2/4] Readiness Check: HTTP 200 -> {'status': 'ready', 'database': 'connected', 'default_model': 'qwen3-0.6b-gguf'}
[3/4] Models Endpoint: HTTP 200 -> Registered models listed
[4/4] Sending Decision Request to POST /v1/systemone ...
✅ Prediction succeeded! (Model: qwen3-0.6b-gguf)
```

---

## 5. API Reference & Contract

### A. Decision Readout: `POST /v1/systemone`

**Headers:**
```http
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

**Request Body:**
```json
{
  "model": "qwen3-0.6b-gguf",
  "state": "Customer has an overdue balance of $450 and their card was declined twice.",
  "questions": {
    "urgency": {
      "type": "score",
      "instructions": "Rate how urgent this issue is from 0 to 4",
      "criteria": ["trivial", "low", "medium", "high", "critical"]
    },
    "category": {
      "type": "choice",
      "instructions": "Select the primary ticket category",
      "criteria": {
        "billing": "Invoices, payments, credit card failures",
        "technical": "Software bugs, system outages",
        "general": "General questions"
      }
    },
    "should_escalate": {
      "type": "noul",
      "instructions": "Should this ticket be escalated to a human supervisor?"
    }
  }
}
```

**Response (Exact TypeSafe Jev Contract):**
```json
{
  "model": "qwen3-0.6b-gguf",
  "answers": {
    "urgency": {
      "type": "score",
      "score": 3.24,
      "confidence": 0.74,
      "probabilities": {
        "0": 0.01,
        "1": 0.03,
        "2": 0.12,
        "3": 0.44,
        "4": 0.40
      },
      "legend": {
        "0": "trivial",
        "1": "low",
        "2": "medium",
        "3": "high",
        "4": "critical"
      }
    },
    "category": {
      "type": "choice",
      "choice": "billing",
      "confidence": 1.0,
      "probabilities": {
        "billing": 1.0,
        "technical": 0.0,
        "general": 0.0
      }
    },
    "should_escalate": {
      "type": "noul",
      "noul": 0.1091
    }
  },
  "usage": {
    "input_tokens": 1248,
    "output_tokens": 3
  }
}
```

### B. Health Probes
- `GET /health`: Liveness probe (returns `{"status": "ok"}`).
- `GET /ready`: Readiness probe (checks SQLite connectivity and default model status).

### C. Models Endpoint: `GET /v1/models`
Returns list of registered models, backend types, and active hardware acceleration (CUDA/CPU).

---

## 6. Verification & Automated Testing

Run the full pytest suite (15/15 tests passing):
```powershell
python -m pytest nirnaya_server/tests -v
```

Run the end-to-end integration test client against the live server:
```powershell
python nirnaya_server/scripts/test_client.py
```

---

## 7. Interactive Streamlit Showcase & Decision Studio

Nirnaya Server includes a standalone, production-ready Streamlit frontend designed to showcase Nirnaya's System-One typed probabilistic decisions (`choice`, `score`, `noul`) with built-in user authentication and live backend health monitoring.

### A. Key Features
- **User Authentication:** Form-based login with session management. Default credentials: `admin` / `nirnaya`.
- **CPU Timeout Protection:** Long-running CPU inference processes won't drop or time out. Configurable via `NIRNAYA_API_TIMEOUT_SECONDS` (default: **300 seconds** / 5 minutes) in `.env.streamlit`.
- **Live Backend Health Monitoring:** Continuously probes `GET /health` and `GET /ready` with a pulsing status indicator, reporting connectivity, latency, and default model.
- **Curated Showcase Scenarios:** Preloaded realistic presets for:
  - 🎫 Customer Support & Churn Risk
  - 🚨 Production Outage & SRE Escalation
  - 💳 Commercial Loan Underwriting & Risk Tiering
  - 🛡️ AI Content Moderation & Prompt Injection Guardrails
  - ⚙️ Custom Freeform JSON/Text Playground
- **Rich Decision Visualization:** Real-time probability bar charts, continuous 1D Center-of-Gravity scoring, and exact TypeSafe AI Jev JSON contract export.
- **Live Audit & Telemetry:** Inspect recent requests, status codes, and latency distributions directly from SQLite.

### B. Configuration (`.env`)
All backend and Streamlit settings are unified directly in `nirnaya_server/.env`:
```ini
# Backend Server Connection
NIRNAYA_HOST=0.0.0.0
NIRNAYA_PORT=8000
NIRNAYA_DEFAULT_MODEL=qwen3-0.6b-gguf
INITIAL_ADMIN_KEY=nir_live_root_secret_key_change_me
REQUIRE_AUTH=true

# Streamlit UI Configuration
SERVICE_TYPE=all
STREAMLIT_PORT=8501
NIRNAYA_API_URL=http://localhost:8000

# Request Timeouts (in seconds)
# Configured high (300s) to prevent timeouts during long-running CPU inferences
NIRNAYA_API_TIMEOUT_SECONDS=300
NIRNAYA_CONNECT_TIMEOUT_SECONDS=15

# Streamlit Authentication Credentials
STREAMLIT_AUTH_USERNAME=admin
STREAMLIT_AUTH_PASSWORD=nirnaya
```

### C. Launching the Streamlit App

**Via PowerShell Launcher:**
```powershell
.\nirnaya_server\scripts\run_streamlit.ps1
```

**Via Windows Batch Launcher:**
```cmd
nirnaya_server\scripts\run_streamlit.bat
```

**Direct Python Command:**
```powershell
python -m streamlit run nirnaya_server/streamlit_app/app.py --server.port 8501
```

Then open your browser at **`http://localhost:8501`** and sign in with `admin` / `nirnaya`.

