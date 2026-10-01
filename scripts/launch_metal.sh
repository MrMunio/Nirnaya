#!/usr/bin/env bash
# ==============================================================================
# Nirnaya Server - Native macOS Metal GPU Launcher
# Enables native Apple Silicon Metal (MPS / Metal framework) GPU acceleration
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "========================================================================"
echo "🚀 Starting Nirnaya Server with Native macOS Metal Acceleration"
echo "========================================================================"

cd "${PROJECT_ROOT}"

# 1. Check Python installation
if ! command -v python3 &>/dev/null; then
    echo "❌ Error: python3 is not installed. Please install Python 3.10+."
    exit 1
fi

# 2. Check virtual environment or create one
VENV_DIR="${PROJECT_ROOT}/.venv_metal"
if [ ! -d "${VENV_DIR}" ]; then
    echo "📦 Creating virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip setuptools wheel
    
    echo "⚙️ Building llama-cpp-python with Apple Metal acceleration enabled..."
    CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python==0.3.35 --no-cache-dir
    pip install -r nirnaya_server/requirements.txt
else
    source "${VENV_DIR}/bin/activate"
fi

# 3. Export configuration
export NIRNAYA_DEFAULT_MODEL="${NIRNAYA_DEFAULT_MODEL:-qwen3.5-4b-gguf}"
export NIRNAYA_HOST="${NIRNAYA_HOST:-0.0.0.0}"
export NIRNAYA_PORT="${NIRNAYA_PORT:-8000}"

echo "✨ Metal acceleration active. Serving on http://${NIRNAYA_HOST}:${NIRNAYA_PORT} (Model: ${NIRNAYA_DEFAULT_MODEL})"
python -m uvicorn nirnaya_server.app.main:app --host "${NIRNAYA_HOST}" --port "${NIRNAYA_PORT}"
