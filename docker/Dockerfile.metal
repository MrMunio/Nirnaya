# ==============================================================================
# Nirnaya Server - Apple Silicon ARM64 Containerized Deployment
# Note: Docker Desktop on macOS runs an arm64 Linux VM where Apple Metal GPU
# passthrough is not supported by macOS hypervisor. This Dockerfile optimizes
# ARM64 NEON CPU matrix acceleration.
# For 100% Native Metal GPU acceleration, use `scripts/launch_metal.sh`.
# ==============================================================================
FROM python:3.11-slim-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Build with ARM NEON optimizations
ENV CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"
RUN pip install --upgrade pip setuptools wheel && \
    pip install llama-cpp-python==0.3.35 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu || \
    pip install llama-cpp-python==0.3.35

# ------------------------------------------------------------------------------
# Final Runtime Stage
# ------------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NIRNAYA_HOST=0.0.0.0 \
    NIRNAYA_PORT=8000 \
    NIRNAYA_ENV=production \
    NIRNAYA_DEFAULT_MODEL=qwen3-0.6b-gguf \
    NIRNAYA_DB_PATH=/app/nirnaya_server/data/nirnaya.db \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/nirnaya_server
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

RUN mkdir -p /app/nirnaya_server/data /app/models

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000 8501

ENTRYPOINT ["/app/entrypoint.sh"]
