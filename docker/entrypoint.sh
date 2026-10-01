#!/usr/bin/env bash
# ==============================================================================
# Nirnaya Docker Unified Entrypoint Script
# Supports:
#   SERVICE_TYPE=all (Default: Starts FastAPI on port 8000 & Streamlit UI on 8501)
#   SERVICE_TYPE=api (Starts only FastAPI backend)
#   SERVICE_TYPE=ui  (Starts only Streamlit frontend)
# ==============================================================================
set -e

SERVICE_TYPE="${SERVICE_TYPE:-all}"
PORT="${NIRNAYA_PORT:-8000}"
HOST="${NIRNAYA_HOST:-0.0.0.0}"
UI_PORT="${STREAMLIT_PORT:-8501}"

echo "=================================================================="
echo "⚡ Starting Nirnaya Decision Platform (Mode: $SERVICE_TYPE)"
echo "📍 API Port: $PORT | UI Port: $UI_PORT | Env: ${NIRNAYA_ENV:-production}"
echo "=================================================================="

# Function for clean shutdown of background children
cleanup() {
    echo "🛑 Caught signal. Terminating child processes..."
    if [ -n "$API_PID" ]; then
        kill -TERM "$API_PID" 2>/dev/null || true
    fi
    if [ -n "$UI_PID" ]; then
        kill -TERM "$UI_PID" 2>/dev/null || true
    fi
    wait
    echo "✅ Shutdown complete."
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

if [ "$SERVICE_TYPE" = "api" ]; then
    echo "🚀 Launching FastAPI backend only..."
    exec python -m uvicorn nirnaya_server.app.main:app --host "$HOST" --port "$PORT" --workers 1

elif [ "$SERVICE_TYPE" = "ui" ]; then
    echo "🚀 Launching Streamlit Decision Studio UI only..."
    exec streamlit run nirnaya_server/streamlit_app/app.py \
        --server.port "$UI_PORT" \
        --server.address 0.0.0.0 \
        --server.headless true

elif [ "$SERVICE_TYPE" = "all" ]; then
    echo "🚀 [1/2] Starting FastAPI backend on http://${HOST}:${PORT}..."
    python -m uvicorn nirnaya_server.app.main:app --host "$HOST" --port "$PORT" --workers 1 &
    API_PID=$!

    # Wait for FastAPI backend to become healthy
    echo "⏳ Waiting for backend API to report healthy..."
    MAX_RETRIES=40
    RETRY_COUNT=0
    HEALTHY=0

    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -s -f "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            HEALTHY=1
            echo "✅ Backend API is healthy!"
            break
        fi
        sleep 1
        RETRY_COUNT=$((RETRY_COUNT + 1))
    done

    if [ $HEALTHY -ne 1 ]; then
        echo "⚠️ Warning: Backend health check did not pass within 40s. Proceeding with UI launch..."
    fi

    echo "🚀 [2/2] Starting Streamlit Decision Studio UI on http://0.0.0.0:${UI_PORT}..."
    streamlit run nirnaya_server/streamlit_app/app.py \
        --server.port "$UI_PORT" \
        --server.address 0.0.0.0 \
        --server.headless true &
    UI_PID=$!

    echo "=================================================================="
    echo "🎉 Nirnaya Full-Stack Service is Running:"
    echo "   - FastAPI Swagger API: http://localhost:${PORT}/docs"
    echo "   - Streamlit Studio UI: http://localhost:${UI_PORT}"
    echo "=================================================================="

    # Wait for both background processes
    wait -n "$API_PID" "$UI_PID"
    cleanup
else
    # Allow custom commands (e.g. bash, pytest)
    exec "$@"
fi
