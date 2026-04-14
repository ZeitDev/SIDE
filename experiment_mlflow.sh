#!/bin/bash

MLFLOW_HOST="127.0.0.1"
MLFLOW_PORT="5000"
DEFAULT_NAME="exp1"

read -p "Enter experiment name to view (default: $DEFAULT_NAME): " EXPERIMENT_NAME
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$DEFAULT_NAME}
TRACKING_DIR="./mlflow/$EXPERIMENT_NAME"

if [ ! -d "$TRACKING_DIR" ]; then
    echo "Error: MLflow tracking folder not found at '$TRACKING_DIR'"
    echo "Make sure the experiment has been started with run_experiment.py first."
    exit 1
fi

if lsof -t -i:$MLFLOW_PORT > /dev/null 2>&1; then
    echo "Error: Port $MLFLOW_PORT is already in use. Killing the process."
    kill -9 $(lsof -t -i:$MLFLOW_PORT)
    sleep 1
fi

cleanup() {
    echo -e "\nShutting down MLflow server..."
    if [ -n "$MLFLOW_PID" ]; then
        kill -9 $MLFLOW_PID 2>/dev/null
    fi
    echo "Server stopped."
}

trap cleanup EXIT

echo "🚀 Starting MLflow UI for experiment: $EXPERIMENT_NAME..."

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated."
fi

echo "👉 MLflow UI will be available at: http://$MLFLOW_HOST:$MLFLOW_PORT"
echo "Reading tracking data from: $TRACKING_DIR"
echo "Press Ctrl+C to stop the server."

uv run mlflow ui --host $MLFLOW_HOST --port $MLFLOW_PORT --backend-store-uri "file:$TRACKING_DIR" &
MLFLOW_PID=$!
wait "$MLFLOW_PID"