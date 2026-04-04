#!/bin/bash

# This script starts the MLflow UI server with robust cleanup.

# --- Configuration ---
MLFLOW_HOST="127.0.0.1"
MLFLOW_PORT="5000"

# --- Pre-flight check for the port ---
if lsof -t -i:$MLFLOW_PORT > /dev/null; then
    echo "Error: Port $MLFLOW_PORT is already in use. Killing the process."
    # Use the more robust process group kill here as well
    kill -- -$(lsof -t -i:$MLFLOW_PORT)
    sleep 1 # Give the OS a moment to release the port
fi

# --- Cleanup function and trap ---
cleanup() {
    echo -e "\nShutting down MLflow server and all its children..."
    if [ -n "$MLFLOW_PID" ]; then
        # Kill the entire process group by using a negative PID
        # The '--' ensures that the negative PID isn't treated as an option.
        kill -- "-$MLFLOW_PID"
    fi
    echo "Server stopped."
}

# Trap the EXIT signal to call the cleanup function.
trap cleanup EXIT

# --- Start MLflow UI ---
echo "🚀 Starting MLflow UI..."

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated."
else
    echo "Warning: .venv/bin/activate not found. Assuming mlflow is in PATH."
fi

echo "👉 MLflow UI will be available at: http://$MLFLOW_HOST:$MLFLOW_PORT"
echo "Press Ctrl+C to stop the server."

# Start the server in the background, get its PID, and wait for it.
mlflow ui --host $MLFLOW_HOST --port $MLFLOW_PORT &
MLFLOW_PID=$!
wait "$MLFLOW_PID"