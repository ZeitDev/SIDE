#!/bin/bash

# This script uses nohup to start a truly detached tmux session,
# even when run from within a VS Code integrated terminal.

# --- Configuration ---
SESSION_NAME="zeitler"
CONFIG_DIR="configs"
DEFAULT_NAME="base"
DEFAULT_CONFIG="$CONFIG_DIR/$DEFAULT_NAME.yaml"

# MLflow UI configuration
MLFLOW_HOST="127.0.0.1"
MLFLOW_PORT="5000"

# --- Pre-flight check for the port ---
LISTENING_PID=$(lsof -t -iTCP:$MLFLOW_PORT -sTCP:LISTEN)
if [ -n "$LISTENING_PID" ]; then
    echo "Error: Port $MLFLOW_PORT is already in use. Killing process group $LISTENING_PID."
    kill -9 -- "-$LISTENING_PID"
    sleep 1
fi

# --- Ask for config name ---
read -p "Enter config name (default: $DEFAULT_NAME): " CONFIG_NAME
CONFIG_NAME=${CONFIG_NAME:-$DEFAULT_NAME}
CONFIG_FILE="$CONFIG_DIR/$CONFIG_NAME.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at '$CONFIG_FILE'"
    exit 1
fi

echo "Using config file: $CONFIG_FILE"

# --- Check for existing session ---
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "Error: A tmux session named '$SESSION_NAME' already exists."
    echo "Attach to it with: tmux attach -t $SESSION_NAME"
    exit 1
fi

# --- Define the commands for the panes ---
MLFLOW_CMD="
cleanup() {
    echo 'Shutting down MLflow server process group...';
    # Kill the entire process group using the negative PID
    kill -- -\$MLFLOW_PID;
};
trap cleanup EXIT;
uv run mlflow ui --host $MLFLOW_HOST --port $MLFLOW_PORT &
MLFLOW_PID=\$!;
wait \$MLFLOW_PID;
"

TRAIN_CMD="
echo 'Waiting for MLflow server...'; sleep 5;
uv run python main.py --config '$CONFIG_FILE';
echo '✅ Training finished. This pane can be closed with Ctrl+D.';
# Keep the pane open after the script finishes
exec bash;
"

# --- The Core Solution: Use nohup to start tmux ---
echo "🚀 Launching detached tmux session '$SESSION_NAME' with nohup..."
nohup tmux new-session -d -s "$SESSION_NAME" -n "main" \; \
    split-window -h -p 95 \; \
    send-keys -t "$SESSION_NAME:main.0" "$MLFLOW_CMD" C-m \; \
    send-keys -t "$SESSION_NAME:main.1" "$TRAIN_CMD" C-m > /dev/null 2>&1 &

# --- Final output ---
sleep 2
echo "✅ Session '$SESSION_NAME' is running in the background, detached from this terminal."
echo "You can now safely close VS Code."
echo "👉 MLflow UI should be available at: http://$MLFLOW_HOST:$MLFLOW_PORT"
echo "Kill the session with:      tmux kill-session -t $SESSION_NAME"
echo "Attach to the session with: tmux attach -t $SESSION_NAME"