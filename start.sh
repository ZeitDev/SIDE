#!/bin/bash

# This script creates a new detached tmux session, splits it into two panes,
# and starts the MLflow UI in one and the training script in the other.

# --- Configuration ---
SESSION_NAME="zeitler"
DEFAULT_CONFIG="configs/default.yaml"
CONFIG_FILE=${1:-$DEFAULT_CONFIG}

# --- Check if config file exists ---
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at '$CONFIG_FILE'"
    exit 1
fi

# --- Check if a tmux session with the same name already exists ---
tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "Error: A tmux session named '$SESSION_NAME' already exists."
    echo "Attach to it with: tmux attach -t $SESSION_NAME"
    echo "Or kill it with:   tmux kill -t $SESSION_NAME"
    exit 1
fi

# --- Create the tmux session and panes ---
echo "🚀 Creating new tmux session: '$SESSION_NAME'"
# Create a new detached session
tmux new-session -d -s "$SESSION_NAME"

# Split the window into two horizontal panes
tmux split-window -h -t "$SESSION_NAME"

# --- Send commands to the panes ---

# Pane 0 (Left): MLflow UI Server
tmux send-keys -t "$SESSION_NAME:0.0" "source .venv/bin/activate" C-m
tmux send-keys -t "$SESSION_NAME:0.0" "echo 'Starting MLflow UI...'; mlflow ui" C-m

# Pane 1 (Right): Training Script
tmux send-keys -t "$SESSION_NAME:0.1" "source .venv/bin/activate" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "echo 'Waiting for MLflow server...'; sleep 5" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "python main.py --config '$CONFIG_FILE'" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "echo '✅ Training finished. This pane can be closed with Ctrl+D.'" C-m

echo "✅ Session '$SESSION_NAME' is running in the background."
echo "Attach to it with the command: tmux attach -t $SESSION_NAME"