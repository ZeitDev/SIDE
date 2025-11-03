#!/bin/bash

# This script creates a new detached tmux session, splits it into two panes,
# and starts the MLflow UI in one and the training script in the other.

# --- Configuration ---
SESSION_NAME="zeitler"
CONFIG_DIR="configs"         # Directory where configs are stored
DEFAULT_NAME="base"       # The name of the default config *without* path or extension
DEFAULT_CONFIG="$CONFIG_DIR/$DEFAULT_NAME.yaml" # Full path to default

# MLflow UI configuration
MLFLOW_HOST="127.0.0.1"
MLFLOW_PORT="5000"

# --- Ask for config name ---
echo "Default config is: $DEFAULT_CONFIG"
# Ask the user for just the name
read -p "Enter config name (default: $DEFAULT_NAME): " CONFIG_NAME

# If the user just pressed Enter, CONFIG_NAME will be empty.
# This line sets it to the DEFAULT_NAME in that case.
CONFIG_NAME=${CONFIG_NAME:-$DEFAULT_NAME}

# --- Construct the full path and check if it exists ---
CONFIG_FILE="$CONFIG_DIR/$CONFIG_NAME.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at '$CONFIG_FILE'"
    echo "(Looked for '$CONFIG_NAME' in '$CONFIG_DIR' with .yaml extension)"
    exit 1
fi

# --- Feedback to user ---
echo "Using config file: $CONFIG_FILE"

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
tmux split-window -h -p 95 -t "$SESSION_NAME"

# --- Send commands to the panes ---

# Pane 0 (Left): MLflow UI Server
tmux send-keys -t "$SESSION_NAME:0.0" "source .venv/bin/activate" C-m
# Use the variables to start the server
tmux send-keys -t "$SESSION_NAME:0.0" "echo 'Starting MLflow UI...'; mlflow ui --host $MLFLOW_HOST --port $MLFLOW_PORT" C-m

# Pane 1 (Right): Training Script
tmux send-keys -t "$SESSION_NAME:0.1" "source .venv/bin/activate" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "echo 'Waiting for MLflow server...'; sleep 5" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "python main.py --config '$CONFIG_FILE'" C-m
tmux send-keys -t "$SESSION_NAME:0.1" "echo '✅ Training finished. This pane can be closed with Ctrl+D.'" C-m

# --- Final output to the main terminal ---
echo "✅ Session '$SESSION_NAME' is running in the background."
echo "👉 MLflow UI should be available at: http://$MLFLOW_HOST:$MLFLOW_PORT"
echo "Kill session with: tmux kill-session -t $SESSION_NAME"
echo "Detach from session with: Ctrl+b then d"
echo "Attach to the session with: tmux attach -t $SESSION_NAME"