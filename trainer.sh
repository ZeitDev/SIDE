#!/bin/bash

SESSION_NAME=$(date +"%d%H%M")
CONFIG_DIR="configs"
DEFAULT_NAME="base"
DEFAULT_CONFIG="$CONFIG_DIR/$DEFAULT_NAME.yaml"

read -p "Enter config name (default: $DEFAULT_NAME): " CONFIG_NAME
CONFIG_NAME=${CONFIG_NAME:-$DEFAULT_NAME}
CONFIG_FILE="$CONFIG_DIR/$CONFIG_NAME.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at '$CONFIG_FILE'"
    exit 1
fi

CUDA_DEVICE=1
read -p "Enter cuda device number: " CUDA_DEVICE

echo "Using config file: $CONFIG_FILE on CUDA device $CUDA_DEVICE"

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "Error: A tmux session named '$SESSION_NAME' already exists."
    echo "Attach to it with: tmux attach -t $SESSION_NAME"
    exit 1
fi

nohup tmux new-session -d -s "$SESSION_NAME" -n "main" \; \
    send-keys -t "$SESSION_NAME:main.0" "uv run python main.py --config '$CONFIG_FILE' --cuda_device $CUDA_DEVICE" C-m > /dev/null 2>&1 &
    
echo "List all sessions with:     tmux ls"
echo "Kill the session with:      tmux kill-session -t $SESSION_NAME"
echo "Attach to the session with: tmux attach -t $SESSION_NAME"