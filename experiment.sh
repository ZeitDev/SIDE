#!/bin/bash

SESSION_NAME=$(date +"%d%H%M")
CONFIG_DIR="configs"
DEFAULT_NAME="exp1"

read -p "Enter experiment name (default: $DEFAULT_NAME): " EXPERIMENT_NAME
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$DEFAULT_NAME}
EXP_DIR="$CONFIG_DIR/$EXPERIMENT_NAME"

if [ ! -d "$EXP_DIR" ]; then
    echo "Error: Experiment folder not found at '$EXP_DIR'"
    exit 1
fi

CUDA_DEVICE=1
read -p "Enter cuda device number (default: $CUDA_DEVICE): " INPUT_DEVICE
CUDA_DEVICE=${INPUT_DEVICE:-$CUDA_DEVICE}

echo "Starting experiment loop: $EXPERIMENT_NAME on CUDA device $CUDA_DEVICE"

tmux has-session -t $SESSION_NAME 2>/dev/null
if [ $? == 0 ]; then
    echo "Error: A tmux session named '$SESSION_NAME' already exists."
    echo "Attach to it with: tmux attach -t $SESSION_NAME"
    exit 1
fi

nohup tmux new-session -d -s "$SESSION_NAME" -n "experiment" \; \
    send-keys -t "$SESSION_NAME:experiment.0" "uv run python notebooks/train/run_experiment.py --experiment '$EXPERIMENT_NAME' --cuda_device $CUDA_DEVICE" C-m > /dev/null 2>&1 &
    
echo "List all sessions with:     tmux ls"
echo "Kill the session with:      tmux kill-session -t $SESSION_NAME"
echo "Attach to the session with: tmux attach -t $SESSION_NAME"