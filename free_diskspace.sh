#!/bin/bash

# Define the directories to be cleaned
DIRECTORIES=("cache" "mlruns" "logs")

echo "This script will permanently delete the contents of the following directories:"
for dir in "${DIRECTORIES[@]}"; do
    if [ -d "$dir" ]; then
        echo " - $dir"
    fi
done
echo ""

# Ask for user confirmation
read -p "Are you sure you want to continue? [y/N] " -n 1 -r
echo    # Move to a new line

# Check the user's input
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Starting cleanup..."
    for dir in "${DIRECTORIES[@]}"; do
        if [ -d "$dir" ]; then
            echo "Cleaning '$dir'..."
            # Remove all files and subdirectories within the directory
            rm -rf "$dir"/*
            echo "Finished cleaning '$dir'."
        else
            echo "Directory '$dir' not found, skipping."
        fi
    done
    echo "Cleanup complete."
else
    echo "Cleanup aborted by user."
fi