#!/bin/bash

# This script archives the 'mlruns' directory into the '.temp' directory.

# Define source and destination
SOURCE_DIR="mlruns"
DEST_DIR="cache"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="mlruns_${TIMESTAMP}.zip"
ARCHIVE_PATH="${DEST_DIR}/${ARCHIVE_NAME}"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '${SOURCE_DIR}' not found in the current path."
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Create the compressed tarball
echo "Archiving '${SOURCE_DIR}' to '${ARCHIVE_PATH}'..."
zip -r "${ARCHIVE_PATH}" "${SOURCE_DIR}"

echo "Archive complete."
echo "File is located at: ${ARCHIVE_PATH}"
echo ""
echo "To download, run the following command from your LOCAL machine's terminal:"
echo "scp <your_user>@<your_host>:/home/Zeitler/code/SIDE/${ARCHIVE_PATH} ."