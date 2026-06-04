#!/bin/bash
cd /data/Zeitler/masterthesis || exit 1
git add .
git commit -m "."
git push

echo "✅ Figures pushed to Overleaf!"