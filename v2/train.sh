#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if present
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi

# Resume from latest checkpoint if one exists
latest=$(ls -t runs/poke_*.zip 2>/dev/null | head -1 | sed 's/\.zip$//')
if [ -n "$latest" ]; then
    echo "Resuming from checkpoint: $latest"
    echo "$latest" | python baseline_fast_v2.py
else
    echo "Starting fresh training"
    python baseline_fast_v2.py
fi
