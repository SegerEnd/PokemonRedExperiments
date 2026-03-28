#!/bin/bash

# Start the Rust WebSocket server
echo "Starting pokemap-ws..."
cd /Users/seger/Documents/GitHub/pokemap-ws && docker compose up -d --build

# Start training
echo "Starting training..."
cd /Users/seger/Documents/GitHub/PokemonRedExperiments/v2
source venv/bin/activate

# Resume from latest checkpoint if one exists (includes both periodic and graceful-shutdown saves)
latest=$(ls -t runs/poke_*.zip 2>/dev/null | head -1 | sed 's/\.zip$//')
if [ -n "$latest" ]; then
    echo "Resuming from checkpoint: $latest"
    echo "$latest" | python baseline_fast_v2.py
else
    echo "Starting fresh"
    python baseline_fast_v2.py
fi

# When training stops, stop the server
echo "Shutting down pokemap-ws..."
cd /Users/seger/Documents/GitHub/pokemap-ws && docker compose down
