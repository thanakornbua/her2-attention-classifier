#!/usr/bin/env bash
set -e

FLAG_FILE="/workspace/.initialized"

if [ ! -f "$FLAG_FILE" ]; then
    echo "🛠️ Running one-time setup inside container..."

    conda init bash
    conda activate her2-class

    pip install -r /workspace/docker/requirements.txt

    # Accept Anaconda ToS (if needed)
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true


    # Mark as initialized
    touch "$FLAG_FILE"

    echo "Setup complete!"
else
    echo "Environment initialized. Skipping bootstrap."
fi

# Keep shell open
exec bash
