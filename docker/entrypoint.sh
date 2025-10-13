#!/bin/bash
set -e

# Print environment info
echo "-----------------------------------------------------"
echo " 🧠 Launching YSC Training Environment"
echo "-----------------------------------------------------"
nvidia-smi
nvidia-smi
python3 - <<EOF
import torch, torchvision, torchaudio
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch vision version:", torchvision.__version__)
print("Torchaudio version:", torchaudio.__version__)
EOF
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "-----------------------------------------------------"

# Start JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser \
    --NotebookApp.token='' \
    --NotebookApp.password=''
