#!/bin/bash
# ============================================================================
# GNN Cancer Driver Gene Identification Pipeline — Dependency Installer
# ============================================================================
# This script handles the complex installation of PyTorch and PyTorch Geometric
# libraries, ensuring version compatibility and CUDA support.
# ============================================================================

set -e

echo "============================================================================"
echo "  Starting Dependency Installation"
echo "============================================================================"

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed."
    exit 1
fi

# 1. Install/Update PyTorch
echo "--> Installing PyTorch (>=2.0.0)..."
pip install "torch>=2.0.0" torchvision torchaudio

# 2. Get PyTorch and CUDA versions for PyG installation
echo "--> Detecting PyTorch and CUDA versions..."
PYTHON_INFO=$(python3 -c "
import torch
import sys

torch_version = torch.__version__.split('+')[0]
if torch.cuda.is_available():
    cuda_version = 'cu' + torch.version.cuda.replace('.', '')
else:
    cuda_version = 'cpu'

print(f'{torch_version}|{cuda_version}')
" 2>/dev/null)

if [ -z "$PYTHON_INFO" ]; then
    echo "Error: Failed to detect PyTorch version."
    exit 1
fi

TORCH_VERSION=$(echo $PYTHON_INFO | cut -d'|' -f1)
CUDA_VERSION=$(echo $PYTHON_INFO | cut -d'|' -f2)

echo "    PyTorch version: $TORCH_VERSION"
echo "    CUDA version:    $CUDA_VERSION"

# 3. Install PyTorch Geometric dependencies
# Note: torch-scatter and torch-sparse require matching wheels
echo "--> Installing PyTorch Geometric dependencies..."
WHL_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html"
echo "    Using wheel index: $WHL_URL"

pip install torch-scatter torch-sparse -f "$WHL_URL"
pip install torch-geometric

# 4. Install remaining requirements
echo "--> Installing remaining requirements from requirements.txt..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found."
fi

echo "============================================================================"
echo "  Dependency Installation Completed Successfully!"
echo "============================================================================"
echo "To verify the installation, run:"
echo "python3 -c \"import torch, torch_geometric; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__)\""
echo "============================================================================"
