#!/bin/bash
# ============================================================================
# GNN Cancer Driver Gene Identification Pipeline — Run Script
# ============================================================================
# Usage:
#   ./run_pipeline.sh                              # Run full pipeline with defaults
#   ./run_pipeline.sh --cancer TCGA-LUSC            # Specify cancer type
#   ./run_pipeline.sh --stage preprocess            # Run single stage
#   ./run_pipeline.sh --epochs 500 --cv_folds 5     # Override training params
#
# Stages: preprocess, cluster, network, train, evaluate, all (default)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default configuration
CONFIG_FILE="configs/config.yaml"
STAGE="all"
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo -e "${CYAN}============================================================================${NC}"
echo -e "${CYAN}  GNN Cancer Driver Gene Identification Pipeline${NC}"
echo -e "${CYAN}============================================================================${NC}"
echo -e "${GREEN}Config:${NC} $CONFIG_FILE"
echo -e "${GREEN}Stage:${NC}  $STAGE"
echo -e "${GREEN}Extra:${NC}  $EXTRA_ARGS"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}Python version:${NC} $PYTHON_VERSION"

if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}PyTorch not installed. Please run 'make setup' or './install_dependencies.sh' first.${NC}"
    # Optional: read -p "Would you like to install dependencies now? (y/n) " -n 1 -r
    # if [[ $REPLY =~ ^[Yy]$ ]]; then ./install_dependencies.sh; else exit 1; fi
else
    python3 -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('CUDA not available, using CPU')
"
fi

echo ""

# Execute pipeline
run_stage() {
    local stage_name=$1
    echo -e "${CYAN}──────────────────────────────────────────────────────────────────────────${NC}"
    echo -e "${GREEN}▶ Running stage: ${stage_name}${NC}"
    echo -e "${CYAN}──────────────────────────────────────────────────────────────────────────${NC}"
    python3 main.py --config "$CONFIG_FILE" --stage "$stage_name" $EXTRA_ARGS
    echo -e "${GREEN}✓ Stage ${stage_name} completed${NC}"
    echo ""
}

case $STAGE in
    preprocess)
        run_stage "preprocess"
        ;;
    cluster)
        run_stage "cluster"
        ;;
    network)
        run_stage "network"
        ;;
    train)
        run_stage "train"
        ;;
    evaluate)
        run_stage "evaluate"
        ;;
    all)
        run_stage "preprocess"
        run_stage "cluster"
        run_stage "network"
        run_stage "train"
        run_stage "evaluate"
        ;;
    *)
        echo -e "${RED}Unknown stage: $STAGE${NC}"
        echo "Valid stages: preprocess, cluster, network, train, evaluate, all"
        exit 1
        ;;
esac

echo -e "${CYAN}============================================================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${GREEN}Results saved to: ./results/${NC}"
echo -e "${GREEN}Execution log: ./execution.log${NC}"
echo -e "${CYAN}============================================================================${NC}"
