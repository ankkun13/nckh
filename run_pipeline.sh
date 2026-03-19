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

# ✅ OPTIMIZATION: Prevent CUDA memory fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

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
ALL_CANCERS=false
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
        --all-cancers)
            ALL_CANCERS=true
            shift
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
echo -e "${GREEN}All Cancers:${NC} $ALL_CANCERS"
echo -e "${GREEN}Extra:${NC}  $EXTRA_ARGS"
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}Python version:${NC} $PYTHON_VERSION"

if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}PyTorch not installed. Please run 'make setup' or './install_dependencies.sh' first.${NC}"
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
    local current_cancer=${2:-}
    local cancer_arg=""
    if [[ -n "$current_cancer" ]]; then
        cancer_arg="--cancer $current_cancer"
    fi

    echo -e "${CYAN}──────────────────────────────────────────────────────────────────────────${NC}"
    echo -e "${GREEN}▶ Running stage: ${stage_name}${NC} ${YELLOW}${current_cancer:+[Cancer: $current_cancer]}${NC}"
    echo -e "${CYAN}──────────────────────────────────────────────────────────────────────────${NC}"
    python3 main.py --config "$CONFIG_FILE" --stage "$stage_name" $cancer_arg $EXTRA_ARGS
    echo -e "${GREEN}✓ Stage ${stage_name} completed${NC}"
    echo ""
}

execute_all_stages() {
    local cancer=$1
    run_stage "preprocess" "$cancer"
    run_stage "cluster" "$cancer"
    run_stage "network" "$cancer"
    run_stage "train" "$cancer"
    run_stage "evaluate" "$cancer"
}

if [[ "$ALL_CANCERS" == "true" ]]; then
    CANCER_TYPES=$(python3 -c "import yaml; config = yaml.safe_load(open('$CONFIG_FILE')); print(' '.join(config['data']['supported_cancers']))")
    echo -e "${YELLOW}Running pipeline for ALL cancers: $CANCER_TYPES${NC}"
    for CANCER in $CANCER_TYPES; do
        echo -e "${YELLOW}>>> PROCESSING CANCER: $CANCER <<<${NC}"
        if [[ "$STAGE" == "all" ]]; then
            execute_all_stages "$CANCER"
        else
            run_stage "$STAGE" "$CANCER"
        fi
    done
else
    # Single cancer run (uses config default or --cancer in EXTRA_ARGS)
    if [[ "$STAGE" == "all" ]]; then
        execute_all_stages ""
    else
        run_stage "$STAGE" ""
    fi
fi

echo -e "${CYAN}============================================================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${GREEN}Results saved to: ./results/${NC}"
echo -e "${GREEN}Execution log: ./execution.log${NC}"
echo -e "${CYAN}============================================================================${NC}"
