# ============================================================================
# GNN Cancer Driver Gene Identification Pipeline — Makefile
# ============================================================================
# Usage:
#   make all                    # Run full pipeline
#   make preprocess             # Run preprocessing only
#   make train CANCER=TCGA-LUSC # Override cancer type
#   make clean                  # Clean generated outputs
# ============================================================================

PYTHON := python3
CONFIG := configs/config.yaml
CANCER ?= TCGA-LUAD
EXTRA_ARGS ?=

.PHONY: all setup preprocess cluster network train evaluate clean help

help: ## Show available targets
	@echo "GNN Cancer Driver Gene Pipeline — Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

all: preprocess cluster network train evaluate ## Run full pipeline

setup: ## Install dependencies
	pip install -r requirements.txt
	@echo "✓ Dependencies installed"

preprocess: ## Stage 2: Data loading & preprocessing
	$(PYTHON) main.py --config $(CONFIG) --stage preprocess --cancer $(CANCER) $(EXTRA_ARGS)

cluster: ## Stage 3: Patient subgroup clustering
	$(PYTHON) main.py --config $(CONFIG) --stage cluster --cancer $(CANCER) $(EXTRA_ARGS)

network: ## Stage 4: Build robust co-association network
	$(PYTHON) main.py --config $(CONFIG) --stage network --cancer $(CANCER) $(EXTRA_ARGS)

train: ## Stage 5: Train GNN model
	$(PYTHON) main.py --config $(CONFIG) --stage train --cancer $(CANCER) $(EXTRA_ARGS)

evaluate: ## Stage 6: Evaluate and generate plots
	$(PYTHON) main.py --config $(CONFIG) --stage evaluate --cancer $(CANCER) $(EXTRA_ARGS)

clean: ## Remove generated outputs
	rm -rf results/
	rm -f execution.log
	@echo "✓ Cleaned generated outputs"
