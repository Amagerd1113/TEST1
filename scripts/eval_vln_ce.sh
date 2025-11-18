#!/bin/bash
# Gravitational Slingshot Navigation - VLN-CE Evaluation Script
# Evaluate on val_unseen and compute distractor-specific metrics

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}==== VLN-CE Evaluation for Slingshot Navigation ====${NC}"
echo ""

# Arguments
CHECKPOINT_PATH="${1}"
SPLIT="${2:-val_unseen}"
CONFIG_PATH="${3:-configs/train_vln_ce.yaml}"
OUTPUT_DIR="${4:-experiments/eval_$(date +%Y%m%d_%H%M%S)}"

if [ -z "${CHECKPOINT_PATH}" ]; then
    echo -e "${RED}Error: Checkpoint path required${NC}"
    echo "Usage: $0 <checkpoint_path> [split] [config_path] [output_dir]"
    echo "Example: $0 experiments/best_model.pt val_unseen"
    exit 1
fi

if [ ! -f "${CHECKPOINT_PATH}" ]; then
    echo -e "${RED}Error: Checkpoint not found: ${CHECKPOINT_PATH}${NC}"
    exit 1
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Split: ${SPLIT}"
echo "  Config: ${CONFIG_PATH}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run evaluation
echo -e "${GREEN}Running evaluation on ${SPLIT}...${NC}"
python src/evaluate.py \
    --checkpoint="${CHECKPOINT_PATH}" \
    --config="${CONFIG_PATH}" \
    --split="${SPLIT}" \
    --output_dir="${OUTPUT_DIR}" \
    --save_trajectories=true \
    --save_metric_fields=true \
    --compute_distractor_metrics=true

# Check success
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Evaluation completed successfully!${NC}"
    echo ""
    echo "Results saved to: ${OUTPUT_DIR}/results.json"
    echo ""

    # Display key metrics if results.json exists
    if [ -f "${OUTPUT_DIR}/results.json" ]; then
        echo -e "${YELLOW}Key Metrics:${NC}"
        python -c "
import json
with open('${OUTPUT_DIR}/results.json') as f:
    results = json.load(f)
    print(f\"  Success Rate (SR): {results.get('success_rate', 0)*100:.2f}%\")
    print(f\"  SPL: {results.get('spl', 0):.4f}\")
    print(f\"  Oracle Success: {results.get('oracle_success', 0)*100:.2f}%\")
    print(f\"  Path Length: {results.get('path_length', 0):.2f}m\")
    if 'distractor_metrics' in results:
        print(f\"\\n  High-Distractor Subset:\")
        print(f\"    SR: {results['distractor_metrics'].get('success_rate', 0)*100:.2f}%\")
        print(f\"    SPL: {results['distractor_metrics'].get('spl', 0):.4f}\")
" || echo "Could not parse results"
    fi

    echo ""
    echo "Visualizations available in: ${OUTPUT_DIR}/visualizations"
else
    echo -e "${RED}Evaluation failed${NC}"
    exit 1
fi
