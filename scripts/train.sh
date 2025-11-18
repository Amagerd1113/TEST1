#!/bin/bash
# Gravitational Slingshot Navigation - Training Script
# Target: IROS 2026 submission with 82%+ SR on VLN-CE val_unseen

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

echo -e "${GREEN}==== Gravitational Slingshot Navigation Training ====${NC}"
echo "Training OpenVLA-7B with dual scalar field (φ+, φ-) for VLN-CE"
echo ""

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. CUDA required for training.${NC}"
    exit 1
fi

echo "CUDA devices available:"
nvidia-smi --list-gpus

# Configuration
CONFIG_PATH="${1:-configs/train_vln_ce.yaml}"
NUM_GPUS="${2:-1}"
PORT="${3:-29500}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Config: ${CONFIG_PATH}"
echo "  GPUs: ${NUM_GPUS}"
echo "  Master port: ${PORT}"
echo ""

# Check if config exists
if [ ! -f "${CONFIG_PATH}" ]; then
    echo -e "${RED}Error: Config file not found: ${CONFIG_PATH}${NC}"
    exit 1
fi

# Set environment variables for optimal performance
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=OFF
export NCCL_DEBUG=INFO

# Verify data directory
echo -e "${YELLOW}Checking data directories...${NC}"
if [ ! -d "data/datasets/vln_ce" ]; then
    echo -e "${YELLOW}Warning: VLN-CE dataset not found at data/datasets/vln_ce${NC}"
    echo "Please download VLN-CE dataset first:"
    echo "  mkdir -p data/datasets && cd data/datasets"
    echo "  wget https://dl.fbaipublicfiles.com/habitat/data/datasets/vln_ce/v1.zip"
    echo "  unzip v1.zip && rm v1.zip"
    echo ""
fi

if [ ! -d "data/scene_datasets/mp3d" ]; then
    echo -e "${YELLOW}Warning: Matterport3D scenes not found at data/scene_datasets/mp3d${NC}"
    echo "Please download Matterport3D dataset first (requires registration)"
    echo ""
fi

# Create experiment directory
EXPERIMENT_DIR="experiments/$(date +%Y%m%d_%H%M%S)_slingshot_vln"
mkdir -p "${EXPERIMENT_DIR}"
echo "Experiment directory: ${EXPERIMENT_DIR}"

# Copy config to experiment directory
cp "${CONFIG_PATH}" "${EXPERIMENT_DIR}/config.yaml"

# Training command
echo -e "${GREEN}Starting training...${NC}"
echo ""

if [ "${NUM_GPUS}" -gt 1 ]; then
    # Multi-GPU training with torchrun
    echo "Using multi-GPU training with ${NUM_GPUS} GPUs"
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${PORT}" \
        src/train.py \
        --config="${CONFIG_PATH}" \
        --experiment_dir="${EXPERIMENT_DIR}"
else
    # Single-GPU training
    echo "Using single-GPU training"
    python src/train.py \
        --config="${CONFIG_PATH}" \
        --experiment_dir="${EXPERIMENT_DIR}"
fi

# Check training success
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo "Checkpoints saved in: ${EXPERIMENT_DIR}/checkpoints"
    echo "Logs available in: ${EXPERIMENT_DIR}/logs"
    echo ""
    echo "To evaluate the model, run:"
    echo "  bash scripts/eval_vln_ce.sh ${EXPERIMENT_DIR}/checkpoints/best_model.pt"
else
    echo -e "${RED}Training failed. Check logs in ${EXPERIMENT_DIR}/logs${NC}"
    exit 1
fi
