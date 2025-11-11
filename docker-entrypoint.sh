#!/bin/bash
set -e

# VLA-GR Docker Entrypoint Script

echo "==================================="
echo "VLA-GR Navigation Framework"
echo "==================================="

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "No GPU detected, running on CPU"
fi

# Set default environment variables if not set
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONPATH=/app:${PYTHONPATH}

# Create necessary directories
mkdir -p /app/checkpoints /app/logs /app/outputs

echo "Environment ready!"
echo "==================================="

# Execute the command passed to the container
exec "$@"
