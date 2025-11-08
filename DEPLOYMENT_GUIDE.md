# VLA-GR Navigation Framework - Complete Deployment Guide

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Habitat Installation](#habitat-installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Training](#model-training)
6. [Evaluation & Benchmarking](#evaluation--benchmarking)
7. [Deployment](#deployment)
8. [Troubleshooting](#troubleshooting)
9. [Performance Optimization](#performance-optimization)

---

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04 or 22.04 (WSL2 supported for Windows)
- **CPU**: 8-core processor
- **RAM**: 16GB
- **GPU**: NVIDIA GPU with 8GB VRAM (GTX 1080 or better)
- **Storage**: 100GB free space
- **CUDA**: 11.7 or higher

### Recommended Requirements
- **CPU**: 16-core processor
- **RAM**: 32GB
- **GPU**: NVIDIA RTX 3090 or better (24GB VRAM)
- **Storage**: 500GB SSD
- **CUDA**: 12.0+

### Software Prerequisites
```bash
# Check CUDA version
nvidia-smi
nvcc --version

# Check Python version (3.8+ required)
python --version

# Check git
git --version
```

---

## 🔧 Environment Setup

### Step 1: Clone Repository

```bash
# Create workspace
mkdir -p ~/vla_gr_workspace
cd ~/vla_gr_workspace

# Clone repository
git clone https://github.com/your-org/vla-gr-navigation.git
cd vla-gr-navigation
```

### Step 2: Create Conda Environment

```bash
# Install Miniconda if not present
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source ~/miniconda3/bin/activate

# Create environment
conda create -n vla_gr python=3.8 cmake=3.14.0 -y
conda activate vla_gr

# Verify environment
which python  # Should point to conda environment
```

### Step 3: Install PyTorch

```bash
# For CUDA 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

# For CUDA 12.1
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🏠 Habitat Installation

### Step 1: Install Habitat-Sim

```bash
# Install Habitat-Sim with CUDA support
conda install habitat-sim withbullet headless -c conda-forge -c aihabitat

# Alternative: Install from source for latest features
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --bullet --with-cuda

# Verify installation
python -c "import habitat_sim; print(habitat_sim.__version__)"
```

### Step 2: Install Habitat-Lab

```bash
# Clone Habitat-Lab
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab

# Checkout stable version
git checkout v0.2.4

# Install habitat-lab
pip install -e habitat-lab

# Install habitat-baselines
pip install -e habitat-baselines

# Verify
python -c "import habitat; print(habitat.__version__)"
```

### Step 3: Test Habitat Installation

```bash
# Run test script
python -m habitat_sim.utils.datasets_download --uids ci_test_assets --data-path data/

# Run example
python examples/shortest_path_follower_example.py
```

---

## 📊 Dataset Preparation

### Step 1: Download Scene Datasets

#### HM3D (Habitat-Matterport 3D) Dataset

```bash
# Create data directory
mkdir -p data/scene_datasets

# Download HM3D scenes (requires agreement to terms)
# Go to: https://aihabitat.org/datasets/hm3d/
# Sign agreement and get download script

# Download minival split (smaller, for testing)
python -m habitat_sim.utils.datasets_download \
    --uids hm3d_minival \
    --data-path data/

# Download full train/val splits
python scripts/download_hm3d.py \
    --split train \
    --output data/scene_datasets/hm3d/
```

#### MP3D (Matterport3D) Dataset

```bash
# Requires academic agreement
# Apply at: https://niessner.github.io/Matterport/

# After approval, download using provided script
python scripts/download_mp3d.py \
    --task_data habitat \
    --output data/scene_datasets/mp3d/
```

#### Gibson Dataset

```bash
# Download Gibson dataset
python -m habitat_sim.utils.datasets_download \
    --uids gibson_habitat \
    --data-path data/
```

#### Replica Dataset

```bash
# Download Replica scenes
python -m habitat_sim.utils.datasets_download \
    --uids replica_cad_dataset \
    --data-path data/
```

### Step 2: Generate Episode Datasets

```bash
# Generate PointNav episodes
python scripts/generate_episodes.py \
    --dataset hm3d \
    --task pointnav \
    --split train \
    --num_episodes 10000 \
    --output data/datasets/pointnav/hm3d/v1/

# Generate ObjectNav episodes
python scripts/generate_episodes.py \
    --dataset hm3d \
    --task objectnav \
    --split train \
    --num_episodes 10000 \
    --object_categories "chair,table,bed,toilet,couch,sofa,plant" \
    --output data/datasets/objectnav/hm3d/v1/
```

### Step 3: Verify Dataset Structure

```bash
# Expected structure
tree data/ -L 3

data/
├── scene_datasets/
│   ├── hm3d/
│   │   ├── train/
│   │   ├── val/
│   │   └── minival/
│   ├── mp3d/
│   │   └── ...
│   └── gibson/
│       └── ...
└── datasets/
    ├── pointnav/
    │   └── hm3d/
    └── objectnav/
        └── hm3d/
```

---

## 🎯 Model Training

### Step 1: Install VLA-GR Dependencies

```bash
cd ~/vla_gr_workspace/vla-gr-navigation

# Install all dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Verify installation
python -c "from src.core.vla_gr_agent import VLAGRAgent; print('VLA-GR imported successfully')"
```

### Step 2: Configure Training

Edit `config.yaml`:

```yaml
# Key configurations to modify
dataset:
  scene_dataset: "hm3d"  # Your dataset
  data_path: "data/"      # Path to data

training:
  batch_size: 8           # Adjust based on GPU memory
  num_workers: 4          # Parallel data loading
  learning_rate: 5e-5
  max_steps: 100000
  
hardware:
  device: "cuda"
  num_gpus: 1            # Number of GPUs
```

### Step 3: Start Training

#### Single GPU Training

```bash
# Basic training
python src/training/train.py \
    --config config.yaml \
    --experiment_name vla_gr_experiment

# With specific hyperparameters
python src/training/train.py \
    --config config.yaml \
    training.batch_size=16 \
    training.learning_rate=1e-4
```

#### Multi-GPU Training

```bash
# Using DataParallel (single node)
python src/training/train.py \
    --config config.yaml \
    hardware.num_gpus=4

# Using DistributedDataParallel
torchrun --nproc_per_node=4 \
    src/training/train.py \
    --config config.yaml \
    hardware.distributed.enabled=true
```

#### SLURM Cluster Training

```bash
# Create SLURM script
cat > train_vla_gr.sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=vla_gr_train
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --mem=128G

# Setup environment
module load cuda/11.7
source ~/miniconda3/bin/activate vla_gr

# Training command
srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HOSTNAME:29500 \
    src/training/train.py \
        --config config.yaml \
        hardware.distributed.enabled=true
EOF

# Submit job
sbatch train_vla_gr.sbatch
```

### Step 4: Monitor Training

```bash
# TensorBoard
tensorboard --logdir logs/ --port 6006

# Weights & Biases (if configured)
wandb login
# Training will auto-log to W&B

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f logs/vla_gr.log
```

---

## 📈 Evaluation & Benchmarking

### Step 1: Run Standard Evaluation

```bash
# Evaluate on validation set
python src/evaluation/evaluate.py \
    --config config.yaml \
    --checkpoint checkpoints/best.pt \
    --split val \
    --num_episodes 1000

# Evaluate on test set
python src/evaluation/evaluate.py \
    --config config.yaml \
    --checkpoint checkpoints/best.pt \
    --split test \
    --num_episodes 1000
```

### Step 2: Run Ablation Studies

```bash
# Full ablation study suite
python scripts/run_ablations.py \
    --checkpoint checkpoints/best.pt \
    --output results/ablations/

# Specific ablation
python src/evaluation/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --ablation no_gr_field
```

### Step 3: Benchmark Comparison

```bash
# Compare with baselines
python scripts/benchmark_comparison.py \
    --methods vla_gr,ppo,dd_ppo,shortest_path \
    --episodes 1000 \
    --output results/comparison/
```

### Step 4: Generate Results

```bash
# Generate comprehensive results report
python scripts/generate_results.py \
    --eval_dir results/ \
    --output report.html

# Generate LaTeX tables for paper
python scripts/generate_latex_tables.py \
    --results results/comparison/ \
    --output tables/
```

---

## 🚀 Deployment

### Option 1: Python API Deployment

```python
# server.py
from fastapi import FastAPI, File, UploadFile
from src.inference import VLAGRInference

app = FastAPI()
model = VLAGRInference("checkpoints/best.pt")

@app.post("/navigate")
async def navigate(
    rgb_image: UploadFile = File(...),
    depth_map: UploadFile = File(...),
    instruction: str = "Navigate to goal"
):
    action = model.predict(rgb_image, depth_map, instruction)
    return {"action": action.tolist()}

# Run: uvicorn server:app --port 8080
```

### Option 2: ROS2 Deployment

```bash
# Install ROS2 dependencies
sudo apt install ros-humble-desktop

# Build ROS2 package
cd ros2_vla_gr
colcon build --packages-select vla_gr_nav

# Launch navigation node
ros2 launch vla_gr_nav navigation.launch.py
```

### Option 3: Docker Deployment

```dockerfile
# Dockerfile
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.8 python3-pip git wget

# Install Habitat
RUN pip install habitat-sim habitat-lab

# Copy project
COPY . /app
WORKDIR /app

# Install VLA-GR
RUN pip install -r requirements.txt
RUN pip install -e .

# Entry point
CMD ["python", "demo.py", "--model", "checkpoints/best.pt"]
```

```bash
# Build and run
docker build -t vla-gr:latest .
docker run --gpus all -p 8080:8080 vla-gr:latest
```

### Option 4: ONNX Export for Edge Deployment

```bash
# Export to ONNX
python scripts/export_onnx.py \
    --checkpoint checkpoints/best.pt \
    --output models/vla_gr.onnx \
    --opset 13 \
    --optimize

# Quantize for edge devices
python scripts/quantize_model.py \
    --model models/vla_gr.onnx \
    --output models/vla_gr_int8.onnx \
    --calibration_data data/calibration/
```

---

## 🔍 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory

```bash
# Reduce batch size
python src/training/train.py training.batch_size=4

# Enable gradient checkpointing
python src/training/train.py \
    hardware.memory.gradient_checkpointing=true

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 2. Habitat Import Error

```bash
# Reinstall with specific versions
conda install -c aihabitat -c conda-forge habitat-sim=0.2.4 headless

# Set environment variables
export HABITAT_SIM_LOG=Debug
export MAGNUM_LOG=verbose
```

#### 3. Dataset Not Found

```bash
# Verify paths in config
python -c "import yaml; c=yaml.safe_load(open('config.yaml')); print(c['dataset']['data_path'])"

# Create symlinks if needed
ln -s /path/to/actual/data data/scene_datasets
```

#### 4. Slow Training

```bash
# Profile training
python -m torch.utils.bottleneck src/training/train.py

# Increase workers
python src/training/train.py training.num_workers=8

# Enable mixed precision
python src/training/train.py training.mixed_precision=true
```

---

## ⚡ Performance Optimization

### Training Optimization

```python
# config_optimized.yaml
training:
  mixed_precision: true
  gradient_accumulation: 4
  num_workers: 8
  pin_memory: true
  prefetch_factor: 2
  
hardware:
  memory:
    gradient_checkpointing: true
    empty_cache_every: 100
```

### Inference Optimization

```bash
# TorchScript compilation
python scripts/compile_model.py \
    --checkpoint checkpoints/best.pt \
    --output models/vla_gr_scripted.pt

# TensorRT optimization
python scripts/tensorrt_optimize.py \
    --model models/vla_gr.onnx \
    --output models/vla_gr_trt.engine \
    --precision fp16
```

### Profiling

```bash
# PyTorch profiler
python scripts/profile_model.py \
    --checkpoint checkpoints/best.pt \
    --num_iterations 100

# Memory profiling
python -m memory_profiler src/training/train.py
```

---

## 📊 Expected Performance Metrics

After successful deployment, you should achieve:

| Metric | Expected Value | Notes |
|--------|---------------|-------|
| Success Rate | 65-75% | On HM3D val set |
| SPL | 0.55-0.65 | Success weighted by path length |
| Collision Rate | 15-20% | In cluttered environments |
| Inference Time | 3-5ms | On RTX 3090 |
| Training Time | 24-48 hours | 100k steps on 4 GPUs |
| Model Size | <100MB | Compressed checkpoint |

---

## 📚 Additional Resources

- [Habitat Documentation](https://aihabitat.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Paper Implementation](https://arxiv.org/abs/your-paper)
- [Video Demo](https://youtube.com/your-demo)

---

## 💬 Support

- **GitHub Issues**: [Report bugs](https://github.com/your-org/vla-gr/issues)
- **Discord**: [Join community](https://discord.gg/vla-gr)
- **Email**: support@vla-gr.ai

---

**Last Updated**: October 2024
**Version**: 1.0.0
