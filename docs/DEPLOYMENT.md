# VLA-GR Deployment Guide

Complete deployment guide for production, development, and research environments.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation Methods](#installation-methods)
3. [Docker Deployment](#docker-deployment)
4. [Hardware Requirements](#hardware-requirements)
5. [Configuration](#configuration)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Installation

```bash
# Clone repository
git clone https://github.com/your-org/vla-gr-navigation.git
cd vla-gr-navigation

# Create environment
conda create -n vla_gr python=3.8
conda activate vla_gr

# Install dependencies
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install -e .

# Verify installation
python scripts/verify_installation.py
```

### Quick Test

```bash
# Run demo
python demo.py --config config.yaml --no-viz

# Run evaluation
python scripts/run_evaluation.py --checkpoint checkpoints/best.pt
```

---

## Installation Methods

### Method 1: Conda (Recommended)

```bash
# Create environment
conda create -n vla_gr python=3.8
conda activate vla_gr

# Install PyTorch with CUDA
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu117

# Install Habitat-Sim
conda install habitat-sim -c conda-forge -c aihabitat

# Install other dependencies
pip install -r requirements.txt

# Install VLA-GR
pip install -e .
```

### Method 2: Docker (Production)

```bash
# Build Docker image
docker build -t vla-gr:latest .

# Run container
docker run --gpus all -it \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/data:/app/data \
    vla-gr:latest
```

### Method 3: Docker Compose (Development)

```bash
# Start development environment
docker-compose up vla-gr-dev

# Access container
docker exec -it vla-gr-dev bash
```

---

## Docker Deployment

### Building Images

```bash
# Production image
docker build -t vla-gr:prod .

# Development image with additional tools
docker build --target dev -t vla-gr:dev .
```

### Running Containers

#### Basic Run

```bash
docker run --gpus all -it \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/data:/app/data \
    -p 8000:8000 \
    -p 6006:6006 \
    vla-gr:latest
```

#### With Docker Compose

```yaml
# docker-compose.yml
version: '3.8'
services:
  vla-gr:
    image: vla-gr:latest
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./checkpoints:/app/checkpoints
      - ./logs:/app/logs
      - ./data:/app/data
    ports:
      - "8000:8000"
      - "6006:6006"
    shm_size: '8gb'
```

```bash
# Start
docker-compose up -d

# Stop
docker-compose down
```

### GPU Support

Ensure NVIDIA Docker runtime is installed:

```bash
# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu22.04 nvidia-smi

# If not working, install:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

---

## Hardware Requirements

### Minimum Requirements

| Component | Specification |
|-----------|--------------|
| CPU | 4 cores, 2.5 GHz |
| RAM | 16 GB |
| GPU | NVIDIA GPU with 6GB VRAM (e.g., GTX 1060) |
| Storage | 50 GB SSD |
| OS | Ubuntu 20.04+ / Windows 10+ / macOS 11+ |

### Recommended Requirements

| Component | Specification |
|-----------|--------------|
| CPU | 8+ cores, 3.0+ GHz |
| RAM | 32 GB+ |
| GPU | NVIDIA RTX 3090 / 4090 (24GB+ VRAM) |
| Storage | 500 GB NVMe SSD |
| OS | Ubuntu 22.04 LTS |

### Tested Configurations

#### RTX 4060 (16GB VRAM)

```bash
# Use optimized config
python demo.py --config config_rtx4060.yaml
```

#### Server (Multiple GPUs)

```bash
# Use server config with DDP
torchrun --nproc_per_node=4 src/training/train.py \
    --config config_server.yaml \
    hardware.distributed.enabled=true
```

---

## Configuration

### Environment Variables

```bash
# GPU selection
export CUDA_VISIBLE_DEVICES=0

# Performance tuning
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Configuration Files

#### Main Config (`config.yaml`)

```yaml
model:
  vla:
    hidden_dim: 768
    num_layers: 12
    num_heads: 8
  gr_field:
    grid_size: [64, 64, 32]
    lambda_curvature: 0.1

training:
  batch_size: 32
  learning_rate: 5e-5
  max_steps: 100000

hardware:
  device: "cuda"
  mixed_precision: true
  num_workers: 4
```

#### GPU-Specific Configs

**RTX 4060** (`config_rtx4060.yaml`):
- Smaller batch size: 16
- Gradient accumulation: 2
- Mixed precision: FP16

**Server** (`config_server.yaml`):
- Larger batch size: 64
- Multi-GPU support
- Distributed training

---

## Production Deployment

### Step 1: Prepare Environment

```bash
# Set up production server
ssh user@production-server

# Install Docker and NVIDIA runtime
sudo apt-get update
sudo apt-get install -y docker.io nvidia-docker2

# Pull VLA-GR image
docker pull ghcr.io/your-org/vla-gr:latest
```

### Step 2: Download Models and Data

```bash
# Download pre-trained checkpoints
./scripts/download_models.sh

# Download Habitat scenes
./scripts/download_datasets.sh
```

### Step 3: Configure for Production

```yaml
# production.yaml
model:
  checkpoint: "checkpoints/best.pt"

inference:
  batch_size: 1
  deterministic: true

logging:
  level: "INFO"
  save_outputs: true
```

### Step 4: Deploy with Docker

```bash
# Run production container
docker run -d \
    --name vla-gr-prod \
    --gpus all \
    --restart unless-stopped \
    -v /data/checkpoints:/app/checkpoints:ro \
    -v /data/logs:/app/logs \
    -p 8000:8000 \
    -e DEPLOYMENT_ENV=production \
    vla-gr:latest \
    python -m src.deployment.serve --config production.yaml
```

### Step 5: Health Monitoring

```bash
# Check container status
docker ps | grep vla-gr-prod

# View logs
docker logs -f vla-gr-prod

# Check GPU usage
nvidia-smi -l 1
```

---

## ONNX Deployment

### Export to ONNX

```python
from src.deployment.export import export_to_onnx

export_to_onnx(
    model_path="checkpoints/best.pt",
    output_path="models/vla_gr.onnx",
    optimize=True,
    opset_version=14
)
```

### Run ONNX Inference

```python
import onnxruntime as ort

# Create session
session = ort.InferenceSession(
    "models/vla_gr.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# Run inference
outputs = session.run(None, {
    'rgb': rgb_array,
    'depth': depth_array,
    'language': language_tokens
})
```

### Performance

- **Speedup**: 1.5-2x faster than PyTorch
- **Memory**: 30-40% less memory
- **Compatibility**: Works on edge devices

---

## ROS2 Integration

### Install ROS2

```bash
# Add ROS2 repository
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y

# Install ROS2 Humble
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop
```

### VLA-GR ROS2 Node

```python
# vla_gr_ros_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Twist
from vla_gr import VLAGRAgent

class VLAGRNode(Node):
    def __init__(self):
        super().__init__('vla_gr_node')
        self.agent = VLAGRAgent.from_pretrained("checkpoints/best.pt")

        # Subscribers
        self.rgb_sub = self.create_subscription(Image, '/camera/rgb', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth', self.depth_callback, 10)

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
```

### Launch File

```xml
<!-- vla_gr.launch.py -->
<launch>
  <node pkg="vla_gr_nav" exec="vla_gr_node" name="vla_gr">
    <param name="checkpoint" value="checkpoints/best.pt"/>
    <param name="config" value="config.yaml"/>
  </node>
</launch>
```

```bash
# Launch
ros2 launch vla_gr_nav vla_gr.launch.py
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
--training.batch_size 8

# Enable gradient checkpointing
--model.gradient_checkpointing true

# Use smaller model
--model.vla.hidden_dim 512
```

#### 2. Habitat Installation Issues

**Symptoms**: `ImportError: cannot import name 'Simulator'`

**Solutions**:
```bash
# Reinstall Habitat
conda uninstall habitat-sim habitat-lab
conda install habitat-sim=0.3.2 habitat-lab=0.3.3 -c conda-forge -c aihabitat

# Or build from source
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install
```

#### 3. Transformers Model Download Fails

**Symptoms**: Connection timeout when downloading Phi-2

**Solutions**:
```bash
# Use mirror
export HF_ENDPOINT=https://hf-mirror.com

# Or download manually
mkdir -p ~/.cache/huggingface/hub
cd ~/.cache/huggingface/hub
git clone https://huggingface.co/microsoft/phi-2
```

#### 4. Docker GPU Not Accessible

**Symptoms**: `nvidia-smi` not working in container

**Solutions**:
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu22.04 nvidia-smi

# Reinstall nvidia-docker2
sudo apt-get purge nvidia-docker2
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

---

## Performance Optimization

### GPU Optimization

```yaml
# Use mixed precision
hardware:
  mixed_precision: true

# Enable TF32
hardware:
  allow_tf32: true

# Optimize CUDA allocator
hardware:
  cuda_malloc_async: true
```

### CPU Optimization

```bash
# Set thread count
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Use optimized BLAS
pip install intel-extension-for-pytorch
```

### Memory Optimization

```yaml
# Enable gradient checkpointing
model:
  gradient_checkpointing: true

# Reduce cache size
training:
  dataloader_num_workers: 2
  prefetch_factor: 2
```

---

## Monitoring and Logging

### TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/ --port 6006

# Access at http://localhost:6006
```

### Weights & Biases

```python
# Enable in config
logging:
  wandb:
    enabled: true
    project: "vla-gr"
    entity: "your-team"
```

### Prometheus Metrics

```python
# Export metrics
from prometheus_client import start_http_server, Counter, Gauge

inference_counter = Counter('vla_gr_inferences_total', 'Total inferences')
inference_time = Gauge('vla_gr_inference_seconds', 'Inference time')

# Start metrics server
start_http_server(9090)
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] All tests passing (`make test`)
- [ ] Code formatted and linted (`make ci`)
- [ ] Dependencies verified (`pip check`)
- [ ] Models downloaded
- [ ] Configuration validated
- [ ] GPU access confirmed
- [ ] Disk space sufficient (>50GB)

### Deployment

- [ ] Container running
- [ ] Health check passing
- [ ] Logs accessible
- [ ] Metrics being collected
- [ ] Backups configured

### Post-Deployment

- [ ] Monitor GPU utilization
- [ ] Check inference latency
- [ ] Verify success rates
- [ ] Review error logs
- [ ] Test rollback procedure

---

## Security Considerations

### Container Security

```dockerfile
# Run as non-root user
USER app

# Read-only filesystem where possible
volumes:
  - ./checkpoints:/app/checkpoints:ro
```

### Network Security

```yaml
# Restrict network access
networks:
  vla-gr-net:
    internal: true
```

### Secrets Management

```bash
# Use environment files
docker run --env-file .env.secret vla-gr:latest

# Or Docker secrets
echo "api_key_value" | docker secret create api_key -
```

---

## Scaling

### Horizontal Scaling

```yaml
# docker-compose.yml
services:
  vla-gr:
    deploy:
      replicas: 4
    environment:
      - CUDA_VISIBLE_DEVICES=${GPU_ID}
```

### Load Balancing

```nginx
# nginx.conf
upstream vla_gr_backend {
    least_conn;
    server vla-gr-1:8000;
    server vla-gr-2:8000;
    server vla-gr-3:8000;
    server vla-gr-4:8000;
}
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vla-gr-deployment
spec:
  replicas: 4
  selector:
    matchLabels:
      app: vla-gr
  template:
    metadata:
      labels:
        app: vla-gr
    spec:
      containers:
      - name: vla-gr
        image: vla-gr:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

---

## Support and Resources

- **Documentation**: https://vla-gr.readthedocs.io
- **Issues**: https://github.com/your-org/vla-gr-navigation/issues
- **Discussions**: https://github.com/your-org/vla-gr-navigation/discussions

---

**Last Updated**: November 11, 2025
**Document Version**: 1.0
