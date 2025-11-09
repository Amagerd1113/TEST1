# VLA-GR Habitat 0.3.3 详细部署 Checklist 和教程

> **版本**: 1.0.0
> **适用环境**: Linux (Ubuntu 20.04/22.04), CUDA 11.8+
> **最后更新**: 2025-11-09

---

## 📋 部署 Checklist

### 阶段一：环境准备 ✅

- [ ] **1.1** 检查系统要求（GPU、内存、存储）
- [ ] **1.2** 安装 CUDA 11.8+ 和 cuDNN
- [ ] **1.3** 安装 Python 3.9-3.11
- [ ] **1.4** 创建虚拟环境
- [ ] **1.5** 配置环境变量

### 阶段二：基础依赖安装 ✅

- [ ] **2.1** 安装 PyTorch 2.0+（CUDA版本）
- [ ] **2.2** 安装 Habitat-Sim 0.3.3
- [ ] **2.3** 安装 Habitat-Lab 0.3.3
- [ ] **2.4** 验证 Habitat 安装
- [ ] **2.5** 安装项目依赖（requirements.txt）

### 阶段三：数据集下载和配置 ✅

- [ ] **3.1** 创建数据目录结构
- [ ] **3.2** 下载 Habitat 测试场景（Replica）
- [ ] **3.3** 下载 HM3D 数据集（可选，训练用）
- [ ] **3.4** 下载 ObjectNav 任务数据
- [ ] **3.5** 配置数据集路径
- [ ] **3.6** 验证数据集完整性

### 阶段四：Hugging Face 模型部署 ✅

- [ ] **4.1** 配置 Hugging Face 访问（token）
- [ ] **4.2** 下载 Microsoft Phi-2 模型
- [ ] **4.3** 下载 CLIP 模型
- [ ] **4.4** 下载 BERT 模型（后备）
- [ ] **4.5** 配置模型缓存路径
- [ ] **4.6** 验证模型加载

### 阶段五：项目安装和配置 ✅

- [ ] **5.1** 克隆项目代码
- [ ] **5.2** 安装项目（editable mode）
- [ ] **5.3** 配置 config.yaml
- [ ] **5.4** 设置 Weights & Biases（可选）
- [ ] **5.5** 创建输出目录

### 阶段六：验证和测试 ✅

- [ ] **6.1** 运行导入测试
- [ ] **6.2** 运行 Habitat 环境测试
- [ ] **6.3** 运行模型加载测试
- [ ] **6.4** 运行简单推理测试
- [ ] **6.5** 检查 GPU 内存使用

### 阶段七：训练和评估 ✅

- [ ] **7.1** 准备训练配置
- [ ] **7.2** 运行小规模训练测试
- [ ] **7.3** 启动完整训练
- [ ] **7.4** 监控训练进度
- [ ] **7.5** 运行评估

---

## 🖥️ 系统要求

### 最低配置（测试/开发）

```yaml
硬件要求:
  GPU: NVIDIA RTX 3060 (12GB) 或更高
  CPU: 8核心以上
  内存: 32GB RAM
  存储: 100GB 可用空间（基础环境 + Replica场景）

软件要求:
  操作系统: Ubuntu 20.04/22.04 LTS
  CUDA: 11.8 或 12.1
  cuDNN: 8.x
  Python: 3.9, 3.10, 或 3.11
  GCC: 7.x 或更高（编译Habitat-Sim需要）
```

### 推荐配置（训练）

```yaml
RTX 4060 配置（8GB显存）:
  GPU: NVIDIA RTX 4060
  内存: 32GB RAM
  存储: 500GB SSD（包含部分训练数据）
  预期训练时间: 48-72小时
  预期成功率: ~80%

服务器配置（生产级）:
  GPU: 4x NVIDIA A100 (80GB) 或 H100
  内存: 256GB+ RAM
  存储: 4TB NVMe SSD（完整数据集）
  预期训练时间: 18-24小时
  预期成功率: 85-90%
```

---

## 📦 阶段一：环境准备

### 1.1 检查系统信息

```bash
# 检查操作系统
cat /etc/os-release

# 检查 GPU
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查可用存储
df -h

# 检查内存
free -h
```

### 1.2 安装 CUDA 和 cuDNN

如果还没有安装 CUDA：

```bash
# Ubuntu 22.04 安装 CUDA 12.1（推荐）
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1

# 或者安装 CUDA 11.8（兼容性更好）
sudo apt-get -y install cuda-toolkit-11-8

# 安装 cuDNN
# 从 NVIDIA 官网下载对应版本：https://developer.nvidia.com/cudnn
# 然后安装：
sudo dpkg -i libcudnn8_*.deb
sudo dpkg -i libcudnn8-dev_*.deb
```

### 1.3 安装 Python 3.9-3.11

```bash
# Ubuntu 22.04 自带 Python 3.10
python3 --version

# 如果需要安装其他版本：
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10 python3.10-dev python3.10-venv

# 安装 pip
sudo apt install python3-pip
```

### 1.4 创建虚拟环境

```bash
# 创建项目目录
mkdir -p ~/vla-gr-workspace
cd ~/vla-gr-workspace

# 创建虚拟环境
python3.10 -m venv vla-gr-env

# 激活虚拟环境
source vla-gr-env/bin/activate

# 升级 pip
pip install --upgrade pip setuptools wheel
```

### 1.5 配置环境变量

```bash
# 编辑 ~/.bashrc
nano ~/.bashrc

# 添加以下内容（根据你的 CUDA 版本调整）：
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Habitat 数据路径（稍后会创建）
export HABITAT_DATA_DIR=~/vla-gr-workspace/habitat-data
export HABITAT_SCENE_DATASETS_DIR=$HABITAT_DATA_DIR/scene_datasets

# Hugging Face 缓存路径
export HF_HOME=~/vla-gr-workspace/huggingface-cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# 应用更改
source ~/.bashrc
```

---

## 🔧 阶段二：基础依赖安装

### 2.1 安装 PyTorch 2.0+

```bash
# 激活虚拟环境
source ~/vla-gr-workspace/vla-gr-env/bin/activate

# 安装 PyTorch 2.1.0 with CUDA 12.1（推荐）
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 或者使用 CUDA 11.8
# pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

**预期输出**:
```
PyTorch: 2.1.0+cu121
CUDA Available: True
CUDA Version: 12.1
```

### 2.2 安装 Habitat-Sim 0.3.3

这是最关键的一步！Habitat-Sim 需要从源码编译。

#### 安装依赖

```bash
# 安装编译工具
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    wget \
    unzip \
    libjpeg-dev \
    libpng-dev \
    ninja-build
```

#### 从源码安装 Habitat-Sim

```bash
# 创建构建目录
mkdir -p ~/vla-gr-workspace/habitat-build
cd ~/vla-gr-workspace/habitat-build

# 克隆 Habitat-Sim 仓库（使用 0.3.3 分支）
git clone --branch v0.3.3 https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim

# 安装 Python 依赖
pip install -r requirements.txt

# 构建和安装（这一步会花费 15-30 分钟）
# 使用 --headless 模式（服务器环境）
python setup.py install --headless --with-cuda

# 或者使用 --bullet（如果需要物理引擎）
# python setup.py install --headless --with-cuda --with-bullet
```

**编译选项说明**:
- `--headless`: 无头模式（无需X11显示）
- `--with-cuda`: 启用 CUDA 加速
- `--with-bullet`: 启用 Bullet 物理引擎（可选）

#### 验证安装

```bash
python -c "import habitat_sim; print(f'Habitat-Sim version: {habitat_sim.__version__}')"
```

**预期输出**: `Habitat-Sim version: 0.3.3`

### 2.3 安装 Habitat-Lab 0.3.3

```bash
# 返回构建目录
cd ~/vla-gr-workspace/habitat-build

# 克隆 Habitat-Lab 仓库
git clone --branch v0.3.3 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab

# 安装
pip install -e habitat-lab
pip install -e habitat-baselines
```

### 2.4 验证 Habitat 安装

创建测试脚本 `test_habitat.py`:

```python
import habitat
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat_sim import make_sim

print(f"Habitat version: {habitat.__version__}")

# 创建基础配置
with read_write(habitat.get_config()):
    config = habitat.get_config()
    print("✓ Config created successfully")

print("✓ Habitat installation verified!")
```

运行测试：

```bash
python test_habitat.py
```

### 2.5 安装项目依赖

```bash
# 返回到项目目录（稍后会克隆）
# 这里先安装核心依赖

# 安装 Transformers 和相关库
pip install transformers>=4.30.0 tokenizers>=0.13.0 accelerate>=0.20.0

# 安装 3D 处理库
pip install open3d>=0.17.0 trimesh>=3.21.0

# 安装优化库
pip install cvxpy>=1.3.0 osqp>=0.6.2 sympy>=1.11.0

# 安装可视化库
pip install matplotlib>=3.7.0 wandb>=0.15.0 tensorboard>=2.12.0

# 安装其他依赖
pip install \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    scikit-learn>=1.2.0 \
    pillow>=9.5.0 \
    opencv-python>=4.7.0 \
    pyyaml>=6.0 \
    tqdm>=4.65.0 \
    hydra-core>=1.3.0 \
    omegaconf>=2.3.0

# 安装部署相关
pip install onnx>=1.14.0 onnxruntime-gpu>=1.14.0 fastapi>=0.95.0 uvicorn>=0.21.0

# 安装 PyBullet（物理模拟）
pip install pybullet>=3.2.5
```

---

## 📁 阶段三：数据集下载和配置

### 3.1 创建数据目录结构

```bash
# 创建主数据目录
mkdir -p $HABITAT_DATA_DIR
cd $HABITAT_DATA_DIR

# 创建子目录
mkdir -p scene_datasets
mkdir -p datasets/objectnav/hm3d/v1
mkdir -p datasets/pointnav/habitat_test_scenes
mkdir -p objects
mkdir -p episodes

# 目录结构：
# habitat-data/
# ├── scene_datasets/          # 3D 场景数据
# │   ├── hm3d/               # HM3D 数据集
# │   ├── mp3d/               # Matterport3D
# │   └── replica/            # Replica（测试用）
# ├── datasets/               # 任务数据（episodes）
# │   ├── objectnav/
# │   └── pointnav/
# └── objects/                # 对象模型
```

### 3.2 下载 Habitat 测试场景（Replica - 必需）

**Replica 是小型高质量场景，用于测试和开发**：

```bash
cd $HABITAT_DATA_DIR/scene_datasets

# 下载 Replica 数据集（~2GB）
# 方法1：使用官方脚本
python -m habitat_sim.utils.datasets_download \
    --username <你的用户名> \
    --password <你的密码> \
    --uids replica_cad_dataset

# 方法2：手动下载
# 访问：https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md
# 下载 replica_v1.zip
wget https://dl.fbaipublicfiles.com/habitat/replica_cad_dataset.zip

# 解压
unzip replica_cad_dataset.zip -d replica/
rm replica_cad_dataset.zip

# 验证（应该包含 18 个场景）
ls replica/
# 预期输出：apartment_0, frl_apartment_0, hotel_0, office_0, room_0, 等
```

### 3.3 下载 HM3D 数据集（可选，训练用）

**⚠️ HM3D 非常大（~2.5TB），仅在训练时需要**：

```bash
# HM3D 需要申请访问权限
# 1. 访问：https://aihabitat.org/datasets/hm3d/
# 2. 填写申请表格（学术用途）
# 3. 获得下载链接

# 下载 HM3D minival（较小，用于快速测试，~10GB）
cd $HABITAT_DATA_DIR/scene_datasets
mkdir -p hm3d

# 使用提供的下载脚本（需要替换你的 token）
python -m habitat_sim.utils.datasets_download \
    --username <你的用户名> \
    --password <你的密码> \
    --uids hm3d_minival

# 或者下载完整训练集（~2.5TB，需要大量时间和存储）
# python -m habitat_sim.utils.datasets_download \
#     --username <用户名> \
#     --password <密码> \
#     --uids hm3d_train_v0.2

# 验证
ls hm3d/
```

### 3.4 下载 ObjectNav 任务数据

```bash
cd $HABITAT_DATA_DIR/datasets

# 下载 ObjectNav episodes（HM3D版本）
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip

# 解压
unzip objectnav_hm3d_v1.zip -d objectnav/hm3d/v1/
rm objectnav_hm3d_v1.zip

# 验证
ls objectnav/hm3d/v1/
# 预期：train/, val/, val_mini/ 等目录
```

### 3.5 配置数据集路径

创建 Habitat 配置文件 `~/.habitat/habitat.yaml`:

```bash
mkdir -p ~/.habitat
cat > ~/.habitat/habitat.yaml << 'EOF'
# Habitat 数据路径配置
data_path: ~/vla-gr-workspace/habitat-data

# 场景数据集路径
scene_datasets:
  hm3d: ~/vla-gr-workspace/habitat-data/scene_datasets/hm3d
  mp3d: ~/vla-gr-workspace/habitat-data/scene_datasets/mp3d
  replica: ~/vla-gr-workspace/habitat-data/scene_datasets/replica

# 任务数据集路径
datasets:
  objectnav:
    hm3d: ~/vla-gr-workspace/habitat-data/datasets/objectnav/hm3d/v1
  pointnav:
    gibson: ~/vla-gr-workspace/habitat-data/datasets/pointnav/gibson/v1
    mp3d: ~/vla-gr-workspace/habitat-data/datasets/pointnav/mp3d/v1
EOF
```

### 3.6 验证数据集完整性

创建验证脚本 `test_datasets.py`:

```python
import os
import habitat
from pathlib import Path

DATA_DIR = Path(os.environ['HABITAT_DATA_DIR'])

print("🔍 检查数据集...")

# 检查场景数据集
replica_path = DATA_DIR / "scene_datasets" / "replica"
if replica_path.exists():
    scenes = list(replica_path.glob("*.glb"))
    print(f"✓ Replica: {len(scenes)} 个场景")
else:
    print("✗ Replica 未找到")

hm3d_path = DATA_DIR / "scene_datasets" / "hm3d"
if hm3d_path.exists():
    scenes = list(hm3d_path.glob("*/*.glb"))
    print(f"✓ HM3D: {len(scenes)} 个场景")
else:
    print("⚠ HM3D 未找到（可选）")

# 检查任务数据
objectnav_path = DATA_DIR / "datasets" / "objectnav" / "hm3d" / "v1"
if objectnav_path.exists():
    print(f"✓ ObjectNav 数据集存在")
else:
    print("✗ ObjectNav 数据集未找到")

print("\n✅ 数据集验证完成")
```

运行：

```bash
python test_datasets.py
```

---

## 🤗 阶段四：Hugging Face 模型部署

### 4.1 配置 Hugging Face 访问

```bash
# 安装 Hugging Face CLI
pip install huggingface-hub

# 登录（可选，用于私有模型）
huggingface-cli login
# 输入你的 token（从 https://huggingface.co/settings/tokens 获取）

# 设置缓存目录
export HF_HOME=~/vla-gr-workspace/huggingface-cache
mkdir -p $HF_HOME
```

### 4.2 下载 Microsoft Phi-2 模型

**Phi-2 是主要的语言模型（2.7B 参数）**：

```bash
# 方法1：使用 Python 预下载
python << 'EOF'
from transformers import AutoModel, AutoTokenizer

print("📥 下载 Phi-2 模型...")
model_name = "microsoft/phi-2"

# 下载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir="~/vla-gr-workspace/huggingface-cache"
)
print(f"✓ Tokenizer 下载完成")

# 下载模型
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir="~/vla-gr-workspace/huggingface-cache"
)
print(f"✓ Phi-2 模型下载完成")
print(f"   参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
EOF
```

**模型文件位置**：`~/vla-gr-workspace/huggingface-cache/models--microsoft--phi-2/`

**文件大小**：约 5.5GB

### 4.3 下载 CLIP 模型

**CLIP 用于视觉-语言对齐**：

```bash
python << 'EOF'
from transformers import CLIPModel, CLIPProcessor

print("📥 下载 CLIP 模型...")
model_name = "openai/clip-vit-base-patch32"

processor = CLIPProcessor.from_pretrained(
    model_name,
    cache_dir="~/vla-gr-workspace/huggingface-cache"
)
print("✓ CLIP Processor 下载完成")

model = CLIPModel.from_pretrained(
    model_name,
    cache_dir="~/vla-gr-workspace/huggingface-cache"
)
print("✓ CLIP 模型下载完成")
EOF
```

**模型文件位置**：`~/vla-gr-workspace/huggingface-cache/models--openai--clip-vit-base-patch32/`

**文件大小**：约 600MB

### 4.4 下载 BERT 模型（后备）

```bash
python << 'EOF'
from transformers import BertModel, BertTokenizer

print("📥 下载 BERT 模型...")
model_name = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(
    model_name,
    cache_dir="~/vla-gr-workspace/huggingface-cache"
)
model = BertModel.from_pretrained(
    model_name,
    cache_dir="~/vla-gr-workspace/huggingface-cache"
)
print("✓ BERT 模型下载完成")
EOF
```

### 4.5 配置模型缓存路径

在项目的 `config.yaml` 中（稍后会创建），确保设置：

```yaml
model:
  language:
    model: "microsoft/phi-2"
    cache_dir: "~/vla-gr-workspace/huggingface-cache"
    local_files_only: false  # 首次下载设为 false，之后可以设为 true
```

### 4.6 验证模型加载

创建测试脚本 `test_models.py`:

```python
import torch
from transformers import AutoModel, AutoTokenizer, CLIPModel, BertModel

print("🧪 测试模型加载...")

# 测试 Phi-2
print("\n1. Phi-2...")
phi_tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True,
    cache_dir="~/vla-gr-workspace/huggingface-cache"
)
phi_model = AutoModel.from_pretrained(
    "microsoft/phi-2",
    trust_remote_code=True,
    cache_dir="~/vla-gr-workspace/huggingface-cache"
)
print(f"   ✓ Phi-2 加载成功 ({sum(p.numel() for p in phi_model.parameters()) / 1e9:.2f}B 参数)")

# 测试 CLIP
print("\n2. CLIP...")
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    cache_dir="~/vla-gr-workspace/huggingface-cache"
)
print(f"   ✓ CLIP 加载成功")

# 测试 BERT
print("\n3. BERT...")
bert_model = BertModel.from_pretrained(
    "bert-base-uncased",
    cache_dir="~/vla-gr-workspace/huggingface-cache"
)
print(f"   ✓ BERT 加载成功")

# GPU 测试
if torch.cuda.is_available():
    print("\n4. GPU 测试...")
    device = torch.device("cuda")
    bert_model.to(device)
    print(f"   ✓ 模型成功加载到 GPU")
    print(f"   GPU 内存使用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("\n✅ 所有模型验证通过！")
```

运行：

```bash
python test_models.py
```

---

## 🚀 阶段五：项目安装和配置

### 5.1 克隆项目代码

```bash
cd ~/vla-gr-workspace

# 克隆你的 VLA-GR 仓库
git clone https://github.com/Amagerd1113/VLA-GR.git
cd VLA-GR

# 检查分支
git branch -a
git checkout claude/habitat-deployment-guide-011CUwwV7U9zmVCFZwVjTzCu
```

### 5.2 安装项目（editable mode）

```bash
# 确保在虚拟环境中
source ~/vla-gr-workspace/vla-gr-env/bin/activate

# 安装项目依赖
pip install -r requirements.txt

# 以可编辑模式安装项目
pip install -e .

# 验证安装
vla-gr-train --help
vla-gr-evaluate --help
```

### 5.3 配置 config.yaml

项目已包含多个配置文件：
- `config.yaml`: 基础配置
- `config_rtx4060.yaml`: RTX 4060 优化配置（8GB 显存）
- `config_server.yaml`: 服务器配置（多GPU）

根据你的硬件选择合适的配置：

**对于 RTX 4060 (8GB 显存)**：

```bash
# 复制并编辑配置
cp config_rtx4060.yaml config_active.yaml
nano config_active.yaml
```

**修改关键路径**：

```yaml
# 数据路径
data:
  habitat_data_dir: ~/vla-gr-workspace/habitat-data
  scene_dataset: "replica"  # 或 "hm3d" （如果下载了）
  episodes_dir: ~/vla-gr-workspace/habitat-data/datasets/objectnav/hm3d/v1

# 模型缓存
model:
  language:
    model: "microsoft/phi-2"
    cache_dir: ~/vla-gr-workspace/huggingface-cache

# 输出目录
training:
  output_dir: ~/vla-gr-workspace/outputs
  checkpoint_dir: ~/vla-gr-workspace/checkpoints
  log_dir: ~/vla-gr-workspace/logs
```

**对于服务器（多GPU）**：

```bash
cp config_server.yaml config_active.yaml
# 同样修改路径
```

### 5.4 设置 Weights & Biases（可选）

如果要使用 W&B 进行训练监控：

```bash
# 安装（已在 requirements.txt 中）
pip install wandb

# 登录
wandb login
# 输入你的 API key（从 https://wandb.ai/authorize 获取）

# 在 config_active.yaml 中启用
# wandb:
#   enabled: true
#   project: "vla-gr-navigation"
#   entity: "your-username"
```

### 5.5 创建输出目录

```bash
# 创建所有必需的输出目录
mkdir -p ~/vla-gr-workspace/outputs
mkdir -p ~/vla-gr-workspace/checkpoints
mkdir -p ~/vla-gr-workspace/logs
mkdir -p ~/vla-gr-workspace/visualizations
mkdir -p ~/vla-gr-workspace/exports

# 链接到项目目录（可选）
ln -s ~/vla-gr-workspace/outputs ~/vla-gr-workspace/VLA-GR/outputs
ln -s ~/vla-gr-workspace/checkpoints ~/vla-gr-workspace/VLA-GR/checkpoints
ln -s ~/vla-gr-workspace/logs ~/vla-gr-workspace/VLA-GR/logs
```

---

## ✅ 阶段六：验证和测试

### 6.1 运行导入测试

创建 `test_imports.py`:

```python
"""测试所有关键导入"""
import sys

def test_imports():
    tests = []

    # 基础库
    try:
        import torch
        tests.append(("PyTorch", torch.__version__, True))
    except Exception as e:
        tests.append(("PyTorch", str(e), False))

    # Habitat
    try:
        import habitat
        import habitat_sim
        tests.append(("Habitat-Sim", habitat_sim.__version__, True))
        tests.append(("Habitat-Lab", habitat.__version__, True))
    except Exception as e:
        tests.append(("Habitat", str(e), False))

    # Transformers
    try:
        import transformers
        tests.append(("Transformers", transformers.__version__, True))
    except Exception as e:
        tests.append(("Transformers", str(e), False))

    # 项目模块
    try:
        from src.core.vla_gr_agent import ConferenceVLAGRAgent
        tests.append(("VLA-GR Agent", "OK", True))
    except Exception as e:
        tests.append(("VLA-GR Agent", str(e), False))

    try:
        from src.environments.habitat_env_v3 import HabitatNavigationEnv
        tests.append(("Habitat Env V3", "OK", True))
    except Exception as e:
        tests.append(("Habitat Env V3", str(e), False))

    try:
        from src.training.train import TrainingPipeline
        tests.append(("Training Pipeline", "OK", True))
    except Exception as e:
        tests.append(("Training Pipeline", str(e), False))

    # 打印结果
    print("\n" + "="*60)
    print("导入测试结果")
    print("="*60)

    all_passed = True
    for name, version, passed in tests:
        status = "✓" if passed else "✗"
        print(f"{status} {name:20s} {version}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("✅ 所有导入测试通过！")
        return 0
    else:
        print("❌ 部分导入测试失败，请检查安装")
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())
```

运行：

```bash
cd ~/vla-gr-workspace/VLA-GR
python test_imports.py
```

### 6.2 运行 Habitat 环境测试

创建 `test_habitat_env.py`:

```python
"""测试 Habitat 环境"""
import os
os.environ['HABITAT_DATA_DIR'] = os.path.expanduser('~/vla-gr-workspace/habitat-data')

from src.environments.habitat_env_v3 import HabitatNavigationEnv
import numpy as np

print("🧪 测试 Habitat 环境...")

# 创建环境
env = HabitatNavigationEnv(
    scene_id="replica/apartment_0.glb",
    task_type="objectnav",
    max_episode_steps=100
)

print(f"✓ 环境创建成功")
print(f"  观察空间: {env.observation_space}")
print(f"  动作空间: {env.action_space}")

# 重置环境
obs, info = env.reset()
print(f"✓ 环境重置成功")
print(f"  RGB shape: {obs['rgb'].shape}")
print(f"  Depth shape: {obs['depth'].shape}")
print(f"  任务指令: {obs.get('instruction', 'N/A')}")

# 运行几步
for i in range(5):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    print(f"  Step {i+1}: reward={reward:.3f}, done={done}")
    if done:
        break

env.close()
print("✅ Habitat 环境测试通过！")
```

运行：

```bash
python test_habitat_env.py
```

### 6.3 运行模型加载测试

创建 `test_agent.py`:

```python
"""测试 VLA-GR Agent"""
import os
os.environ['HABITAT_DATA_DIR'] = os.path.expanduser('~/vla-gr-workspace/habitat-data')
os.environ['HF_HOME'] = os.path.expanduser('~/vla-gr-workspace/huggingface-cache')

import torch
from src.core.vla_gr_agent import ConferenceVLAGRAgent
from omegaconf import OmegaConf

print("🧪 测试 VLA-GR Agent...")

# 加载配置
config = OmegaConf.load("config_active.yaml")

# 创建 Agent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = ConferenceVLAGRAgent(config, device=device)

print(f"✓ Agent 创建成功")
print(f"  设备: {device}")
print(f"  语言模型: {config.model.language.model}")

# 测试前向传播（使用随机数据）
batch_size = 2
rgb = torch.randn(batch_size, 3, 224, 224).to(device)
depth = torch.randn(batch_size, 1, 224, 224).to(device)
instruction = ["go to the chair"] * batch_size

print(f"✓ 准备测试输入")

with torch.no_grad():
    output = agent.forward(rgb, depth, instruction)

print(f"✓ 前向传播成功")
print(f"  输出 keys: {output.keys()}")
print(f"  Action shape: {output['action'].shape}")

# 检查 GPU 内存
if torch.cuda.is_available():
    print(f"  GPU 内存使用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print("✅ Agent 测试通过！")
```

运行：

```bash
python test_agent.py
```

### 6.4 运行简单推理测试

创建 `test_inference.py`:

```python
"""端到端推理测试"""
import os
os.environ['HABITAT_DATA_DIR'] = os.path.expanduser('~/vla-gr-workspace/habitat-data')
os.environ['HF_HOME'] = os.path.expanduser('~/vla-gr-workspace/huggingface-cache')

import torch
from src.core.vla_gr_agent import ConferenceVLAGRAgent
from src.environments.habitat_env_v3 import HabitatNavigationEnv
from omegaconf import OmegaConf

print("🧪 端到端推理测试...")

# 加载配置
config = OmegaConf.load("config_active.yaml")

# 创建环境和 Agent
env = HabitatNavigationEnv(
    scene_id="replica/apartment_0.glb",
    task_type="objectnav",
    max_episode_steps=50
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = ConferenceVLAGRAgent(config, device=device)
agent.eval()

print(f"✓ 环境和 Agent 就绪")

# 运行一个 episode
obs, info = env.reset()
episode_reward = 0
steps = 0

print(f"  任务: {obs.get('instruction', 'navigate')}")

for step in range(10):  # 运行 10 步
    # 准备输入
    rgb = torch.from_numpy(obs['rgb']).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    depth = torch.from_numpy(obs['depth']).unsqueeze(0).unsqueeze(0).float().to(device)
    instruction = [obs.get('instruction', 'go forward')]

    # Agent 推理
    with torch.no_grad():
        output = agent.forward(rgb, depth, instruction)

    # 获取动作
    action_probs = torch.softmax(output['action'], dim=-1)
    action = torch.argmax(action_probs, dim=-1).item()

    # 执行动作
    obs, reward, done, truncated, info = env.step(action)
    episode_reward += reward
    steps += 1

    print(f"  Step {steps}: action={action}, reward={reward:.3f}, done={done}")

    if done or truncated:
        break

env.close()

print(f"✓ Episode 完成")
print(f"  总步数: {steps}")
print(f"  总奖励: {episode_reward:.3f}")
print("✅ 端到端推理测试通过！")
```

运行：

```bash
python test_inference.py
```

### 6.5 检查 GPU 内存使用

创建 `test_gpu_memory.py`:

```python
"""GPU 内存使用测试"""
import torch
import os
os.environ['HABITAT_DATA_DIR'] = os.path.expanduser('~/vla-gr-workspace/habitat-data')
os.environ['HF_HOME'] = os.path.expanduser('~/vla-gr-workspace/huggingface-cache')

from src.core.vla_gr_agent import ConferenceVLAGRAgent
from omegaconf import OmegaConf

if not torch.cuda.is_available():
    print("❌ CUDA 不可用")
    exit(1)

print("🔍 GPU 内存分析...")

device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats()

# 加载配置
config = OmegaConf.load("config_active.yaml")

print(f"初始 GPU 内存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# 创建 Agent
agent = ConferenceVLAGRAgent(config, device=device)
print(f"Agent 加载后: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# 模拟训练批次
batch_size = config.training.batch_size
rgb = torch.randn(batch_size, 3, 224, 224).to(device)
depth = torch.randn(batch_size, 1, 224, 224).to(device)
instruction = ["test"] * batch_size

# 前向传播
output = agent.forward(rgb, depth, instruction)
print(f"前向传播后: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# 模拟反向传播
loss = output['action'].sum()
loss.backward()
print(f"反向传播后: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# 峰值内存
peak_memory = torch.cuda.max_memory_allocated() / 1e9
print(f"\n峰值 GPU 内存: {peak_memory:.2f} GB")

# 检查是否超出显存
gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU 总内存: {gpu_memory:.2f} GB")

if peak_memory < gpu_memory * 0.9:
    print("✅ 内存使用正常")
else:
    print("⚠️  内存使用接近上限，考虑减小 batch size 或启用梯度检查点")
```

运行：

```bash
python test_gpu_memory.py
```

---

## 🏋️ 阶段七：训练和评估

### 7.1 准备训练配置

确保你的 `config_active.yaml` 配置正确：

**RTX 4060 (8GB) 配置**：

```yaml
training:
  batch_size: 4              # 小批次
  gradient_accumulation: 8   # 累积梯度模拟大批次
  mixed_precision: true      # FP16
  gradient_checkpointing: true  # 减少内存
  max_steps: 100000
  learning_rate: 3e-5

model:
  use_lora: true             # LoRA 微调
  lora_rank: 16
  lora_alpha: 32
```

**服务器配置**：

```yaml
training:
  batch_size: 32
  gradient_accumulation: 1
  mixed_precision: "bf16"    # BF16（A100/H100）
  distributed: true
  num_gpus: 4
  max_steps: 100000
  learning_rate: 5e-5
```

### 7.2 运行小规模训练测试

首先运行一个短时间的训练测试，确保一切正常：

```bash
# 修改配置进行快速测试
cp config_active.yaml config_test.yaml

# 编辑 config_test.yaml，设置：
# training.max_steps: 100
# training.eval_every: 50
# training.save_every: 50

# 运行测试训练
vla-gr-train \
    --config config_test.yaml \
    --output-dir ~/vla-gr-workspace/outputs/test_run \
    --num-episodes 10

# 或者直接使用 Python
python -m src.training.train \
    --config config_test.yaml \
    --output-dir ~/vla-gr-workspace/outputs/test_run
```

**预期输出**：

```
🚀 VLA-GR Training Pipeline
📋 Configuration:
   - Device: cuda
   - Batch size: 4
   - Learning rate: 3e-05
   - Mixed precision: True

📊 Loading datasets...
✓ Train dataset: 10 episodes
✓ Val dataset: 2 episodes

🏗️ Initializing model...
✓ VLA-GR Agent created
   Parameters: 2.8B total, 45M trainable (LoRA)

🎯 Starting training...

Step 1/100 | Loss: 2.456 | LR: 3.00e-05 | Time: 2.3s
Step 10/100 | Loss: 1.823 | LR: 3.00e-05 | Time: 1.8s
...
Step 50/100 | Loss: 0.945 | LR: 3.00e-05 | Time: 1.7s
📊 Validation | Val Loss: 1.123 | Success Rate: 15.0%
...
Step 100/100 | Loss: 0.712 | LR: 3.00e-05 | Time: 1.6s

✅ Training test completed successfully!
```

### 7.3 启动完整训练

**单 GPU 训练**：

```bash
# 使用命令行工具
vla-gr-train \
    --config config_active.yaml \
    --output-dir ~/vla-gr-workspace/outputs/full_training \
    --resume-from-checkpoint ~/vla-gr-workspace/checkpoints/latest.pt

# 或使用 Python
python -m src.training.train \
    --config config_active.yaml \
    --output-dir ~/vla-gr-workspace/outputs/full_training

# 使用 nohup 后台运行
nohup vla-gr-train \
    --config config_active.yaml \
    --output-dir ~/vla-gr-workspace/outputs/full_training \
    > ~/vla-gr-workspace/logs/training.log 2>&1 &

# 查看训练日志
tail -f ~/vla-gr-workspace/logs/training.log
```

**多 GPU 分布式训练**：

```bash
# 使用 torchrun（推荐）
torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    -m src.training.train \
    --config config_server.yaml \
    --output-dir ~/vla-gr-workspace/outputs/distributed_training

# 或使用 accelerate
accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    -m src.training.train \
    --config config_server.yaml \
    --output-dir ~/vla-gr-workspace/outputs/distributed_training
```

### 7.4 监控训练进度

**使用 TensorBoard**：

```bash
# 启动 TensorBoard
tensorboard --logdir ~/vla-gr-workspace/logs --port 6006

# 在浏览器中打开：http://localhost:6006
```

**使用 Weights & Biases**：

```bash
# 如果启用了 W&B，访问：
# https://wandb.ai/<your-username>/vla-gr-navigation
```

**使用命令行工具**：

```bash
# 查看最新检查点
ls -lht ~/vla-gr-workspace/checkpoints/

# 检查 GPU 使用
watch -n 1 nvidia-smi

# 监控日志
tail -f ~/vla-gr-workspace/logs/training.log | grep -E "(Step|Loss|Success)"
```

### 7.5 运行评估

在训练过程中或之后运行评估：

```bash
# 评估最新检查点
vla-gr-evaluate \
    --config config_active.yaml \
    --checkpoint ~/vla-gr-workspace/checkpoints/latest.pt \
    --output-dir ~/vla-gr-workspace/outputs/evaluation \
    --num-episodes 100

# 或使用 Python
python -m src.evaluation.evaluator \
    --config config_active.yaml \
    --checkpoint ~/vla-gr-workspace/checkpoints/latest.pt \
    --num-episodes 100 \
    --save-trajectories

# 会议级评估（详细指标）
python scripts/run_evaluation.py \
    --config config_active.yaml \
    --checkpoint ~/vla-gr-workspace/checkpoints/best.pt \
    --output-dir ~/vla-gr-workspace/outputs/conference_eval \
    --split val \
    --num-episodes 500
```

**预期评估输出**：

```
🎯 VLA-GR Evaluation
📊 Loading checkpoint: checkpoints/latest.pt

🏃 Running evaluation on 100 episodes...

Episode 1/100 | Success: True | SPL: 0.85 | Steps: 45
Episode 10/100 | Success: False | SPL: 0.00 | Steps: 500
...
Episode 100/100 | Success: True | SPL: 0.72 | Steps: 67

📈 Evaluation Results:
   - Success Rate: 77.4%
   - SPL: 0.645
   - Collision Rate: 16.5%
   - Avg Steps: 124.5
   - Avg Distance to Goal: 0.38m

✅ Evaluation complete!
Results saved to: outputs/evaluation/results.json
```

---

## 🔧 常见问题和解决方案

### Q1: Habitat-Sim 编译失败

**问题**：编译 Habitat-Sim 时出错

**解决方案**：

```bash
# 确保安装了所有编译依赖
sudo apt-get install -y build-essential cmake git ninja-build

# 检查 GCC 版本（需要 7.x 或更高）
gcc --version

# 清理并重新编译
cd ~/vla-gr-workspace/habitat-build/habitat-sim
rm -rf build
python setup.py clean
python setup.py install --headless --with-cuda
```

### Q2: CUDA out of memory

**问题**：训练时 GPU 内存不足

**解决方案**：

```yaml
# 在 config_active.yaml 中调整：
training:
  batch_size: 2              # 减小批次
  gradient_accumulation: 16  # 增加累积步数
  gradient_checkpointing: true  # 启用梯度检查点

model:
  use_lora: true             # 使用 LoRA
  freeze_vision_encoder: true  # 冻结视觉编码器
```

### Q3: Hugging Face 模型下载失败

**问题**：网络问题导致下载失败

**解决方案**：

```bash
# 使用镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载后设置本地路径
# 在 config.yaml 中：
model:
  language:
    model: "/path/to/local/phi-2"
    local_files_only: true
```

### Q4: Habitat 场景未找到

**问题**：`Scene dataset not found`

**解决方案**：

```bash
# 检查环境变量
echo $HABITAT_DATA_DIR

# 验证场景文件
ls $HABITAT_DATA_DIR/scene_datasets/replica/

# 在 Python 中明确设置路径
import os
os.environ['HABITAT_DATA_DIR'] = '/your/path/to/habitat-data'
```

### Q5: 训练速度慢

**优化建议**：

```yaml
# 启用数据加载优化
data:
  num_workers: 4              # 多进程数据加载
  prefetch_factor: 2          # 预取批次
  pin_memory: true            # 固定内存（GPU 传输更快）

# 启用编译优化（PyTorch 2.0+）
training:
  compile: true               # torch.compile
  compile_mode: "reduce-overhead"
```

### Q6: 多 GPU 训练问题

**问题**：分布式训练卡住或出错

**解决方案**：

```bash
# 检查 NCCL（多 GPU 通信）
python -c "import torch; print(torch.cuda.nccl.version())"

# 设置环境变量
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # 如果遇到 InfiniBand 问题

# 使用正确的启动方式
torchrun --nproc_per_node=4 ...
```

---

## 📚 目录结构总结

**最终的工作空间结构**：

```
~/vla-gr-workspace/
├── vla-gr-env/                    # 虚拟环境
├── VLA-GR/                        # 项目代码
│   ├── src/                       # 源代码
│   ├── scripts/                   # 脚本
│   ├── config_active.yaml         # 活动配置
│   └── ...
├── habitat-data/                  # Habitat 数据（100GB - 3.5TB）
│   ├── scene_datasets/
│   │   ├── replica/              # ~2GB（必需）
│   │   └── hm3d/                 # ~2.5TB（可选）
│   └── datasets/
│       └── objectnav/
├── huggingface-cache/             # HF 模型缓存（~10GB）
│   ├── models--microsoft--phi-2/
│   ├── models--openai--clip-vit-base-patch32/
│   └── models--bert-base-uncased/
├── outputs/                       # 训练输出
├── checkpoints/                   # 模型检查点
├── logs/                          # 日志文件
└── habitat-build/                 # Habitat 构建目录（可删除）
```

**存储需求总结**：

```
基础环境（测试/开发）:
  - 虚拟环境: ~5GB
  - Habitat-Sim/Lab: ~2GB
  - PyTorch + 依赖: ~8GB
  - Replica 场景: ~2GB
  - HF 模型: ~10GB
  - 总计: ~30GB

完整训练环境:
  - 基础环境: ~30GB
  - HM3D 数据集: ~2.5TB
  - 训练输出/检查点: ~50GB
  - 总计: ~2.6TB
```

---

## 🎉 完成部署！

恭喜！你已经完成了 VLA-GR Habitat 0.3.3 的完整部署。

**下一步**：

1. ✅ 运行所有验证测试（阶段六）
2. 🧪 运行小规模训练测试
3. 🚀 开始完整训练
4. 📊 监控和评估结果
5. 🔧 根据性能调优配置

**有用的命令**：

```bash
# 快速启动训练（RTX 4060）
vla-gr-train --config config_rtx4060.yaml

# 快速启动训练（服务器）
torchrun --nproc_per_node=4 -m src.training.train --config config_server.yaml

# 评估
vla-gr-evaluate --checkpoint checkpoints/best.pt --num-episodes 100

# 监控
tensorboard --logdir logs

# GPU 监控
watch -n 1 nvidia-smi
```

**获取帮助**：

- 📖 查看项目文档：`README.md`, `DEPLOYMENT_GUIDE.md`
- 🐛 遇到问题：检查 `BUG_FIXES_SUMMARY.md`
- 📚 API 参考：`HABITAT_TRANSFORMERS_QUICK_REFERENCE.md`
- 🔬 理论背景：`THEORETICAL_CONTRIBUTIONS.md`

祝训练顺利！🚀
