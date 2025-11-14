# VLA-GR 完整部署指南 (RTX 4060)

> 📅 最后更新: 2025-11-14
> 🎯 目标显卡: NVIDIA RTX 4060 (8GB VRAM)
> 💻 推荐系统: Ubuntu 20.04/22.04

---

## 📑 目录

1. [系统要求](#1-系统要求)
2. [克隆仓库](#2-克隆仓库)
3. [环境配置](#3-环境配置)
4. [安装 Habitat 0.3.3](#4-安装-habitat-033)
5. [安装依赖包](#5-安装依赖包)
6. [下载数据集](#6-下载数据集)
7. [下载预训练模型](#7-下载预训练模型)
8. [验证安装](#8-验证安装)
9. [运行评估](#9-运行评估)
10. [常见问题](#10-常见问题)

---

## 1. 系统要求

### 硬件配置

| 组件 | RTX 4060 最低要求 | 推荐配置 |
|------|------------------|---------|
| **GPU** | NVIDIA RTX 4060 (8GB VRAM) | RTX 4060 Ti (16GB) |
| **CPU** | 4核 2.5GHz | 8核 3.0GHz+ |
| **内存** | 16GB RAM | 32GB RAM |
| **存储** | 100GB 可用空间 | 500GB SSD |
| **系统** | Ubuntu 20.04+ | Ubuntu 22.04 LTS |

### 软件要求

- NVIDIA 驱动 >= 525.x
- CUDA 11.7 或 11.8
- Python 3.8 或 3.9
- Conda/Miniconda
- Git

### 检查 NVIDIA 驱动和 CUDA

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 应该看到类似输出:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
# +-----------------------------------------------------------------------------+
# |   0  NVIDIA GeForce RTX 4060     Off  | 00000000:01:00.0 Off |          N/A |
# +-----------------------------------------------------------------------------+
```

---

## 2. 克隆仓库

### 2.1 创建工作目录

```bash
# 创建统一的工作空间
mkdir -p ~/vla-gr-workspace
cd ~/vla-gr-workspace

# 设置环境变量（添加到 ~/.bashrc 以永久保存）
export VLA_GR_ROOT="$HOME/vla-gr-workspace"
export HABITAT_DATA_DIR="$VLA_GR_ROOT/habitat-data"
export HF_HOME="$VLA_GR_ROOT/huggingface-cache"
```

### 2.2 克隆 VLA-GR 仓库

```bash
cd $VLA_GR_ROOT

# 从 GitHub 克隆仓库
git clone https://github.com/Amagerd1113/VLA-GR.git
cd VLA-GR

# 查看当前分支
git branch

# 切换到开发分支（如果需要）
# git checkout claude/deployment-guide-habitat-013AqUjE5WkWP6Nv6phELeFJ
```

### 项目目录结构

```
~/vla-gr-workspace/
├── VLA-GR/                      # 项目代码
│   ├── src/                     # 源代码
│   ├── scripts/                 # 脚本工具
│   ├── config.yaml              # 主配置
│   ├── config_rtx4060.yaml      # RTX 4060 优化配置 ⭐
│   └── requirements.txt         # Python 依赖
├── habitat-data/                # Habitat 数据集
│   ├── scene_datasets/          # 3D 场景
│   │   ├── replica/             # Replica 数据集 (~2GB)
│   │   └── hm3d/                # HM3D 数据集
│   └── datasets/                # 任务数据
│       └── objectnav/           # ObjectNav 任务
└── huggingface-cache/           # HuggingFace 模型缓存
    └── models--*/               # 各个预训练模型
```

---

## 3. 环境配置

### 3.1 创建 Conda 环境

```bash
cd $VLA_GR_ROOT/VLA-GR

# 创建 Python 3.9 环境
conda create -n vla_gr python=3.9 cmake=3.14.0 -y
conda activate vla_gr

# 验证 Python 版本
python --version  # 应显示 Python 3.9.x
```

### 3.2 安装 PyTorch (CUDA 11.7)

```bash
# 针对 RTX 4060 安装 PyTorch 2.0 + CUDA 11.7
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu117

# 验证 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# 期望输出:
# PyTorch: 2.0.1+cu117
# CUDA available: True
# CUDA version: 11.7
# Device name: NVIDIA GeForce RTX 4060
```

---

## 4. 安装 Habitat 0.3.3

### 4.1 安装 Habitat-Sim 0.3.3

```bash
# 激活环境
conda activate vla_gr

# 从 conda-forge 安装 Habitat-Sim 0.3.3 (带物理引擎)
conda install habitat-sim=0.3.3 withbullet -c conda-forge -c aihabitat -y

# 验证安装
python -c "import habitat_sim; print(f'Habitat-Sim version: {habitat_sim.__version__}')"
# 期望输出: Habitat-Sim version: 0.3.3
```

### 4.2 安装 Habitat-Lab 0.3.3

```bash
# Habitat-Lab 通常需要从 pip 安装
pip install habitat-lab==0.3.3

# 或者从源码安装最新版本
# git clone --branch v0.3.3 https://github.com/facebookresearch/habitat-lab.git
# cd habitat-lab
# pip install -e habitat-lab

# 验证安装
python -c "import habitat; print(f'Habitat-Lab version: {habitat.__version__}')"
# 期望输出: Habitat-Lab version: 0.3.3
```

### 4.3 测试 Habitat 环境

```bash
# 测试基本导入
python << EOF
import habitat_sim
import habitat

# 创建简单配置
backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = "NONE"

agent_cfg = habitat_sim.agent.AgentConfiguration()
cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])

# 创建模拟器
sim = habitat_sim.Simulator(cfg)
print("✓ Habitat 环境测试成功！")
sim.close()
EOF
```

---

## 5. 安装依赖包

### 5.1 安装主要依赖

```bash
cd $VLA_GR_ROOT/VLA-GR

# 安装 requirements.txt 中的所有依赖
pip install -r requirements.txt

# 主要包括:
# - transformers>=4.30.0       # Hugging Face 模型
# - opencv-python>=4.8.0       # 图像处理
# - Pillow>=10.0.0             # 图像操作
# - wandb>=0.15.0              # 实验跟踪
# - tensorboard>=2.13.0        # 可视化
# - cvxpy>=1.3.0               # 凸优化
# - qpsolvers>=3.4.0           # 二次规划
# - hydra-core>=1.3.0          # 配置管理
# - omegaconf>=2.3.0           # 配置解析
# - einops>=0.6.0              # 张量操作
```

### 5.2 安装开发依赖（可选）

```bash
# 如果需要开发和测试
pip install -r requirements-dev.txt

# 包括:
# - pytest>=7.4.0              # 测试框架
# - black>=23.7.0              # 代码格式化
# - flake8>=6.0.0              # 代码检查
# - mypy>=1.4.0                # 类型检查
```

### 5.3 安装 VLA-GR 包

```bash
# 以开发模式安装
pip install -e .

# 验证安装
python -c "import vla_gr; print('✓ VLA-GR 安装成功！')"
```

---

## 6. 下载数据集

### 6.1 配置数据目录

```bash
# 确保环境变量已设置
export HABITAT_DATA_DIR="$HOME/vla-gr-workspace/habitat-data"
mkdir -p $HABITAT_DATA_DIR

# 添加到 ~/.bashrc 以永久保存
echo 'export HABITAT_DATA_DIR="$HOME/vla-gr-workspace/habitat-data"' >> ~/.bashrc
```

### 6.2 下载 Replica 数据集（必需，~2GB）

```bash
cd $VLA_GR_ROOT/VLA-GR

# 使用项目提供的下载脚本
bash scripts/download_datasets.sh

# 选择: 1) Replica 测试场景（必需，~2GB）
# 或者手动下载:

mkdir -p $HABITAT_DATA_DIR/scene_datasets/replica
cd $HABITAT_DATA_DIR/scene_datasets

# 方法1: 从 Hugging Face 下载（推荐）
python << EOF
from huggingface_hub import snapshot_download
import os

print("从 Hugging Face 下载 Replica...")
snapshot_download(
    repo_id="ai-habitat/replica_cad_dataset",
    repo_type="dataset",
    local_dir="replica",
    local_dir_use_symlinks=False
)
print("✓ Replica 下载完成")
EOF

# 方法2: 从官方源下载
# wget https://dl.fbaipublicfiles.com/habitat/replica_cad_dataset.zip
# unzip replica_cad_dataset.zip -d replica/
# rm replica_cad_dataset.zip
```

### 6.3 下载 HM3D minival（测试用，~10GB）

**注意**: HM3D 数据集需要注册和申请访问权限

```bash
# 1. 访问 https://aihabitat.org/datasets/hm3d/
# 2. 注册并申请访问权限
# 3. 获取用户名和密码

# 下载 minival 数据集
python -m habitat_sim.utils.datasets_download \
    --username YOUR_USERNAME \
    --password YOUR_PASSWORD \
    --uids hm3d_minival \
    --data-path $HABITAT_DATA_DIR
```

### 6.4 下载 ObjectNav 任务数据（~500MB）

```bash
mkdir -p $HABITAT_DATA_DIR/datasets/objectnav/hm3d/v1
cd $HABITAT_DATA_DIR/datasets/objectnav/hm3d/v1

# 下载 ObjectNav episodes
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
unzip objectnav_hm3d_v1.zip
rm objectnav_hm3d_v1.zip

echo "✓ ObjectNav 数据已下载"
```

### 数据集安装位置总结

```
$HABITAT_DATA_DIR (~/vla-gr-workspace/habitat-data/)
│
├── scene_datasets/              # 3D 场景数据
│   ├── replica/                 # ✓ 必需: Replica 场景 (~2GB)
│   │   ├── apartment_0/
│   │   ├── apartment_1/
│   │   ├── frl_apartment_0/
│   │   └── ...                  # 18个场景
│   │
│   └── hm3d/                    # ⭐ 推荐: HM3D 场景
│       ├── minival/             # 快速测试 (~10GB)
│       ├── val/                 # 验证集 (~50GB)
│       └── train/               # 完整训练集 (~2.5TB，可选)
│
└── datasets/                    # 任务数据
    ├── objectnav/               # ObjectNav 导航任务
    │   └── hm3d/v1/
    │       ├── train/
    │       ├── val/
    │       └── test/
    │
    └── pointnav/                # PointNav 导航任务（可选）
        └── gibson/v1/
```

### RTX 4060 存储建议

由于 RTX 4060 显卡显存有限，建议：
- ✅ **必须下载**: Replica (~2GB)
- ✅ **推荐下载**: HM3D minival (~10GB)
- ✅ **推荐下载**: ObjectNav 任务数据 (~500MB)
- ❌ **不推荐**: HM3D 完整训练集 (~2.5TB) - 仅在有充足存储和训练需求时

**总存储需求**: 约 15-20GB（最小配置）

---

## 7. 下载预训练模型

### 7.1 配置 Hugging Face 缓存

```bash
# 设置缓存目录
export HF_HOME="$HOME/vla-gr-workspace/huggingface-cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

mkdir -p $HF_HOME

# 永久保存到 ~/.bashrc
echo 'export HF_HOME="$HOME/vla-gr-workspace/huggingface-cache"' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE="$HF_HOME/transformers"' >> ~/.bashrc

# 如果在中国，使用镜像加速
export HF_ENDPOINT=https://hf-mirror.com
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
```

### 7.2 下载所有必需模型

```bash
cd $VLA_GR_ROOT/VLA-GR

# 使用项目提供的下载脚本
bash scripts/download_models.sh

# 选择: 5) 全部下载

# 或者手动下载每个模型（见下方）
```

### 7.3 模型详细下载

#### (1) Microsoft Phi-2 语言模型 (~5.5GB)

```bash
python << EOF
from transformers import AutoModel, AutoTokenizer

model_name = "microsoft/phi-2"
print("下载 Phi-2 模型...")

# 下载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 下载模型
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True
)

print(f"✓ Phi-2 已下载到 {HF_HOME}")
EOF
```

#### (2) DINOv2 视觉编码器 (~340MB)

```bash
python << EOF
from transformers import AutoModel

model_name = "facebook/dinov2-base"
print("下载 DINOv2 模型...")

model = AutoModel.from_pretrained(model_name)
print(f"✓ DINOv2 已下载")
EOF
```

#### (3) OpenAI CLIP 视觉-语言模型 (~600MB)

```bash
python << EOF
from transformers import CLIPModel, CLIPProcessor

model_name = "openai/clip-vit-base-patch32"
print("下载 CLIP 模型...")

processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)

print(f"✓ CLIP 已下载")
EOF
```

#### (4) BERT Base (后备语言模型, ~440MB)

```bash
python << EOF
from transformers import BertModel, BertTokenizer

model_name = "bert-base-uncased"
print("下载 BERT 模型...")

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

print(f"✓ BERT 已下载")
EOF
```

### 模型安装位置总结

```
$HF_HOME (~/vla-gr-workspace/huggingface-cache/)
│
├── models--microsoft--phi-2/           # Phi-2 语言模型 (5.5GB)
│   └── snapshots/
│       └── xxxxx/
│           ├── config.json
│           ├── pytorch_model.bin
│           └── tokenizer.json
│
├── models--facebook--dinov2-base/      # DINOv2 视觉编码器 (340MB)
│   └── snapshots/
│
├── models--openai--clip-vit-base-patch32/  # CLIP 模型 (600MB)
│   └── snapshots/
│
└── models--bert-base-uncased/          # BERT 模型 (440MB)
    └── snapshots/
```

**总存储需求**: 约 7-8GB

---

## 8. 验证安装

### 8.1 使用验证脚本

```bash
cd $VLA_GR_ROOT/VLA-GR

# 完整验证
python scripts/verify_installation.py

# 分项验证
python scripts/verify_installation.py --check-datasets
python scripts/verify_installation.py --check-models
```

### 8.2 手动验证关键组件

```bash
# 创建验证脚本
python << 'EOF'
import sys

def check_component(name, import_fn):
    try:
        import_fn()
        print(f"✓ {name}")
        return True
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False

print("=" * 50)
print("VLA-GR 安装验证")
print("=" * 50)

all_pass = True

# 1. PyTorch + CUDA
def check_pytorch():
    import torch
    assert torch.cuda.is_available(), "CUDA 不可用"
    assert torch.version.cuda == "11.7", f"CUDA 版本不匹配: {torch.version.cuda}"
    print(f"   PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

all_pass &= check_component("1. PyTorch + CUDA", check_pytorch)

# 2. Habitat-Sim
def check_habitat_sim():
    import habitat_sim
    assert habitat_sim.__version__ == "0.3.3", f"版本不匹配: {habitat_sim.__version__}"
    print(f"   Habitat-Sim {habitat_sim.__version__}")

all_pass &= check_component("2. Habitat-Sim 0.3.3", check_habitat_sim)

# 3. Habitat-Lab
def check_habitat_lab():
    import habitat
    print(f"   Habitat-Lab {habitat.__version__}")

all_pass &= check_component("3. Habitat-Lab", check_habitat_lab)

# 4. Transformers
def check_transformers():
    from transformers import AutoModel, AutoTokenizer
    print(f"   Transformers 已安装")

all_pass &= check_component("4. Transformers", check_transformers)

# 5. VLA-GR
def check_vla_gr():
    from vla_gr import VLAGRAgent
    print(f"   VLA-GR 核心模块已加载")

all_pass &= check_component("5. VLA-GR", check_vla_gr)

# 6. 数据集
def check_datasets():
    import os
    data_dir = os.path.expanduser("~/vla-gr-workspace/habitat-data")
    replica_path = os.path.join(data_dir, "scene_datasets/replica")
    assert os.path.exists(replica_path), f"Replica 数据集未找到: {replica_path}"

    scene_count = len([f for f in os.listdir(replica_path) if os.path.isdir(os.path.join(replica_path, f))])
    print(f"   Replica: {scene_count} 个场景")

all_pass &= check_component("6. 数据集 (Replica)", check_datasets)

# 7. 模型
def check_models():
    import os
    cache_dir = os.path.expanduser("~/vla-gr-workspace/huggingface-cache")
    assert os.path.exists(cache_dir), f"模型缓存目录未找到: {cache_dir}"

    model_dirs = [d for d in os.listdir(cache_dir) if d.startswith("models--")]
    print(f"   已缓存 {len(model_dirs)} 个模型")

all_pass &= check_component("7. 预训练模型", check_models)

print("=" * 50)
if all_pass:
    print("✅ 所有组件验证通过！")
    sys.exit(0)
else:
    print("❌ 部分组件验证失败，请检查安装")
    sys.exit(1)
EOF
```

### 8.3 GPU 内存测试

```bash
# 测试 RTX 4060 是否能加载模型
python << EOF
import torch

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# 测试分配
x = torch.randn(1000, 1000, device='cuda')
print(f"已分配: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
print(f"已缓存: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

print("✓ GPU 内存测试通过")
EOF
```

---

## 9. 运行评估

### 9.1 快速评估（使用 RTX 4060 配置）

```bash
cd $VLA_GR_ROOT/VLA-GR

# 使用 RTX 4060 优化配置
python scripts/run_evaluation.py \
    --config config_rtx4060.yaml \
    --num-episodes 10 \
    --no-viz

# 期望输出:
# Evaluation Results:
# - Success Rate: XX%
# - SPL: XX%
# - Average Episode Length: XX steps
```

### 9.2 查看 RTX 4060 配置

```bash
cat config_rtx4060.yaml

# 主要优化:
# - batch_size: 8 (减小批大小)
# - gradient_accumulation_steps: 4 (梯度累积)
# - mixed_precision: true (混合精度)
# - gradient_checkpointing: true (梯度检查点，节省显存)
# - model.vla.hidden_dim: 512 (减小模型容量)
```

### 9.3 运行完整评估

```bash
# 运行完整评估套件（会生成论文图表）
python scripts/run_complete_evaluation.py \
    --config config_rtx4060.yaml \
    --output-dir results/rtx4060_eval \
    --num-episodes 100

# 生成的文件:
# results/rtx4060_eval/
# ├── metrics.json           # 评估指标
# ├── figures/               # 生成的图表
# │   ├── success_rate.png
# │   ├── spl_comparison.png
# │   └── trajectory_vis.png
# └── tables/                # LaTeX 表格
#     └── results_table.tex
```

---

## 10. 常见问题

### 问题 1: CUDA Out of Memory

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:

```bash
# 1. 使用更小的批大小
python scripts/run_evaluation.py \
    --config config_rtx4060.yaml \
    training.batch_size=4

# 2. 启用梯度检查点
python scripts/run_evaluation.py \
    --config config_rtx4060.yaml \
    model.gradient_checkpointing=true

# 3. 减小模型大小
python scripts/run_evaluation.py \
    --config config_rtx4060.yaml \
    model.vla.hidden_dim=256

# 4. 设置 CUDA 内存分配策略
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 问题 2: Habitat 导入错误

**症状**: `ImportError: cannot import name 'Simulator'`

**解决方案**:

```bash
# 重新安装 Habitat
conda uninstall habitat-sim habitat-lab -y
conda install habitat-sim=0.3.3 withbullet -c conda-forge -c aihabitat -y
pip install habitat-lab==0.3.3
```

### 问题 3: Hugging Face 模型下载失败

**症状**: `Connection timeout` 或下载缓慢

**解决方案**:

```bash
# 使用中国镜像站
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型
mkdir -p ~/.cache/huggingface/hub
cd ~/.cache/huggingface/hub
git lfs install
git clone https://hf-mirror.com/microsoft/phi-2
```

### 问题 4: Replica 数据集加载失败

**症状**: `Scene file not found`

**解决方案**:

```bash
# 检查数据集路径
ls -lh $HABITAT_DATA_DIR/scene_datasets/replica

# 重新下载 Replica
cd $HABITAT_DATA_DIR/scene_datasets
rm -rf replica
wget https://dl.fbaipublicfiles.com/habitat/replica_cad_dataset.zip
unzip replica_cad_dataset.zip -d replica/
```

### 问题 5: RTX 4060 性能较慢

**优化建议**:

```bash
# 1. 启用 TF32 加速（Ampere 架构）
export NVIDIA_TF32_OVERRIDE=1

# 2. 启用 cuDNN benchmark
# 在代码中添加:
# torch.backends.cudnn.benchmark = True

# 3. 使用混合精度训练
# config_rtx4060.yaml 中已默认启用:
# hardware.mixed_precision: true

# 4. 减少数据加载器进程数
# training.dataloader_num_workers: 2
```

### 问题 6: 验证脚本报错

**症状**: `verify_installation.py` 失败

**解决方案**:

```bash
# 逐步检查
python -c "import torch; print(torch.cuda.is_available())"
python -c "import habitat_sim; print(habitat_sim.__version__)"
python -c "import habitat; print(habitat.__version__)"
python -c "from transformers import AutoModel"

# 查看详细错误
python scripts/verify_installation.py --verbose
```

---

## 附录 A: 完整环境变量配置

将以下内容添加到 `~/.bashrc`:

```bash
# VLA-GR 工作空间
export VLA_GR_ROOT="$HOME/vla-gr-workspace"

# Habitat 数据目录
export HABITAT_DATA_DIR="$VLA_GR_ROOT/habitat-data"

# Hugging Face 缓存
export HF_HOME="$VLA_GR_ROOT/huggingface-cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

# Hugging Face 镜像（中国用户）
export HF_ENDPOINT=https://hf-mirror.com

# CUDA 优化
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# 启用 TF32（RTX 30/40 系列）
export NVIDIA_TF32_OVERRIDE=1
```

应用配置:
```bash
source ~/.bashrc
```

---

## 附录 B: RTX 4060 vs 其他显卡对比

| 显卡型号 | VRAM | 批大小 | 训练速度 | 推理速度 | 推荐用途 |
|---------|------|--------|---------|---------|---------|
| **RTX 4060** | 8GB | 8 | 1.0x | 1.0x | ✅ 评估、推理 |
| RTX 4060 Ti | 16GB | 16 | 1.3x | 1.2x | ✅ 轻量训练 |
| RTX 4070 | 12GB | 12 | 1.5x | 1.4x | ✅ 中等训练 |
| RTX 4080 | 16GB | 24 | 2.0x | 1.8x | ✅ 完整训练 |
| RTX 4090 | 24GB | 64 | 3.0x | 2.5x | ✅ 大规模训练 |

**RTX 4060 建议**:
- ✅ 适合: 模型评估、推理、小规模实验
- ⚠️ 可行: 轻量级微调（使用 LoRA/PEFT）
- ❌ 不适合: 大规模从头训练

---

## 附录 C: 快速参考命令

```bash
# 激活环境
conda activate vla_gr

# 快速评估
cd ~/vla-gr-workspace/VLA-GR
python scripts/run_evaluation.py --config config_rtx4060.yaml --num-episodes 10

# 查看 GPU 使用
watch -n 1 nvidia-smi

# 检查数据集
ls -lh $HABITAT_DATA_DIR/scene_datasets/replica

# 检查模型
ls -lh $HF_HOME/models--*

# 运行测试
pytest tests/

# 代码格式化
make format

# 代码检查
make lint
```

---

## 附录 D: 存储空间规划

| 组件 | 大小 | 必需性 | 位置 |
|------|------|--------|------|
| VLA-GR 代码 | ~100MB | ✅ 必需 | `~/vla-gr-workspace/VLA-GR/` |
| Conda 环境 | ~3GB | ✅ 必需 | `~/miniconda3/envs/vla_gr/` |
| **数据集** | | | |
| └ Replica | ~2GB | ✅ 必需 | `$HABITAT_DATA_DIR/scene_datasets/replica/` |
| └ HM3D minival | ~10GB | ⭐ 推荐 | `$HABITAT_DATA_DIR/scene_datasets/hm3d/minival/` |
| └ ObjectNav 任务 | ~500MB | ⭐ 推荐 | `$HABITAT_DATA_DIR/datasets/objectnav/` |
| └ HM3D 完整 | ~2.5TB | ❌ 可选 | `$HABITAT_DATA_DIR/scene_datasets/hm3d/train/` |
| **模型** | | | |
| └ Phi-2 | ~5.5GB | ✅ 必需 | `$HF_HOME/models--microsoft--phi-2/` |
| └ DINOv2 | ~340MB | ✅ 必需 | `$HF_HOME/models--facebook--dinov2-base/` |
| └ CLIP | ~600MB | ⭐ 推荐 | `$HF_HOME/models--openai--clip-vit-base-patch32/` |
| └ BERT | ~440MB | ❌ 可选 | `$HF_HOME/models--bert-base-uncased/` |
| **日志和结果** | ~5GB | - | `~/vla-gr-workspace/VLA-GR/logs/` |

**RTX 4060 最小配置总计**: 约 25GB
**RTX 4060 推荐配置总计**: 约 35GB

---

## 联系和支持

- 📖 **文档**: 查看 `docs/` 目录
- 🐛 **问题反馈**: https://github.com/Amagerd1113/VLA-GR/issues
- 💬 **讨论**: https://github.com/Amagerd1113/VLA-GR/discussions

---

**祝您使用愉快！ 🚀**
