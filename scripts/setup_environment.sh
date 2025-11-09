#!/bin/bash
################################################################################
# VLA-GR 环境自动化设置脚本
# 用途：自动创建虚拟环境、安装依赖、配置路径
# 使用：bash scripts/setup_environment.sh
################################################################################

set -e  # 遇到错误立即退出

echo "=========================================="
echo "🚀 VLA-GR 环境设置脚本"
echo "=========================================="

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Python 版本
echo -e "\n${YELLOW}1. 检查 Python 版本...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python 版本: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) and sys.version_info < (3, 12) else 1)"; then
    echo -e "${RED}   ✗ 需要 Python 3.9-3.11${NC}"
    exit 1
fi
echo -e "${GREEN}   ✓ Python 版本满足要求${NC}"

# 检查 CUDA
echo -e "\n${YELLOW}2. 检查 CUDA...${NC}"
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo "   CUDA 版本: $cuda_version"
    echo -e "${GREEN}   ✓ CUDA 已安装${NC}"
else
    echo -e "${YELLOW}   ⚠ CUDA 未检测到，将使用 CPU 模式${NC}"
fi

# 检查 GPU
echo -e "\n${YELLOW}3. 检查 GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo -e "${GREEN}   ✓ GPU 已检测到${NC}"
else
    echo -e "${YELLOW}   ⚠ nvidia-smi 未找到，请确保安装了 NVIDIA 驱动${NC}"
fi

# 设置工作空间路径
WORKSPACE_DIR="$HOME/vla-gr-workspace"
ENV_DIR="$WORKSPACE_DIR/vla-gr-env"

echo -e "\n${YELLOW}4. 创建工作空间...${NC}"
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"
echo "   工作空间路径: $WORKSPACE_DIR"
echo -e "${GREEN}   ✓ 工作空间已创建${NC}"

# 创建虚拟环境
echo -e "\n${YELLOW}5. 创建 Python 虚拟环境...${NC}"
if [ -d "$ENV_DIR" ]; then
    echo -e "${YELLOW}   虚拟环境已存在，跳过创建${NC}"
else
    python3 -m venv "$ENV_DIR"
    echo -e "${GREEN}   ✓ 虚拟环境已创建: $ENV_DIR${NC}"
fi

# 激活虚拟环境
source "$ENV_DIR/bin/activate"
echo -e "${GREEN}   ✓ 虚拟环境已激活${NC}"

# 升级 pip
echo -e "\n${YELLOW}6. 升级 pip, setuptools, wheel...${NC}"
pip install --upgrade pip setuptools wheel --quiet
echo -e "${GREEN}   ✓ 已升级到最新版本${NC}"

# 安装 PyTorch
echo -e "\n${YELLOW}7. 安装 PyTorch...${NC}"
echo "   这可能需要几分钟..."

# 检测 CUDA 版本并安装对应的 PyTorch
if command -v nvcc &> /dev/null; then
    cuda_major=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1 | cut -d'.' -f1)
    if [ "$cuda_major" == "12" ]; then
        echo "   安装 PyTorch with CUDA 12.1..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 --quiet
    else
        echo "   安装 PyTorch with CUDA 11.8..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 --quiet
    fi
else
    echo "   安装 CPU 版本的 PyTorch..."
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu --quiet
fi
echo -e "${GREEN}   ✓ PyTorch 已安装${NC}"

# 验证 PyTorch
python -c "import torch; print(f'   PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 创建数据目录
echo -e "\n${YELLOW}8. 创建数据目录...${NC}"
mkdir -p "$WORKSPACE_DIR/habitat-data/scene_datasets"
mkdir -p "$WORKSPACE_DIR/habitat-data/datasets/objectnav/hm3d/v1"
mkdir -p "$WORKSPACE_DIR/habitat-data/datasets/pointnav"
mkdir -p "$WORKSPACE_DIR/huggingface-cache"
mkdir -p "$WORKSPACE_DIR/outputs"
mkdir -p "$WORKSPACE_DIR/checkpoints"
mkdir -p "$WORKSPACE_DIR/logs"
echo -e "${GREEN}   ✓ 数据目录已创建${NC}"

# 设置环境变量
echo -e "\n${YELLOW}9. 配置环境变量...${NC}"

ENV_FILE="$HOME/.vla_gr_env"
cat > "$ENV_FILE" << EOF
# VLA-GR 环境变量
export VLA_GR_WORKSPACE="$WORKSPACE_DIR"
export HABITAT_DATA_DIR="$WORKSPACE_DIR/habitat-data"
export HABITAT_SCENE_DATASETS_DIR="\$HABITAT_DATA_DIR/scene_datasets"
export HF_HOME="$WORKSPACE_DIR/huggingface-cache"
export TRANSFORMERS_CACHE="\$HF_HOME/transformers"

# 激活虚拟环境
alias vla-gr-env="source $ENV_DIR/bin/activate"
EOF

# 添加到 .bashrc（如果还没有）
if ! grep -q "source $ENV_FILE" "$HOME/.bashrc"; then
    echo "" >> "$HOME/.bashrc"
    echo "# VLA-GR 环境" >> "$HOME/.bashrc"
    echo "source $ENV_FILE" >> "$HOME/.bashrc"
    echo -e "${GREEN}   ✓ 环境变量已添加到 .bashrc${NC}"
else
    echo -e "${YELLOW}   环境变量已在 .bashrc 中${NC}"
fi

# 加载环境变量
source "$ENV_FILE"
echo -e "${GREEN}   ✓ 环境变量已加载${NC}"

# 显示摘要
echo -e "\n=========================================="
echo -e "${GREEN}✅ 环境设置完成！${NC}"
echo "=========================================="
echo ""
echo "📁 工作空间位置:"
echo "   $WORKSPACE_DIR"
echo ""
echo "🐍 虚拟环境:"
echo "   $ENV_DIR"
echo "   激活命令: source $ENV_DIR/bin/activate"
echo "   或使用别名: vla-gr-env"
echo ""
echo "📊 数据目录:"
echo "   Habitat: $HABITAT_DATA_DIR"
echo "   HuggingFace: $HF_HOME"
echo ""
echo "🔧 下一步:"
echo "   1. 激活环境: source $ENV_DIR/bin/activate"
echo "   2. 安装 Habitat: bash scripts/install_habitat.sh"
echo "   3. 克隆项目: git clone <your-repo> $WORKSPACE_DIR/VLA-GR"
echo "   4. 安装项目: cd $WORKSPACE_DIR/VLA-GR && pip install -e ."
echo ""
echo "💡 提示: 重新打开终端后环境变量会自动加载"
echo ""
