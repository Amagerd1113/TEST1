#!/bin/bash
################################################################################
# Habitat-Sim & Habitat-Lab 0.3.3 自动安装脚本
# 用途：从源码编译并安装 Habitat
# 使用：bash scripts/install_habitat.sh
################################################################################

set -e

echo "=========================================="
echo "🏗️ Habitat-Sim & Habitat-Lab 安装脚本"
echo "=========================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}✗ 请先激活虚拟环境${NC}"
    echo "   运行: source ~/vla-gr-workspace/vla-gr-env/bin/activate"
    exit 1
fi

echo -e "${GREEN}✓ 虚拟环境已激活: $VIRTUAL_ENV${NC}"

# 安装系统依赖
echo -e "\n${YELLOW}1. 安装系统依赖...${NC}"
echo "   这需要 sudo 权限"

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
    ninja-build \
    libglfw3-dev \
    libglu1-mesa-dev

echo -e "${GREEN}✓ 系统依赖已安装${NC}"

# 创建构建目录
WORKSPACE_DIR="${VLA_GR_WORKSPACE:-$HOME/vla-gr-workspace}"
BUILD_DIR="$WORKSPACE_DIR/habitat-build"

echo -e "\n${YELLOW}2. 创建构建目录...${NC}"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
echo "   构建目录: $BUILD_DIR"

# 安装 Habitat-Sim
echo -e "\n${YELLOW}3. 安装 Habitat-Sim 0.3.3...${NC}"

if [ -d "habitat-sim" ]; then
    echo -e "${YELLOW}   habitat-sim 目录已存在，是否重新下载？(y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm -rf habitat-sim
    else
        cd habitat-sim
        git pull
        cd ..
    fi
fi

if [ ! -d "habitat-sim" ]; then
    echo "   克隆 Habitat-Sim 仓库..."
    git clone --branch v0.3.3 https://github.com/facebookresearch/habitat-sim.git
fi

cd habitat-sim

# 安装 Python 依赖
echo "   安装 Python 依赖..."
pip install -r requirements.txt

# 构建选项
echo -e "\n${YELLOW}   选择构建选项:${NC}"
echo "   1) 无头模式 + CUDA（推荐，用于服务器）"
echo "   2) 无头模式 + CUDA + Bullet（物理引擎）"
echo "   3) 标准模式（需要显示器）"
echo -n "   选择 (1-3) [默认: 1]: "
read -r build_option
build_option=${build_option:-1}

case $build_option in
    1)
        BUILD_FLAGS="--headless --with-cuda"
        ;;
    2)
        BUILD_FLAGS="--headless --with-cuda --with-bullet"
        ;;
    3)
        BUILD_FLAGS="--with-cuda"
        ;;
    *)
        echo -e "${RED}   无效选择，使用默认选项${NC}"
        BUILD_FLAGS="--headless --with-cuda"
        ;;
esac

echo -e "\n${YELLOW}   开始编译 Habitat-Sim...${NC}"
echo "   这将花费 15-30 分钟，请耐心等待..."
echo "   构建标志: $BUILD_FLAGS"

# 清理之前的构建
python setup.py clean

# 构建和安装
python setup.py install $BUILD_FLAGS

echo -e "${GREEN}✓ Habitat-Sim 已安装${NC}"

# 验证 Habitat-Sim
echo -e "\n${YELLOW}   验证 Habitat-Sim...${NC}"
python -c "import habitat_sim; print(f'   Habitat-Sim version: {habitat_sim.__version__}')"

# 安装 Habitat-Lab
echo -e "\n${YELLOW}4. 安装 Habitat-Lab 0.3.3...${NC}"
cd "$BUILD_DIR"

if [ -d "habitat-lab" ]; then
    echo -e "${YELLOW}   habitat-lab 目录已存在，是否重新下载？(y/N)${NC}"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm -rf habitat-lab
    else
        cd habitat-lab
        git pull
        cd ..
    fi
fi

if [ ! -d "habitat-lab" ]; then
    echo "   克隆 Habitat-Lab 仓库..."
    git clone --branch v0.3.3 https://github.com/facebookresearch/habitat-lab.git
fi

cd habitat-lab

echo "   安装 Habitat-Lab..."
pip install -e habitat-lab

echo "   安装 Habitat-Baselines..."
pip install -e habitat-baselines

echo -e "${GREEN}✓ Habitat-Lab 已安装${NC}"

# 验证 Habitat-Lab
echo -e "\n${YELLOW}   验证 Habitat-Lab...${NC}"
python -c "import habitat; print(f'   Habitat-Lab version: {habitat.__version__}')"

# 完成
echo -e "\n=========================================="
echo -e "${GREEN}✅ Habitat 安装完成！${NC}"
echo "=========================================="
echo ""
echo "📦 已安装组件:"
echo "   - Habitat-Sim 0.3.3"
echo "   - Habitat-Lab 0.3.3"
echo "   - Habitat-Baselines 0.3.3"
echo ""
echo "🔧 构建选项: $BUILD_FLAGS"
echo ""
echo "📁 安装位置:"
echo "   构建目录: $BUILD_DIR"
echo "   Python 包: $VIRTUAL_ENV/lib/python*/site-packages/"
echo ""
echo "🔍 运行验证测试:"
echo "   python -c 'import habitat_sim; import habitat; print(\"✓ Habitat OK\")'"
echo ""
echo "🗑️ 清理构建文件（可选，释放空间）:"
echo "   rm -rf $BUILD_DIR"
echo ""
echo "🎯 下一步:"
echo "   1. 下载场景数据: bash scripts/download_datasets.sh"
echo "   2. 下载 HF 模型: bash scripts/download_models.sh"
echo "   3. 运行验证脚本: python scripts/verify_installation.py"
echo ""
