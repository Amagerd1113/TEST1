#!/bin/bash
################################################################################
# Habitat 数据集下载脚本
# 用途：下载 Replica、HM3D 场景和任务数据
# 使用：bash scripts/download_datasets.sh
################################################################################

set -e

echo "=========================================="
echo "📥 Habitat 数据集下载脚本"
echo "=========================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 检查环境变量
if [ -z "$HABITAT_DATA_DIR" ]; then
    HABITAT_DATA_DIR="$HOME/vla-gr-workspace/habitat-data"
    echo -e "${YELLOW}⚠ HABITAT_DATA_DIR 未设置，使用默认值: $HABITAT_DATA_DIR${NC}"
fi

mkdir -p "$HABITAT_DATA_DIR"
cd "$HABITAT_DATA_DIR"

echo -e "\n${BLUE}数据目录: $HABITAT_DATA_DIR${NC}"

# 菜单
echo -e "\n${YELLOW}请选择要下载的数据集:${NC}"
echo "   1) Replica 测试场景（必需，~2GB）"
echo "   2) HM3D minival（测试用，~10GB）"
echo "   3) HM3D 完整训练集（~2.5TB，需要申请）"
echo "   4) ObjectNav 任务数据（~500MB）"
echo "   5) 全部下载（除了 HM3D 完整版）"
echo "   6) 退出"
echo -n "选择 (1-6): "
read -r choice

download_replica() {
    echo -e "\n${YELLOW}📥 下载 Replica 数据集...${NC}"

    mkdir -p scene_datasets
    cd scene_datasets

    if [ -d "replica" ] && [ "$(ls -A replica)" ]; then
        echo -e "${YELLOW}   Replica 已存在，跳过下载${NC}"
        cd ..
        return
    fi

    mkdir -p replica

    echo "   方式 1: 从 Hugging Face 下载（推荐）"
    echo "   方式 2: 从官方源下载"
    echo -n "   选择方式 (1-2) [默认: 1]: "
    read -r method
    method=${method:-1}

    if [ "$method" == "1" ]; then
        # 使用 Python 从 HuggingFace 下载
        python << 'EOF'
from huggingface_hub import snapshot_download
import os

print("   从 Hugging Face 下载 Replica...")
snapshot_download(
    repo_id="ai-habitat/replica_cad_dataset",
    repo_type="dataset",
    local_dir="replica",
    local_dir_use_symlinks=False
)
print("   ✓ 下载完成")
EOF
    else
        # 官方下载
        echo "   从官方源下载..."
        wget -c https://dl.fbaipublicfiles.com/habitat/replica_cad_dataset.zip
        unzip -q replica_cad_dataset.zip -d replica/
        rm replica_cad_dataset.zip
    fi

    # 验证
    scene_count=$(find replica -name "*.glb" | wc -l)
    echo -e "${GREEN}   ✓ Replica 已下载: $scene_count 个场景${NC}"

    cd ..
}

download_hm3d_minival() {
    echo -e "\n${YELLOW}📥 下载 HM3D minival...${NC}"

    mkdir -p scene_datasets/hm3d
    cd scene_datasets/hm3d

    if [ -d "minival" ] && [ "$(ls -A minival)" ]; then
        echo -e "${YELLOW}   HM3D minival 已存在，跳过下载${NC}"
        cd ../..
        return
    fi

    echo -e "${BLUE}   HM3D 需要注册和访问权限${NC}"
    echo "   1. 访问: https://aihabitat.org/datasets/hm3d/"
    echo "   2. 注册并申请访问"
    echo "   3. 获取下载凭证"
    echo ""
    echo -n "   是否已有访问权限？(y/N): "
    read -r has_access

    if [[ ! "$has_access" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo -e "${YELLOW}   请先申请访问权限${NC}"
        cd ../..
        return
    fi

    echo -n "   输入用户名: "
    read -r username
    echo -n "   输入密码: "
    read -rs password
    echo ""

    # 使用 Habitat 下载工具
    python -m habitat_sim.utils.datasets_download \
        --username "$username" \
        --password "$password" \
        --uids hm3d_minival

    echo -e "${GREEN}   ✓ HM3D minival 已下载${NC}"

    cd ../..
}

download_hm3d_full() {
    echo -e "\n${YELLOW}📥 下载 HM3D 完整训练集...${NC}"
    echo -e "${RED}   ⚠️ 警告: 这是一个非常大的数据集（~2.5TB）${NC}"
    echo -e "${RED}   ⚠️ 下载可能需要数小时到数天${NC}"
    echo ""

    # 检查可用空间
    available_space=$(df -BG "$HABITAT_DATA_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
    echo "   可用存储空间: ${available_space}GB"

    if [ "$available_space" -lt 2600 ]; then
        echo -e "${RED}   ✗ 存储空间不足，需要至少 2.6TB${NC}"
        return
    fi

    echo -n "   确认下载完整 HM3D 数据集？(yes/NO): "
    read -r confirm

    if [[ ! "$confirm" == "yes" ]]; then
        echo "   取消下载"
        return
    fi

    mkdir -p scene_datasets/hm3d
    cd scene_datasets/hm3d

    echo -n "   输入用户名: "
    read -r username
    echo -n "   输入密码: "
    read -rs password
    echo ""

    # 下载完整训练集
    python -m habitat_sim.utils.datasets_download \
        --username "$username" \
        --password "$password" \
        --uids hm3d_train_v0.2

    echo -e "${GREEN}   ✓ HM3D 完整数据集已下载${NC}"

    cd ../..
}

download_objectnav() {
    echo -e "\n${YELLOW}📥 下载 ObjectNav 任务数据...${NC}"

    mkdir -p datasets/objectnav/hm3d/v1
    cd datasets/objectnav/hm3d/v1

    if [ -f "train/train.json.gz" ]; then
        echo -e "${YELLOW}   ObjectNav 数据已存在，跳过下载${NC}"
        cd ../../../..
        return
    fi

    # 下载 ObjectNav episodes
    echo "   下载 ObjectNav HM3D v1..."
    wget -c https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip

    echo "   解压..."
    unzip -q objectnav_hm3d_v1.zip
    rm objectnav_hm3d_v1.zip

    echo -e "${GREEN}   ✓ ObjectNav 数据已下载${NC}"

    # 显示内容
    echo "   包含的 splits:"
    ls -1

    cd ../../../..
}

# 根据选择执行
case $choice in
    1)
        download_replica
        ;;
    2)
        download_hm3d_minival
        ;;
    3)
        download_hm3d_full
        ;;
    4)
        download_objectnav
        ;;
    5)
        download_replica
        download_objectnav
        download_hm3d_minival
        ;;
    6)
        echo "退出"
        exit 0
        ;;
    *)
        echo -e "${RED}无效选择${NC}"
        exit 1
        ;;
esac

# 创建配置文件
echo -e "\n${YELLOW}📝 创建 Habitat 配置文件...${NC}"
mkdir -p "$HOME/.habitat"

cat > "$HOME/.habitat/habitat.yaml" << EOF
# Habitat 数据路径配置
# 自动生成于 $(date)

data_path: $HABITAT_DATA_DIR

scene_datasets:
  replica: $HABITAT_DATA_DIR/scene_datasets/replica
  hm3d: $HABITAT_DATA_DIR/scene_datasets/hm3d

datasets:
  objectnav:
    hm3d: $HABITAT_DATA_DIR/datasets/objectnav/hm3d/v1
  pointnav:
    gibson: $HABITAT_DATA_DIR/datasets/pointnav/gibson/v1
    mp3d: $HABITAT_DATA_DIR/datasets/pointnav/mp3d/v1
EOF

echo -e "${GREEN}   ✓ 配置文件已创建: ~/.habitat/habitat.yaml${NC}"

# 显示摘要
echo -e "\n=========================================="
echo -e "${GREEN}✅ 数据集下载完成！${NC}"
echo "=========================================="
echo ""
echo "📁 数据位置: $HABITAT_DATA_DIR"
echo ""
echo "📊 已下载数据集:"

if [ -d "scene_datasets/replica" ]; then
    replica_count=$(find scene_datasets/replica -name "*.glb" 2>/dev/null | wc -l)
    echo "   ✓ Replica: $replica_count 个场景"
fi

if [ -d "scene_datasets/hm3d/minival" ]; then
    echo "   ✓ HM3D minival"
fi

if [ -d "scene_datasets/hm3d/train" ]; then
    echo "   ✓ HM3D 完整训练集"
fi

if [ -d "datasets/objectnav" ]; then
    echo "   ✓ ObjectNav 任务数据"
fi

echo ""
echo "💾 存储使用:"
du -sh "$HABITAT_DATA_DIR" 2>/dev/null || echo "   计算中..."
echo ""
echo "🔍 验证数据集:"
echo "   python scripts/verify_installation.py --check-datasets"
echo ""
