#!/bin/bash
################################################################################
# HM3D 数据集手动下载和解压脚本
# 用于下载 minival 和 val 数据集
################################################################################

set -e

echo "=========================================="
echo "📥 HM3D v0.2 手动下载脚本"
echo "=========================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 设置数据目录
HABITAT_DATA_DIR="${HABITAT_DATA_DIR:-$HOME/vla-gr-workspace/habitat-data}"
HM3D_DIR="$HABITAT_DATA_DIR/scene_datasets/hm3d"

echo -e "${BLUE}数据目录: $HM3D_DIR${NC}"
echo ""

# 创建目录
mkdir -p "$HM3D_DIR"/{minival,val}
cd "$HM3D_DIR"

# ============================================================================
# MINIVAL 数据集下载
# ============================================================================

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}📦 下载 MINIVAL 数据集 (~1.1GB)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 1. Minival Habitat (必需)
echo -e "${BLUE}[1/4] 下载 minival-habitat (390M)${NC}"
if [ ! -f "hm3d-minival-habitat-v0.2.tar" ]; then
    wget -c https://api.matterport.com/resources/habitat/hm3d-minival-habitat-v0.2.tar
    echo -e "${GREEN}✓ 下载完成${NC}"
else
    echo -e "${YELLOW}⊙ 文件已存在，跳过下载${NC}"
fi

echo -e "${BLUE}  解压到 minival/...${NC}"
tar -xf hm3d-minival-habitat-v0.2.tar -C minival/
echo -e "${GREEN}✓ 解压完成${NC}"
echo ""

# 2. Minival GLB (推荐)
echo -e "${BLUE}[2/4] 下载 minival-glb (464M)${NC}"
if [ ! -f "hm3d-minival-glb-v0.2.tar" ]; then
    wget -c https://api.matterport.com/resources/habitat/hm3d-minival-glb-v0.2.tar
    echo -e "${GREEN}✓ 下载完成${NC}"
else
    echo -e "${YELLOW}⊙ 文件已存在，跳过下载${NC}"
fi

echo -e "${BLUE}  解压到 minival/...${NC}"
tar -xf hm3d-minival-glb-v0.2.tar -C minival/
echo -e "${GREEN}✓ 解压完成${NC}"
echo ""

# 3. Minival Semantic Annotations (推荐)
echo -e "${BLUE}[3/4] 下载 minival-semantic-annots (240.6M)${NC}"
if [ ! -f "hm3d-minival-semantic-annots-v0.2.tar" ]; then
    wget -c https://api.matterport.com/resources/habitat/hm3d-minival-semantic-annots-v0.2.tar
    echo -e "${GREEN}✓ 下载完成${NC}"
else
    echo -e "${YELLOW}⊙ 文件已存在，跳过下载${NC}"
fi

echo -e "${BLUE}  解压到 minival/...${NC}"
tar -xf hm3d-minival-semantic-annots-v0.2.tar -C minival/
echo -e "${GREEN}✓ 解压完成${NC}"
echo ""

# 4. Minival Semantic Configs (必需)
echo -e "${BLUE}[4/4] 下载 minival-semantic-configs (30K)${NC}"
if [ ! -f "hm3d-minival-semantic-configs-v0.2.tar" ]; then
    wget -c https://api.matterport.com/resources/habitat/hm3d-minival-semantic-configs-v0.2.tar
    echo -e "${GREEN}✓ 下载完成${NC}"
else
    echo -e "${YELLOW}⊙ 文件已存在，跳过下载${NC}"
fi

echo -e "${BLUE}  解压...${NC}"
tar -xf hm3d-minival-semantic-configs-v0.2.tar
echo -e "${GREEN}✓ 解压完成${NC}"
echo ""

echo -e "${GREEN}✅ MINIVAL 数据集下载完成！${NC}"
echo ""

# ============================================================================
# VAL 数据集下载 (可选)
# ============================================================================

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}📦 下载 VAL 数据集 (~9.3GB)${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -n "是否下载 VAL 数据集？(y/N): "
read -r download_val

if [[ ! "$download_val" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${YELLOW}跳过 VAL 数据集下载${NC}"
else
    # 1. Val Habitat (必需)
    echo -e "${BLUE}[1/4] 下载 val-habitat (3.3G)${NC}"
    if [ ! -f "hm3d-val-habitat-v0.2.tar" ]; then
        wget -c https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar
        echo -e "${GREEN}✓ 下载完成${NC}"
    else
        echo -e "${YELLOW}⊙ 文件已存在，跳过下载${NC}"
    fi

    echo -e "${BLUE}  解压到 val/...${NC}"
    tar -xf hm3d-val-habitat-v0.2.tar -C val/
    echo -e "${GREEN}✓ 解压完成${NC}"
    echo ""

    # 2. Val GLB (推荐)
    echo -e "${BLUE}[2/4] 下载 val-glb (4G)${NC}"
    if [ ! -f "hm3d-val-glb-v0.2.tar" ]; then
        wget -c https://api.matterport.com/resources/habitat/hm3d-val-glb-v0.2.tar
        echo -e "${GREEN}✓ 下载完成${NC}"
    else
        echo -e "${YELLOW}⊙ 文件已存在，跳过下载${NC}"
    fi

    echo -e "${BLUE}  解压到 val/...${NC}"
    tar -xf hm3d-val-glb-v0.2.tar -C val/
    echo -e "${GREEN}✓ 解压完成${NC}"
    echo ""

    # 3. Val Semantic Annotations (推荐)
    echo -e "${BLUE}[3/4] 下载 val-semantic-annots (2.0G)${NC}"
    if [ ! -f "hm3d-val-semantic-annots-v0.2.tar" ]; then
        wget -c https://api.matterport.com/resources/habitat/hm3d-val-semantic-annots-v0.2.tar
        echo -e "${GREEN}✓ 下载完成${NC}"
    else
        echo -e "${YELLOW}⊙ 文件已存在，跳过下载${NC}"
    fi

    echo -e "${BLUE}  解压到 val/...${NC}"
    tar -xf hm3d-val-semantic-annots-v0.2.tar -C val/
    echo -e "${GREEN}✓ 解压完成${NC}"
    echo ""

    # 4. Val Semantic Configs (必需)
    echo -e "${BLUE}[4/4] 下载 val-semantic-configs (40K)${NC}"
    if [ ! -f "hm3d-val-semantic-configs-v0.2.tar" ]; then
        wget -c https://api.matterport.com/resources/habitat/hm3d-val-semantic-configs-v0.2.tar
        echo -e "${GREEN}✓ 下载完成${NC}"
    else
        echo -e "${YELLOW}⊙ 文件已存在，跳过下载${NC}"
    fi

    echo -e "${BLUE}  解压...${NC}"
    tar -xf hm3d-val-semantic-configs-v0.2.tar
    echo -e "${GREEN}✓ 解压完成${NC}"
    echo ""

    echo -e "${GREEN}✅ VAL 数据集下载完成！${NC}"
fi

# ============================================================================
# 清理和验证
# ============================================================================

echo ""
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}🧹 清理和验证${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 询问是否删除 tar 文件
echo -n "是否删除下载的 .tar 文件以节省空间？(y/N): "
read -r clean_tars

if [[ "$clean_tars" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${BLUE}删除 tar 文件...${NC}"
    rm -f *.tar
    echo -e "${GREEN}✓ 清理完成${NC}"
else
    echo -e "${YELLOW}保留 tar 文件${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✅ HM3D 数据集安装完成！${NC}"
echo "=========================================="
echo ""

# 显示目录结构
echo "📁 数据集结构:"
echo ""
tree -L 2 "$HM3D_DIR" 2>/dev/null || find "$HM3D_DIR" -maxdepth 2 -type d

echo ""
echo "📊 存储使用:"
du -sh "$HM3D_DIR"/* 2>/dev/null

echo ""
echo "💾 总大小:"
du -sh "$HM3D_DIR" 2>/dev/null

echo ""
echo "📝 验证文件:"
echo "  minival 场景数: $(find "$HM3D_DIR/minival" -name "*.glb" 2>/dev/null | wc -l)"
if [ -d "$HM3D_DIR/val" ]; then
    echo "  val 场景数: $(find "$HM3D_DIR/val" -name "*.glb" 2>/dev/null | wc -l)"
fi

echo ""
echo -e "${BLUE}下一步:${NC}"
echo "  1. 验证安装: python scripts/verify_installation.py --check-datasets"
echo "  2. 查看配置: cat config.yaml"
echo ""
