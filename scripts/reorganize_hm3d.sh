#!/bin/bash
################################################################################
# HM3D 数据集目录结构整理脚本
# 将分散的文件合并到统一的场景目录中
################################################################################

set -e

echo "=========================================="
echo "📁 HM3D 数据集目录结构整理"
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

# 检查是否存在原始解压目录
if [ ! -d "$HM3D_DIR/minival" ]; then
    echo -e "${RED}✗ 未找到 minival 目录${NC}"
    echo "请先运行: bash scripts/download_hm3d_manual.sh"
    exit 1
fi

cd "$HM3D_DIR/minival"

# ============================================================================
# 整理 MINIVAL 数据集
# ============================================================================

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}🔧 整理 MINIVAL 数据集${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 统计场景数量
total_scenes=$(ls -d hm3d-minival-glb-v0.2/*/  2>/dev/null | wc -l)
echo -e "${BLUE}找到 $total_scenes 个场景${NC}"
echo ""

processed=0

# 遍历所有场景
for scene_dir in hm3d-minival-glb-v0.2/*/; do
    scene_id=$(basename "$scene_dir")

    echo -e "${BLUE}[$(($processed + 1))/$total_scenes] 处理场景: $scene_id${NC}"

    # 创建场景目录（如果不存在）
    mkdir -p "$scene_id"

    # 1. 复制 GLB 文件
    if [ -f "hm3d-minival-glb-v0.2/$scene_id/$scene_id.glb" ]; then
        cp -v "hm3d-minival-glb-v0.2/$scene_id/$scene_id.glb" "$scene_id/"
        echo -e "  ${GREEN}✓ GLB 文件${NC}"
    else
        echo -e "  ${YELLOW}⊙ GLB 文件不存在${NC}"
    fi

    # 2. 复制 Habitat 文件 (basis.glb 和 navmesh)
    if [ -f "hm3d-minival-habitat-v0.2/$scene_id/$scene_id.basis.glb" ]; then
        cp -v "hm3d-minival-habitat-v0.2/$scene_id/$scene_id.basis.glb" "$scene_id/"
        echo -e "  ${GREEN}✓ Basis GLB 文件${NC}"
    else
        echo -e "  ${YELLOW}⊙ Basis GLB 文件不存在${NC}"
    fi

    if [ -f "hm3d-minival-habitat-v0.2/$scene_id/$scene_id.basis.navmesh" ]; then
        cp -v "hm3d-minival-habitat-v0.2/$scene_id/$scene_id.basis.navmesh" "$scene_id/"
        echo -e "  ${GREEN}✓ Navmesh 文件${NC}"
    else
        echo -e "  ${YELLOW}⊙ Navmesh 文件不存在${NC}"
    fi

    # 3. 复制语义标注文件（如果存在）
    if [ -d "hm3d-minival-semantic-annots-v0.2/$scene_id" ]; then
        if [ -f "hm3d-minival-semantic-annots-v0.2/$scene_id/$scene_id.semantic.glb" ]; then
            cp -v "hm3d-minival-semantic-annots-v0.2/$scene_id/$scene_id.semantic.glb" "$scene_id/"
            echo -e "  ${GREEN}✓ Semantic GLB 文件${NC}"
        fi

        if [ -f "hm3d-minival-semantic-annots-v0.2/$scene_id/$scene_id.semantic.txt" ]; then
            cp -v "hm3d-minival-semantic-annots-v0.2/$scene_id/$scene_id.semantic.txt" "$scene_id/"
            echo -e "  ${GREEN}✓ Semantic TXT 文件${NC}"
        fi
    else
        echo -e "  ${YELLOW}⊙ 语义标注不存在（部分场景没有）${NC}"
    fi

    echo ""
    processed=$(($processed + 1))
done

# 4. 复制配置文件到父目录
echo -e "${BLUE}复制配置文件...${NC}"
if [ -d "hm3d-minival-semantic-configs-v0.2" ]; then
    cp -v hm3d-minival-semantic-configs-v0.2/*.json ./
    echo -e "${GREEN}✓ 配置文件已复制${NC}"
fi
echo ""

# ============================================================================
# 清理原始解压目录
# ============================================================================

echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}🧹 清理原始解压目录${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

echo -n "是否删除原始解压目录以节省空间？(y/N): "
read -r clean_dirs

if [[ "$clean_dirs" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo -e "${BLUE}删除原始目录...${NC}"
    rm -rf hm3d-minival-glb-v0.2/
    rm -rf hm3d-minival-habitat-v0.2/
    rm -rf hm3d-minival-semantic-annots-v0.2/
    rm -rf hm3d-minival-semantic-configs-v0.2/
    echo -e "${GREEN}✓ 清理完成${NC}"
else
    echo -e "${YELLOW}保留原始目录${NC}"
fi

echo ""

# ============================================================================
# 整理 VAL 数据集（如果存在）
# ============================================================================

if [ -d "$HM3D_DIR/val/hm3d-val-glb-v0.2" ]; then
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}🔧 整理 VAL 数据集${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""

    cd "$HM3D_DIR/val"

    total_scenes=$(ls -d hm3d-val-glb-v0.2/*/  2>/dev/null | wc -l)
    echo -e "${BLUE}找到 $total_scenes 个场景${NC}"
    echo ""

    processed=0

    for scene_dir in hm3d-val-glb-v0.2/*/; do
        scene_id=$(basename "$scene_dir")

        echo -e "${BLUE}[$(($processed + 1))/$total_scenes] 处理场景: $scene_id${NC}"

        mkdir -p "$scene_id"

        # GLB
        if [ -f "hm3d-val-glb-v0.2/$scene_id/$scene_id.glb" ]; then
            cp -v "hm3d-val-glb-v0.2/$scene_id/$scene_id.glb" "$scene_id/"
        fi

        # Habitat files
        if [ -f "hm3d-val-habitat-v0.2/$scene_id/$scene_id.basis.glb" ]; then
            cp -v "hm3d-val-habitat-v0.2/$scene_id/$scene_id.basis.glb" "$scene_id/"
        fi

        if [ -f "hm3d-val-habitat-v0.2/$scene_id/$scene_id.basis.navmesh" ]; then
            cp -v "hm3d-val-habitat-v0.2/$scene_id/$scene_id.basis.navmesh" "$scene_id/"
        fi

        # Semantic files
        if [ -d "hm3d-val-semantic-annots-v0.2/$scene_id" ]; then
            if [ -f "hm3d-val-semantic-annots-v0.2/$scene_id/$scene_id.semantic.glb" ]; then
                cp -v "hm3d-val-semantic-annots-v0.2/$scene_id/$scene_id.semantic.glb" "$scene_id/"
            fi

            if [ -f "hm3d-val-semantic-annots-v0.2/$scene_id/$scene_id.semantic.txt" ]; then
                cp -v "hm3d-val-semantic-annots-v0.2/$scene_id/$scene_id.semantic.txt" "$scene_id/"
            fi
        fi

        echo ""
        processed=$(($processed + 1))
    done

    # Copy config files
    if [ -d "hm3d-val-semantic-configs-v0.2" ]; then
        cp -v hm3d-val-semantic-configs-v0.2/*.json ./
    fi

    # Clean up
    echo -n "是否删除 val 原始解压目录？(y/N): "
    read -r clean_val

    if [[ "$clean_val" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        rm -rf hm3d-val-glb-v0.2/
        rm -rf hm3d-val-habitat-v0.2/
        rm -rf hm3d-val-semantic-annots-v0.2/
        rm -rf hm3d-val-semantic-configs-v0.2/
        echo -e "${GREEN}✓ Val 目录清理完成${NC}"
    fi
fi

# ============================================================================
# 验证整理结果
# ============================================================================

echo ""
echo "=========================================="
echo -e "${GREEN}✅ 目录结构整理完成！${NC}"
echo "=========================================="
echo ""

echo "📁 整理后的结构:"
echo ""
tree -L 2 "$HM3D_DIR" 2>/dev/null || {
    echo "Minival 场景:"
    ls -1 "$HM3D_DIR/minival" | grep -E "^[0-9]" | head -5
    echo "..."

    if [ -d "$HM3D_DIR/val" ]; then
        echo ""
        echo "Val 场景:"
        ls -1 "$HM3D_DIR/val" | grep -E "^[0-9]" | head -5
        echo "..."
    fi
}

echo ""
echo "📊 统计信息:"
minival_count=$(find "$HM3D_DIR/minival" -maxdepth 1 -type d -name "00*" 2>/dev/null | wc -l)
echo "  Minival 场景数: $minival_count"
echo "  Minival navmesh 文件: $(find "$HM3D_DIR/minival" -name "*.navmesh" 2>/dev/null | wc -l)"
echo "  Minival glb 文件: $(find "$HM3D_DIR/minival" -name "*.glb" 2>/dev/null | wc -l)"

if [ -d "$HM3D_DIR/val" ]; then
    val_count=$(find "$HM3D_DIR/val" -maxdepth 1 -type d -name "00*" 2>/dev/null | wc -l)
    echo ""
    echo "  Val 场景数: $val_count"
    echo "  Val navmesh 文件: $(find "$HM3D_DIR/val" -name "*.navmesh" 2>/dev/null | wc -l)"
    echo "  Val glb 文件: $(find "$HM3D_DIR/val" -name "*.glb" 2>/dev/null | wc -l)"
fi

echo ""
echo "💾 存储使用:"
du -sh "$HM3D_DIR"/* 2>/dev/null

echo ""
echo "✅ 正确的目录结构示例:"
echo ""
echo "  minival/"
echo "  ├── 00800-TEEsavR23oF/"
echo "  │   ├── TEEsavR23oF.glb                 (3D 场景)"
echo "  │   ├── TEEsavR23oF.basis.glb           (压缩纹理)"
echo "  │   ├── TEEsavR23oF.basis.navmesh       (导航网格 ✅)"
echo "  │   ├── TEEsavR23oF.semantic.glb        (语义模型)"
echo "  │   └── TEEsavR23oF.semantic.txt        (语义标签)"
echo "  ├── 00801-HaxA7YrQdEC/"
echo "  │   └── ..."
echo "  └── hm3d_annotated_basis.scene_dataset_config.json"
echo ""

echo -e "${BLUE}下一步:${NC}"
echo "  1. 验证 Habitat 加载: python scripts/verify_installation.py --check-datasets"
echo "  2. 测试场景加载:"
echo "     python -c \"import habitat_sim; print('Habitat OK')\""
echo ""
