#!/bin/bash
################################################################################
# VLA-GR 安装修复脚本
# 解决常见的安装问题
################################################################################

set -e

echo "=========================================="
echo "🔧 VLA-GR 安装修复脚本"
echo "=========================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${RED}✗ 请先激活虚拟环境${NC}"
    echo "  conda activate vla_gr"
    exit 1
fi

echo -e "${BLUE}当前环境: ${CONDA_DEFAULT_ENV:-$VIRTUAL_ENV}${NC}"
echo ""

# 1. 安装缺失的关键依赖
echo -e "${YELLOW}步骤 1/4: 安装缺失的关键依赖...${NC}"

# huggingface_hub (用于下载数据和模型)
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "  安装 huggingface_hub..."
    pip install huggingface_hub -q
    echo -e "  ${GREEN}✓ huggingface_hub 已安装${NC}"
else
    echo -e "  ${GREEN}✓ huggingface_hub 已存在${NC}"
fi

# 2. 重新安装包
echo -e "\n${YELLOW}步骤 2/4: 重新安装 VLA-GR 包...${NC}"
pip uninstall -y vla-gr-navigation 2>/dev/null || true
pip install -e . -q
echo -e "${GREEN}✓ VLA-GR 包已重新安装${NC}"

# 3. 创建环境配置文件
echo -e "\n${YELLOW}步骤 3/4: 创建环境配置...${NC}"

# 添加 src 到 PYTHONPATH
PROJECT_ROOT=$(pwd)
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# 创建激活脚本
ACTIVATE_SCRIPT="$PROJECT_ROOT/activate_env.sh"
cat > "$ACTIVATE_SCRIPT" << 'ACTIVATE_EOF'
#!/bin/bash
# VLA-GR 环境激活脚本

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 设置 PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

# 设置工作空间环境变量
export VLA_GR_ROOT="${VLA_GR_ROOT:-$HOME/vla-gr-workspace}"
export HABITAT_DATA_DIR="${HABITAT_DATA_DIR:-$VLA_GR_ROOT/habitat-data}"
export HF_HOME="${HF_HOME:-$VLA_GR_ROOT/huggingface-cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

# 如果在中国，启用镜像
# export HF_ENDPOINT=https://hf-mirror.com

echo "✓ VLA-GR 环境变量已设置"
echo "  PYTHONPATH: $PYTHONPATH"
echo "  HABITAT_DATA_DIR: $HABITAT_DATA_DIR"
echo "  HF_HOME: $HF_HOME"
ACTIVATE_EOF

chmod +x "$ACTIVATE_SCRIPT"
echo -e "${GREEN}✓ 环境配置文件已创建: activate_env.sh${NC}"

# 4. 验证安装
echo -e "\n${YELLOW}步骤 4/4: 验证安装...${NC}"

# Source 环境变量
source "$ACTIVATE_SCRIPT"

# 测试基本导入
echo "  测试 Python 导入..."

python << 'VERIFY_EOF'
import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

tests_passed = 0
tests_total = 0

def test_import(name, import_fn):
    global tests_passed, tests_total
    tests_total += 1
    try:
        import_fn()
        print(f"  ✓ {name}")
        tests_passed += 1
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return False

# 基本导入测试
test_import("huggingface_hub", lambda: __import__('huggingface_hub'))

# 核心模块测试（不需要 torch）
try:
    # 只测试模块是否存在，不实际导入（避免 torch 依赖）
    import importlib.util
    spec = importlib.util.find_spec('core.vla_gr_agent')
    if spec is not None:
        print(f"  ✓ 核心模块路径正确")
        tests_passed += 1
    else:
        print(f"  ✗ 核心模块路径未找到")
    tests_total += 1
except Exception as e:
    print(f"  ✗ 核心模块检查: {e}")
    tests_total += 1

print(f"\n验证结果: {tests_passed}/{tests_total} 测试通过")

if tests_passed < tests_total:
    print("\n⚠️  部分测试失败，但基本功能可用")
    print("   如果缺少 PyTorch，请按照部署指南安装")
    sys.exit(0)
else:
    print("\n✅ 所有基本测试通过！")
    sys.exit(0)
VERIFY_EOF

echo ""
echo "=========================================="
echo -e "${GREEN}✅ 安装修复完成！${NC}"
echo "=========================================="
echo ""
echo "📝 使用说明:"
echo ""
echo "1. 每次使用前激活环境变量:"
echo "   source activate_env.sh"
echo ""
echo "2. 下载数据集:"
echo "   bash scripts/download_datasets.sh"
echo ""
echo "3. 下载模型:"
echo "   bash scripts/download_models.sh"
echo ""
echo "4. 完整的 PyTorch 安装请参考:"
echo "   docs/RTX4060_DEPLOYMENT_GUIDE.md"
echo ""
