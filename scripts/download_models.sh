#!/bin/bash
################################################################################
# Hugging Face 模型下载脚本
# 用途：预下载所有需要的 HuggingFace 模型
# 使用：bash scripts/download_models.sh
################################################################################

set -e

echo "=========================================="
echo "🤗 Hugging Face 模型下载脚本"
echo "=========================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 检查虚拟环境
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${RED}✗ 请先激活虚拟环境${NC}"
    exit 1
fi

# 设置缓存目录
if [ -z "$HF_HOME" ]; then
    HF_HOME="$HOME/vla-gr-workspace/huggingface-cache"
    export HF_HOME
    export TRANSFORMERS_CACHE="$HF_HOME/transformers"
    echo -e "${YELLOW}⚠ HF_HOME 未设置，使用默认值: $HF_HOME${NC}"
fi

mkdir -p "$HF_HOME"
echo -e "${BLUE}缓存目录: $HF_HOME${NC}"

# 检查网络连接
echo -e "\n${YELLOW}检查网络连接...${NC}"
if ! curl -s --head https://huggingface.co | head -n 1 | grep "200 OK" > /dev/null; then
    echo -e "${RED}✗ 无法连接到 huggingface.co${NC}"
    echo "  提示: 如果在中国，考虑使用镜像站"
    echo "  export HF_ENDPOINT=https://hf-mirror.com"
    exit 1
fi
echo -e "${GREEN}✓ 网络连接正常${NC}"

# 菜单
echo -e "\n${YELLOW}请选择要下载的模型:${NC}"
echo "   1) Microsoft Phi-2（语言模型，~5.5GB）"
echo "   2) OpenAI CLIP（视觉-语言，~600MB）"
echo "   3) BERT base（后备语言模型，~440MB）"
echo "   4) DINOv2（视觉编码器，~340MB）"
echo "   5) 全部下载"
echo "   6) 退出"
echo -n "选择 (1-6): "
read -r choice

download_phi2() {
    echo -e "\n${YELLOW}📥 下载 Microsoft Phi-2...${NC}"
    echo "   模型大小: ~5.5GB"
    echo "   参数量: 2.7B"

    python << 'EOF'
from transformers import AutoModel, AutoTokenizer
import os

model_name = "microsoft/phi-2"
cache_dir = os.environ.get("HF_HOME", "~/.cache/huggingface")

print(f"   下载到: {cache_dir}")
print("   正在下载 tokenizer...")

try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    print("   ✓ Tokenizer 已下载")

    print("   正在下载模型（这可能需要几分钟）...")
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir
    )
    print("   ✓ Phi-2 模型已下载")

    # 显示模型信息
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"   参数量: {param_count:.2f}B")

except Exception as e:
    print(f"   ✗ 下载失败: {e}")
    exit(1)
EOF

    echo -e "${GREEN}✓ Phi-2 下载完成${NC}"
}

download_clip() {
    echo -e "\n${YELLOW}📥 下载 OpenAI CLIP...${NC}"
    echo "   模型大小: ~600MB"

    python << 'EOF'
from transformers import CLIPModel, CLIPProcessor
import os

model_name = "openai/clip-vit-base-patch32"
cache_dir = os.environ.get("HF_HOME", "~/.cache/huggingface")

print("   正在下载 CLIP processor...")
try:
    processor = CLIPProcessor.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    print("   ✓ Processor 已下载")

    print("   正在下载 CLIP 模型...")
    model = CLIPModel.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    print("   ✓ CLIP 模型已下载")

except Exception as e:
    print(f"   ✗ 下载失败: {e}")
    exit(1)
EOF

    echo -e "${GREEN}✓ CLIP 下载完成${NC}"
}

download_bert() {
    echo -e "\n${YELLOW}📥 下载 BERT base uncased...${NC}"
    echo "   模型大小: ~440MB"

    python << 'EOF'
from transformers import BertModel, BertTokenizer
import os

model_name = "bert-base-uncased"
cache_dir = os.environ.get("HF_HOME", "~/.cache/huggingface")

print("   正在下载 BERT tokenizer...")
try:
    tokenizer = BertTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    print("   ✓ Tokenizer 已下载")

    print("   正在下载 BERT 模型...")
    model = BertModel.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    print("   ✓ BERT 模型已下载")

except Exception as e:
    print(f"   ✗ 下载失败: {e}")
    exit(1)
EOF

    echo -e "${GREEN}✓ BERT 下载完成${NC}"
}

download_dinov2() {
    echo -e "\n${YELLOW}📥 下载 DINOv2...${NC}"
    echo "   模型大小: ~340MB"

    python << 'EOF'
from transformers import AutoModel
import os

model_name = "facebook/dinov2-base"
cache_dir = os.environ.get("HF_HOME", "~/.cache/huggingface")

print("   正在下载 DINOv2 模型...")
try:
    model = AutoModel.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    print("   ✓ DINOv2 模型已下载")

except Exception as e:
    print(f"   ✗ 下载失败: {e}")
    exit(1)
EOF

    echo -e "${GREEN}✓ DINOv2 下载完成${NC}"
}

# 根据选择执行
case $choice in
    1)
        download_phi2
        ;;
    2)
        download_clip
        ;;
    3)
        download_bert
        ;;
    4)
        download_dinov2
        ;;
    5)
        echo -e "${BLUE}下载所有模型...${NC}"
        download_phi2
        download_clip
        download_bert
        download_dinov2
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

# 显示摘要
echo -e "\n=========================================="
echo -e "${GREEN}✅ 模型下载完成！${NC}"
echo "=========================================="
echo ""
echo "📁 缓存位置: $HF_HOME"
echo ""
echo "📊 已下载模型:"

# 列出已下载的模型
if [ -d "$HF_HOME" ]; then
    echo ""
    for model_dir in "$HF_HOME"/models--*; do
        if [ -d "$model_dir" ]; then
            model_name=$(basename "$model_dir" | sed 's/models--//' | sed 's/--/\//')
            size=$(du -sh "$model_dir" 2>/dev/null | cut -f1)
            echo "   ✓ $model_name ($size)"
        fi
    done
fi

echo ""
echo "💾 总存储使用:"
du -sh "$HF_HOME" 2>/dev/null || echo "   计算中..."

echo ""
echo "🔍 验证模型加载:"
echo "   python scripts/verify_installation.py --check-models"
echo ""
echo "💡 提示:"
echo "   - 模型会自动从缓存加载，无需重复下载"
echo "   - 可以在 config.yaml 中设置 local_files_only: true"
echo "   - 使用镜像站: export HF_ENDPOINT=https://hf-mirror.com"
echo ""
