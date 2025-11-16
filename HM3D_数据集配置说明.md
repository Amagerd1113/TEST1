# HM3D 数据集配置说明

> 📅 创建日期: 2025-11-14
> 🎯 适用版本: HM3D v0.2

---

## 📦 8个文件的具体存放位置

### Minival 数据集 (4个文件, ~1.1GB)

| 文件名 | 大小 | 解压后位置 | 说明 |
|--------|------|-----------|------|
| `hm3d-minival-habitat-v0.2.tar` | 390M | `~/vla-gr-workspace/habitat-data/scene_datasets/hm3d/minival/` | ✅ 必需：导航网格文件 |
| `hm3d-minival-glb-v0.2.tar` | 464M | `~/vla-gr-workspace/habitat-data/scene_datasets/hm3d/minival/` | ⭐ 推荐：3D场景模型 |
| `hm3d-minival-semantic-annots-v0.2.tar` | 240.6M | `~/vla-gr-workspace/habitat-data/scene_datasets/hm3d/minival/` | ⭐ 推荐：语义标注 |
| `hm3d-minival-semantic-configs-v0.2.tar` | 30K | `~/vla-gr-workspace/habitat-data/scene_datasets/hm3d/` | ✅ 必需：配置文件 |

### Val 数据集 (4个文件, ~9.3GB)

| 文件名 | 大小 | 解压后位置 | 说明 |
|--------|------|-----------|------|
| `hm3d-val-habitat-v0.2.tar` | 3.3G | `~/vla-gr-workspace/habitat-data/scene_datasets/hm3d/val/` | ✅ 必需：导航网格文件 |
| `hm3d-val-glb-v0.2.tar` | 4G | `~/vla-gr-workspace/habitat-data/scene_datasets/hm3d/val/` | ⭐ 推荐：3D场景模型 |
| `hm3d-val-semantic-annots-v0.2.tar` | 2.0G | `~/vla-gr-workspace/habitat-data/scene_datasets/hm3d/val/` | ⭐ 推荐：语义标注 |
| `hm3d-val-semantic-configs-v0.2.tar` | 40K | `~/vla-gr-workspace/habitat-data/scene_datasets/hm3d/` | ✅ 必需：配置文件 |

---

## 📁 目标目录结构（整理后）

**重要**：解压后的文件需要重新组织！每个 tar 文件会创建自己的子目录，需要运行整理脚本。

### 整理后的正确结构：

```bash
~/vla-gr-workspace/habitat-data/
└── scene_datasets/
    └── hm3d/
        ├── minival/                                    # Minival 数据集目录
        │   ├── 00800-TEEsavR23oF/                     # 示例场景1
        │   │   ├── TEEsavR23oF.basis.glb              # 压缩纹理
        │   │   ├── TEEsavR23oF.glb                    # 3D 场景模型
        │   │   ├── TEEsavR23oF.basis.navmesh          # 导航网格 ✅
        │   │   ├── TEEsavR23oF.semantic.glb           # 语义几何
        │   │   └── TEEsavR23oF.semantic.txt           # 语义标签
        │   ├── 00801-HaxA7YrQdEC/                     # 示例场景2
        │   ├── 00802-wcojb4TFT35/                     # 示例场景3
        │   ├── ...                                     # 更多场景
        │   └── hm3d_annotated_basis.scene_dataset_config.json  # 配置文件
        │
        └── val/                                        # Val 数据集目录
            ├── 00009-vLpv2VX547B/                     # Val场景1
            ├── 00153-9ks21UvVQjL/                     # Val场景2
            ├── ...                                     # 更多场景
            └── hm3d_annotated_basis.scene_dataset_config.json  # 配置文件
```

### 解压后的原始结构（需要整理）：

```bash
minival/
├── hm3d-minival-glb-v0.2/              # ❌ 需要整理
│   ├── 00800-TEEsavR23oF/
│   │   └── TEEsavR23oF.glb
│   └── ...
├── hm3d-minival-habitat-v0.2/          # ❌ 需要整理
│   ├── 00800-TEEsavR23oF/
│   │   ├── TEEsavR23oF.basis.glb
│   │   └── TEEsavR23oF.basis.navmesh
│   └── ...
├── hm3d-minival-semantic-annots-v0.2/  # ❌ 需要整理
│   ├── 00800-TEEsavR23oF/
│   │   ├── TEEsavR23oF.semantic.glb
│   │   └── TEEsavR23oF.semantic.txt
│   └── ...
└── hm3d-minival-semantic-configs-v0.2/ # ❌ 需要整理
    └── *.scene_dataset_config.json
```

### 文件类型说明

- **`.navmesh`**: 导航网格文件，Habitat 模拟器必需
- **`.glb`**: 3D 场景模型，用于渲染和可视化
- **`.basis.glb`**: 压缩版纹理，加载更快
- **`.semantic.glb`**: 语义分割的 3D 模型
- **`.semantic.txt`**: 语义类别标签映射
- **`.scene_dataset_config.json`**: Habitat 场景数据集配置

---

## 🚀 快速下载和安装

### 方法1: 使用下载脚本（推荐）

```bash
cd ~/vla-gr-workspace/VLA-GR

# 步骤1: 下载数据集
bash scripts/download_hm3d_manual.sh
# 脚本会自动：
# 1. 下载所有 minival 文件
# 2. 询问是否下载 val 数据集
# 3. 自动解压到正确位置
# 4. 询问是否删除 .tar 文件节省空间

# 步骤2: ⭐ 重新整理目录结构（重要！）
bash scripts/reorganize_hm3d.sh
# 脚本会：
# 1. 将分散的文件合并到统一的场景目录
# 2. 复制配置文件到正确位置
# 3. 询问是否删除原始解压目录
# 4. 验证整理结果
```

### 方法2: 手动下载

```bash
# 设置数据目录
export HABITAT_DATA_DIR="$HOME/vla-gr-workspace/habitat-data"
mkdir -p "$HABITAT_DATA_DIR/scene_datasets/hm3d"/{minival,val}
cd "$HABITAT_DATA_DIR/scene_datasets/hm3d"

# 下载 Minival
wget https://api.matterport.com/resources/habitat/hm3d-minival-habitat-v0.2.tar
wget https://api.matterport.com/resources/habitat/hm3d-minival-glb-v0.2.tar
wget https://api.matterport.com/resources/habitat/hm3d-minival-semantic-annots-v0.2.tar
wget https://api.matterport.com/resources/habitat/hm3d-minival-semantic-configs-v0.2.tar

# 解压 Minival
tar -xf hm3d-minival-habitat-v0.2.tar -C minival/
tar -xf hm3d-minival-glb-v0.2.tar -C minival/
tar -xf hm3d-minival-semantic-annots-v0.2.tar -C minival/
tar -xf hm3d-minival-semantic-configs-v0.2.tar

# （可选）下载和解压 Val
wget https://api.matterport.com/resources/habitat/hm3d-val-habitat-v0.2.tar
wget https://api.matterport.com/resources/habitat/hm3d-val-glb-v0.2.tar
wget https://api.matterport.com/resources/habitat/hm3d-val-semantic-annots-v0.2.tar
wget https://api.matterport.com/resources/habitat/hm3d-val-semantic-configs-v0.2.tar

tar -xf hm3d-val-habitat-v0.2.tar -C val/
tar -xf hm3d-val-glb-v0.2.tar -C val/
tar -xf hm3d-val-semantic-annots-v0.2.tar -C val/
tar -xf hm3d-val-semantic-configs-v0.2.tar

# 清理 tar 文件（可选）
rm *.tar
```

---

## ⚙️ Config 配置文件更新

### 主配置文件 (`config.yaml`)

已自动更新为：

```yaml
environment:
  habitat:
    scene_dataset: "hm3d"
    split: "minival"  # 使用 minival 进行评估

    # 数据路径（自动从环境变量读取）
    data_path: "${HABITAT_DATA_DIR:~/vla-gr-workspace/habitat-data}"
    scenes_dir: "${HABITAT_DATA_DIR:~/vla-gr-workspace/habitat-data}/scene_datasets"

    # HM3D 数据集路径
    hm3d:
      minival_path: "${HABITAT_DATA_DIR:~/vla-gr-workspace/habitat-data}/scene_datasets/hm3d/minival"
      val_path: "${HABITAT_DATA_DIR:~/vla-gr-workspace/habitat-data}/scene_datasets/hm3d/val"
      train_path: "${HABITAT_DATA_DIR:~/vla-gr-workspace/habitat-data}/scene_datasets/hm3d/train"
      semantic_annotations: true
      use_semantic_sensor: true

    # Replica 数据集路径
    replica:
      path: "${HABITAT_DATA_DIR:~/vla-gr-workspace/habitat-data}/scene_datasets/replica"
```

### RTX 4060 配置文件 (`config_rtx4060.yaml`)

同样已更新，默认使用 `minival` 数据集以节省资源。

---

## 🔍 验证安装

### 1. 检查文件结构

```bash
# 查看目录结构
tree -L 2 ~/vla-gr-workspace/habitat-data/scene_datasets/hm3d

# 或使用 find
find ~/vla-gr-workspace/habitat-data/scene_datasets/hm3d -type d -maxdepth 2
```

### 2. 统计场景数量

```bash
# Minival 场景数
find ~/vla-gr-workspace/habitat-data/scene_datasets/hm3d/minival -name "*.glb" | wc -l

# Val 场景数（如果已下载）
find ~/vla-gr-workspace/habitat-data/scene_datasets/hm3d/val -name "*.glb" | wc -l
```

### 3. 检查存储空间

```bash
# 查看各数据集大小
du -sh ~/vla-gr-workspace/habitat-data/scene_datasets/hm3d/*

# 查看总大小
du -sh ~/vla-gr-workspace/habitat-data/
```

### 4. 运行验证脚本

```bash
cd ~/vla-gr-workspace/VLA-GR

# 验证数据集
python scripts/verify_installation.py --check-datasets

# 测试 Habitat 加载
python -c "
import habitat_sim
import os

data_dir = os.path.expanduser('~/vla-gr-workspace/habitat-data')
minival_dir = os.path.join(data_dir, 'scene_datasets/hm3d/minival')

# 查找第一个场景
import glob
scenes = glob.glob(os.path.join(minival_dir, '**/*.glb'), recursive=True)
print(f'找到 {len(scenes)} 个场景')
if scenes:
    print(f'第一个场景: {scenes[0]}')
"
```

---

## 📊 存储空间需求总结

| 数据集 | 压缩包大小 | 解压后大小 | 推荐 |
|--------|-----------|-----------|------|
| **Minival** | ~1.1GB | ~1.5GB | ✅ RTX 4060 必需 |
| **Val** | ~9.3GB | ~12GB | ⭐ 可选（存储充足时） |
| **Train** | ~67GB | ~85GB | ❌ 不推荐（RTX 4060） |

### RTX 4060 推荐配置

```
Replica:    ~2GB      ✅ 快速测试
Minival:    ~1.5GB    ✅ 标准评估
ObjectNav:  ~500MB    ✅ 任务数据
───────────────────────
总计:       ~4GB
```

---

## 🎯 使用数据集

### 切换数据集

编辑 `config.yaml` 或通过命令行参数：

```bash
# 使用 minival
python scripts/run_evaluation.py --config config.yaml environment.habitat.split=minival

# 使用 val
python scripts/run_evaluation.py --config config.yaml environment.habitat.split=val

# 使用 Replica（快速测试）
python scripts/run_evaluation.py --config config.yaml environment.habitat.scene_dataset=replica
```

### 指定场景数量（RTX 4060）

```bash
# 只使用前5个场景
python scripts/run_evaluation.py \
    --config config_rtx4060.yaml \
    environment.habitat.num_scenes=5
```

---

## 🐛 常见问题

### 问题1: 找不到场景文件

**解决方案**:
```bash
# 确认环境变量
echo $HABITAT_DATA_DIR

# 如果未设置
export HABITAT_DATA_DIR="$HOME/vla-gr-workspace/habitat-data"

# 或运行环境激活脚本
source activate_env.sh
```

### 问题2: 语义标注缺失

**解决方案**:
```bash
# 检查是否下载了 semantic-annots
ls ~/vla-gr-workspace/habitat-data/scene_datasets/hm3d/minival/*.semantic.glb

# 如果缺失，下载
cd ~/vla-gr-workspace/habitat-data/scene_datasets/hm3d
wget https://api.matterport.com/resources/habitat/hm3d-minival-semantic-annots-v0.2.tar
tar -xf hm3d-minival-semantic-annots-v0.2.tar -C minival/
```

### 问题3: Habitat 加载失败

**解决方案**:
```bash
# 确认 Habitat 版本
python -c "import habitat_sim; print(habitat_sim.__version__)"

# 应该显示 0.3.3

# 如果版本不对，重新安装
conda install habitat-sim=0.3.3 withbullet -c conda-forge -c aihabitat
```

---

## 📚 相关文档

- **完整部署指南**: [docs/RTX4060_DEPLOYMENT_GUIDE.md](docs/RTX4060_DEPLOYMENT_GUIDE.md)
- **数据规范**: [docs/TRAINING_DATA_SPEC.md](docs/TRAINING_DATA_SPEC.md)
- **Habitat 官方文档**: https://aihabitat.org/docs/habitat-sim/

---

**更新日期**: 2025-11-14

**维护者**: VLA-GR Team
