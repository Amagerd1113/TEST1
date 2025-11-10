# Habitat 环境设置指南

## 📁 目录结构

正确的项目目录结构应该是：

```
VLA-GR/
├── data/
│   ├── scene_datasets/          # 场景数据集（需要下载）
│   │   ├── hm3d/                # HM3D 场景数据
│   │   ├── mp3d/                # Matterport3D 场景数据（可选）
│   │   └── gibson/              # Gibson 场景数据（可选）
│   ├── datasets/                # 任务数据集（需要下载）
│   │   ├── objectnav/
│   │   └── pointnav/
│   └── episodes_train.json      # 训练数据（自动生成）
│   └── episodes_val.json        # 验证数据（自动生成）
├── src/
├── scripts/
├── config.yaml
└── ...
```

## 🔧 1. 安装 Habitat-Sim 和 Habitat-Lab

### 方法 A: 使用 Conda（推荐）

```bash
# 创建环境
conda create -n vla_gr python=3.8
conda activate vla_gr

# 安装 Habitat-Sim（带 GPU 支持）
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat

# 安装 Habitat-Lab
pip install habitat-lab==0.3.3

# 验证安装
python -c "import habitat; import habitat_sim; print('Habitat installed successfully!')"
```

### 方法 B: 使用 Pip

```bash
# 安装 Habitat-Sim（预编译版本）
pip install habitat-sim==0.3.3

# 安装 Habitat-Lab
pip install habitat-lab==0.3.3
```

**注意**：habitat-sim 和 habitat-lab 会安装到 Python 环境的 site-packages 目录，不需要手动放置文件夹！

## 📦 2. 下载场景数据集

### HM3D 数据集（推荐，最大最真实）

```bash
# 创建数据目录
mkdir -p data/scene_datasets data/datasets

# 下载 HM3D 场景数据（需要先注册）
# 1. 访问 https://aihabitat.org/datasets/hm3d/
# 2. 注册并获取下载权限
# 3. 下载 minival 数据集（较小，适合快速测试）

# 使用官方脚本下载
python -m habitat_sim.utils.datasets_download --username <your-username> --password <your-password> --uids hm3d_minival_v0.2

# 或者使用 Habitat 提供的下载工具
python scripts/download_habitat_data.py
```

### Matterport3D 数据集（备选）

```bash
# 需要签署协议：https://niessner.github.io/Matterport/
# 下载后解压到 data/scene_datasets/mp3d/
```

### Gibson 数据集（备选，较小）

```bash
# 下载 Gibson tiny 数据集（仅用于测试）
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
```

### 最小测试配置（快速开始）

```bash
# 如果只是想快速测试，下载测试场景
mkdir -p data/scene_datasets
cd data/scene_datasets

# 下载 Habitat 测试场景（约 40MB）
wget https://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip
unzip habitat-test-scenes.zip
mv habitat-test-scenes hm3d_test

cd ../..
```

## 🎯 3. 下载任务数据集（Episode 定义）

```bash
# 下载 ObjectNav 数据集（任务定义，非场景）
mkdir -p data/datasets/objectnav

# HM3D ObjectNav v1
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
unzip objectnav_hm3d_v1.zip -d data/datasets/objectnav/

# 或者使用 Habitat 工具
python -m habitat.datasets.download_data --task objectnav-hm3d-v1 --data-path data/
```

## 🔄 4. 生成训练数据

**是的，你需要生成训练数据！** VLA-GR 使用自定义的导航episodes。有两种方式：

### 方法 A: 自动生成（推荐）

数据集类会在首次运行时自动生成 episodes：

```python
# 第一次运行训练脚本时，会自动生成并保存 episodes
python src/training/train.py --config config.yaml

# episodes 会保存在：
# - data/episodes_train.json  (训练集)
# - data/episodes_val.json    (验证集)
```

### 方法 B: 手动生成（更可控）

创建一个生成脚本：

```bash
# 创建数据生成脚本
cat > scripts/generate_episodes.py << 'EOF'
#!/usr/bin/env python3
"""
生成 VLA-GR 训练和验证数据集
"""

import sys
import json
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import habitat
from habitat.config.default import get_config
from habitat.sims import make_sim
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_habitat_config(scene_dataset="hm3d"):
    """创建 Habitat 配置"""
    config = get_config()
    config.defrost()

    # 场景设置
    if scene_dataset == "hm3d":
        config.SIMULATOR.SCENE_DATASET = "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
    elif scene_dataset == "mp3d":
        config.SIMULATOR.SCENE_DATASET = "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
    else:
        config.SIMULATOR.SCENE = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"

    config.SIMULATOR.TURN_ANGLE = 10
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.25

    # 传感器
    config.SIMULATOR.RGB_SENSOR.WIDTH = 640
    config.SIMULATOR.RGB_SENSOR.HEIGHT = 480
    config.SIMULATOR.DEPTH_SENSOR.WIDTH = 640
    config.SIMULATOR.DEPTH_SENSOR.HEIGHT = 480

    config.freeze()
    return config


def generate_episodes(
    num_episodes: int,
    split: str,
    scene_dataset: str = "hm3d",
    success_distance: float = 0.2
):
    """生成导航 episodes"""

    logger.info(f"生成 {num_episodes} 个 {split} episodes...")

    # 创建模拟器
    config = create_habitat_config(scene_dataset)
    simulator = make_sim(
        id_sim=config.SIMULATOR.TYPE,
        config=config.SIMULATOR
    )

    episodes = []

    for i in range(num_episodes):
        try:
            # 获取场景 ID
            if hasattr(simulator, 'semantic_scene') and simulator.semantic_scene:
                scene_id = simulator.semantic_scene.levels[0].id
            else:
                scene_id = f"scene_{i % 10}"

            # 随机起点
            start_position = simulator.sample_navigable_point()
            start_rotation = [0, random.uniform(0, 2 * np.pi), 0, 1]

            # 随机目标点（确保与起点有一定距离）
            max_attempts = 20
            for attempt in range(max_attempts):
                goal_position = simulator.sample_navigable_point()
                distance = np.linalg.norm(
                    np.array(start_position) - np.array(goal_position)
                )

                # 确保距离在合理范围内（2-10米）
                if 2.0 <= distance <= 10.0:
                    break

            # 创建 episode
            episode = NavigationEpisode(
                episode_id=f"{split}_{i:05d}",
                scene_id=scene_id,
                start_position=start_position.tolist(),
                start_rotation=start_rotation,
                goals=[NavigationGoal(
                    position=goal_position.tolist(),
                    radius=success_distance
                )]
            )

            episodes.append(episode)

            if (i + 1) % 100 == 0:
                logger.info(f"已生成 {i + 1}/{num_episodes} episodes")

        except Exception as e:
            logger.warning(f"生成 episode {i} 失败: {e}")
            continue

    simulator.close()
    logger.info(f"成功生成 {len(episodes)} 个 episodes")

    return episodes


def save_episodes(episodes, output_path):
    """保存 episodes 到 JSON 文件"""

    episodes_data = []
    for ep in episodes:
        episodes_data.append({
            'episode_id': ep.episode_id,
            'scene_id': ep.scene_id,
            'start_position': ep.start_position,
            'start_rotation': ep.start_rotation,
            'goals': [
                {
                    'position': g.position,
                    'radius': g.radius
                }
                for g in ep.goals
            ]
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(episodes_data, f, indent=2)

    logger.info(f"Episodes 已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="生成 VLA-GR 训练数据")
    parser.add_argument('--num_train', type=int, default=10000, help='训练集大小')
    parser.add_argument('--num_val', type=int, default=1000, help='验证集大小')
    parser.add_argument('--scene_dataset', type=str, default='hm3d',
                       choices=['hm3d', 'mp3d', 'gibson', 'test'],
                       help='场景数据集')
    parser.add_argument('--output_dir', type=str, default='data', help='输出目录')

    args = parser.parse_args()

    # 生成训练集
    train_episodes = generate_episodes(
        num_episodes=args.num_train,
        split='train',
        scene_dataset=args.scene_dataset
    )
    save_episodes(
        train_episodes,
        f"{args.output_dir}/episodes_train.json"
    )

    # 生成验证集
    val_episodes = generate_episodes(
        num_episodes=args.num_val,
        split='val',
        scene_dataset=args.scene_dataset
    )
    save_episodes(
        val_episodes,
        f"{args.output_dir}/episodes_val.json"
    )

    logger.info("✅ 数据生成完成！")


if __name__ == "__main__":
    main()
EOF

chmod +x scripts/generate_episodes.py
```

运行生成脚本：

```bash
# 生成训练数据
python scripts/generate_episodes.py \
    --num_train 10000 \
    --num_val 1000 \
    --scene_dataset hm3d

# 使用测试场景（快速测试）
python scripts/generate_episodes.py \
    --num_train 100 \
    --num_val 20 \
    --scene_dataset test
```

## ✅ 5. 验证设置

创建验证脚本确保一切正常：

```bash
python << 'EOF'
import sys
import logging
logging.basicConfig(level=logging.INFO)

print("🔍 检查 Habitat 安装...")

# 检查包导入
try:
    import habitat
    import habitat_sim
    print(f"✅ Habitat-Lab {habitat.__version__}")
    print(f"✅ Habitat-Sim {habitat_sim.__version__}")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# 检查数据目录
import os
data_dirs = [
    'data/scene_datasets',
    'data/datasets',
]
for d in data_dirs:
    if os.path.exists(d):
        print(f"✅ 目录存在: {d}")
    else:
        print(f"⚠️  目录不存在: {d}")

# 检查 episodes 文件
episode_files = [
    'data/episodes_train.json',
    'data/episodes_val.json'
]
for f in episode_files:
    if os.path.exists(f):
        print(f"✅ Episodes 文件存在: {f}")
    else:
        print(f"ℹ️  Episodes 文件不存在（首次运行时会自动生成）: {f}")

# 尝试创建模拟器
try:
    from habitat.config.default import get_config
    from habitat.sims import make_sim

    config = get_config()
    config.defrost()
    config.SIMULATOR.SCENE = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"
    config.freeze()

    sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)
    print("✅ 模拟器创建成功")
    sim.close()
except Exception as e:
    print(f"⚠️  模拟器创建失败: {e}")

print("\n✅ 设置验证完成！")
EOF
```

## 📝 6. 配置文件更新

确保 `config.yaml` 中的路径正确：

```yaml
environment:
  habitat:
    scene_dataset: "hm3d"  # 或 "mp3d", "gibson"
    split: "train"
    max_episode_steps: 500
    success_distance: 0.2

  # 如果使用特定场景文件
  scene_path: "data/scene_datasets/hm3d/"  # 根据实际下载的数据集调整
```

## 🚀 7. 开始训练

```bash
# 小规模测试（使用少量数据）
python src/training/train.py \
    --config config.yaml \
    training.max_steps=1000

# 完整训练
python src/training/train.py \
    --config config.yaml
```

## 🐛 常见问题

### Q1: ModuleNotFoundError: No module named 'habitat'
```bash
# 确保在正确的环境中
conda activate vla_gr
pip install habitat-lab==0.3.3
```

### Q2: 找不到场景文件
```bash
# 检查场景数据集路径
ls -la data/scene_datasets/
# 应该看到 hm3d/ 或其他数据集目录
```

### Q3: Episodes 文件为空或生成失败
```bash
# 手动运行生成脚本
python scripts/generate_episodes.py --num_train 100 --num_val 20 --scene_dataset test
```

### Q4: GPU 内存不足
```bash
# 使用较小的 batch size
python src/training/train.py --config config.yaml training.batch_size=8
```

## 📚 参考资源

- [Habitat-Lab 文档](https://aihabitat.org/docs/habitat-lab/)
- [Habitat-Sim 文档](https://aihabitat.org/docs/habitat-sim/)
- [HM3D 数据集](https://aihabitat.org/datasets/hm3d/)
- [Habitat 数据集下载](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md)

## 📊 数据集大小参考

| 数据集 | 场景数 | 大小 | 用途 |
|--------|--------|------|------|
| Habitat Test Scenes | 3 | ~40MB | 快速测试 |
| HM3D minival | ~100 | ~2GB | 小规模训练 |
| HM3D train | ~800 | ~15GB | 完整训练 |
| Matterport3D | ~90 | ~10GB | 真实场景 |
| Gibson | ~572 | ~8GB | 多样性 |

## 🎯 推荐工作流程

1. **快速测试**（第一天）
   - 使用 Habitat Test Scenes
   - 生成 100 个训练 episodes
   - 运行 1000 步训练验证代码

2. **小规模实验**（1-2周）
   - 下载 HM3D minival
   - 生成 1000-5000 个 episodes
   - 调整超参数

3. **完整训练**（发表论文）
   - 下载完整 HM3D 或 MP3D
   - 生成 10000+ episodes
   - 运行完整训练流程

---

有任何问题，请查看日志文件或提 Issue！
