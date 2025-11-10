#!/usr/bin/env python3
"""
下载 Habitat 场景和数据集
Download Habitat scenes and datasets for VLA-GR training
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
import urllib.request
import zipfile
import gzip
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HabitatDataDownloader:
    """Habitat 数据下载器"""

    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.scene_dir = self.data_dir / "scene_datasets"
        self.dataset_dir = self.data_dir / "datasets"

        # 创建目录
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, output_path, description="文件"):
        """下载文件并显示进度"""

        logger.info(f"📥 下载 {description}...")
        logger.info(f"   URL: {url}")
        logger.info(f"   保存到: {output_path}")

        try:
            def reporthook(count, block_size, total_size):
                if total_size > 0:
                    percent = min(int(count * block_size * 100 / total_size), 100)
                    sys.stdout.write(f"\r   进度: {percent}%")
                    sys.stdout.flush()

            urllib.request.urlretrieve(url, output_path, reporthook)
            print()  # 换行
            logger.info(f"✅ 下载完成")
            return True

        except Exception as e:
            logger.error(f"❌ 下载失败: {e}")
            return False

    def extract_zip(self, zip_path, extract_dir):
        """解压 ZIP 文件"""

        logger.info(f"📦 解压 {zip_path.name}...")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            logger.info(f"✅ 解压完成")
            return True

        except Exception as e:
            logger.error(f"❌ 解压失败: {e}")
            return False

    def download_test_scenes(self):
        """下载 Habitat 测试场景（约 40MB）"""

        logger.info("\n" + "=" * 60)
        logger.info("下载 Habitat 测试场景")
        logger.info("=" * 60)

        url = "https://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip"
        zip_path = self.scene_dir / "habitat-test-scenes.zip"
        extract_dir = self.scene_dir

        # 检查是否已存在
        if (self.scene_dir / "habitat-test-scenes").exists():
            logger.info("⚠️  测试场景已存在，跳过下载")
            return True

        # 下载
        if not self.download_file(url, zip_path, "测试场景"):
            return False

        # 解压
        if not self.extract_zip(zip_path, extract_dir):
            return False

        # 清理
        zip_path.unlink()
        logger.info("✅ 测试场景安装完成")

        return True

    def download_hm3d_minival(self):
        """下载 HM3D minival 数据集"""

        logger.info("\n" + "=" * 60)
        logger.info("下载 HM3D Minival 数据集")
        logger.info("=" * 60)
        logger.info("⚠️  HM3D 数据集需要注册和授权")
        logger.info("   请访问: https://aihabitat.org/datasets/hm3d/")
        logger.info("   使用 Habitat 官方工具下载:")
        logger.info("")
        logger.info("   python -m habitat_sim.utils.datasets_download \\")
        logger.info("       --username <your-username> \\")
        logger.info("       --password <your-password> \\")
        logger.info("       --uids hm3d_minival_v0.2")
        logger.info("")

        return False

    def download_with_habitat_tool(self, task="pointnav", dataset="mp3d"):
        """使用 Habitat 官方工具下载数据"""

        logger.info("\n" + "=" * 60)
        logger.info(f"使用 Habitat 工具下载 {task}-{dataset}")
        logger.info("=" * 60)

        try:
            cmd = [
                sys.executable, "-m", "habitat.datasets.download_data",
                "--task", f"{task}-{dataset}",
                "--data-path", str(self.data_dir)
            ]

            logger.info(f"运行命令: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            logger.info("✅ 下载完成")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 下载失败: {e}")
            logger.info("提示: 某些数据集需要注册和授权")
            return False

        except Exception as e:
            logger.error(f"❌ 错误: {e}")
            return False

    def verify_installation(self):
        """验证数据安装"""

        logger.info("\n" + "=" * 60)
        logger.info("验证数据安装")
        logger.info("=" * 60)

        # 检查场景数据
        scene_datasets = {
            'habitat-test-scenes': self.scene_dir / "habitat-test-scenes",
            'hm3d': self.scene_dir / "hm3d",
            'mp3d': self.scene_dir / "mp3d",
            'gibson': self.scene_dir / "gibson",
        }

        logger.info("\n📁 场景数据集:")
        found_scenes = False
        for name, path in scene_datasets.items():
            if path.exists():
                logger.info(f"   ✅ {name}: {path}")
                found_scenes = True
            else:
                logger.info(f"   ❌ {name}: 未安装")

        if not found_scenes:
            logger.warning("⚠️  没有找到任何场景数据集")

        # 检查任务数据
        logger.info("\n📊 任务数据集:")
        task_datasets = {
            'objectnav': self.dataset_dir / "objectnav",
            'pointnav': self.dataset_dir / "pointnav",
        }

        for name, path in task_datasets.items():
            if path.exists():
                logger.info(f"   ✅ {name}: {path}")
            else:
                logger.info(f"   ❌ {name}: 未安装")

        # 测试 Habitat 导入
        logger.info("\n🔍 测试 Habitat 导入:")
        try:
            import habitat
            import habitat_sim
            logger.info(f"   ✅ Habitat-Lab: {habitat.__version__}")
            logger.info(f"   ✅ Habitat-Sim: {habitat_sim.__version__}")
        except ImportError as e:
            logger.error(f"   ❌ 导入失败: {e}")

        logger.info("\n" + "=" * 60)

    def show_download_guide(self):
        """显示下载指南"""

        guide = """
╔══════════════════════════════════════════════════════════════╗
║           Habitat 数据下载指南                               ║
╚══════════════════════════════════════════════════════════════╝

1. 快速测试（推荐新手）
   ------------------------
   使用测试场景，约 40MB:

   python scripts/download_habitat_data.py --test-scenes

2. HM3D 数据集（最真实，推荐）
   ------------------------
   需要注册: https://aihabitat.org/datasets/hm3d/

   a) Minival (约 2GB):
      python -m habitat_sim.utils.datasets_download \\
          --username <your-username> \\
          --password <your-password> \\
          --uids hm3d_minival_v0.2

   b) 完整训练集 (约 15GB):
      python -m habitat_sim.utils.datasets_download \\
          --username <your-username> \\
          --password <your-password> \\
          --uids hm3d_train_v0.2

3. Matterport3D 数据集
   ------------------------
   需要签署协议: https://niessner.github.io/Matterport/

   下载后解压到: data/scene_datasets/mp3d/

4. Gibson 数据集
   ------------------------
   使用 Habitat 工具下载:

   python -m habitat.datasets.download_data \\
       --task pointnav-gibson \\
       --data-path data

5. 任务数据集（Episode 定义）
   ------------------------
   ObjectNav for HM3D:

   python -m habitat.datasets.download_data \\
       --task objectnav-hm3d-v1 \\
       --data-path data

   PointNav for MP3D:

   python -m habitat.datasets.download_data \\
       --task pointnav-mp3d-v1 \\
       --data-path data

╔══════════════════════════════════════════════════════════════╗
║  推荐工作流程                                                ║
╚══════════════════════════════════════════════════════════════╝

第一天（快速验证）:
  1. 下载测试场景: python scripts/download_habitat_data.py --test-scenes
  2. 生成测试数据: python scripts/generate_episodes.py --num_train 100 --num_val 20 --scene_dataset test
  3. 运行训练: python src/training/train.py --config config.yaml training.max_steps=100

1-2周（小规模实验）:
  1. 下载 HM3D minival
  2. 生成 1000-5000 episodes
  3. 完整训练流程

发表论文:
  1. 下载完整 HM3D 或 MP3D
  2. 生成 10000+ episodes
  3. 完整训练和评估

╔══════════════════════════════════════════════════════════════╗
║  数据集大小对比                                              ║
╚══════════════════════════════════════════════════════════════╝

| 数据集            | 场景数 | 大小   | 适用场景          |
|-------------------|--------|--------|-------------------|
| Test Scenes       | 3      | ~40MB  | 代码测试          |
| HM3D minival      | ~100   | ~2GB   | 算法开发          |
| HM3D train        | ~800   | ~15GB  | 完整训练          |
| Matterport3D      | ~90    | ~10GB  | 真实场景          |
| Gibson            | ~572   | ~8GB   | 场景多样性        |

"""
        print(guide)


def main():
    parser = argparse.ArgumentParser(
        description="下载 Habitat 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='数据目录'
    )

    parser.add_argument(
        '--test-scenes',
        action='store_true',
        help='下载测试场景（约 40MB，推荐新手）'
    )

    parser.add_argument(
        '--task',
        type=str,
        choices=['pointnav', 'objectnav', 'imagenav', 'vln'],
        help='任务类型'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mp3d', 'gibson', 'hm3d'],
        help='数据集类型'
    )

    parser.add_argument(
        '--guide',
        action='store_true',
        help='显示完整下载指南'
    )

    parser.add_argument(
        '--verify',
        action='store_true',
        help='验证已安装的数据'
    )

    args = parser.parse_args()

    # 创建下载器
    downloader = HabitatDataDownloader(args.data_dir)

    # 显示指南
    if args.guide or (not args.test_scenes and not args.task and not args.verify):
        downloader.show_download_guide()
        return

    # 验证安装
    if args.verify:
        downloader.verify_installation()
        return

    # 下载测试场景
    if args.test_scenes:
        success = downloader.download_test_scenes()
        if success:
            logger.info("\n✅ 测试场景安装成功！")
            logger.info("\n下一步:")
            logger.info("  1. 生成训练数据:")
            logger.info("     python scripts/generate_episodes.py --num_train 100 --num_val 20 --scene_dataset test")
            logger.info("  2. 运行训练:")
            logger.info("     python src/training/train.py --config config.yaml")
        return

    # 使用 Habitat 工具下载
    if args.task and args.dataset:
        downloader.download_with_habitat_tool(args.task, args.dataset)
        return

    # 验证安装
    downloader.verify_installation()


if __name__ == "__main__":
    main()
