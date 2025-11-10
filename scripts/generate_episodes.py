#!/usr/bin/env python3
"""
生成 VLA-GR 训练和验证数据集
Generate training and validation episodes for VLA-GR navigation
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
    elif scene_dataset == "gibson":
        config.SIMULATOR.SCENE_DATASET = "data/scene_datasets/gibson/gibson.scene_dataset_config.json"
    else:  # test scenes
        config.SIMULATOR.SCENE = "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb"

    config.SIMULATOR.TURN_ANGLE = 10
    config.SIMULATOR.FORWARD_STEP_SIZE = 0.25

    # 传感器配置
    config.SIMULATOR.RGB_SENSOR.WIDTH = 640
    config.SIMULATOR.RGB_SENSOR.HEIGHT = 480
    config.SIMULATOR.RGB_SENSOR.HFOV = 79

    config.SIMULATOR.DEPTH_SENSOR.WIDTH = 640
    config.SIMULATOR.DEPTH_SENSOR.HEIGHT = 480
    config.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.0
    config.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 10.0

    config.freeze()
    return config


def generate_episodes(
    num_episodes: int,
    split: str,
    scene_dataset: str = "hm3d",
    success_distance: float = 0.2,
    min_distance: float = 2.0,
    max_distance: float = 10.0
):
    """
    生成导航 episodes

    Args:
        num_episodes: 要生成的 episode 数量
        split: 数据集划分 ('train' or 'val')
        scene_dataset: 场景数据集名称
        success_distance: 成功到达目标的距离阈值（米）
        min_distance: 起点和终点的最小距离（米）
        max_distance: 起点和终点的最大距离（米）
    """

    logger.info(f"生成 {num_episodes} 个 {split} episodes...")
    logger.info(f"使用场景数据集: {scene_dataset}")

    # 创建模拟器
    try:
        config = create_habitat_config(scene_dataset)
        simulator = make_sim(
            id_sim=config.SIMULATOR.TYPE,
            config=config.SIMULATOR
        )
        logger.info("✅ 模拟器创建成功")
    except Exception as e:
        logger.error(f"❌ 创建模拟器失败: {e}")
        logger.info("提示: 请确保已下载场景数据集")
        sys.exit(1)

    episodes = []
    failed_count = 0
    max_failures = num_episodes // 10  # 允许 10% 的失败率

    for i in range(num_episodes + max_failures):
        if len(episodes) >= num_episodes:
            break

        try:
            # 获取场景 ID
            if hasattr(simulator, 'semantic_scene') and simulator.semantic_scene:
                try:
                    scene_id = simulator.semantic_scene.levels[0].id
                except (AttributeError, IndexError):
                    scene_id = f"scene_{split}_{i % 10}"
            else:
                scene_id = f"scene_{split}_{i % 10}"

            # 随机起点
            start_position = simulator.sample_navigable_point()
            start_rotation = [0, random.uniform(0, 2 * np.pi), 0, 1]

            # 随机目标点（确保与起点有合适的距离）
            max_attempts = 50
            goal_position = None

            for attempt in range(max_attempts):
                candidate_goal = simulator.sample_navigable_point()
                distance = np.linalg.norm(
                    np.array(start_position) - np.array(candidate_goal)
                )

                # 确保距离在合理范围内
                if min_distance <= distance <= max_distance:
                    goal_position = candidate_goal
                    break

            if goal_position is None:
                logger.warning(f"Episode {i}: 无法找到合适的目标点，跳过")
                failed_count += 1
                continue

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

            if (len(episodes)) % 100 == 0:
                logger.info(f"已生成 {len(episodes)}/{num_episodes} episodes")

        except Exception as e:
            logger.warning(f"生成 episode {i} 失败: {e}")
            failed_count += 1

            if failed_count > max_failures:
                logger.error(f"失败次数过多 ({failed_count})，停止生成")
                break
            continue

    simulator.close()
    logger.info(f"✅ 成功生成 {len(episodes)} 个 episodes (失败 {failed_count} 个)")

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

    logger.info(f"📁 Episodes 已保存到: {output_path}")
    logger.info(f"   文件大小: {output_path.stat().st_size / 1024:.2f} KB")


def validate_episodes(episodes):
    """验证生成的 episodes"""

    logger.info("\n🔍 验证生成的 episodes...")

    if len(episodes) == 0:
        logger.error("❌ 没有生成任何 episodes")
        return False

    # 检查距离分布
    distances = []
    for ep in episodes:
        start = np.array(ep.start_position)
        goal = np.array(ep.goals[0].position)
        dist = np.linalg.norm(start - goal)
        distances.append(dist)

    distances = np.array(distances)
    logger.info(f"✅ Episode 数量: {len(episodes)}")
    logger.info(f"✅ 距离统计:")
    logger.info(f"   - 最小距离: {distances.min():.2f}m")
    logger.info(f"   - 最大距离: {distances.max():.2f}m")
    logger.info(f"   - 平均距离: {distances.mean():.2f}m")
    logger.info(f"   - 中位数距离: {np.median(distances):.2f}m")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="生成 VLA-GR 训练数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--num_train',
        type=int,
        default=10000,
        help='训练集大小'
    )
    parser.add_argument(
        '--num_val',
        type=int,
        default=1000,
        help='验证集大小'
    )
    parser.add_argument(
        '--scene_dataset',
        type=str,
        default='test',
        choices=['hm3d', 'mp3d', 'gibson', 'test'],
        help='场景数据集 (test 用于快速测试)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='输出目录'
    )
    parser.add_argument(
        '--min_distance',
        type=float,
        default=2.0,
        help='起点和终点的最小距离（米）'
    )
    parser.add_argument(
        '--max_distance',
        type=float,
        default=10.0,
        help='起点和终点的最大距离（米）'
    )
    parser.add_argument(
        '--success_distance',
        type=float,
        default=0.2,
        help='成功到达目标的距离阈值（米）'
    )
    parser.add_argument(
        '--skip_train',
        action='store_true',
        help='跳过训练集生成'
    )
    parser.add_argument(
        '--skip_val',
        action='store_true',
        help='跳过验证集生成'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("VLA-GR 数据集生成工具")
    logger.info("=" * 60)

    # 生成训练集
    if not args.skip_train:
        logger.info("\n📊 生成训练集...")
        train_episodes = generate_episodes(
            num_episodes=args.num_train,
            split='train',
            scene_dataset=args.scene_dataset,
            success_distance=args.success_distance,
            min_distance=args.min_distance,
            max_distance=args.max_distance
        )

        if validate_episodes(train_episodes):
            save_episodes(
                train_episodes,
                f"{args.output_dir}/episodes_train.json"
            )

    # 生成验证集
    if not args.skip_val:
        logger.info("\n📊 生成验证集...")
        val_episodes = generate_episodes(
            num_episodes=args.num_val,
            split='val',
            scene_dataset=args.scene_dataset,
            success_distance=args.success_distance,
            min_distance=args.min_distance,
            max_distance=args.max_distance
        )

        if validate_episodes(val_episodes):
            save_episodes(
                val_episodes,
                f"{args.output_dir}/episodes_val.json"
            )

    logger.info("\n" + "=" * 60)
    logger.info("✅ 数据生成完成！")
    logger.info("=" * 60)
    logger.info(f"\n生成的文件:")
    if not args.skip_train:
        logger.info(f"  - {args.output_dir}/episodes_train.json")
    if not args.skip_val:
        logger.info(f"  - {args.output_dir}/episodes_val.json")
    logger.info(f"\n现在可以运行训练:")
    logger.info(f"  python src/training/train.py --config config.yaml")


if __name__ == "__main__":
    main()
