#!/usr/bin/env python3
"""
VLA-GR 安装验证脚本

用途：全面验证环境、依赖、数据集和模型安装
使用：python scripts/verify_installation.py [--check-all] [--check-env] [--check-habitat] [--check-datasets] [--check-models]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

# ANSI 颜色代码
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'  # No Color

def print_header(text: str):
    """打印标题"""
    print(f"\n{Colors.CYAN}{'='*60}{Colors.NC}")
    print(f"{Colors.CYAN}{text}{Colors.NC}")
    print(f"{Colors.CYAN}{'='*60}{Colors.NC}\n")

def print_success(text: str):
    """打印成功消息"""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")

def print_error(text: str):
    """打印错误消息"""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")

def print_warning(text: str):
    """打印警告消息"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")

def print_info(text: str):
    """打印信息"""
    print(f"{Colors.BLUE}  {text}{Colors.NC}")


class InstallationVerifier:
    """安装验证器"""

    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = 0

    def check_environment(self) -> bool:
        """检查系统环境"""
        print_header("1. 系统环境检查")

        all_passed = True

        # 检查 Python 版本
        try:
            py_version = sys.version.split()[0]
            py_major, py_minor = sys.version_info[:2]

            if 9 <= py_minor <= 11 and py_major == 3:
                print_success(f"Python 版本: {py_version}")
                self.passed_tests += 1
            else:
                print_warning(f"Python 版本: {py_version} (推荐 3.9-3.11)")
                self.warnings += 1
        except Exception as e:
            print_error(f"Python 检查失败: {e}")
            all_passed = False
            self.failed_tests += 1

        # 检查 PyTorch
        try:
            import torch
            print_success(f"PyTorch: {torch.__version__}")

            if torch.cuda.is_available():
                print_success(f"CUDA 可用: {torch.version.cuda}")
                print_info(f"GPU 设备: {torch.cuda.get_device_name(0)}")
                print_info(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                self.passed_tests += 2
            else:
                print_warning("CUDA 不可用（将使用 CPU 模式）")
                self.warnings += 1
        except ImportError:
            print_error("PyTorch 未安装")
            all_passed = False
            self.failed_tests += 1

        # 检查环境变量
        env_vars = {
            'HABITAT_DATA_DIR': 'Habitat 数据目录',
            'HF_HOME': 'HuggingFace 缓存目录',
        }

        for var, desc in env_vars.items():
            value = os.environ.get(var)
            if value:
                print_success(f"{desc}: {value}")
                self.passed_tests += 1
            else:
                print_warning(f"{desc} ({var}) 未设置")
                self.warnings += 1

        return all_passed

    def check_habitat(self) -> bool:
        """检查 Habitat 安装"""
        print_header("2. Habitat 安装检查")

        all_passed = True

        # 检查 Habitat-Sim
        try:
            import habitat_sim
            print_success(f"Habitat-Sim: {habitat_sim.__version__}")

            # 检查 CUDA 支持
            if habitat_sim.built_with_cuda:
                print_success("Habitat-Sim CUDA 支持已启用")
            else:
                print_warning("Habitat-Sim CUDA 支持未启用")
                self.warnings += 1

            self.passed_tests += 1
        except ImportError as e:
            print_error(f"Habitat-Sim 未安装: {e}")
            all_passed = False
            self.failed_tests += 1

        # 检查 Habitat-Lab
        try:
            import habitat
            print_success(f"Habitat-Lab: {habitat.__version__}")

            # 测试配置创建
            try:
                from habitat import get_config
                from habitat.config import read_write

                with read_write(get_config()):
                    config = get_config()
                print_success("Habitat 配置系统正常")
                self.passed_tests += 1
            except Exception as e:
                print_warning(f"Habitat 配置测试失败: {e}")
                self.warnings += 1

        except ImportError as e:
            print_error(f"Habitat-Lab 未安装: {e}")
            all_passed = False
            self.failed_tests += 1

        return all_passed

    def check_datasets(self) -> bool:
        """检查数据集"""
        print_header("3. 数据集检查")

        habitat_data_dir = os.environ.get('HABITAT_DATA_DIR',
                                          os.path.expanduser('~/vla-gr-workspace/habitat-data'))
        data_path = Path(habitat_data_dir)

        if not data_path.exists():
            print_error(f"数据目录不存在: {data_path}")
            self.failed_tests += 1
            return False

        print_info(f"数据目录: {data_path}")

        # 检查 Replica
        replica_path = data_path / "scene_datasets" / "replica"
        if replica_path.exists():
            scenes = list(replica_path.glob("*.glb"))
            if scenes:
                print_success(f"Replica: {len(scenes)} 个场景")
                self.passed_tests += 1
            else:
                print_warning("Replica 目录存在但为空")
                self.warnings += 1
        else:
            print_warning("Replica 数据集未找到（测试必需）")
            self.warnings += 1

        # 检查 HM3D
        hm3d_path = data_path / "scene_datasets" / "hm3d"
        if hm3d_path.exists():
            # 检查不同的 splits
            splits = ['minival', 'train', 'val']
            found_splits = [s for s in splits if (hm3d_path / s).exists()]

            if found_splits:
                print_success(f"HM3D: {', '.join(found_splits)}")

                # 统计场景数量
                total_scenes = 0
                for split in found_splits:
                    scenes = list((hm3d_path / split).glob("**/*.glb"))
                    total_scenes += len(scenes)

                if total_scenes > 0:
                    print_info(f"  总场景数: {total_scenes}")
                    self.passed_tests += 1
            else:
                print_warning("HM3D 目录存在但没有 split 数据")
                self.warnings += 1
        else:
            print_warning("HM3D 数据集未找到（训练可选）")
            self.warnings += 1

        # 检查任务数据
        objectnav_path = data_path / "datasets" / "objectnav" / "hm3d" / "v1"
        if objectnav_path.exists():
            splits = list(objectnav_path.glob("*/"))
            if splits:
                split_names = [s.name for s in splits]
                print_success(f"ObjectNav: {', '.join(split_names)}")
                self.passed_tests += 1
            else:
                print_warning("ObjectNav 目录存在但为空")
                self.warnings += 1
        else:
            print_warning("ObjectNav 任务数据未找到")
            self.warnings += 1

        return True

    def check_models(self) -> bool:
        """检查 HuggingFace 模型"""
        print_header("4. HuggingFace 模型检查")

        all_passed = True

        # 检查 Transformers
        try:
            import transformers
            print_success(f"Transformers: {transformers.__version__}")
            self.passed_tests += 1
        except ImportError:
            print_error("Transformers 未安装")
            self.failed_tests += 1
            return False

        # 设置缓存目录
        cache_dir = os.environ.get('HF_HOME',
                                   os.path.expanduser('~/vla-gr-workspace/huggingface-cache'))

        print_info(f"缓存目录: {cache_dir}")

        # 检查模型
        models_to_check = [
            ("microsoft/phi-2", "Phi-2 语言模型"),
            ("openai/clip-vit-base-patch32", "CLIP 视觉-语言模型"),
            ("bert-base-uncased", "BERT 模型"),
            ("facebook/dinov2-base", "DINOv2 视觉编码器"),
        ]

        for model_name, description in models_to_check:
            try:
                from transformers import AutoModel

                # 尝试加载（仅检查缓存）
                try:
                    model = AutoModel.from_pretrained(
                        model_name,
                        cache_dir=cache_dir,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    print_success(f"{description}")
                    self.passed_tests += 1
                    del model  # 释放内存
                except Exception:
                    print_warning(f"{description} 未缓存（将在首次使用时下载）")
                    self.warnings += 1

            except Exception as e:
                print_warning(f"{description} 检查失败: {e}")
                self.warnings += 1

        return all_passed

    def check_project_modules(self) -> bool:
        """检查项目模块"""
        print_header("5. 项目模块检查")

        all_passed = True

        modules_to_check = [
            ("src.core.vla_gr_agent", "VLA-GR Agent"),
            ("src.core.perception", "感知模块"),
            ("src.core.gr_field", "GR Field"),
            ("src.core.path_optimizer", "路径优化器"),
            ("src.environments.habitat_env_v3", "Habitat 环境 V3"),
            ("src.datasets.habitat_dataset", "Habitat 数据集"),
            ("src.training.train", "训练管道"),
            ("src.evaluation.evaluator", "评估器"),
        ]

        for module_name, description in modules_to_check:
            try:
                __import__(module_name)
                print_success(description)
                self.passed_tests += 1
            except ImportError as e:
                print_error(f"{description}: {e}")
                all_passed = False
                self.failed_tests += 1

        return all_passed

    def run_quick_test(self) -> bool:
        """运行快速功能测试"""
        print_header("6. 功能测试")

        # 设置环境变量
        os.environ['HABITAT_DATA_DIR'] = os.environ.get(
            'HABITAT_DATA_DIR',
            os.path.expanduser('~/vla-gr-workspace/habitat-data')
        )

        # 测试 Habitat 环境
        try:
            from src.environments.habitat_env_v3 import HabitatNavigationEnv

            print_info("创建 Habitat 环境...")
            env = HabitatNavigationEnv(
                scene_id="replica/apartment_0.glb",
                task_type="objectnav",
                max_episode_steps=10
            )

            print_info("重置环境...")
            obs, info = env.reset()

            print_info(f"观察空间: RGB {obs['rgb'].shape}, Depth {obs['depth'].shape}")

            print_info("执行随机动作...")
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            env.close()

            print_success("Habitat 环境测试通过")
            self.passed_tests += 1

        except Exception as e:
            print_error(f"Habitat 环境测试失败: {e}")
            self.failed_tests += 1
            return False

        # 测试模型加载
        try:
            import torch
            from src.core.vla_gr_agent import ConferenceVLAGRAgent
            from omegaconf import OmegaConf

            print_info("加载配置...")

            # 查找配置文件
            config_files = ['config_active.yaml', 'config_rtx4060.yaml', 'config.yaml']
            config_path = None
            for cf in config_files:
                if Path(cf).exists():
                    config_path = cf
                    break

            if not config_path:
                print_warning("未找到配置文件，跳过模型测试")
                self.warnings += 1
                return True

            config = OmegaConf.load(config_path)

            print_info("创建 VLA-GR Agent...")
            device = torch.device("cpu")  # 使用 CPU 避免 GPU 内存问题

            # 临时修改配置以加快测试
            config.model.use_lora = True

            agent = ConferenceVLAGRAgent(config, device=device)

            print_info("测试前向传播...")
            batch_size = 1
            rgb = torch.randn(batch_size, 3, 224, 224)
            depth = torch.randn(batch_size, 1, 224, 224)
            instruction = ["test"]

            with torch.no_grad():
                output = agent.forward(rgb, depth, instruction)

            print_success("模型加载和推理测试通过")
            self.passed_tests += 1

        except Exception as e:
            print_error(f"模型测试失败: {e}")
            self.failed_tests += 1
            return False

        return True

    def print_summary(self):
        """打印测试摘要"""
        print_header("测试摘要")

        total_tests = self.passed_tests + self.failed_tests

        print(f"总测试数: {total_tests}")
        print_success(f"通过: {self.passed_tests}")

        if self.failed_tests > 0:
            print_error(f"失败: {self.failed_tests}")

        if self.warnings > 0:
            print_warning(f"警告: {self.warnings}")

        print()

        if self.failed_tests == 0:
            print_success("🎉 所有测试通过！环境已就绪")
            print()
            print_info("下一步:")
            print_info("  1. 运行训练测试: vla-gr-train --config config_active.yaml")
            print_info("  2. 运行评估: vla-gr-evaluate --checkpoint <path>")
            print_info("  3. 查看文档: less DEPLOYMENT_CHECKLIST_CN.md")
            return 0
        else:
            print_error("❌ 部分测试失败，请检查上述错误")
            print()
            print_info("故障排除:")
            print_info("  1. 查看部署文档: less DEPLOYMENT_CHECKLIST_CN.md")
            print_info("  2. 检查依赖安装: pip list | grep -E '(habitat|torch|transformers)'")
            print_info("  3. 重新运行安装脚本: bash scripts/install_habitat.sh")
            return 1


def main():
    parser = argparse.ArgumentParser(description="VLA-GR 安装验证")
    parser.add_argument('--check-all', action='store_true', help='运行所有检查（默认）')
    parser.add_argument('--check-env', action='store_true', help='仅检查环境')
    parser.add_argument('--check-habitat', action='store_true', help='仅检查 Habitat')
    parser.add_argument('--check-datasets', action='store_true', help='仅检查数据集')
    parser.add_argument('--check-models', action='store_true', help='仅检查模型')
    parser.add_argument('--quick-test', action='store_true', help='运行快速功能测试')

    args = parser.parse_args()

    # 如果没有指定任何检查，默认运行所有
    if not any([args.check_env, args.check_habitat, args.check_datasets,
                args.check_models, args.quick_test]):
        args.check_all = True

    verifier = InstallationVerifier()

    print_header("🔍 VLA-GR 安装验证")

    try:
        if args.check_all or args.check_env:
            verifier.check_environment()

        if args.check_all or args.check_habitat:
            verifier.check_habitat()

        if args.check_all or args.check_datasets:
            verifier.check_datasets()

        if args.check_all or args.check_models:
            verifier.check_models()

        if args.check_all:
            verifier.check_project_modules()

        if args.check_all or args.quick_test:
            verifier.run_quick_test()

    except KeyboardInterrupt:
        print("\n\n中断测试")
        return 1
    except Exception as e:
        print_error(f"验证过程出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return verifier.print_summary()


if __name__ == "__main__":
    sys.exit(main())
