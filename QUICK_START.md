# VLA-GR 快速入门指南

## 🚀 5分钟快速开始

如果你想快速验证代码能否运行，按照以下步骤：

### 1. 安装依赖

```bash
# 创建环境
conda create -n vla_gr python=3.8
conda activate vla_gr

# 安装 PyTorch（根据你的 CUDA 版本调整）
pip install torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu117

# 安装 Habitat
conda install habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat
pip install habitat-lab==0.3.3

# 安装项目依赖
pip install -r requirements.txt
```

### 2. 下载测试场景（约 40MB）

```bash
# 方法 A: 使用我们的脚本
python scripts/download_habitat_data.py --test-scenes

# 方法 B: 手动下载
mkdir -p data/scene_datasets
cd data/scene_datasets
wget https://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip
unzip habitat-test-scenes.zip
cd ../..
```

### 3. 生成少量训练数据

```bash
# 生成 100 个训练样本和 20 个验证样本
python scripts/generate_episodes.py \
    --num_train 100 \
    --num_val 20 \
    --scene_dataset test
```

这将创建：
- `data/episodes_train.json` - 训练数据
- `data/episodes_val.json` - 验证数据

### 4. 运行快速训练测试

```bash
# 运行 100 步训练验证代码能否工作
python src/training/train.py \
    --config config.yaml \
    training.max_steps=100 \
    training.batch_size=4
```

### 5. （可选）运行演示

```bash
# 如果有预训练模型
python demo.py --checkpoint checkpoints/best.pt
```

---

## 📚 接下来做什么？

### 选项 A: 小规模实验（推荐）

1. **下载更大的数据集**

   下载 HM3D minival（约 2GB）：
   ```bash
   # 需要先注册 https://aihabitat.org/datasets/hm3d/
   python -m habitat_sim.utils.datasets_download \
       --username <your-username> \
       --password <your-password> \
       --uids hm3d_minival_v0.2
   ```

2. **生成更多训练数据**

   ```bash
   python scripts/generate_episodes.py \
       --num_train 5000 \
       --num_val 500 \
       --scene_dataset hm3d
   ```

3. **完整训练**

   ```bash
   python src/training/train.py --config config.yaml
   ```

### 选项 B: 完整训练（发表论文）

1. 下载完整 HM3D 或 Matterport3D 数据集（详见 `HABITAT_SETUP_GUIDE.md`）
2. 生成 10000+ episodes
3. 运行完整训练流程（可能需要数天）

---

## 🐛 遇到问题？

### 问题 1: 找不到 habitat 模块

```bash
# 确保在正确的环境中
conda activate vla_gr
pip install habitat-lab==0.3.3 habitat-sim==0.3.3
```

### 问题 2: 找不到场景文件

```bash
# 检查场景是否下载
ls -la data/scene_datasets/

# 如果为空，重新下载
python scripts/download_habitat_data.py --test-scenes
```

### 问题 3: Episodes 生成失败

```bash
# 查看详细日志
python scripts/generate_episodes.py \
    --num_train 10 \
    --num_val 5 \
    --scene_dataset test
```

### 问题 4: GPU 内存不足

```bash
# 减小 batch size
python src/training/train.py \
    --config config.yaml \
    training.batch_size=4
```

### 问题 5: 训练非常慢

可能原因：
- num_workers 设置过高（改为 2-4）
- 数据增强过多（关闭部分增强）
- 场景加载慢（使用更小的场景数据集）

修改 `config.yaml`:
```yaml
training:
  num_workers: 2  # 减少 worker 数量
  batch_size: 8   # 减小 batch size
```

---

## 📖 更多文档

- **完整安装指南**: `HABITAT_SETUP_GUIDE.md` - 详细的 Habitat 环境设置
- **项目文档**: `README.md` - 项目概览和架构说明
- **部署指南**: `DEPLOYMENT_GUIDE.md` - 生产环境部署
- **API 文档**: `docs/` - 详细的 API 文档

---

## ✅ 验证检查清单

在开始训练前，确保：

- [ ] Habitat-Sim 和 Habitat-Lab 安装成功
- [ ] 至少有一个场景数据集（test scenes 即可）
- [ ] Episodes 文件已生成（`data/episodes_*.json`）
- [ ] 能够成功运行 100 步训练
- [ ] GPU 正常工作（如果使用 GPU）

运行验证脚本：
```bash
python scripts/verify_installation.py
```

---

## 🎯 推荐学习路径

**第 1 天**: 环境搭建
- 安装 Habitat
- 下载测试场景
- 运行快速测试

**第 2-3 天**: 熟悉代码
- 阅读 `README.md` 了解架构
- 运行 demo 查看效果
- 修改配置文件实验

**第 1 周**: 小规模训练
- 下载 HM3D minival
- 生成 1000-5000 episodes
- 调整超参数

**第 2-4 周**: 完整训练
- 下载完整数据集
- 生成完整 episodes
- 运行完整训练流程
- 评估和分析结果

---

## 💡 提示

1. **先用小数据集测试**: 用 100 个 episodes 确保代码能跑通
2. **监控训练过程**: 使用 TensorBoard 或 W&B 查看训练曲线
3. **定期保存检查点**: 训练可能需要很长时间
4. **阅读日志**: 日志文件包含重要的调试信息

---

## 🆘 获取帮助

- 查看 Issues: https://github.com/your-org/vla-gr-navigation/issues
- 阅读文档: `docs/` 目录
- 检查日志: `logs/vla_gr.log`

祝训练顺利！🚀
