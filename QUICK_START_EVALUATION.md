# 快速开始 - 运行完整评估

本指南帮助你快速运行VLA-GR的完整评估,生成可用于IROS/ICRA投稿的实验结果。

---

## 🎯 目标

生成满足顶会投稿标准的实验结果,包括:
- ✅ 主实验结果 (Table 1)
- ✅ 消融实验 (Table 2)
- ✅ Baseline对比 (Table 3)
- ✅ 统计显著性检验
- ✅ Publication-quality 图表
- ✅ LaTeX表格

---

## 📋 前置准备

### 方案A: 使用模拟评估 (演示框架)

如果你还没有Habitat环境,可以先使用模拟评估来演示整个框架:

```bash
# 不需要安装Habitat,直接运行
python scripts/run_complete_evaluation.py --num-episodes 500
```

**注意**: 这会生成模拟数据以演示评估框架。实际投稿**必须**使用真实Habitat环境。

### 方案B: 使用真实Habitat环境 (投稿必需)

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载Habitat数据
python scripts/download_habitat_data.py

# 3. 运行真实评估
python scripts/run_habitat_evaluation.py --dataset hm3d --num-episodes 500
```

---

## 🚀 运行评估

### 1. 基础评估 (500 episodes)

```bash
python scripts/run_complete_evaluation.py \
    --num-episodes 500 \
    --num-ablation-episodes 200 \
    --output-dir evaluation_results
```

**预计时间**:
- 模拟模式: ~5-10分钟
- 真实Habitat: ~2-4小时 (取决于硬件)

### 2. 快速评估 (调试用)

```bash
python scripts/run_complete_evaluation.py \
    --num-episodes 50 \
    --num-ablation-episodes 20 \
    --output-dir quick_test
```

### 3. 完整评估 (投稿用)

```bash
python scripts/run_complete_evaluation.py \
    --num-episodes 1000 \
    --num-ablation-episodes 500 \
    --output-dir final_results
```

---

## 📊 查看结果

评估完成后,检查输出目录:

```bash
cd evaluation_results/

# 查看主报告
cat EVALUATION_REPORT.md

# 查看生成的文件
tree -L 2
```

### 目录结构

```
evaluation_results/
├── EVALUATION_REPORT.md           # 主报告
├── figures/                        # 图表 (PDF + PNG)
│   ├── baseline_comparison.pdf
│   ├── ablation_study.pdf
│   └── performance_distribution.pdf
├── tables/                         # LaTeX表格
│   ├── main_results.tex
│   ├── ablation_study.tex
│   └── baseline_comparison.tex
├── raw_data/                       # 原始数据 (CSV)
│   ├── main_experiments.csv
│   ├── ablation_experiments.csv
│   └── baseline_comparisons.csv
└── statistics/                     # 统计分析
    └── statistical_tests.json
```

---

## 📈 关键输出文件

### 1. 图表 (用于论文)

- `figures/baseline_comparison.pdf` → Figure 1 in paper
- `figures/ablation_study.pdf` → Figure 2 in paper
- `figures/performance_distribution.pdf` → Supplementary

**使用方式**:
```latex
\begin{figure}
  \includegraphics[width=\columnwidth]{figures/baseline_comparison.pdf}
  \caption{Comparison with baseline methods on HM3D ObjectNav.}
  \label{fig:baselines}
\end{figure}
```

### 2. LaTeX表格

直接复制到论文:

```latex
\input{tables/main_results.tex}
\input{tables/ablation_study.tex}
\input{tables/baseline_comparison.tex}
```

### 3. 统计结果

```bash
# 查看统计检验结果
cat statistics/statistical_tests.json
```

示例输出:
```json
{
  "vs_dd_ppo": {
    "t_statistic": 12.456,
    "p_value": 0.0001,
    "significant": true,
    "effect_size": 0.823
  }
}
```

---

## 🔬 运行理论分析

```bash
python src/theory/theoretical_analysis.py
```

这会生成:
- `theoretical_analysis_report.md`: 详细理论分析
- 收敛性证明
- 样本复杂度分析
- 信息论界限

**用途**: 用于论文的理论部分,特别是投稿NeurIPS/ICRA时。

---

## ✅ 投稿前Checklist

使用以下checklist确保评估完整:

```bash
# 运行checklist
python scripts/check_evaluation_completeness.py
```

手动检查:

- [ ] **主实验**: ≥500 episodes on HM3D
- [ ] **消融实验**: ≥200 episodes per configuration
- [ ] **Baseline对比**: 至少3个方法对比
- [ ] **统计检验**: p-value < 0.05 vs baseline
- [ ] **Confidence intervals**: Bootstrap 95% CI
- [ ] **可视化**: 所有图表清晰,publication quality
- [ ] **LaTeX表格**: 格式正确,可直接使用
- [ ] **原始数据**: CSV文件完整,可复现

---

## 🎯 根据目标会议调整

### IROS 2025 (推荐)

```bash
# IROS要求相对宽松,500 episodes足够
python scripts/run_complete_evaluation.py \
    --num-episodes 500 \
    --output-dir iros2025_results
```

**强调**: 方法新颖性 + 合理实验

### ICRA 2026

```bash
# ICRA要求更严格,建议1000 episodes
python scripts/run_complete_evaluation.py \
    --num-episodes 1000 \
    --num-ablation-episodes 500 \
    --output-dir icra2026_results
```

**强调**: 系统实现 + 充分实验 + (最好)真实机器人

### RA-L (备选)

```bash
# RA-L格式较短,300 episodes可能足够
python scripts/run_complete_evaluation.py \
    --num-episodes 300 \
    --output-dir ral_results
```

**强调**: 技术贡献清晰 + 合理实验

---

## 🐛 常见问题

### Q1: 评估运行很慢

**A**: 减少episodes数量进行快速测试:
```bash
python scripts/run_complete_evaluation.py --num-episodes 50
```

### Q2: 内存不足

**A**: 减小batch size或分批运行:
```bash
python scripts/run_complete_evaluation.py --batch-size 16
```

### Q3: 如何添加更多baseline?

**A**: 编辑 `scripts/run_complete_evaluation.py`, 在 `run_baseline_comparisons()` 中添加:

```python
methods = {
    'VLA-GR (Ours)': 'vla_gr_full',
    'DD-PPO': 'dd_ppo',
    'CLIP-Nav': 'clip_nav',  # 添加新方法
    'Your-Method': 'your_method',
}
```

### Q4: 如何导出到Excel?

**A**: 使用pandas:
```python
import pandas as pd
df = pd.read_csv('evaluation_results/raw_data/main_experiments.csv')
df.to_excel('results.xlsx', index=False)
```

---

## 📝 生成论文材料

### 1. 导出所有表格

```bash
python scripts/export_paper_materials.py \
    --input-dir evaluation_results \
    --output-dir paper_materials \
    --format latex
```

### 2. 生成补充材料

```bash
python scripts/generate_supplementary.py \
    --results-dir evaluation_results \
    --output supplementary.pdf
```

### 3. 录制Demo视频 (如果有Habitat)

```bash
python scripts/record_demo_videos.py \
    --checkpoint checkpoints/best.pt \
    --num-videos 5 \
    --output-dir videos/
```

---

## 🎓 下一步

1. **运行评估**: 使用本指南运行完整评估
2. **检查结果**: 确保所有指标合理
3. **撰写论文**: 使用生成的表格和图表
4. **准备投稿**:
   - 修改latex模板
   - 插入生成的表格和图表
   - 引用统计结果
   - 讨论局限性

5. **投稿目标** (按优先级):
   - 首选: **IROS 2025** (Deadline: ~March 2025)
   - 备选: **RA-L** (Rolling submission)
   - 长期: **ICRA 2026** (需要更多工作)

---

## 📞 获取帮助

- **评估框架问题**: 查看 `scripts/run_complete_evaluation.py` 代码
- **理论分析**: 查看 `src/theory/theoretical_analysis.py`
- **项目评估**: 查看 `PROJECT_EVALUATION_REPORT.md`
- **投稿建议**: 查看 `PUBLICATION_RECOMMENDATIONS.md`

---

## ⚠️ 重要提醒

1. **模拟 vs 真实**: 当前框架使用模拟数据演示。投稿前**必须**在真实Habitat上运行!

2. **诚实报告**: 如果性能不如预期,诚实报告。强调方法新颖性而非SOTA性能。

3. **统计检验**: 确保所有对比都有p-value和confidence intervals。

4. **可复现性**: 保存所有随机种子和配置文件,确保结果可复现。

---

**祝评估顺利! 🚀**

*如有问题,请查阅详细文档或提出issue。*
