# 结果索引

## 一、表格（`results/tables/`）

| 文件 | 行数 | 说明 |
|---|---|---|
| `label1_distribution_summary.csv` | 7 | 源域 7 类故障样本数与占比（PCA 后 26795 条） |
| `oversampling_comparison_report.csv` | 7 | SMOTE 前后各类样本数、增长率、新增样本数 |
| `model_performance_comparison.csv` | 5 | 五种模型的 Accuracy / Precision / Recall / F1 |
| `detailed_class_performance.csv` | 35 | 五个模型 × 7 类的逐类性能 |
| `target_domain_predictions.csv` | 16 | 目标域 16 个文件的主预测标签与平均置信度（每文件 247 窗） |
| `target_domain_window_predictions.csv` | 3952 | 目标域逐窗口预测明细（`file / prediction / confidence`） |

### 关于 `target_domain_predictions.csv` 与论文表 9 的差异

`results/tables/` 是实验脚本**直接落盘**的原始记录；论文表 9 是最终成稿时整理的结果。
两者 **16 个文件的标签完全一致**，置信度差异 ≤ 0.03（见下表），可以直接对照引用。

| 文件 | 落盘置信度 | 论文表 9 | 文件 | 落盘置信度 | 论文表 9 |
|---|---|---|---|---|---|
| A | 0.521 | 0.546 | I | 0.885 | 0.911 |
| B | 0.988 | 0.986 | J | 0.813 | 0.824 |
| C | 0.838 | 0.853 | K | 0.250 | 0.288 |
| D | 0.954 | 0.965 | L | 0.720 | 0.721 |
| E | 0.977 | 0.978 | M | 0.448 | 0.484 |
| F | 0.897 | 0.926 | N | 0.958 | 0.957 |
| G | 0.866 | 0.871 | O | 0.976 | 0.980 |
| H | 0.973 | 0.977 | P | 0.956 | 0.966 |

## 二、图表（`results/figures/`）

| 目录 | 文件 | 内容 |
|---|---|---|
| `boxplots/` | `B007_0_boxplot.png`、`IR007_0_boxplot.png`、`OR007@6_0_boxplot.png` | 三类故障样本 DE / FE / BA 三通道箱线图，用于异常值识别（论文图 2） |
| `denoised/` | `B007_0_DE_denoised.png`、`B007_0_FE_denoised.png`、`IR007_0_DE_denoised.png` | 5 点中心滑动平均去噪前后对比（论文图 3） |
| `detrended/` | `B007_0_DE_detrended.png`、`IR007_0_DE_detrended.png` | 线性去趋势前后对比（论文图 3） |
| `target_pca/` | `pca_variance_explained.png` | 目标域各主成分方差解释率 |
| | `pca_cumulative_variance.png` | 目标域累计方差解释率（13 维达 99.9%） |
| | `pca_2d_scatter.png` / `pca_3d_scatter.png` | 目标域 PC1-PC2 / PC1-PC3 分布（论文图 15、16） |
| | `pca_boxplots.png` | 目标域各主成分箱线图 |
| | `pca_loading_heatmap.png` | 主成分-特征载荷热力图（论文图 7） |

## 三、关键数值速查

- 源域：161 个文件 → 26795 个窗口 → 41 维 → 13 维（累计方差 96.48%）
- SMOTE 后：每类 8038 条，共 56266 条；训练 / 测试 = 39386 / 16880
- 源域最佳：XGBoost，Accuracy 0.9991，AUC 1.000
- 目标域：16 个文件 → 3952 个窗口（247 / 文件）
- 迁移：UTL-ABC MMD 0.2962 / 置信度 0.853；CTL-ABC MMD 0.2886 / 置信度 0.865；一致率 92.7%
- 可靠性：高 11 个（68.8%）、中 2 个、低 3 个

## 四、复现与随机性说明

迁移算法每一轮都会从源域**随机抽取** `min(2000, n_s)` 条样本与高置信伪标签样本一起重训
XGBoost，因此它是随机算法。`code/config.py` 已固定全局随机种子（`RANDOM_SEED = 42`），
同一次代码、同一份数据多次运行结果一致；但换种子或换环境重新跑，得到的是**另一条收敛轨迹**：

- 结论稳定的部分：**9 个文件（B、D、E、F、G、H、J、O、P）**在复跑中标签与可靠性等级不变；
- 可能漂移的部分：**7 个边界文件（A、C、I、K、L、M、N）**置信度落在 0.25–0.9，
  复跑时可能在相邻类别间变化（例如 A 在 DE-B / DE-OR 之间、I/M 在 DE-B / FE-B / DE-OR 之间）；
- 整体趋势稳定：最终 MMD 都落在 0.29–0.30，平均置信度 0.79–0.87，类别分布不会偏离到
  “全部判为同一类”的程度，但逐类占比（DE-B / DE-OR 的此消彼长）会有波动。

论文与 `results/tables/` 记录的是成稿那一次运行；若你复跑得到不同的边界结果，属于算法
随机性而非代码错误。
