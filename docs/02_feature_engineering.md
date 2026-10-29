# 02 特征工程

对应的代码：`code/data_pipeline/sliding_window.py`、`code/data_pipeline/pca_reduce.py`。

## 1. 故障特征频率

轴承局部缺陷会产生周期性冲击，其重复频率只取决于几何参数与转速：

转频 `fr = n / 60`（Hz，n 为 rpm），滚动体数 `Nb`、滚动体直径 `d`、节径 `D`：

```
BPFO = fr · (Nb/2) · (1 - d/D)          外圈
BPFI = fr · (Nb/2) · (1 + d/D)          内圈
BSF  = fr · (D/d) · [1 - (d/D)^2]       滚动体
FTF  = fr/2 · (1 - d/D)                 保持架（滚动体公转）
```

对应的周期 `T = 1/f`。这些量既作为特征进入模型，也是下面自适应滑窗的依据。

## 2. 按机理自适应滑窗

固定窗口会把不同故障的周期结构切碎。我们让窗口大小、步长跟随故障机理变化
（`calculate_window_params`）：

| 故障 | 窗口 W | 步长 S | 依据 |
|---|---|---|---|
| 外圈 OR | `Tr · fs` | `T_BPFO · fs` | 冲击以 T_BPFO 为间隔，幅值受转频调制 |
| 内圈 IR | `Tr · fs` | `T_BPFI · fs` | 故障点周期性进出载荷区 |
| 滚动体 B | `T_FTF · fs` | `T_BSF · fs` | 冲击受保持架公转频率调制 |
| 正常 N | `rand(0.01, 0.09) · fs` | `rand(0.001, 0.009) · fs` | 无周期结构，随机扰动增强多样性 |

窗口内样本数：`N_windows = floor((N - W)/S) + 1`。

这样做的好处是每个窗口都**至少覆盖一个完整的故障冲击周期**，时域统计量（峭度、峰值因子）
与频域能量才真正与故障类型相关。

## 3. 41 维特征

每个窗口计算三组特征（`calculate_time_domain_features / calculate_frequency_domain_features
/ calculate_time_frequency_features`）：

**时域（13）**：均值、标准差、方差、均方根、峰值、峰峰值、平均绝对值、波形系数、
峰值因子、脉冲指标、裕度指标、峭度、偏度。

**频域（6）**：频谱质心、频谱带宽、频谱偏度、频谱峭度、频谱熵、1–3 kHz 带通能量。

**时频域（4）**：分段频谱熵（`tf_entropy`）、平均频率（`tf_mean_freq`）、
小波能量及其能量占比（`wavelet_energy / wavelet_energy_ratio`）。

在此基础上补充故障特征频率族（BPFO、BPFI、BSF、FTF 及对应周期、转频周期）与位置 / 载荷
的独热编码，剔除重复标签与诊断无关列后，最终保留 **41 维**特征空间。

## 4. PCA 降维

41 维存在较强共线性（热力图可见 PC 与 `position_Orthogonal`、`Load_2.0` 完全相关）。
对标准化后的特征矩阵求协方差矩阵、分解特征值，取累计方差 ≥ 95% 的前 k 个主成分。
源域结果：**13 维，累计方差 96.48%**。

| 主成分 | 方差贡献率 | 累计 | 主成分 | 方差贡献率 | 累计 |
|---|---|---|---|---|---|
| PC1 | 22.79% | 22.79% | PC8 | 3.17% | 83.49% |
| PC2 | 19.65% | 42.45% | PC9 | 3.11% | 86.60% |
| PC3 | 14.68% | 57.13% | PC10 | 2.78% | 89.38% |
| PC4 | 9.67% | 66.80% | PC11 | 2.45% | 91.83% |
| PC5 | 5.80% | 72.60% | PC12 | 2.23% | 94.06% |
| PC6 | 4.42% | 77.02% | PC13 | 2.42% | **96.48%** |
| PC7 | 3.30% | 80.32% | | | |

前两个主成分的散点图中，除 DE-B 相对独立成簇外，其余类内集中、类间可分，说明降维没有
破坏可分性（`results/figures/target_pca/pca_2d_scatter.png` 展示了目标域版本）。

**目标域**用同一套 41 维特征 + PCA 流程，13 维累计方差达 **99.9%**；PC1 单独解释 57.9%，
PC1 与 PC3 合计约 89.6%。

## 5. 产物

| 产物 | 位置 |
|---|---|
| 逐个窗口的 41 维特征表 | `work/windows/windows_summary_complete.csv` |
| 13 维 PCA 特征 | `work/df2_pca_features.csv`（仓库副本 `data/processed/df2_pca_features.csv`） |
| 目标域 13 维 PCA | `work/target_windows/target_domain_pca_13d.csv`（仓库副本 `data/processed/target_domain_pca13.csv`） |
