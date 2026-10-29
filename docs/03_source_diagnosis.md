# 03 源域故障诊断

对应代码：`code/source_domain/train_classifiers.py`
结果表：`results/tables/label1_distribution_summary.csv`、`oversampling_comparison_report.csv`、
`model_performance_comparison.csv`、`detailed_class_performance.csv`

## 1. 类别不均衡与 SMOTE

PCA 后源域共 **26795** 条样本，7 类的分布高度倾斜：

| 类别 | 样本数 | 占比 |
|---|---|---|
| DE-OR | 8038 | 30.00% |
| DE-IR | 5767 | 21.52% |
| DE-B | 4997 | 18.65% |
| N | 2929 | 10.93% |
| FE-OR | 2004 | 7.48% |
| FE-IR | 1826 | 6.81% |
| FE-B | 1234 | 4.61% |

以样本最多的 DE-OR（8038）为目标数量，对少数类做 SMOTE 合成：

```
x_new = x + rand(0,1) · (x_neighbor - x)      # k = 5 邻居
```

| 类别 | 原始 | 平衡后 | 新增 |
|---|---|---|---|
| DE-B | 4997 | 8038 | 3041 |
| DE-IR | 5767 | 8038 | 2271 |
| DE-OR | 8038 | 8038 | 0 |
| N | 2929 | 8038 | 5109 |
| FE-OR | 2004 | 8038 | 6034 |
| FE-IR | 1826 | 8038 | 6212 |
| FE-B | 1234 | 8038 | 6804 |
| **合计** | **26795** | **56266** | **29471** |

注意：SMOTE 只在**训练集内部**做，测试集保持真实分布，避免"合成样本泄漏进测试集"造成虚高。
按 7:3 分层划分后：训练 **39386** 条、测试 **16880** 条。

## 2. 五种模型

| 模型 | 关键配置 |
|---|---|
| XGBoost | `n_estimators=100, max_depth=6, learning_rate=0.1, objective=multi:softprob, random_state=42` |
| 逻辑回归 | L2 正则，`C=1.0`，`max_iter=1000`，ovr 多分类 |
| 朴素贝叶斯 | 高斯 NB，`var_smoothing=1e-9` |
| SVM | RBF 核，`C=1.0`，`probability=True` |
| BP 神经网络 | 隐层 `(100, 50)`，Adam，`lr=1e-3`，早停 |

XGBoost 的加法模型与目标函数：

```
ŷ_i = Σ_{k=1..K} f_k(x_i),  f_k ∈ F
Obj = Σ_i l(y_i, ŷ_i^(t-1) + f_t(x_i)) + Σ_k Ω(f_k)
Ω(f) = γT + 1/2 λ Σ_j w_j^2
```

节点增益用于后续的特征重要性量化：

```
G = 1/2 [ G_L^2/(H_L+λ) + G_R^2/(H_R+λ) - (G_L+G_R)^2/(H_L+H_R+λ) ]
```

## 3. 结果（16880 条测试样本）

| 模型 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **XGBoost** | **0.9991** | **0.9991** | **0.9991** | **0.9991** |
| BP 神经网络 | 0.9972 | 0.9972 | 0.9972 | 0.9972 |
| SVM | 0.9655 | 0.9680 | 0.9655 | 0.9654 |
| 逻辑回归 | 0.9167 | 0.9256 | 0.9167 | 0.9162 |
| 朴素贝叶斯 | 0.8889 | 0.8968 | 0.8889 | 0.8879 |

XGBoost 的 AUC 为 **1.000**。逐类来看，7 类的 Precision 都 ≥ 0.997，Recall 都 ≥ 0.997，
其中 DE-OR、DE-IR、FE-B、N 等类别的 F1 均超过 0.997——13 维 PCA 特征对 7 类故障的
区分能力已经接近上限，误差主要来自同端不同尺寸之间的边界样本。

## 4. 产物

| 产物 | 位置 |
|---|---|
| 平衡后的特征集 | `work/balanced_feature_set.csv`（仓库副本 `data/processed/balanced_feature_set.csv`） |
| 五模型指标 | `work/model_performance_comparison.csv` |
| 逐类指标 | `work/detailed_class_performance.csv` |
| 模型与预处理器 | `work/saved_models/`（`best_model_xgboost.pkl`、`scaler.pkl`、`label_encoder.pkl`、`model_metadata.json`）|

> 仓库只随附最终使用的 XGBoost 模型。其余四个模型的权重文件（`all_trained_models.pkl`）
> 在原始实验产物中已损坏无法反序列化，因此未纳入仓库；需要时重跑
> `train_classifiers.py` 即可完整复现（数据已随仓库提供）。

`code/models/` 里的同名文件就是这次训练的产物，可直接用于推理，见
`code/models/model_usage_example.py`。
