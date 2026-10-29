# code/models 文件说明

| 文件 | 内容 | 类型 |
|---|---|---|
| `best_model_xgboost.pkl` | 源域最佳模型（XGBoost，100 棵树，max_depth=6，lr=0.1） | `XGBClassifier` pickle |
| `scaler.pkl` | 13 个主成分名（`PC1`~`PC13`） | `numpy.ndarray`，历史命名 |
| `label_encoder.pkl` | 7 个类别名（`DE-B / DE-IR / DE-OR / FE-B / FE-IR / FE-OR / N`） | `numpy.ndarray`，历史命名 |
| `model_metadata.json` | 五模型指标、特征列、类别名、训练/测试规模、各模型超参数 | JSON |
| `model_usage_example.py` | 加载模型 + 对目标域 3952 个窗口推理的示例脚本 | Python |

## 三点说明

1. **模型是在“标准化后”的 13 维 PCA 特征上训练的，输入必须同尺度。**
   XGBoost 本身对特征单调缩放不敏感，但训练时特征先经 `StandardScaler` 处理，
   树的分裂阈值因此落在标准化空间里。直接把 `data/processed/*.csv` 的原始 PCA 值喂进去，
   会得到明显偏低的结果（实测源域准确率仅 0.82）。
   `model_usage_example.py` 里的 `build_scaler()` 用随仓库提供的训练特征表
   （`balanced_feature_set.csv`，缺失时退回 `df2_pca_features.csv`）重新拟合标准化器，
   实测源域准确率恢复到 **0.9994**，与论文的 0.9991 同一量级。

2. **`scaler.pkl` / `label_encoder.pkl` 里只存了名称，没有保存真正的缩放参数。**
   原始实验里它们被写成了特征名数组与类别名数组（文件名沿用当时的命名）。
   迁移脚本对这两种情况都做了兼容：能取到 `transform` 就直接用，否则现场 `fit`。
   所以这三个 pickle 只是“名字对齐”用的，不是可直接调用的 sklearn 对象。

3. **未随仓库提供 `all_trained_models.pkl`。**
   原始实验产物中该文件已损坏（反序列化时报 `invalid load key`），只保留了最终使用的
   XGBoost 模型。需要对比 LR / NB / SVM / BP 四个模型时，用
   `code/source_domain/train_classifiers.py` 重新训练即可（训练数据已随仓库提供）。

## 直接推理 vs 迁移结果

`python code/models/model_usage_example.py` 跑的是**未经迁移**的源域模型直接推理
（即迁移学习的起点基线），16 个文件的预测如下表。它与论文表 9（CTL-ABC 迁移后）有 9 个文件
不同，另有 7 个（C、E、I、J、K、M、N）一致：

| 文件 | 直接推理（迁移前） | 置信度 | CTL-ABC 迁移后 | 迁移后置信度 |
|---|---|---|---|---|
| A | DE-OR | 0.174 | DE-B | 0.546 |
| B | N | 0.236 | DE-B | 0.986 |
| D | N | 0.368 | DE-OR | 0.965 |
| F | N | 0.177 | DE-OR | 0.926 |
| G | N | 0.179 | DE-B | 0.871 |
| H | DE-OR | 0.592 | DE-OR | 0.977 |
| L | N | 0.160 | DE-B | 0.721 |
| O | N | 0.274 | DE-B | 0.980 |
| P | N | 0.268 | DE-OR | 0.966 |

迁移前 16 个文件的**平均置信度只有 0.387**，迁移后为 **0.865**。这组对比正好说明迁移学习
在这里的作用：源域模型在目标域上大量判为 `N`（无故障）且置信度普遍偏低，
经过 MMD/CORAL 对齐与置信度伪标签迭代后，分布差异被压缩、判别边界重新校准，
同时给出可用于可靠性分级的置信度。
