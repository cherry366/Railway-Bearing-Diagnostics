# 基于迁移学习的轴承跨域故障诊断

> 🏆 **华为杯研究生竞赛国奖作品** · 2025 年“华为杯”第二十二届中国研究生数学建模竞赛 · 国家级奖项（三等奖）
> 题目：高速列车轴承智能故障诊断 · 赛题 E
> 完整论文：[`docs/paper/award_paper.pdf`](docs/paper/award_paper.pdf)

**一句话**：用试验台架（源域）的振动数据训练出一个高精度诊断模型，再通过带约束的迁移学习把它“搬”到无标签的实际列车轴承（目标域）上，并为迁移的每一个环节给出可解释的证据链。

---

## 亮点

| # | 内容 | 关键结果 |
|---|---|---|
| 1 | **41 维多域特征**（时域 / 频域 / 时频域）→ PCA 降维 | 26795 × 41 → 13 维，累计方差 **96.48%** |
| 2 | **按故障机理自适应滑窗**（外圈用 T_BPFO、内圈用 T_BPFI、滚动体用 T_FTF/T_BSF） | 窗口/步长随故障类型变化，而非固定参数 |
| 3 | **五模型对比**：XGBoost / LR / NB / SVM / BP | XGBoost 准确率 **0.9991**，AUC **1.000** |
| 4 | **两种迁移算法**：UTL-ABC（无约束）/ CTL-ABC（类别平衡 + 特征一致性约束） | MMD **0.2962 → 0.2886**，平均置信度 **0.853 → 0.865** |
| 5 | **三层可解释性框架**（事前 / 迁移过程 / 事后） | 16 个目标域文件中 **11 个（68.8%）** 高置信度 |

---

## 系统架构

```
源域 161 个台架数据文件（有标签，7 类）
        │
        ▼
①  数据清洗：缺失 FE 通道相似样本填充 → 箱线图异常值修复 → 线性去趋势 → 5 点滑动均值去噪 → Z-score
        │
        ▼
②  特征工程：时域 + 频域 + 时频域 共 41 维  →  PCA  →  13 维（96.48%）
        │
        ▼
③  源域诊断：SMOTE 平衡至每类 8038 条（新增 29471 条）→ 7:3 划分 → 五模型对比 → **XGBoost**（Acc 0.9991）
        │
        ├──────────────────────────────┐
        ▼                              ▼
④ 目标域适配                         数据侧：16 个无标签文件 → 1 万点窗 / 1000 步长 → 3952 窗 → 13 维
        │
        ▼
⑤ 跨域迁移（起点为源域 XGBoost）
   ├── UTL-ABC：MMD(RBF) + CORAL 对齐 + 置信度伪标签 + 动态阈值 τ   →  MMD 0.2962
   └── CTL-ABC：+ 类别平衡约束 V_balance + 特征一致性约束 V_consistency →  MMD 0.2886
        │
        ▼
⑥ 三层可解释性：树结构透明 + 特征重要性 → t-SNE 迁移路径 + 约束量化 → 决策路径回溯 + 故障机理映射 + ECE
        │
        ▼
16 个目标域文件标签 + 可靠性分级（高 11 / 中 2 / 低 3）
```

---

## 目录结构

```
bearing-transfer-diagnosis/
├── code/
│   ├── config.py                  # 统一路径（只读仓库 + work/ 运行目录）
│   ├── requirements.txt
│   ├── data_pipeline/             # ① ② 特征工程
│   │   ├── build_label_table.py       # 标签/尺寸/载荷/故障特征频率表
│   │   ├── denoise_detrend_normalize.py
│   │   ├── sliding_window.py          # 机理驱动的自适应滑窗
│   │   └── pca_reduce.py              # 41 维 → 13 维
│   ├── source_domain/
│   │   └── train_classifiers.py   # ③ SMOTE + 五模型训练评估
│   ├── transfer/
│   │   ├── target_preprocess.py   # ④ 目标域滑窗 + PCA
│   │   ├── utl_abc.py             # ⑤ 无约束迁移
│   │   ├── ctl_abc.py             # ⑤ 有约束迁移（最终模型）
│   │   └── compare_utl_vs_ctl.py  # 两算法对比实验
│   ├── explainability/
│   │   ├── explain_three_levels.py # ⑥ 三层可解释性
│   │   └── constraint_analysis.py  # 约束机制量化
│   └── models/                    # 已训练模型（开箱即用）
├── data/
│   ├── processed/                 # 仓库自带的中间结果（可直接跑训练/迁移）
│   └── README.md
├── docs/
│   ├── paper/                     # 论文 PDF + 赛题说明
│   ├── 01_problem_and_data.md
│   ├── 02_feature_engineering.md
│   ├── 03_source_diagnosis.md
│   ├── 04_cross_domain_transfer.md
│   └── 05_explainability.md
├── results/
│   ├── tables/                    # 6 张结果表
│   └── figures/                   # 14 张关键图
└── README.md
```

---

## 快速开始

```bash
pip install -r code/requirements.txt

# 批量推理：加载已训练 XGBoost，输出目标域 16 个文件在“迁移前”的预测与置信度
python code/models/model_usage_example.py

# 目标域迁移（CTL-ABC，论文最终方案）
python code/transfer/ctl_abc.py

# 两种迁移算法对比
python code/transfer/compare_utl_vs_ctl.py
```

运行约定：仓库目录**只读**，所有产物写到 `work/`（`code/config.py` 会在首次运行时把
`data/processed/*.csv` 与 `code/models/*.pkl` 复制进去，并 `chdir` 到该目录）。
需要换盘或换输出位置时设置环境变量 `BD_ROOT`。

> `model_usage_example.py` 跑的是**未经迁移**的源域模型直接推理（迁移前基线）：16 个文件里
> 9 个与论文表 9（CTL-ABC 迁移后）不同，大量文件被直接判为 `N`，平均置信度只有 **0.387**；
> 迁移后平均置信度升到 **0.865**。这组差异本身就是"为什么必须做迁移"的最直观证据，
> 详见 [`code/models/README.md`](code/models/README.md)。

**开箱可跑**的部分：源域训练（③）、目标域迁移（④⑤）、可解释性分析（⑥）——所需中间结果与模型均已随仓库提供。
需要自备原始数据的部分：赛道 ① ②（原始 `.mat`/xlsx 信号文件，见 [`data/README.md`](data/README.md)）。

---

## 关键结果

### 源域五模型对比（13 维 PCA 特征，16880 条测试样本）

| 模型 | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **XGBoost** | **0.9991** | **0.9991** | **0.9991** | **0.9991** |
| BP 神经网络 | 0.9972 | 0.9972 | 0.9972 | 0.9972 |
| SVM | 0.9655 | 0.9680 | 0.9655 | 0.9654 |
| 逻辑回归 | 0.9167 | 0.9256 | 0.9167 | 0.9162 |
| 朴素贝叶斯 | 0.8889 | 0.8968 | 0.8889 | 0.8879 |

### 两种迁移算法（目标域 3952 个窗口）

| 指标 | CTL-ABC | UTL-ABC |
|---|---|---|
| 最终 MMD | **0.2886** | 0.2962 |
| 平均置信度 | **0.865** | 0.853 |
| 最高置信度 | **0.992** | 0.991 |
| 置信度标准差 | **0.281** | 0.290 |
| 最终约束违反 | **1.1667** | — |
| MMD 下降率 | **9.32%** | 5.43% |

两算法预测一致率 **92.7%**。

### 目标域 16 个文件的最终标定（CTL-ABC）

| 文件 | 标签 | 置信度 | 可靠性 | 文件 | 标签 | 置信度 | 可靠性 |
|---|---|---|---|---|---|---|---|---|
| A | DE-B | 0.546 | 低 | I | FE-B | 0.911 | 高 |
| B | DE-B | 0.986 | 高 | J | DE-B | 0.824 | 高 |
| C | DE-B | 0.853 | 高 | K | FE-B | 0.288 | 低 |
| D | DE-OR | 0.965 | 高 | L | DE-B | 0.721 | 中 |
| E | DE-IR | 0.978 | 高 | M | FE-B | 0.484 | 低 |
| F | DE-OR | 0.926 | 高 | N | DE-IR | 0.957 | 高 |
| G | DE-B | 0.871 | 高 | O | DE-B | 0.980 | 高 |
| H | DE-OR | 0.977 | 高 | P | DE-OR | 0.966 | 高 |

> 可靠性分级：置信度 > 0.8 为高，0.6–0.8 为中，< 0.6 为低。
> `results/tables/` 中的 CSV 是同一次实验直接落盘的记录，因随机种子与迭代截断，
> 个别置信度与论文表格存在 0.0X 量级差异，标签完全一致（见 [`results/README.md`](results/README.md)）。

---

## 文档

| 文档 | 内容 |
|---|---|
| [docs/01_problem_and_data.md](docs/01_problem_and_data.md) | 任务背景、源域/目标域数据规格、清洗流程 |
| [docs/02_feature_engineering.md](docs/02_feature_engineering.md) | 41 维特征、故障特征频率、自适应滑窗、PCA |
| [docs/03_source_diagnosis.md](docs/03_source_diagnosis.md) | SMOTE 平衡、五模型训练与评估 |
| [docs/04_cross_domain_transfer.md](docs/04_cross_domain_transfer.md) | UTL-ABC / CTL-ABC 原理、参数、结果对比 |
| [docs/05_explainability.md](docs/05_explainability.md) | 事前 / 迁移过程 / 事后三层可解释性 |

---

## 技术栈

Python 3.10+ · NumPy · pandas · SciPy · scikit-learn · XGBoost · Matplotlib / Seaborn · joblib

## 引用

若本项目对你的工作有帮助，请引用该获奖论文：

```
《基于迁移学习的轴承跨域故障诊断研究》.
2025 年“华为杯”第二十二届中国研究生数学建模竞赛 E 题, 国家级奖项（三等奖）, 2025.
```
