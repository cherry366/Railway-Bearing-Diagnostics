# 数据说明

## 一、原始数据从哪里来

赛题给出的数据下载地址（公开、长期有效）：

- 链接：<https://pan.baidu.com/s/1H8nHXyMTv085jGRaiBOr0Q?pwd=anih>
- 提取码：`anih`

下载后按下述目录放置即可（目录名不要改，脚本靠文件名匹配）：

```
data/
├── source/          # 源域：161 个台架轴承数据文件
│   └── ...
├── target/          # 目标域：16 个列车轴承文件 A~P（.xlsx，各 8 s @ 32 kHz）
└── raw/
    └── e1.csv       # 源域原始信号汇总表（由 source/ 下文件逐个读取 DE/FE/BA 通道后汇总）
```

> `data/raw/` 与原始信号文件体积很大（数 GB），已被 `.gitignore` 忽略，不随仓库分发。

### 源域（试验台架，有标签）

| 项 | 说明 |
|---|---|
| 采集位置 | 驱动端 DE、风扇端 FE、基座 BA（本项目只用 DE / FE） |
| 轴承型号 | DE：SKF6205（滚动体数 9，d = 0.3126 in，D = 1.537 in）；FE：SKF6203（9，0.2656 in，1.122 in） |
| 采样率 | 驱动端 12 kHz / 48 kHz；风扇端 12 kHz |
| 故障类型 | 外圈 OR、内圈 IR、滚动体 B、正常 N（单一故障，无复合故障） |
| 故障尺寸 | OR：0.007 / 0.014 / 0.021 in；IR、B：0.007 / 0.014 / 0.021 / 0.028 in |
| 载荷 | 0 / 1 / 2 / 3 马力；转速记录在 RPM 变量中 |
| 文件命名 | 如 `B007_0` = 0 载荷下 0.007 in 的滚动体故障 |

### 目标域（实际列车轴承，无标签）

| 项 | 说明 |
|---|---|
| 文件数 | 16 个，以 `A`~`P` 编号 |
| 采集时长 | 8 秒 |
| 采样率 | 32 kHz |
| 工况 | 车速约 90 km/h，轴承转速约 600 rpm |
| 标签 | **未知**（本题要标定的对象） |

## 二、仓库自带的中间结果（`data/processed/`）

为了让训练、迁移、可解释性三个阶段在没有原始信号的情况下也能直接复现，仓库提供了 4 个中间文件：

| 文件 | 规模 | 内容 |
|---|---|---|
| `df1.csv` | 161 行 | 源域文件级标签表：`Source / frequency / Data length / RPM / Fault type / size / Load / position / label1 / label2 / BPFO / BPFI / BSF / 各周期` |
| `df2_pca_features.csv` | 26795 × 15 | 源域 41 维特征降维后的 13 个主成分 + `label1 / label2` |
| `balanced_feature_set.csv` | 56266 × 15 | SMOTE 平衡后（每类 8038 条）的 13 维特征集，训练脚本直接使用 |
| `target_domain_pca13.csv` | 3952 × 14 | 目标域 3952 个窗口的 13 维主成分 + `file_label`（A~P） |

## 三、各阶段输入 / 输出对照

| 阶段 | 输入 | 输出（均在 `work/`） |
|---|---|---|
| ① 标签与特征频率 | `data/raw/e1.csv` + `data/source/` | `df1.csv`、`signals_clean/` |
| ② 去噪/去趋势/标准化 | `work/df1.csv`、`work/signals_clean/` | `signals_ready/` |
| ③ 自适应滑窗 | `work/df1.csv`、`work/signals_ready/` | `windows/windows_summary_complete.csv` |
| ④ PCA | `work/windows/windows_summary_complete.csv` | `df2_pca_features.csv` |
| ⑤ 训练 | `work/balanced_feature_set.csv`（缺失时回退到 `df2_pca_features.csv`） | 结果表 + `work/saved_models/` |
| ⑥ 目标域预处理 | `data/target/*.xlsx` | `target_windows/target_domain_pca_13d.csv` |
| ⑦ 迁移 | `work/df2_pca_features.csv` + `data/processed/target_domain_pca13.csv` | 预测 CSV |

> 未下载原始数据时，可从阶段 ⑤ 开始跑：`code/config.py` 会自动把 `data/processed/` 与
> `code/models/` 的文件复制到 `work/`。
