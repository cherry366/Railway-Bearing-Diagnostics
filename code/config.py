# -*- coding: utf-8 -*-
"""统一路径约定。

仓库目录只当“只读素材”：所有脚本运行产物都落在 work/ 下，仓库自带的示例数据与
已训练模型会在首次运行时复制进 work/。需要换盘或换输出位置时，设置环境变量
BD_ROOT 指向新的工作根目录即可。
"""
import os
import shutil

# 脚本里有大量 plt.show()，统一使用非交互后端，避免在无显示环境（服务器 / CI）下阻塞
os.environ.setdefault("MPLBACKEND", "Agg")

# 迁移算法每轮会随机抽取源域样本，固定全局种子让多次运行结果一致
RANDOM_SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))     # code/
ROOT = os.path.dirname(HERE)                          # 仓库根目录
BASE = os.environ.get("BD_ROOT") or ROOT

DATA = os.path.join(BASE, "data")                     # 原始 / 公开数据
RAW = os.path.join(DATA, "raw")                       # 源域原始信号表
PROCESSED = os.path.join(DATA, "processed")           # 仓库自带的中间结果
WORK = os.path.join(BASE, "work")                     # 全部运行产物
MODEL_SRC = os.path.join(HERE, "models")              # 仓库自带的已训练模型
MODEL_DIR = os.path.join(WORK, "saved_models")        # 运行期模型目录

# 发布版中间结果：直接平铺到 work/ 下，脚本按文件名读取
_PROCESSED_FILES = [
    "df1.csv",
    "df2_pca_features.csv",
    "balanced_feature_set.csv",
    "target_domain_pca13.csv",
]

# 发布版模型：复制到 work/saved_models，训练脚本的产物也写在这里
_MODEL_FILES = [
    "best_model_xgboost.pkl",
    "scaler.pkl",
    "label_encoder.pkl",
    "model_metadata.json",
]


def _copy_missing(src_dir, dst_dir, names):
    os.makedirs(dst_dir, exist_ok=True)
    for name in names:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)
        if os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)


def use_workdir():
    """准备工作目录并切换过去，保证所有相对路径的读写都落在 work/。"""
    import numpy as np
    np.random.seed(RANDOM_SEED)
    os.makedirs(WORK, exist_ok=True)
    _copy_missing(PROCESSED, WORK, _PROCESSED_FILES)
    _copy_missing(MODEL_SRC, MODEL_DIR, _MODEL_FILES)
    os.chdir(WORK)
    return WORK
