# -*- coding: utf-8 -*-
"""模型加载与使用示例。

仓库自带的模型是 pickle 格式（非 joblib 压缩），统一用 `pickle` 读取。
模型是在**标准化后的 13 维 PCA 特征**上训练的（树的分裂阈值位于标准化空间），
因此推理前必须用训练数据（`balanced_feature_set.csv`）重新拟合一个 StandardScaler
并做同样的变换——`scaler.pkl` 里只存了特征名，原始缩放参数没有保留。

脚本会对 `data/processed/target_domain_pca13.csv` 的 3952 个窗口做一次**未经迁移**的
直接推理，并按文件聚合（众数投票 + 平均置信度）。结果就是"迁移前"的基线：
大量文件被判成 N、置信度普遍偏低，这正好量化了源域与目标域之间的分布偏移。
"""
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import MODEL_DIR, PROCESSED, WORK, use_workdir

use_workdir()

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_xgboost.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

FEATURE_COLUMNS = ["PC%d" % i for i in range(1, 14)]
CLASS_NAMES = ["DE-B", "DE-IR", "DE-OR", "FE-B", "FE-IR", "FE-OR", "N"]


def load_pickle(path):
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_metadata():
    """优先从 model_metadata.json 取特征列与类别名。"""
    global FEATURE_COLUMNS, CLASS_NAMES
    if os.path.isfile(METADATA_PATH):
        with open(METADATA_PATH, encoding="utf-8") as fh:
            meta = json.load(fh)
        FEATURE_COLUMNS = meta.get("feature_columns", FEATURE_COLUMNS)
        CLASS_NAMES = meta.get("class_names", CLASS_NAMES)
        return meta
    return {}


meta = load_metadata()
best_model = load_pickle(BEST_MODEL_PATH)
encoder = load_pickle(ENCODER_PATH)       # 历史文件：存的是 7 个类别名
scaler_file = load_pickle(SCALER_PATH)    # 历史文件：存的是 13 个特征名


def build_scaler():
    """用训练数据重拟合标准化器；取不到训练数据时退回源域全量特征。"""
    if hasattr(scaler_file, "transform"):
        return scaler_file
    for name in ("balanced_feature_set.csv", "df2_pca_features.csv"):
        path = os.path.join(WORK, name)
        if os.path.isfile(path):
            df = pd.read_csv(path, encoding="utf-8-sig", usecols=FEATURE_COLUMNS)
            return StandardScaler().fit(df.values)
    raise RuntimeError("找不到用于拟合标准化器的特征表")


scaler = build_scaler()


def decode(values):
    """把模型输出的整数编码还原成类别名。"""
    values = np.asarray(values).ravel()
    if hasattr(encoder, "inverse_transform"):
        return encoder.inverse_transform(values)
    names = encoder if isinstance(encoder, (list, tuple, np.ndarray)) else CLASS_NAMES
    return np.array(names)[values]


def predict_with_best_model(new_data):
    """输入含 13 个主成分列的 DataFrame，返回 (类别标签, 概率矩阵)。"""
    X = scaler.transform(new_data[FEATURE_COLUMNS].values)
    y_pred = best_model.predict(X)
    y_proba = best_model.predict_proba(X)
    return decode(y_pred), y_proba


def predict_single_sample(sample_dict):
    """输入单个样本的特征字典，返回 (类别标签, 概率向量)。"""
    sample_df = pd.DataFrame([sample_dict])
    pred, proba = predict_with_best_model(sample_df)
    return pred[0], proba[0]


def demo_target_domain():
    """对目标域 16 个文件的全部窗口推理，并聚合成文件级"迁移前"基线。"""
    path = os.path.join(PROCESSED, "target_domain_pca13.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    pred, proba = predict_with_best_model(df)
    group_col = "file_label" if "file_label" in df.columns else df.columns[0]

    detail = pd.DataFrame({
        "file": df[group_col],
        "pred": pred,
        "confidence": proba.max(axis=1),
    })
    agg = (detail.groupby("file")
                 .agg(n_windows=("pred", "size"),
                      prediction=("pred", lambda s: s.value_counts().idxmax()),
                      confidence=("confidence", "mean"))
                 .reset_index())
    print(agg.to_string(index=False))
    print("\n迁移前平均置信度: %.3f（迁移后 CTL-ABC 可达 0.865，见 docs/04）"
          % detail["confidence"].mean())
    return agg


if __name__ == "__main__":
    if meta:
        print("最佳模型:", meta.get("best_model"))
        print("测试集准确率: %.4f" % meta.get("best_accuracy", float("nan")))
        print("训练时间:", meta.get("training_time"))
    print("加载模型:", os.path.basename(BEST_MODEL_PATH))
    print("特征列:", FEATURE_COLUMNS)
    print("类别:", CLASS_NAMES)
    print("标准化器: 由训练数据重新拟合\n")
    demo_target_domain()

