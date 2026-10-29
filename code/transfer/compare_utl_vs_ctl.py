# -*- coding: utf-8 -*-
# 路径约定：仓库目录只读，运行产物统一写入 work/（可用环境变量 BD_ROOT 换盘）。
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import WORK, DATA, PROCESSED, MODEL_DIR, use_workdir  # noqa: E402

use_workdir()

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from scipy.special import softmax
import warnings
import time
from datetime import datetime

warnings.filterwarnings('ignore')

# 固定随机种子
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 数据加载器（共用）
# ============================================================================
class DataLoader:
    """数据加载器"""
    
    def __init__(self, model_path=None, scaler_path=None, encoder_path=None):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        
        self.xgb_model = None
        self.scaler = None
        self.label_encoder = None
        
        if model_path:
            self.load_pretrained_tools()
        else:
            print("未指定模型路径，将创建新的编码器和标准化器")
            self.label_encoder = LabelEncoder()
            self.scaler = StandardScaler()
    
    def load_pretrained_tools(self):
        """加载预训练的模型和工具"""
        if self.model_path:
            try:
                with open(self.model_path, 'rb') as f:
                    self.xgb_model = pickle.load(f)
                print(f"✓ 成功加载XGBoost模型: {self.model_path}")
            except Exception as e:
                print(f"⚠ 加载XGBoost模型失败: {e}")
                self.xgb_model = None
        
        if self.scaler_path:
            try:
                with open(self.scaler_path, 'rb') as f:
                    loaded_scaler = pickle.load(f)
                    if hasattr(loaded_scaler, 'transform'):
                        self.scaler = loaded_scaler
                        print(f"✓ 成功加载标准化器: {self.scaler_path}")
                    else:
                        print(f"⚠ 标准化器格式不正确，创建新的标准化器")
                        self.scaler = StandardScaler()
            except Exception as e:
                print(f"⚠ 加载标准化器失败: {e}")
                self.scaler = StandardScaler()
        else:
            self.scaler = StandardScaler()
        
        if self.encoder_path:
            try:
                with open(self.encoder_path, 'rb') as f:
                    loaded_encoder = pickle.load(f)
                    if isinstance(loaded_encoder, LabelEncoder):
                        self.label_encoder = loaded_encoder
                        print(f"✓ 成功加载标签编码器: {self.encoder_path}")
                    elif isinstance(loaded_encoder, (list, np.ndarray)):
                        print(f"⚠ 检测到类别列表，创建新的标签编码器")
                        self.label_encoder = LabelEncoder()
                        self.label_encoder.classes_ = np.array(loaded_encoder)
                    else:
                        print(f"⚠ 标签编码器格式不正确: {type(loaded_encoder)}")
                        self.label_encoder = LabelEncoder()
            except Exception as e:
                print(f"⚠ 加载标签编码器失败: {e}")
                self.label_encoder = LabelEncoder()
        else:
            self.label_encoder = LabelEncoder()
    
    def load_data(self, source_path, target_path):
        """加载源域和目标域数据"""
        print("\n加载数据集...")
        
        try:
            source_df = pd.read_csv(source_path)
            print(f"源域数据形状: {source_df.shape}")
        except Exception as e:
            print(f"✗ 无法加载源域数据: {e}")
            return None, None, None, None
        
        feature_cols = [f'PC{i}' for i in range(1, 14)]
        X_source = source_df[feature_cols].values
        y_source_labels = source_df['label1'].values
        
        unique_labels = np.unique(y_source_labels)
        print(f"源域标签类别: {unique_labels}")
        label_counts = pd.Series(y_source_labels).value_counts()
        print(f"源域类别分布:\n{label_counts}")
        
        try:
            if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
                unknown_labels = set(y_source_labels) - set(self.label_encoder.classes_)
                if unknown_labels:
                    print(f"⚠ 发现未知标签: {unknown_labels}，重新训练编码器")
                    self.label_encoder = LabelEncoder()
                    y_source = self.label_encoder.fit_transform(y_source_labels)
                else:
                    y_source = self.label_encoder.transform(y_source_labels)
            else:
                print("训练新的标签编码器...")
                y_source = self.label_encoder.fit_transform(y_source_labels)
            
            print(f"编码后的标签范围: {np.min(y_source)} - {np.max(y_source)}")
            
        except Exception as e:
            print(f"✗ 标签编码失败: {e}")
            label_map = {label: i for i, label in enumerate(unique_labels)}
            y_source = np.array([label_map[label] for label in y_source_labels])
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = unique_labels
        
        try:
            target_df = pd.read_csv(target_path)
            print(f"\n目标域数据形状: {target_df.shape}")
        except Exception as e:
            print(f"✗ 无法加载目标域数据: {e}")
            return X_source, y_source, None, None
        
        X_target = target_df[feature_cols].values
        target_files = target_df['file_label'].values
        
        file_counts = pd.Series(target_files).value_counts()
        print(f"目标域文件分布:\n{file_counts}")
        
        if self.scaler is not None:
            try:
                if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
                    print("\n使用已训练的标准化器...")
                    X_source = self.scaler.transform(X_source)
                    X_target = self.scaler.transform(X_target)
                else:
                    print("\n训练新的标准化器...")
                    X_source = self.scaler.fit_transform(X_source)
                    X_target = self.scaler.transform(X_target)
                print("✓ 数据标准化完成")
            except Exception as e:
                print(f"⚠ 标准化失败: {e}，使用原始数据")
        
        return X_source, y_source, X_target, target_files

# ============================================================================
# 带约束的迁移学习 (CTL-ABC)
# ============================================================================
class ConstrainedTransferLearning:
    """带约束的迁移学习"""
    
    def __init__(self, base_model=None, label_encoder=None, epsilon=0.15, delta=1.0):
        self.base_model = base_model
        self.label_encoder = label_encoder if label_encoder is not None else LabelEncoder()
        self.epsilon = epsilon
        self.delta = delta
        
        if hasattr(self.label_encoder, 'classes_'):
            self.n_classes = len(self.label_encoder.classes_)
        else:
            self.n_classes = 4
        
        self.lambda1 = 0.3
        self.lambda2 = 0.5
        self.lambda3 = 0.1
        self.alpha1 = 1.0
        self.alpha2 = 0.5
        
        self.history = {
            'loss': [],
            'mmd': [],
            'constraint_violation': [],
            'pseudo_labels': [],
            'confidence': [],
            'time': []
        }
    
    def compute_mmd(self, X_source, X_target, gamma=1.0):
        """计算最大均值差异(MMD)"""
        n_s = len(X_source)
        n_t = len(X_target)
        
        max_samples = 500
        if n_s > max_samples:
            idx_s = np.random.choice(n_s, max_samples, replace=False)
            X_source = X_source[idx_s]
            n_s = max_samples
        if n_t > max_samples:
            idx_t = np.random.choice(n_t, max_samples, replace=False)
            X_target = X_target[idx_t]
            n_t = max_samples
        
        K_ss = rbf_kernel(X_source, X_source, gamma=gamma)
        K_tt = rbf_kernel(X_target, X_target, gamma=gamma)
        K_st = rbf_kernel(X_source, X_target, gamma=gamma)
        
        mmd = np.sum(K_ss)/(n_s**2) + np.sum(K_tt)/(n_t**2) - 2*np.sum(K_st)/(n_s*n_t)
        
        return np.sqrt(max(0, mmd))
    
    def coral_alignment(self, X_source, X_target, lambda_reg=0.01):
        """CORAL特征对齐"""
        d = X_source.shape[1]
        
        C_s = np.cov(X_source.T) + lambda_reg * np.eye(d)
        C_t = np.cov(X_target.T) + lambda_reg * np.eye(d)
        
        try:
            U_s, S_s, V_s = np.linalg.svd(C_s)
            C_s_sqrt_inv = U_s @ np.diag(1/np.sqrt(S_s + 1e-5)) @ V_s
            
            U_t, S_t, V_t = np.linalg.svd(C_t)
            C_t_sqrt = U_t @ np.diag(np.sqrt(S_t + 1e-5)) @ V_t
            
            X_source_aligned = (X_source - np.mean(X_source, axis=0)) @ C_s_sqrt_inv @ C_t_sqrt
            X_source_aligned += np.mean(X_target, axis=0)
            
        except Exception as e:
            X_source_aligned = (X_source - np.mean(X_source, axis=0)) / (np.std(X_source, axis=0) + 1e-6)
            X_source_aligned = X_source_aligned * np.std(X_target, axis=0) + np.mean(X_target, axis=0)
        
        return X_source_aligned, X_target
    
    def compute_class_balance_constraint(self, pseudo_probs, source_class_dist):
        """计算类别平衡约束违反程度"""
        target_dist = np.mean(pseudo_probs, axis=0)
        
        violation = 0
        for k in range(min(len(target_dist), len(source_class_dist))):
            diff = abs(target_dist[k] - source_class_dist[k])
            if diff > self.epsilon:
                violation += (diff - self.epsilon) ** 2
        
        return violation, target_dist
    
    def compute_feature_consistency_constraint(self, X_source, y_source, X_target, pseudo_labels):
        """计算特征一致性约束违反程度"""
        violation = 0
        
        for k in range(self.n_classes):
            source_k = X_source[y_source == k]
            target_k = X_target[pseudo_labels == k]
            
            if len(source_k) > 0 and len(target_k) > 0:
                source_center = np.mean(source_k, axis=0)
                target_center = np.mean(target_k, axis=0)
                
                dist = np.linalg.norm(source_center - target_center)
                if dist > self.delta:
                    violation += (dist - self.delta) ** 2
        
        return violation
    
    def iterative_optimization(self, X_source, y_source, X_target, max_iter=10, confidence_threshold=0.7):
        """迭代优化主流程"""
        print("\n开始带约束的迁移学习优化 (CTL-ABC)")
        
        unique_classes = np.unique(y_source)
        self.n_classes = len(unique_classes)
        source_class_dist = np.bincount(y_source, minlength=self.n_classes) / len(y_source)
        
        X_s_aligned = X_source.copy()
        X_t_aligned = X_target.copy()
        prev_labels = None
        
        if self.base_model is None:
            print("训练初始XGBoost模型...")
            self.base_model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
            train_size = min(5000, len(X_source))
            idx = np.random.choice(len(X_source), train_size, replace=False)
            self.base_model.fit(X_source[idx], y_source[idx])
        
        start_time = time.time()
        
        for iteration in range(max_iter):
            iter_start = time.time()
            
            mmd = self.compute_mmd(X_s_aligned, X_t_aligned)
            self.history['mmd'].append(mmd)
            
            if iteration > 0:
                X_s_aligned, X_t_aligned = self.coral_alignment(X_source, X_target)
            
            pseudo_probs = self.base_model.predict_proba(X_t_aligned)
            pseudo_labels = np.argmax(pseudo_probs, axis=1)
            confidence = np.max(pseudo_probs, axis=1)
            
            self.history['pseudo_labels'].append(pseudo_labels)
            self.history['confidence'].append(confidence)
            
            balance_violation, target_dist = self.compute_class_balance_constraint(
                pseudo_probs, source_class_dist
            )
            
            consistency_violation = self.compute_feature_consistency_constraint(
                X_s_aligned, y_source, X_t_aligned, pseudo_labels
            )
            
            self.history['constraint_violation'].append(
                balance_violation + consistency_violation
            )
            
            high_conf_mask = confidence > confidence_threshold
            n_high_conf = np.sum(high_conf_mask)
            
            if n_high_conf > 10:
                n_source_sample = min(2000, len(X_source))
                idx_source = np.random.choice(len(X_source), n_source_sample, replace=False)
                
                X_train = np.vstack([X_s_aligned[idx_source], X_t_aligned[high_conf_mask]])
                y_train = np.hstack([y_source[idx_source], pseudo_labels[high_conf_mask]])
                
                source_weights = np.ones(n_source_sample)
                target_weights = confidence[high_conf_mask]
                
                if balance_violation > 0.01:
                    for k in range(self.n_classes):
                        if k < len(target_dist) and k < len(source_class_dist):
                            if target_dist[k] > source_class_dist[k] + self.epsilon:
                                mask = pseudo_labels[high_conf_mask] == k
                                target_weights[mask] *= 0.8
                            elif target_dist[k] < source_class_dist[k] - self.epsilon:
                                mask = pseudo_labels[high_conf_mask] == k
                                target_weights[mask] *= 1.2
                
                weights = np.hstack([source_weights, target_weights])
                
                self.base_model = xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=RANDOM_SEED,
                    n_jobs=-1
                )
                self.base_model.fit(X_train, y_train, sample_weight=weights)
            
            total_loss = (self.lambda1 * mmd + 
                         self.alpha1 * balance_violation + 
                         self.alpha2 * consistency_violation)
            self.history['loss'].append(total_loss)
            
            iter_time = time.time() - iter_start
            self.history['time'].append(iter_time)
            
            if prev_labels is not None:
                label_changes = np.sum(pseudo_labels != prev_labels)
                change_rate = label_changes / len(pseudo_labels)
                
                if change_rate < 0.01 and balance_violation < 0.1:
                    break
            
            prev_labels = pseudo_labels.copy()
            
            if iteration > 3 and n_high_conf < len(X_target) * 0.5:
                confidence_threshold = max(0.5, confidence_threshold - 0.05)
        
        total_time = time.time() - start_time
        print(f"CTL-ABC总训练时间: {total_time:.2f}秒")
        
        return pseudo_labels, confidence

# ============================================================================
# 无约束的迁移学习 (UTL-ABC)
# ============================================================================
class UnconstrainedTransferLearning:
    """无约束的迁移学习"""
    
    def __init__(self, base_model=None, label_encoder=None):
        self.base_model = base_model
        self.label_encoder = label_encoder if label_encoder is not None else LabelEncoder()
        
        if hasattr(self.label_encoder, 'classes_'):
            self.n_classes = len(self.label_encoder.classes_)
        else:
            self.n_classes = 4
        
        self.lambda1 = 0.3
        self.lambda2 = 0.5
        self.lambda3 = 0.1
        
        self.history = {
            'loss': [],
            'mmd': [],
            'pseudo_labels': [],
            'confidence': [],
            'label_change_rate': [],
            'time': []
        }
    
    def compute_mmd(self, X_source, X_target, gamma=1.0):
        """计算最大均值差异(MMD)"""
        n_s = len(X_source)
        n_t = len(X_target)
        
        max_samples = 500
        if n_s > max_samples:
            idx_s = np.random.choice(n_s, max_samples, replace=False)
            X_source = X_source[idx_s]
            n_s = max_samples
        if n_t > max_samples:
            idx_t = np.random.choice(n_t, max_samples, replace=False)
            X_target = X_target[idx_t]
            n_t = max_samples
        
        K_ss = rbf_kernel(X_source, X_source, gamma=gamma)
        K_tt = rbf_kernel(X_target, X_target, gamma=gamma)
        K_st = rbf_kernel(X_source, X_target, gamma=gamma)
        
        mmd = np.sum(K_ss)/(n_s**2) + np.sum(K_tt)/(n_t**2) - 2*np.sum(K_st)/(n_s*n_t)
        
        return np.sqrt(max(0, mmd))
    
    def coral_alignment(self, X_source, X_target, lambda_reg=0.01):
        """CORAL特征对齐"""
        d = X_source.shape[1]
        
        C_s = np.cov(X_source.T) + lambda_reg * np.eye(d)
        C_t = np.cov(X_target.T) + lambda_reg * np.eye(d)
        
        try:
            U_s, S_s, V_s = np.linalg.svd(C_s)
            C_s_sqrt_inv = U_s @ np.diag(1/np.sqrt(S_s + 1e-5)) @ V_s
            
            U_t, S_t, V_t = np.linalg.svd(C_t)
            C_t_sqrt = U_t @ np.diag(np.sqrt(S_t + 1e-5)) @ V_t
            
            X_source_aligned = (X_source - np.mean(X_source, axis=0)) @ C_s_sqrt_inv @ C_t_sqrt
            X_source_aligned += np.mean(X_target, axis=0)
            
        except Exception as e:
            X_source_aligned = (X_source - np.mean(X_source, axis=0)) / (np.std(X_source, axis=0) + 1e-6)
            X_source_aligned = X_source_aligned * np.std(X_target, axis=0) + np.mean(X_target, axis=0)
        
        return X_source_aligned, X_target
    
    def iterative_optimization(self, X_source, y_source, X_target, max_iter=10, confidence_threshold=0.7):
        """迭代优化主流程（无约束版本）"""
        print("\n开始无约束迁移学习优化 (UTL-ABC)")
        
        unique_classes = np.unique(y_source)
        self.n_classes = len(unique_classes)
        source_class_dist = np.bincount(y_source, minlength=self.n_classes) / len(y_source)
        
        X_s_aligned = X_source.copy()
        X_t_aligned = X_target.copy()
        prev_labels = None
        
        if self.base_model is None:
            print("训练初始XGBoost模型...")
            self.base_model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=RANDOM_SEED,
                n_jobs=-1
            )
            train_size = min(5000, len(X_source))
            idx = np.random.choice(len(X_source), train_size, replace=False)
            self.base_model.fit(X_source[idx], y_source[idx])
        
        start_time = time.time()
        
        for iteration in range(max_iter):
            iter_start = time.time()
            
            mmd = self.compute_mmd(X_s_aligned, X_t_aligned)
            self.history['mmd'].append(mmd)
            
            if iteration > 0:
                X_s_aligned, X_t_aligned = self.coral_alignment(X_source, X_target)
            
            pseudo_probs = self.base_model.predict_proba(X_t_aligned)
            pseudo_labels = np.argmax(pseudo_probs, axis=1)
            confidence = np.max(pseudo_probs, axis=1)
            
            self.history['pseudo_labels'].append(pseudo_labels)
            self.history['confidence'].append(confidence)
            
            high_conf_mask = confidence > confidence_threshold
            n_high_conf = np.sum(high_conf_mask)
            
            if n_high_conf > 10:
                n_source_sample = min(2000, len(X_source))
                idx_source = np.random.choice(len(X_source), n_source_sample, replace=False)
                
                X_train = np.vstack([X_s_aligned[idx_source], X_t_aligned[high_conf_mask]])
                y_train = np.hstack([y_source[idx_source], pseudo_labels[high_conf_mask]])
                
                source_weights = np.ones(n_source_sample)
                target_weights = confidence[high_conf_mask]
                weights = np.hstack([source_weights, target_weights])
                
                self.base_model = xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=RANDOM_SEED,
                    n_jobs=-1
                )
                self.base_model.fit(X_train, y_train, sample_weight=weights)
            
            total_loss = self.lambda1 * mmd
            self.history['loss'].append(total_loss)
            
            iter_time = time.time() - iter_start
            self.history['time'].append(iter_time)
            
            if prev_labels is not None:
                label_changes = np.sum(pseudo_labels != prev_labels)
                change_rate = label_changes / len(pseudo_labels)
                self.history['label_change_rate'].append(change_rate)
                
                if change_rate < 0.01:
                    break
            
            prev_labels = pseudo_labels.copy()
            
            if iteration > 3 and n_high_conf < len(X_target) * 0.3:
                confidence_threshold = max(0.5, confidence_threshold - 0.05)
        
        total_time = time.time() - start_time
        print(f"UTL-ABC总训练时间: {total_time:.2f}秒")
        
        return pseudo_labels, confidence

# ============================================================================
# 对比实验和可视化
# ============================================================================
def run_comparison_experiment(X_source, y_source, X_target, target_files, 
                            base_model, label_encoder):
    """运行对比实验"""
    
    print("\n" + "="*80)
    print("开始对比实验")
    print("="*80)
    
    # 运行CTL-ABC算法
    print("\n[1] 运行带约束的迁移学习算法 (CTL-ABC)...")
    ctl_learner = ConstrainedTransferLearning(
        base_model=base_model,
        label_encoder=label_encoder,
        epsilon=0.15,
        delta=1.0
    )
    ctl_predictions, ctl_confidence = ctl_learner.iterative_optimization(
        X_source, y_source, X_target,
        max_iter=30,
        confidence_threshold=0.7
    )
    
    # 运行UTL-ABC算法
    print("\n[2] 运行无约束迁移学习算法 (UTL-ABC)...")
    utl_learner = UnconstrainedTransferLearning(
        base_model=base_model,
        label_encoder=label_encoder
    )
    utl_predictions, utl_confidence = utl_learner.iterative_optimization(
        X_source, y_source, X_target,
        max_iter=30,
        confidence_threshold=0.7
    )
    
    # 处理结果
    results = {
        'CTL': {
            'predictions': ctl_predictions,
            'confidence': ctl_confidence,
            'history': ctl_learner.history
        },
        'UTL': {
            'predictions': utl_predictions,
            'confidence': utl_confidence,
            'history': utl_learner.history
        }
    }
    
    return results

def compute_label_change_rate(history):
    """计算标签变化率"""
    label_change_rates = []
    if 'pseudo_labels' in history and len(history['pseudo_labels']) > 1:
        for i in range(1, len(history['pseudo_labels'])):
            prev_labels = history['pseudo_labels'][i-1]
            curr_labels = history['pseudo_labels'][i]
            change_rate = np.sum(prev_labels != curr_labels) / len(curr_labels)
            label_change_rates.append(change_rate)
    return label_change_rates

def compute_label_change_rate(history):
    """计算标签变化率"""
    label_change_rates = []
    if 'pseudo_labels' in history and len(history['pseudo_labels']) > 1:
        for i in range(1, len(history['pseudo_labels'])):
            prev_labels = history['pseudo_labels'][i-1]
            curr_labels = history['pseudo_labels'][i]
            change_rate = np.sum(prev_labels != curr_labels) / len(curr_labels)
            label_change_rates.append(change_rate)
    return label_change_rates

def visualize_comparison(results, X_source, y_source):
    """生成对比可视化（3x3布局）"""
    
    fig = plt.figure(figsize=(15, 12))
    
    # 1. MMD距离对比
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(results['CTL']['history']['mmd'], 'o-', label='CTL-ABC', linewidth=2)
    ax1.plot(results['UTL']['history']['mmd'], 's-', label='UTL-ABC', linewidth=2)
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('MMD距离')
    ax1.set_title('域差异变化对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 总损失对比
    ax2 = plt.subplot(3, 3, 2)
    ax2.plot(results['CTL']['history']['loss'], 'o-', label='CTL-ABC', linewidth=2)
    ax2.plot(results['UTL']['history']['loss'], 's-', label='UTL-ABC', linewidth=2)
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('总损失')
    ax2.set_title('总损失变化对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 平均置信度对比
    ax3 = plt.subplot(3, 3, 3)
    ctl_avg_conf = [np.mean(c) for c in results['CTL']['history']['confidence']]
    utl_avg_conf = [np.mean(c) for c in results['UTL']['history']['confidence']]
    ax3.plot(ctl_avg_conf, 'o-', label='CTL-ABC', linewidth=2)
    ax3.plot(utl_avg_conf, 's-', label='UTL-ABC', linewidth=2)
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('平均置信度')
    ax3.set_title('预测置信度对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 置信度分布对比（直方图）
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(results['CTL']['confidence'], bins=30, alpha=0.5, label='CTL-ABC', edgecolor='blue')
    ax4.hist(results['UTL']['confidence'], bins=30, alpha=0.5, label='UTL-ABC', edgecolor='orange')
    ax4.set_xlabel('置信度')
    ax4.set_ylabel('样本数')
    ax4.set_title('最终置信度分布对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 类别分布对比
    ax5 = plt.subplot(3, 3, 5)
    unique_ctl, counts_ctl = np.unique(results['CTL']['predictions'], return_counts=True)
    unique_utl, counts_utl = np.unique(results['UTL']['predictions'], return_counts=True)
    
    width = 0.35
    
    # 确保两个算法的类别对齐
    all_classes = sorted(set(unique_ctl) | set(unique_utl))
    ctl_counts_aligned = [counts_ctl[np.where(unique_ctl == c)[0][0]] if c in unique_ctl else 0 for c in all_classes]
    utl_counts_aligned = [counts_utl[np.where(unique_utl == c)[0][0]] if c in unique_utl else 0 for c in all_classes]
    
    ax5.bar(np.arange(len(all_classes)) - width/2, ctl_counts_aligned, width, label='CTL-ABC', alpha=0.8)
    ax5.bar(np.arange(len(all_classes)) + width/2, utl_counts_aligned, width, label='UTL-ABC', alpha=0.8)
    ax5.set_xlabel('类别')
    ax5.set_ylabel('样本数')
    ax5.set_title('预测类别分布对比')
    ax5.set_xticks(np.arange(len(all_classes)))
    ax5.set_xticklabels(all_classes)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 训练时间对比
    ax6 = plt.subplot(3, 3, 6)
    if 'time' in results['CTL']['history'] and len(results['CTL']['history']['time']) > 0:
        ctl_cumtime = np.cumsum(results['CTL']['history']['time'])
        utl_cumtime = np.cumsum(results['UTL']['history']['time'])
        ax6.plot(ctl_cumtime, 'o-', label='CTL-ABC', linewidth=2)
        ax6.plot(utl_cumtime, 's-', label='UTL-ABC', linewidth=2)
        ax6.set_xlabel('迭代次数')
        ax6.set_ylabel('累计时间(秒)')
        ax6.set_title('训练时间对比')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 7. MMD减少比例
    ax7 = plt.subplot(3, 3, 7)
    if len(results['CTL']['history']['mmd']) > 0:
        ctl_mmd_reduction = (results['CTL']['history']['mmd'][0] - results['CTL']['history']['mmd'][-1]) / results['CTL']['history']['mmd'][0] * 100
        utl_mmd_reduction = (results['UTL']['history']['mmd'][0] - results['UTL']['history']['mmd'][-1]) / results['UTL']['history']['mmd'][0] * 100
        
        bars = ax7.bar(['CTL-ABC', 'UTL-ABC'], [ctl_mmd_reduction, utl_mmd_reduction], color=['blue', 'orange'])
        ax7.set_ylabel('MMD减少比例(%)')
        ax7.set_title('域适应效果对比')
        ax7.grid(True, alpha=0.3)
        
        # 在条形图上添加数值
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
    
    # 8. 源域类别分布（参考）
    ax8 = plt.subplot(3, 3, 8)
    unique_source, counts_source = np.unique(y_source, return_counts=True)
    ax8.bar(unique_source, counts_source, color='green', alpha=0.7)
    ax8.set_xlabel('类别')
    ax8.set_ylabel('样本数')
    ax8.set_title('源域类别分布（参考）')
    ax8.grid(True, alpha=0.3)
    
    # 9. 预测稳定性（标签变化率）- 包含两种算法
    ax9 = plt.subplot(3, 3, 9)
    
    # 计算CTL-ABC的标签变化率
    ctl_change_rates = compute_label_change_rate(results['CTL']['history'])
    
    # 使用已有的UTL-ABC标签变化率或重新计算
    if 'label_change_rate' in results['UTL']['history']:
        utl_change_rates = results['UTL']['history']['label_change_rate']
    else:
        utl_change_rates = compute_label_change_rate(results['UTL']['history'])
    
    if len(ctl_change_rates) > 0:
        ax9.plot(range(1, len(ctl_change_rates)+1), ctl_change_rates, 'o-', label='CTL-ABC', linewidth=2)
    if len(utl_change_rates) > 0:
        ax9.plot(range(1, len(utl_change_rates)+1), utl_change_rates, 's-', label='UTL-ABC', linewidth=2)
    
    ax9.set_xlabel('迭代次数')
    ax9.set_ylabel('标签变化率')
    ax9.set_title('预测稳定性对比')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('CTL-ABC vs UTL-ABC 迁移学习算法对比分析', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig

def visualize_constraint_satisfaction(results):
    """单独生成约束满足情况图"""
    if 'constraint_violation' not in results['CTL']['history']:
        return None
    
    fig, ax = plt.figure(figsize=(8, 6)), plt.gca()
    ax.plot(results['CTL']['history']['constraint_violation'], 'd-', color='red', linewidth=2, markersize=8)
    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_ylabel('约束违反程度', fontsize=12)
    ax.set_title('CTL-ABC约束满足情况', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 添加趋势线
    x = np.arange(len(results['CTL']['history']['constraint_violation']))
    z = np.polyfit(x, results['CTL']['history']['constraint_violation'], 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), "--", alpha=0.5, color='darkred', label=f'趋势线 (斜率: {z[0]:.4f})')
    ax.legend()
    
    plt.tight_layout()
    return fig

def visualize_performance_summary(results):
    """单独生成算法性能总结表"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = [
        ['指标', 'CTL-ABC', 'UTL-ABC', '优势算法'],
        ['最终MMD', f"{results['CTL']['history']['mmd'][-1]:.4f}", 
         f"{results['UTL']['history']['mmd'][-1]:.4f}",
         'CTL-ABC' if results['CTL']['history']['mmd'][-1] < results['UTL']['history']['mmd'][-1] else 'UTL-ABC'],
        ['平均置信度', f"{np.mean(results['CTL']['confidence']):.3f}", 
         f"{np.mean(results['UTL']['confidence']):.3f}",
         'CTL-ABC' if np.mean(results['CTL']['confidence']) > np.mean(results['UTL']['confidence']) else 'UTL-ABC'],
        ['最高置信度', f"{np.max(results['CTL']['confidence']):.3f}", 
         f"{np.max(results['UTL']['confidence']):.3f}",
         'CTL-ABC' if np.max(results['CTL']['confidence']) > np.max(results['UTL']['confidence']) else 'UTL-ABC'],
        ['置信度标准差', f"{np.std(results['CTL']['confidence']):.3f}",
         f"{np.std(results['UTL']['confidence']):.3f}",
         'CTL-ABC' if np.std(results['CTL']['confidence']) < np.std(results['UTL']['confidence']) else 'UTL-ABC'],
        ['迭代次数', str(len(results['CTL']['history']['mmd'])), 
         str(len(results['UTL']['history']['mmd'])),
         'CTL-ABC' if len(results['CTL']['history']['mmd']) < len(results['UTL']['history']['mmd']) else 'UTL-ABC'],
    ]
    
    # 添加MMD减少率
    if len(results['CTL']['history']['mmd']) > 0 and len(results['UTL']['history']['mmd']) > 0:
        ctl_reduction = (results['CTL']['history']['mmd'][0] - results['CTL']['history']['mmd'][-1]) / results['CTL']['history']['mmd'][0] * 100
        utl_reduction = (results['UTL']['history']['mmd'][0] - results['UTL']['history']['mmd'][-1]) / results['UTL']['history']['mmd'][0] * 100
        summary_data.append(['MMD减少率(%)', f"{ctl_reduction:.2f}", f"{utl_reduction:.2f}",
                           'CTL-ABC' if ctl_reduction > utl_reduction else 'UTL-ABC'])
    
    if 'constraint_violation' in results['CTL']['history'] and len(results['CTL']['history']['constraint_violation']) > 0:
        summary_data.append(['最终约束违反', f"{results['CTL']['history']['constraint_violation'][-1]:.4f}", 'N/A', 'CTL-ABC'])
    
    # 计算总体优势
    ctl_wins = sum(1 for row in summary_data[1:] if row[3] == 'CTL-ABC')
    utl_wins = sum(1 for row in summary_data[1:] if row[3] == 'UTL-ABC')
    
    table = ax.table(cellText=summary_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.3, 2)
    
    # 设置表格样式
    for i in range(len(summary_data)):
        for j in range(4):
            if i == 0:  # 标题行
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                if j == 0:  # 指标列
                    table[(i, j)].set_facecolor('#E0E0E0')
                elif j == 3:  # 优势算法列
                    if summary_data[i][j] == 'CTL-ABC':
                        table[(i, j)].set_facecolor('#BBDEFB')
                    else:
                        table[(i, j)].set_facecolor('#FFE0B2')
                else:
                    table[(i, j)].set_facecolor('#F5F5F5')
    
    ax.set_title(f'算法性能综合对比总结\n(CTL-ABC优势项: {ctl_wins}, UTL-ABC优势项: {utl_wins})', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def visualize_prediction_consistency(results):
    """单独生成预测一致性分析图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 左图：饼图
    agreement = np.sum(results['CTL']['predictions'] == results['UTL']['predictions'])
    disagreement = len(results['CTL']['predictions']) - agreement
    
    labels = ['预测一致', '预测不一致']
    sizes = [agreement, disagreement]
    colors = ['#66BB6A', '#FF7043']
    explode = (0.1, 0)
    
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.set_title(f'两种算法预测一致性分析\n(总样本数: {len(results["CTL"]["predictions"])})', 
                 fontsize=12, fontweight='bold')
    
    # 美化饼图文字
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')
    
    # 右图：置信度对比
    agreement_mask = results['CTL']['predictions'] == results['UTL']['predictions']
    
    # 一致和不一致样本的置信度分布
    ctl_conf_agree = results['CTL']['confidence'][agreement_mask]
    ctl_conf_disagree = results['CTL']['confidence'][~agreement_mask]
    utl_conf_agree = results['UTL']['confidence'][agreement_mask]
    utl_conf_disagree = results['UTL']['confidence'][~agreement_mask]
    
    positions = [1, 2, 4, 5]
    bp = ax2.boxplot([ctl_conf_agree, utl_conf_agree, ctl_conf_disagree, utl_conf_disagree],
                     positions=positions, widths=0.6, patch_artist=True,
                     labels=['CTL\n(一致)', 'UTL\n(一致)', 'CTL\n(不一致)', 'UTL\n(不一致)'])
    
    # 设置箱线图颜色
    colors_box = ['#BBDEFB', '#FFE0B2', '#BBDEFB', '#FFE0B2']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('置信度', fontsize=11)
    ax2.set_title('一致与不一致预测的置信度分布', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加均值标记
    means = [np.mean(ctl_conf_agree), np.mean(utl_conf_agree), 
            np.mean(ctl_conf_disagree) if len(ctl_conf_disagree) > 0 else 0,
            np.mean(utl_conf_disagree) if len(utl_conf_disagree) > 0 else 0]
    ax2.plot(positions, means, 'D', color='red', markersize=6, label='均值')
    ax2.legend()
    
    plt.suptitle('算法预测一致性详细分析', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig

def generate_detailed_report(results, target_files, label_encoder):
    """生成详细的对比报告"""
    
    print("\n" + "="*80)
    print("算法对比详细报告")
    print("="*80)
    
    # 1. 基本统计信息
    print("\n【基本统计信息】")
    print("-"*40)
    print(f"{'指标':<20} {'CTL-ABC':<15} {'UTL-ABC':<15}")
    print("-"*40)
    
    # 迭代次数
    ctl_iters = len(results['CTL']['history']['mmd'])
    utl_iters = len(results['UTL']['history']['mmd'])
    print(f"{'迭代次数':<20} {ctl_iters:<15} {utl_iters:<15}")
    
    # MMD减少
    if ctl_iters > 0 and utl_iters > 0:
        ctl_mmd_init = results['CTL']['history']['mmd'][0]
        ctl_mmd_final = results['CTL']['history']['mmd'][-1]
        utl_mmd_init = results['UTL']['history']['mmd'][0]
        utl_mmd_final = results['UTL']['history']['mmd'][-1]
        
        print(f"{'初始MMD':<20} {ctl_mmd_init:<15.4f} {utl_mmd_init:<15.4f}")
        print(f"{'最终MMD':<20} {ctl_mmd_final:<15.4f} {utl_mmd_final:<15.4f}")
        print(f"{'MMD减少率(%)':<20} {(ctl_mmd_init-ctl_mmd_final)/ctl_mmd_init*100:<15.2f} {(utl_mmd_init-utl_mmd_final)/utl_mmd_init*100:<15.2f}")
    
    # 置信度统计
    ctl_conf_mean = np.mean(results['CTL']['confidence'])
    ctl_conf_std = np.std(results['CTL']['confidence'])
    utl_conf_mean = np.mean(results['UTL']['confidence'])
    utl_conf_std = np.std(results['UTL']['confidence'])
    
    print(f"{'平均置信度':<20} {ctl_conf_mean:<15.3f} {utl_conf_mean:<15.3f}")
    print(f"{'置信度标准差':<20} {ctl_conf_std:<15.3f} {utl_conf_std:<15.3f}")
    print(f"{'最高置信度':<20} {np.max(results['CTL']['confidence']):<15.3f} {np.max(results['UTL']['confidence']):<15.3f}")
    print(f"{'最低置信度':<20} {np.min(results['CTL']['confidence']):<15.3f} {np.min(results['UTL']['confidence']):<15.3f}")
    
    # 2. 预测一致性分析
    print("\n【预测一致性分析】")
    print("-"*40)
    
    agreement_mask = results['CTL']['predictions'] == results['UTL']['predictions']
    agreement_rate = np.mean(agreement_mask) * 100
    
    print(f"预测一致率: {agreement_rate:.2f}%")
    print(f"一致样本数: {np.sum(agreement_mask)}/{len(agreement_mask)}")
    
    # 不一致样本的置信度分析
    if np.sum(~agreement_mask) > 0:
        ctl_conf_disagree = results['CTL']['confidence'][~agreement_mask]
        utl_conf_disagree = results['UTL']['confidence'][~agreement_mask]
        
        print(f"\n不一致样本的置信度分析:")
        print(f"  CTL-ABC平均置信度: {np.mean(ctl_conf_disagree):.3f}")
        print(f"  UTL-ABC平均置信度: {np.mean(utl_conf_disagree):.3f}")
    
    # 3. 按文件的预测结果
    print("\n【按文件预测结果对比】")
    print("-"*60)
    
    # 反向编码标签
    try:
        if hasattr(label_encoder, 'inverse_transform'):
            ctl_labels = label_encoder.inverse_transform(results['CTL']['predictions'])
            utl_labels = label_encoder.inverse_transform(results['UTL']['predictions'])
        else:
            ctl_labels = [f"Class_{i}" for i in results['CTL']['predictions']]
            utl_labels = [f"Class_{i}" for i in results['UTL']['predictions']]
    except:
        ctl_labels = [f"Class_{i}" for i in results['CTL']['predictions']]
        utl_labels = [f"Class_{i}" for i in results['UTL']['predictions']]
    
    # 创建DataFrame
    comparison_df = pd.DataFrame({
        'file': target_files,
        'CTL_pred': ctl_labels,
        'CTL_conf': results['CTL']['confidence'],
        'UTL_pred': utl_labels,
        'UTL_conf': results['UTL']['confidence'],
        'agreement': agreement_mask
    })
    
    # 按文件聚合
    file_comparison = {}
    unique_files = np.unique(target_files)
    
    print(f"{'文件':<10} {'CTL预测':<15} {'CTL置信度':<12} {'UTL预测':<15} {'UTL置信度':<12} {'一致性':<10}")
    print("-"*80)
    
    for file in sorted(unique_files):
        file_data = comparison_df[comparison_df['file'] == file]
        
        # CTL-ABC结果
        ctl_mode = file_data['CTL_pred'].mode()[0] if len(file_data['CTL_pred'].mode()) > 0 else file_data['CTL_pred'].iloc[0]
        ctl_conf = file_data['CTL_conf'].mean()
        
        # UTL-ABC结果
        utl_mode = file_data['UTL_pred'].mode()[0] if len(file_data['UTL_pred'].mode()) > 0 else file_data['UTL_pred'].iloc[0]
        utl_conf = file_data['UTL_conf'].mean()
        
        # 一致性
        agreement = 'Yes' if ctl_mode == utl_mode else 'No'
        
        print(f"{file:<10} {ctl_mode:<15} {ctl_conf:<12.3f} {utl_mode:<15} {utl_conf:<12.3f} {agreement:<10}")
        
        file_comparison[file] = {
            'CTL_prediction': ctl_mode,
            'CTL_confidence': ctl_conf,
            'UTL_prediction': utl_mode,
            'UTL_confidence': utl_conf,
            'agreement': agreement
        }
    
    # 4. 约束满足情况（仅CTL-ABC）
    if 'constraint_violation' in results['CTL']['history'] and len(results['CTL']['history']['constraint_violation']) > 0:
        print("\n【CTL-ABC约束满足情况】")
        print("-"*40)
        
        violations = results['CTL']['history']['constraint_violation']
        print(f"初始约束违反: {violations[0]:.4f}")
        print(f"最终约束违反: {violations[-1]:.4f}")
        print(f"改善率: {(violations[0] - violations[-1])/violations[0]*100:.2f}%")
    
    return file_comparison, comparison_df

# ============================================================================
# 主程序
# ============================================================================
def main():
    """主程序入口"""
    
    # 设置路径
    MODEL_PATH = os.path.join(MODEL_DIR, 'best_model_xgboost.pkl')
    SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
    ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    SOURCE_DATA_PATH = os.path.join(WORK, 'df2_pca_features.csv')
    TARGET_DATA_PATH = os.path.join(PROCESSED, 'target_domain_pca13.csv')
    
    print("="*80)
    print("CTL-ABC vs UTL-ABC 迁移学习算法对比实验")
    print("="*80)
    print(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"随机种子: {RANDOM_SEED}")
    
    # Step 1: 创建数据加载器
    loader = DataLoader(MODEL_PATH, SCALER_PATH, ENCODER_PATH)
    
    # Step 2: 加载数据
    X_source, y_source, X_target, target_files = loader.load_data(
        SOURCE_DATA_PATH, TARGET_DATA_PATH
    )
    
    if X_source is None or X_target is None:
        print("✗ 数据加载失败，程序退出")
        return
    
    # Step 3: 运行对比实验
    results = run_comparison_experiment(
        X_source, y_source, X_target, target_files,
        loader.xgb_model, loader.label_encoder
    )
    
    # Step 4: 生成可视化
    print("\n生成对比可视化...")
    
    # 生成主要对比图（3x3布局）
    fig_main = visualize_comparison(results, X_source, y_source)
    plt.savefig('transfer_learning_comparison_3x3.png', dpi=300, bbox_inches='tight')
    print("✓ 主要对比图已保存: transfer_learning_comparison_3x3.png")
    
    # 生成约束满足情况图（单独）
    fig_constraint = visualize_constraint_satisfaction(results)
    if fig_constraint:
        plt.savefig('ctl_constraint_satisfaction.png', dpi=300, bbox_inches='tight')
        print("✓ 约束满足情况图已保存: ctl_constraint_satisfaction.png")
    
    # 生成算法性能总结表（单独）
    fig_summary = visualize_performance_summary(results)
    plt.savefig('algorithm_performance_summary.png', dpi=300, bbox_inches='tight')
    print("✓ 算法性能总结表已保存: algorithm_performance_summary.png")
    
    # 生成预测一致性分析图（单独）
    fig_consistency = visualize_prediction_consistency(results)
    plt.savefig('prediction_consistency_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ 预测一致性分析图已保存: prediction_consistency_analysis.png")
    
    # Step 5: 生成详细报告
    file_comparison, comparison_df = generate_detailed_report(
        results, target_files, loader.label_encoder
    )
    
    # Step 6: 保存结果
    print("\n保存对比结果...")
    
    # 保存文件级别的对比结果
    file_comp_df = pd.DataFrame(file_comparison).T
    file_comp_df.to_csv('algorithm_comparison_by_file.csv')
    print("✓ 文件级别对比结果已保存: algorithm_comparison_by_file.csv")
    
    # 保存详细对比结果
    comparison_df.to_csv('algorithm_comparison_detailed.csv', index=False)
    print("✓ 详细对比结果已保存: algorithm_comparison_detailed.csv")
    
    # 保存历史记录
    import json
    history_data = {
        'CTL': {
            'mmd': results['CTL']['history']['mmd'],
            'loss': results['CTL']['history']['loss'],
            'constraint_violation': results['CTL']['history'].get('constraint_violation', []),
            'iterations': len(results['CTL']['history']['mmd'])
        },
        'UTL': {
            'mmd': results['UTL']['history']['mmd'],
            'loss': results['UTL']['history']['loss'],
            'iterations': len(results['UTL']['history']['mmd'])
        }
    }
    
    with open('algorithm_history.json', 'w') as f:
        json.dump(history_data, f, indent=2)
    print("✓ 训练历史已保存: algorithm_history.json")
    
    print("\n" + "="*80)
    print("实验完成！")
    print("="*80)
    
    # 显示图表
    plt.show()

# 运行主程序
if __name__ == "__main__":
    main()