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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from scipy.optimize import minimize
from scipy.special import softmax
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 数据加载和预处理（保持不变）
# ============================================================================
class DataLoader:
    """数据加载器"""
    
    def __init__(self, model_path=None, scaler_path=None, encoder_path=None):
        """
        初始化
        
        Parameters:
        -----------
        model_path : str, XGBoost模型路径
        scaler_path : str, 标准化器路径
        encoder_path : str, 标签编码器路径
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.encoder_path = encoder_path
        
        # 初始化工具
        self.xgb_model = None
        self.scaler = None
        self.label_encoder = None
        
        # 尝试加载预训练模型和工具
        if model_path:
            self.load_pretrained_tools()
        else:
            print("未指定模型路径，将创建新的编码器和标准化器")
            self.label_encoder = LabelEncoder()
            self.scaler = StandardScaler()
    
    def load_pretrained_tools(self):
        """加载预训练的模型和工具"""
        # 加载XGBoost模型
        if self.model_path:
            try:
                with open(self.model_path, 'rb') as f:
                    self.xgb_model = pickle.load(f)
                print(f"✓ 成功加载XGBoost模型: {self.model_path}")
            except Exception as e:
                print(f"⚠ 加载XGBoost模型失败: {e}")
                self.xgb_model = None
        
        # 加载标准化器
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
        
        # 加载标签编码器
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
        """
        加载源域和目标域数据
        
        Parameters:
        -----------
        source_path : str, 源域数据路径
        target_path : str, 目标域数据路径
        
        Returns:
        --------
        X_source, y_source, X_target, target_info
        """
        print("\n加载数据集...")
        
        # 加载源域数据
        try:
            source_df = pd.read_csv(source_path)
            print(f"源域数据形状: {source_df.shape}")
        except Exception as e:
            print(f"❌ 无法加载源域数据: {e}")
            return None, None, None, None
        
        # 提取源域特征和标签
        feature_cols = [f'PC{i}' for i in range(1, 14)]
        X_source = source_df[feature_cols].values
        y_source_labels = source_df['label1'].values
        
        # 打印原始标签分布
        unique_labels = np.unique(y_source_labels)
        print(f"源域标签类别: {unique_labels}")
        label_counts = pd.Series(y_source_labels).value_counts()
        print(f"源域类别分布:\n{label_counts}")
        
        # 编码标签
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
            print(f"标签映射: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
            
        except Exception as e:
            print(f"❌ 标签编码失败: {e}")
            print("使用简单的标签映射...")
            label_map = {label: i for i, label in enumerate(unique_labels)}
            y_source = np.array([label_map[label] for label in y_source_labels])
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = unique_labels
        
        # 加载目标域数据
        try:
            target_df = pd.read_csv(target_path)
            print(f"\n目标域数据形状: {target_df.shape}")
        except Exception as e:
            print(f"❌ 无法加载目标域数据: {e}")
            return X_source, y_source, None, None
        
        X_target = target_df[feature_cols].values
        target_files = target_df['file_label'].values
        
        # 统计目标域文件分布
        file_counts = pd.Series(target_files).value_counts()
        print(f"目标域文件分布:\n{file_counts}")
        
        # 数据标准化
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
# 无约束迁移学习算法
# ============================================================================
class UnconstrainedTransferLearning:
    """无约束的迁移学习（用于对比）"""
    
    def __init__(self, base_model=None, label_encoder=None):
        """
        初始化
        
        Parameters:
        -----------
        base_model : 预训练的模型（可选）
        label_encoder : 标签编码器（可选）
        """
        self.base_model = base_model
        self.label_encoder = label_encoder if label_encoder is not None else LabelEncoder()
        
        # 获取类别数
        if hasattr(self.label_encoder, 'classes_'):
            self.n_classes = len(self.label_encoder.classes_)
        else:
            self.n_classes = 4  # 默认4个类别
        
        # 超参数（简化版）
        self.lambda1 = 0.8  # MMD权重
        self.lambda2 = 0.5  # 伪标签权重
        
        # 记录优化历史
        self.history = {
            'loss': [],
            'mmd': [],
            'pseudo_labels': [],
            'confidence': [],
            'target_distributions': []
        }
    
    def compute_mmd(self, X_source, X_target, gamma=1.0):
        """计算最大均值差异(MMD)"""
        n_s = len(X_source)
        n_t = len(X_target)
        
        # 为了计算效率，采样部分数据
        max_samples = 2000
        if n_s > max_samples:
            idx_s = np.random.choice(n_s, max_samples, replace=False)
            X_source = X_source[idx_s]
            n_s = max_samples
        if n_t > max_samples:
            idx_t = np.random.choice(n_t, max_samples, replace=False)
            X_target = X_target[idx_t]
            n_t = max_samples
        
        # 计算核矩阵
        K_ss = rbf_kernel(X_source, X_source, gamma=gamma)
        K_tt = rbf_kernel(X_target, X_target, gamma=gamma)
        K_st = rbf_kernel(X_source, X_target, gamma=gamma)
        
        # MMD计算
        mmd = np.sum(K_ss)/(n_s**2) + np.sum(K_tt)/(n_t**2) - 2*np.sum(K_st)/(n_s*n_t)
        
        return np.sqrt(max(0, mmd))
    
    def coral_alignment(self, X_source, X_target, lambda_reg=0.01):
        """CORAL特征对齐"""
        d = X_source.shape[1]
        
        # 计算协方差矩阵
        C_s = np.cov(X_source.T) + lambda_reg * np.eye(d)
        C_t = np.cov(X_target.T) + lambda_reg * np.eye(d)
        
        try:
            # 使用SVD分解计算矩阵平方根
            U_s, S_s, V_s = np.linalg.svd(C_s)
            C_s_sqrt_inv = U_s @ np.diag(1/np.sqrt(S_s + 1e-5)) @ V_s
            
            U_t, S_t, V_t = np.linalg.svd(C_t)
            C_t_sqrt = U_t @ np.diag(np.sqrt(S_t + 1e-5)) @ V_t
            
            # 对齐变换
            X_source_aligned = (X_source - np.mean(X_source, axis=0)) @ C_s_sqrt_inv @ C_t_sqrt
            X_source_aligned += np.mean(X_target, axis=0)
            
        except Exception as e:
            print(f"⚠ CORAL对齐失败: {e}，使用简化方法")
            X_source_aligned = (X_source - np.mean(X_source, axis=0)) / (np.std(X_source, axis=0) + 1e-6)
            X_source_aligned = X_source_aligned * np.std(X_target, axis=0) + np.mean(X_target, axis=0)
        
        return X_source_aligned, X_target
    
    def iterative_optimization(self, X_source, y_source, X_target, max_iter=100, confidence_threshold=0.7):
        """无约束迭代优化主流程"""
        print("\n" + "="*60)
        print("开始无约束迁移学习优化")
        print("="*60)
        
        # 计算源域类别分布（仅用于记录和分析）
        unique_classes = np.unique(y_source)
        self.n_classes = len(unique_classes)
        source_class_dist = np.bincount(y_source, minlength=self.n_classes) / len(y_source)
        print(f"源域类别分布: {source_class_dist}")
        self.source_class_dist = source_class_dist
        
        # 初始化
        X_s_aligned = X_source.copy()
        X_t_aligned = X_target.copy()
        prev_labels = None
        
        # 如果没有基础模型，训练一个
        if self.base_model is None:
            print("训练初始XGBoost模型...")
            self.base_model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            train_size = min(5000, len(X_source))
            idx = np.random.choice(len(X_source), train_size, replace=False)
            self.base_model.fit(X_source[idx], y_source[idx])
            print("✓ 初始模型训练完成")
        
        for iteration in range(max_iter):
            print(f"\n--- 迭代 {iteration+1}/{max_iter} ---")
            
            # Step 1: 计算MMD
            mmd = self.compute_mmd(X_s_aligned, X_t_aligned)
            self.history['mmd'].append(mmd)
            print(f"MMD距离: {mmd:.4f}")
            
            # Step 2: CORAL特征对齐
            if iteration > 0:
                X_s_aligned, X_t_aligned = self.coral_alignment(X_source, X_target)
            
            # Step 3: 生成伪标签和置信度
            try:
                pseudo_probs = self.base_model.predict_proba(X_t_aligned)
                pseudo_labels = np.argmax(pseudo_probs, axis=1)
                confidence = np.max(pseudo_probs, axis=1)
            except Exception as e:
                print(f"⚠ 预测失败: {e}，使用随机预测")
                pseudo_probs = softmax(np.random.randn(len(X_target), self.n_classes), axis=1)
                pseudo_labels = np.argmax(pseudo_probs, axis=1)
                confidence = np.max(pseudo_probs, axis=1)
            
            self.history['pseudo_labels'].append(pseudo_labels)
            self.history['confidence'].append(confidence)
            
            # 记录目标域分布（仅用于分析，不用于约束）
            target_dist = np.mean(pseudo_probs, axis=0)
            self.history['target_distributions'].append(target_dist)
            print(f"目标域预测分布: {target_dist.round(3)}")
            
            # Step 4: 选择高置信度样本
            high_conf_mask = confidence > confidence_threshold
            n_high_conf = np.sum(high_conf_mask)
            print(f"高置信度样本数: {n_high_conf}/{len(X_target)} ({n_high_conf/len(X_target)*100:.1f}%)")
            print(f"平均置信度: {np.mean(confidence):.3f}")
            
            # Step 5: 更新模型（无约束版本）
            if n_high_conf > 10:
                # 构建增强训练集
                n_source_sample = min(2000, len(X_source))
                idx_source = np.random.choice(len(X_source), n_source_sample, replace=False)
                
                X_train = np.vstack([X_s_aligned[idx_source], X_t_aligned[high_conf_mask]])
                y_train = np.hstack([y_source[idx_source], pseudo_labels[high_conf_mask]])
                
                # 简单的样本权重（无约束调整）
                source_weights = np.ones(n_source_sample)
                target_weights = confidence[high_conf_mask]
                weights = np.hstack([source_weights, target_weights])
                
                # 重新训练模型
                self.base_model = xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
                self.base_model.fit(X_train, y_train, sample_weight=weights)
            
            # Step 6: 计算总损失（简化版，只有MMD）
            total_loss = self.lambda1 * mmd
            self.history['loss'].append(total_loss)
            print(f"总损失: {total_loss:.4f}")
            
            # Step 7: 收敛检查
            if prev_labels is not None:
                label_changes = np.sum(pseudo_labels != prev_labels)
                change_rate = label_changes / len(pseudo_labels)
                print(f"标签变化率: {change_rate:.2%}")
                
                if change_rate < 0.01:
                    print("\n✓ 模型已收敛！")
                    break
            
            prev_labels = pseudo_labels.copy()
            
            # 动态调整置信度阈值
            if iteration > 3 and n_high_conf < len(X_target) * 0.5:
                confidence_threshold = max(0.5, confidence_threshold - 0.05)
                print(f"调整置信度阈值为: {confidence_threshold:.2f}")
        
        return pseudo_labels, confidence

# ============================================================================
# 带约束的迁移学习（保留原始版本用于对比）
# ============================================================================
class ConstrainedTransferLearning:
    """带约束的迁移学习"""
    
    def __init__(self, base_model=None, label_encoder=None, epsilon=0.15, delta=1.0):
        """
        初始化
        
        Parameters:
        -----------
        base_model : 预训练的模型（可选）
        label_encoder : 标签编码器（可选）
        epsilon : float, 类别平衡约束的容忍度
        delta : float, 特征一致性约束的阈值
        """
        self.base_model = base_model
        self.label_encoder = label_encoder if label_encoder is not None else LabelEncoder()
        self.epsilon = epsilon
        self.delta = delta
        
        # 获取类别数
        if hasattr(self.label_encoder, 'classes_'):
            self.n_classes = len(self.label_encoder.classes_)
        else:
            self.n_classes = 4  # 默认4个类别
        
        # 超参数
        self.lambda1 = 0.3  # MMD权重
        self.lambda2 = 0.5  # 伪标签权重
        self.lambda3 = 0.1  # 正则化权重
        self.alpha1 = 0.8   # 类别平衡约束权重
        self.alpha2 = 0.4   # 特征一致性约束权重
        
        # 记录优化历史
        self.history = {
            'loss': [],
            'mmd': [],
            'constraint_violation': [],
            'pseudo_labels': [],
            'confidence': [],
            'target_distributions': []
        }
    
    def compute_mmd(self, X_source, X_target, gamma=1.0):
        """计算最大均值差异(MMD)"""
        n_s = len(X_source)
        n_t = len(X_target)
        
        # 为了计算效率，采样部分数据
        max_samples = 2000
        if n_s > max_samples:
            idx_s = np.random.choice(n_s, max_samples, replace=False)
            X_source = X_source[idx_s]
            n_s = max_samples
        if n_t > max_samples:
            idx_t = np.random.choice(n_t, max_samples, replace=False)
            X_target = X_target[idx_t]
            n_t = max_samples
        
        # 计算核矩阵
        K_ss = rbf_kernel(X_source, X_source, gamma=gamma)
        K_tt = rbf_kernel(X_target, X_target, gamma=gamma)
        K_st = rbf_kernel(X_source, X_target, gamma=gamma)
        
        # MMD计算
        mmd = np.sum(K_ss)/(n_s**2) + np.sum(K_tt)/(n_t**2) - 2*np.sum(K_st)/(n_s*n_t)
        
        return np.sqrt(max(0, mmd))
    
    def coral_alignment(self, X_source, X_target, lambda_reg=0.01):
        """CORAL特征对齐"""
        d = X_source.shape[1]
        
        # 计算协方差矩阵
        C_s = np.cov(X_source.T) + lambda_reg * np.eye(d)
        C_t = np.cov(X_target.T) + lambda_reg * np.eye(d)
        
        try:
            # 使用SVD分解计算矩阵平方根
            U_s, S_s, V_s = np.linalg.svd(C_s)
            C_s_sqrt_inv = U_s @ np.diag(1/np.sqrt(S_s + 1e-5)) @ V_s
            
            U_t, S_t, V_t = np.linalg.svd(C_t)
            C_t_sqrt = U_t @ np.diag(np.sqrt(S_t + 1e-5)) @ V_t
            
            # 对齐变换
            X_source_aligned = (X_source - np.mean(X_source, axis=0)) @ C_s_sqrt_inv @ C_t_sqrt
            X_source_aligned += np.mean(X_target, axis=0)
            
        except Exception as e:
            print(f"⚠ CORAL对齐失败: {e}，使用简化方法")
            X_source_aligned = (X_source - np.mean(X_source, axis=0)) / (np.std(X_source, axis=0) + 1e-6)
            X_source_aligned = X_source_aligned * np.std(X_target, axis=0) + np.mean(X_target, axis=0)
        
        return X_source_aligned, X_target
    
    def compute_class_balance_constraint(self, pseudo_probs, source_class_dist):
        """计算类别平衡约束违反程度"""
        # 计算目标域的预测分布
        target_dist = np.mean(pseudo_probs, axis=0)
        
        # 计算违反程度
        violation = 0
        for k in range(min(len(target_dist), len(source_class_dist))):
            diff = abs(target_dist[k] - source_class_dist[k])
            if diff > self.epsilon:
                violation += (diff - self.epsilon) ** 2
        
        return violation, target_dist
    
    def compute_feature_consistency_constraint(self, X_source, y_source, X_target, pseudo_labels):
        """计算特征一致性约束违反程度"""
        violation = 0
        
        # 对每个类别计算违反程度
        for k in range(self.n_classes):
            # 源域中该类别的样本
            source_k = X_source[y_source == k]
            # 目标域中预测为该类别的样本
            target_k = X_target[pseudo_labels == k]
            
            if len(source_k) > 0 and len(target_k) > 0:
                # 计算类中心
                source_center = np.mean(source_k, axis=0)
                target_center = np.mean(target_k, axis=0)
                
                # 计算中心距离
                dist = np.linalg.norm(source_center - target_center)
                if dist > self.delta:
                    violation += (dist - self.delta) ** 2
        
        return violation
    
    def iterative_optimization(self, X_source, y_source, X_target, max_iter=100, confidence_threshold=0.7):
        """带约束的迭代优化主流程"""
        print("\n" + "="*60)
        print("开始带约束的迁移学习优化")
        print("="*60)
        
        # 计算源域类别分布
        unique_classes = np.unique(y_source)
        self.n_classes = len(unique_classes)
        source_class_dist = np.bincount(y_source, minlength=self.n_classes) / len(y_source)
        print(f"源域类别分布: {source_class_dist}")
        self.source_class_dist = source_class_dist
        
        # 初始化
        X_s_aligned = X_source.copy()
        X_t_aligned = X_target.copy()
        prev_labels = None
        
        # 如果没有基础模型，训练一个
        if self.base_model is None:
            print("训练初始XGBoost模型...")
            self.base_model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            train_size = min(5000, len(X_source))
            idx = np.random.choice(len(X_source), train_size, replace=False)
            self.base_model.fit(X_source[idx], y_source[idx])
            print("✓ 初始模型训练完成")
        
        for iteration in range(max_iter):
            print(f"\n--- 迭代 {iteration+1}/{max_iter} ---")
            
            # Step 1: 计算MMD
            mmd = self.compute_mmd(X_s_aligned, X_t_aligned)
            self.history['mmd'].append(mmd)
            print(f"MMD距离: {mmd:.4f}")
            
            # Step 2: CORAL特征对齐
            if iteration > 0:
                X_s_aligned, X_t_aligned = self.coral_alignment(X_source, X_target)
            
            # Step 3: 生成伪标签和置信度
            try:
                pseudo_probs = self.base_model.predict_proba(X_t_aligned)
                pseudo_labels = np.argmax(pseudo_probs, axis=1)
                confidence = np.max(pseudo_probs, axis=1)
            except Exception as e:
                print(f"⚠ 预测失败: {e}，使用随机预测")
                pseudo_probs = softmax(np.random.randn(len(X_target), self.n_classes), axis=1)
                pseudo_labels = np.argmax(pseudo_probs, axis=1)
                confidence = np.max(pseudo_probs, axis=1)
            
            self.history['pseudo_labels'].append(pseudo_labels)
            self.history['confidence'].append(confidence)
            
            # Step 4: 计算约束违反程度
            balance_violation, target_dist = self.compute_class_balance_constraint(
                pseudo_probs, source_class_dist
            )
            self.history['target_distributions'].append(target_dist)
            
            print(f"目标域预测分布: {target_dist.round(3)}")
            print(f"类别平衡约束违反: {balance_violation:.4f}")
            
            consistency_violation = self.compute_feature_consistency_constraint(
                X_s_aligned, y_source, X_t_aligned, pseudo_labels
            )
            print(f"特征一致性约束违反: {consistency_violation:.4f}")
            
            self.history['constraint_violation'].append(
                balance_violation + consistency_violation
            )
            
            # Step 5: 选择高置信度样本
            high_conf_mask = confidence > confidence_threshold
            n_high_conf = np.sum(high_conf_mask)
            print(f"高置信度样本数: {n_high_conf}/{len(X_target)} ({n_high_conf/len(X_target)*100:.1f}%)")
            print(f"平均置信度: {np.mean(confidence):.3f}")
            
            # Step 6: 更新模型（带约束版本）
            if n_high_conf > 10:
                # 构建增强训练集
                n_source_sample = min(2000, len(X_source))
                idx_source = np.random.choice(len(X_source), n_source_sample, replace=False)
                
                X_train = np.vstack([X_s_aligned[idx_source], X_t_aligned[high_conf_mask]])
                y_train = np.hstack([y_source[idx_source], pseudo_labels[high_conf_mask]])
                
                # 样本权重（带约束调整）
                source_weights = np.ones(n_source_sample)
                target_weights = confidence[high_conf_mask]
                
                # 根据约束调整权重
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
                
                # 重新训练模型
                self.base_model = xgb.XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
                self.base_model.fit(X_train, y_train, sample_weight=weights)
            
            # Step 7: 计算总损失（包含约束）
            total_loss = (self.lambda1 * mmd + 
                         self.alpha1 * balance_violation + 
                         self.alpha2 * consistency_violation)
            self.history['loss'].append(total_loss)
            print(f"总损失: {total_loss:.4f}")
            
            # Step 8: 收敛检查
            if prev_labels is not None:
                label_changes = np.sum(pseudo_labels != prev_labels)
                change_rate = label_changes / len(pseudo_labels)
                print(f"标签变化率: {change_rate:.2%}")
                
                if change_rate < 0.01 and balance_violation < 0.1:
                    print("\n✓ 模型已收敛！")
                    break
            
            prev_labels = pseudo_labels.copy()
            
            # 动态调整置信度阈值
            if iteration > 3 and n_high_conf < len(X_target) * 0.5:
                confidence_threshold = max(0.5, confidence_threshold - 0.05)
                print(f"调整置信度阈值为: {confidence_threshold:.2f}")
        
        return pseudo_labels, confidence

# ============================================================================
# 对比实验类
# ============================================================================
class ComparativeExperiment:
    """约束 vs 无约束迁移学习对比实验"""
    
    def __init__(self, X_source, y_source, X_target, target_files, loader):
        self.X_source = X_source
        self.y_source = y_source
        self.X_target = X_target
        self.target_files = target_files
        self.loader = loader
        
    def run_comparison(self, max_iter=100):
        """运行对比实验"""
        print("\n" + "="*80)
        print("开始对比实验：带约束 vs 无约束迁移学习")
        print("="*80)
        
        # 1. 运行带约束的迁移学习
        print("\n【实验1：带约束迁移学习】")
        constrained_learner = ConstrainedTransferLearning(
            base_model=self.loader.xgb_model,
            label_encoder=self.loader.label_encoder,
            epsilon=0.15,
            delta=1.0
        )
        
        predictions_constrained, confidence_constrained = constrained_learner.iterative_optimization(
            self.X_source, self.y_source, self.X_target,
            max_iter=max_iter,
            confidence_threshold=0.7
        )
        
        # 2. 运行无约束的迁移学习
        print("\n" + "="*60)
        print("\n【实验2：无约束迁移学习】")
        unconstrained_learner = UnconstrainedTransferLearning(
            base_model=self.loader.xgb_model,
            label_encoder=self.loader.label_encoder
        )
        
        predictions_unconstrained, confidence_unconstrained = unconstrained_learner.iterative_optimization(
            self.X_source, self.y_source, self.X_target,
            max_iter=max_iter,
            confidence_threshold=0.7
        )
        
        # 3. 分析对比结果
        results = self.analyze_results(
            constrained_learner, unconstrained_learner,
            predictions_constrained, predictions_unconstrained,
            confidence_constrained, confidence_unconstrained
        )
        
        # 4. 可视化对比
        self.visualize_comparison(constrained_learner, unconstrained_learner)
        
        return results, constrained_learner, unconstrained_learner
    
    def analyze_results(self, constrained_learner, unconstrained_learner,
                        pred_c, pred_u, conf_c, conf_u):
        """分析对比结果"""
        print("\n" + "="*80)
        print("对比分析结果")
        print("="*80)
        
        results = {}
        
        # 1. MMD对比
        if len(constrained_learner.history['mmd']) > 0 and len(unconstrained_learner.history['mmd']) > 0:
            results['mmd_comparison'] = {
                'constrained_initial': constrained_learner.history['mmd'][0],
                'constrained_final': constrained_learner.history['mmd'][-1],
                'unconstrained_initial': unconstrained_learner.history['mmd'][0],
                'unconstrained_final': unconstrained_learner.history['mmd'][-1]
            }
            
            print("\n1. MMD距离对比:")
            print(f"   带约束: {results['mmd_comparison']['constrained_initial']:.4f} → {results['mmd_comparison']['constrained_final']:.4f}")
            print(f"   无约束: {results['mmd_comparison']['unconstrained_initial']:.4f} → {results['mmd_comparison']['unconstrained_final']:.4f}")
        
        # 2. 置信度对比
        results['confidence_comparison'] = {
            'constrained_mean': np.mean(conf_c),
            'constrained_std': np.std(conf_c),
            'unconstrained_mean': np.mean(conf_u),
            'unconstrained_std': np.std(conf_u)
        }
        
        print("\n2. 置信度对比:")
        print(f"   带约束: 均值={results['confidence_comparison']['constrained_mean']:.3f}, 标准差={results['confidence_comparison']['constrained_std']:.3f}")
        print(f"   无约束: 均值={results['confidence_comparison']['unconstrained_mean']:.3f}, 标准差={results['confidence_comparison']['unconstrained_std']:.3f}")
        
        # 3. 预测一致性
        agreement = np.sum(pred_c == pred_u) / len(pred_c)
        results['prediction_agreement'] = agreement
        
        print(f"\n3. 预测一致性: {agreement:.1%}")
        
        # 4. 类别分布对比
        if hasattr(constrained_learner, 'source_class_dist'):
            source_dist = constrained_learner.source_class_dist
            
            # 计算目标域预测分布
            n_classes = len(source_dist)
            constrained_dist = np.bincount(pred_c, minlength=n_classes) / len(pred_c)
            unconstrained_dist = np.bincount(pred_u, minlength=n_classes) / len(pred_u)
            
            # 计算与源域分布的差异（KL散度）
            def kl_divergence(p, q):
                """计算KL散度"""
                p = np.clip(p, 1e-10, 1.0)
                q = np.clip(q, 1e-10, 1.0)
                return np.sum(p * np.log(p / q))
            
            kl_constrained = kl_divergence(constrained_dist, source_dist)
            kl_unconstrained = kl_divergence(unconstrained_dist, source_dist)
            
            results['distribution_comparison'] = {
                'source_dist': source_dist,
                'constrained_dist': constrained_dist,
                'unconstrained_dist': unconstrained_dist,
                'kl_constrained': kl_constrained,
                'kl_unconstrained': kl_unconstrained
            }
            
            print("\n4. 类别分布对比 (与源域的KL散度):")
            print(f"   带约束: {kl_constrained:.4f}")
            print(f"   无约束: {kl_unconstrained:.4f}")
            print(f"   （KL散度越小，说明分布越接近源域）")
        
        # 5. 收敛速度对比
        print(f"\n5. 收敛速度:")
        print(f"   带约束: {len(constrained_learner.history['mmd'])} 次迭代")
        print(f"   无约束: {len(unconstrained_learner.history['mmd'])} 次迭代")
        
        return results
    
    def visualize_comparison(self, constrained_learner, unconstrained_learner):
        """可视化对比结果"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. MMD对比
        ax = axes[0, 0]
        if len(constrained_learner.history['mmd']) > 0:
            ax.plot(constrained_learner.history['mmd'], 'o-', label='带约束', linewidth=2, markersize=6)
        if len(unconstrained_learner.history['mmd']) > 0:
            ax.plot(unconstrained_learner.history['mmd'], 's-', label='无约束', linewidth=2, markersize=6)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('MMD距离')
        ax.set_title('MMD距离收敛对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 损失函数对比
        ax = axes[0, 1]
        if len(constrained_learner.history['loss']) > 0:
            ax.plot(constrained_learner.history['loss'], 'o-', label='带约束', linewidth=2, markersize=6)
        if len(unconstrained_learner.history['loss']) > 0:
            ax.plot(unconstrained_learner.history['loss'], 's-', label='无约束', linewidth=2, markersize=6)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('总损失')
        ax.set_title('损失函数对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 平均置信度对比
        ax = axes[0, 2]
        if len(constrained_learner.history['confidence']) > 0:
            avg_conf_c = [np.mean(conf) for conf in constrained_learner.history['confidence']]
            ax.plot(avg_conf_c, 'o-', label='带约束', linewidth=2, markersize=6)
        if len(unconstrained_learner.history['confidence']) > 0:
            avg_conf_u = [np.mean(conf) for conf in unconstrained_learner.history['confidence']]
            ax.plot(avg_conf_u, 's-', label='无约束', linewidth=2, markersize=6)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('平均置信度')
        ax.set_title('置信度演化对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 类别分布演化（带约束）
        ax = axes[1, 0]
        if len(constrained_learner.history['target_distributions']) > 0:
            distributions = np.array(constrained_learner.history['target_distributions'])
            for i in range(distributions.shape[1]):
                ax.plot(distributions[:, i], label=f'类别{i}', linewidth=2)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('类别比例')
        ax.set_title('带约束：类别分布演化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 类别分布演化（无约束）
        ax = axes[1, 1]
        if len(unconstrained_learner.history['target_distributions']) > 0:
            distributions = np.array(unconstrained_learner.history['target_distributions'])
            for i in range(distributions.shape[1]):
                ax.plot(distributions[:, i], label=f'类别{i}', linewidth=2)
        ax.set_xlabel('迭代次数')
        ax.set_ylabel('类别比例')
        ax.set_title('无约束：类别分布演化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 约束违反度（仅带约束）
        ax = axes[1, 2]
        if hasattr(constrained_learner, 'history') and 'constraint_violation' in constrained_learner.history:
            if len(constrained_learner.history['constraint_violation']) > 0:
                ax.plot(constrained_learner.history['constraint_violation'], 'o-', 
                       color='red', linewidth=2, markersize=6)
                ax.set_xlabel('迭代次数')
                ax.set_ylabel('约束违反度')
                ax.set_title('约束满足过程')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('带约束 vs 无约束迁移学习对比分析', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig

# ============================================================================
# 主程序（增强版，包含对比实验）
# ============================================================================
def main_with_comparison():
    """主程序入口（含对比实验）"""
    
    # 设置路径
    MODEL_PATH = os.path.join(MODEL_DIR, 'best_model_xgboost.pkl')
    SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
    ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')
    SOURCE_DATA_PATH = os.path.join(WORK, 'df2_pca_features.csv')
    TARGET_DATA_PATH = os.path.join(PROCESSED, 'target_domain_pca13.csv')
    
    print("="*80)
    print("带约束 vs 无约束迁移学习对比实验系统")
    print("="*80)
    
    # Step 1: 创建数据加载器
    loader = DataLoader(MODEL_PATH, SCALER_PATH, ENCODER_PATH)
    
    # Step 2: 加载数据
    X_source, y_source, X_target, target_files = loader.load_data(
        SOURCE_DATA_PATH, TARGET_DATA_PATH
    )
    
    if X_source is None or X_target is None:
        print("❌ 数据加载失败，程序退出")
        return None
    
    # Step 3: 运行对比实验
    experiment = ComparativeExperiment(X_source, y_source, X_target, target_files, loader)
    results, constrained_learner, unconstrained_learner = experiment.run_comparison(max_iter=10)
    
    # Step 4: 生成详细报告
    print("\n" + "="*80)
    print("实验总结报告")
    print("="*80)
    
    print("\n关键发现:")
    print("-"*40)
    
    # 分析MMD
    if 'mmd_comparison' in results:
        mmd_c_reduction = (results['mmd_comparison']['constrained_initial'] - 
                          results['mmd_comparison']['constrained_final']) / results['mmd_comparison']['constrained_initial'] * 100
        mmd_u_reduction = (results['mmd_comparison']['unconstrained_initial'] - 
                          results['mmd_comparison']['unconstrained_final']) / results['mmd_comparison']['unconstrained_initial'] * 100
        
        print(f"\n1. 域差异缩小效果:")
        print(f"   - 带约束MMD降低: {mmd_c_reduction:.1f}%")
        print(f"   - 无约束MMD降低: {mmd_u_reduction:.1f}%")
        
        if mmd_c_reduction > mmd_u_reduction:
            print("   → 带约束方法在域对齐方面表现更好")
        else:
            print("   → 无约束方法在域对齐方面表现更好")
    
    # 分析置信度
    if 'confidence_comparison' in results:
        print(f"\n2. 模型置信度:")
        print(f"   - 带约束平均置信度: {results['confidence_comparison']['constrained_mean']:.3f}")
        print(f"   - 无约束平均置信度: {results['confidence_comparison']['unconstrained_mean']:.3f}")
        
        if results['confidence_comparison']['constrained_mean'] > results['confidence_comparison']['unconstrained_mean']:
            print("   → 带约束方法产生更高的置信度")
        else:
            print("   → 无约束方法产生更高的置信度")
    
    # 分析分布差异
    if 'distribution_comparison' in results:
        print(f"\n3. 类别分布保持:")
        print(f"   - 带约束KL散度: {results['distribution_comparison']['kl_constrained']:.4f}")
        print(f"   - 无约束KL散度: {results['distribution_comparison']['kl_unconstrained']:.4f}")
        
        if results['distribution_comparison']['kl_constrained'] < results['distribution_comparison']['kl_unconstrained']:
            print("   → 带约束方法更好地保持了源域类别分布")
        else:
            print("   → 无约束方法的类别分布更接近源域")
    
    # 分析预测一致性
    if 'prediction_agreement' in results:
        print(f"\n4. 预测一致性: {results['prediction_agreement']:.1%}")
        if results['prediction_agreement'] > 0.8:
            print("   → 两种方法的预测结果高度一致")
        elif results['prediction_agreement'] > 0.6:
            print("   → 两种方法的预测结果部分一致")
        else:
            print("   → 两种方法的预测结果差异较大")
    
    print("\n" + "="*80)
    print("结论与建议")
    print("="*80)
    
    print("\n基于对比实验，可以得出以下结论：")
    print("\n• 带约束方法的优势：")
    print("  - 通过类别平衡约束，确保目标域预测分布与源域相似")
    print("  - 通过特征一致性约束，维持类别特征的稳定性")
    print("  - 提供更可控和可解释的迁移过程")
    
    print("\n• 无约束方法的优势：")
    print("  - 算法更简单，计算效率更高")
    print("  - 不受约束限制，可能在某些情况下找到更优解")
    print("  - 适用于源域和目标域分布差异较小的场景")
    
    print("\n• 选择建议：")
    print("  - 当源域和目标域分布差异较大时，推荐使用带约束方法")
    print("  - 当需要保持类别平衡和特征一致性时，使用带约束方法")
    print("  - 当计算资源有限或需要快速部署时，可考虑无约束方法")
    
    # Step 5: 保存对比结果
    print("\n" + "="*80)
    print("保存实验结果")
    print("="*80)
    
    # 保存对比报告
    comparison_report = pd.DataFrame([{
        'Method': 'Constrained',
        'Initial_MMD': results['mmd_comparison']['constrained_initial'] if 'mmd_comparison' in results else None,
        'Final_MMD': results['mmd_comparison']['constrained_final'] if 'mmd_comparison' in results else None,
        'Avg_Confidence': results['confidence_comparison']['constrained_mean'] if 'confidence_comparison' in results else None,
        'KL_Divergence': results['distribution_comparison']['kl_constrained'] if 'distribution_comparison' in results else None
    }, {
        'Method': 'Unconstrained',
        'Initial_MMD': results['mmd_comparison']['unconstrained_initial'] if 'mmd_comparison' in results else None,
        'Final_MMD': results['mmd_comparison']['unconstrained_final'] if 'mmd_comparison' in results else None,
        'Avg_Confidence': results['confidence_comparison']['unconstrained_mean'] if 'confidence_comparison' in results else None,
        'KL_Divergence': results['distribution_comparison']['kl_unconstrained'] if 'distribution_comparison' in results else None
    }])
    
    comparison_report.to_csv('comparison_report.csv', index=False)
    print("✓ 对比报告已保存到: comparison_report.csv")
    
    return results

# 运行主程序
if __name__ == "__main__":
    # 运行对比实验
    results = main_with_comparison()
    
    if results is not None:
        print("\n" + "="*80)
        print("对比实验完成！")
        print("="*80)
        print("\n请查看生成的图表和报告文件以获取详细信息。")
        print("文件列表：")
        print("  - comparison_report.csv: 对比实验数值结果")
        print("  - 可视化图表: 展示了两种方法的详细对比")