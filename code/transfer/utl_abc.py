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
                    # 检查是否是StandardScaler对象
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
                    # 检查加载的对象类型
                    if isinstance(loaded_encoder, LabelEncoder):
                        self.label_encoder = loaded_encoder
                        print(f"✓ 成功加载标签编码器: {self.encoder_path}")
                    elif isinstance(loaded_encoder, (list, np.ndarray)):
                        # 如果是类别列表，创建新的编码器
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
            # 如果编码器已经训练过
            if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
                # 检查是否所有标签都能被编码
                unknown_labels = set(y_source_labels) - set(self.label_encoder.classes_)
                if unknown_labels:
                    print(f"⚠ 发现未知标签: {unknown_labels}，重新训练编码器")
                    self.label_encoder = LabelEncoder()
                    y_source = self.label_encoder.fit_transform(y_source_labels)
                else:
                    y_source = self.label_encoder.transform(y_source_labels)
            else:
                # 训练新的编码器
                print("训练新的标签编码器...")
                y_source = self.label_encoder.fit_transform(y_source_labels)
            
            print(f"编码后的标签范围: {np.min(y_source)} - {np.max(y_source)}")
            print(f"标签映射: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
            
        except Exception as e:
            print(f"❌ 标签编码失败: {e}")
            print("使用简单的标签映射...")
            # 创建简单的标签映射
            label_map = {label: i for i, label in enumerate(unique_labels)}
            y_source = np.array([label_map[label] for label in y_source_labels])
            
            # 更新编码器
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
                # 检查标准化器是否已经训练
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
# 无约束迁移学习核心算法
# ============================================================================
class UnconstrainedTransferLearning:
    """无约束的迁移学习"""
    
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
        
        # 超参数
        self.lambda1 = 0.3  # MMD权重
        self.lambda2 = 0.5  # 伪标签权重
        self.lambda3 = 0.1  # 正则化权重
        
        # 记录优化历史
        self.history = {
            'loss': [],
            'mmd': [],
            'pseudo_labels': [],
            'confidence': [],
            'label_change_rate': []
        }
    
    def compute_mmd(self, X_source, X_target, gamma=1.0):
        """计算最大均值差异(MMD)"""
        n_s = len(X_source)
        n_t = len(X_target)
        
        # 为了计算效率，采样部分数据
        max_samples = 500
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
            # 简化的对齐方法
            X_source_aligned = (X_source - np.mean(X_source, axis=0)) / (np.std(X_source, axis=0) + 1e-6)
            X_source_aligned = X_source_aligned * np.std(X_target, axis=0) + np.mean(X_target, axis=0)
        
        return X_source_aligned, X_target
    
    def iterative_optimization(self, X_source, y_source, X_target, max_iter=10, confidence_threshold=0.7):
        """迭代优化主流程（无约束版本）"""
        print("\n" + "="*60)
        print("开始无约束迁移学习优化")
        print("="*60)
        
        # 计算源域类别分布（仅用于监控，不作为约束）
        unique_classes = np.unique(y_source)
        self.n_classes = len(unique_classes)
        source_class_dist = np.bincount(y_source, minlength=self.n_classes) / len(y_source)
        print(f"源域类别分布: {source_class_dist}")
        
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
            # 使用部分数据训练以加速
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
                # 模拟预测
                pseudo_probs = softmax(np.random.randn(len(X_target), self.n_classes), axis=1)
                pseudo_labels = np.argmax(pseudo_probs, axis=1)
                confidence = np.max(pseudo_probs, axis=1)
            
            self.history['pseudo_labels'].append(pseudo_labels)
            self.history['confidence'].append(confidence)
            
            # 监控目标域预测分布（不作为约束）
            target_dist = np.mean(pseudo_probs, axis=0)
            print(f"目标域预测分布: {target_dist.round(3)}")
            
            # Step 4: 选择高置信度样本
            high_conf_mask = confidence > confidence_threshold
            n_high_conf = np.sum(high_conf_mask)
            print(f"高置信度样本数: {n_high_conf}/{len(X_target)} ({n_high_conf/len(X_target)*100:.1f}%)")
            print(f"平均置信度: {np.mean(confidence):.3f}")
            
            # Step 5: 更新模型
            if n_high_conf > 10:  # 至少有10个高置信度样本
                # 构建增强训练集
                # 采样源域数据以平衡训练集大小
                n_source_sample = min(2000, len(X_source))
                idx_source = np.random.choice(len(X_source), n_source_sample, replace=False)
                
                X_train = np.vstack([X_s_aligned[idx_source], X_t_aligned[high_conf_mask]])
                y_train = np.hstack([y_source[idx_source], pseudo_labels[high_conf_mask]])
                
                # 样本权重（仅基于置信度，无约束调整）
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
                print("✓ 模型更新完成")
            
            # Step 6: 计算总损失（简化版，无约束项）
            total_loss = self.lambda1 * mmd
            self.history['loss'].append(total_loss)
            print(f"总损失: {total_loss:.4f}")
            
            # Step 7: 收敛检查
            if prev_labels is not None:
                label_changes = np.sum(pseudo_labels != prev_labels)
                change_rate = label_changes / len(pseudo_labels)
                self.history['label_change_rate'].append(change_rate)
                print(f"标签变化率: {change_rate:.2%}")
                
                if change_rate < 0.01:
                    print("\n✓ 模型已收敛！")
                    break
            
            prev_labels = pseudo_labels.copy()
            
            # 动态调整置信度阈值
            if iteration > 3 and n_high_conf < len(X_target) * 0.3:
                confidence_threshold = max(0.5, confidence_threshold - 0.05)
                print(f"调整置信度阈值为: {confidence_threshold:.2f}")
        
        return pseudo_labels, confidence

# ============================================================================
# 可视化模块（简化版）
# ============================================================================
def plot_results(X_source, y_source, X_target, predictions, confidence, history):
    """生成结果可视化"""
    
    fig = plt.figure(figsize=(15, 10))
    
    # 图1: MMD变化
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(history['mmd'], 'o-', linewidth=2, markersize=6)
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('MMD距离')
    ax1.set_title('域差异变化')
    ax1.grid(True, alpha=0.3)
    
    # 图2: 标签变化率
    if 'label_change_rate' in history and len(history['label_change_rate']) > 0:
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(history['label_change_rate'], 's-', color='red', linewidth=2, markersize=6)
        ax2.set_xlabel('迭代次数')
        ax2.set_ylabel('标签变化率')
        ax2.set_title('预测稳定性')
        ax2.grid(True, alpha=0.3)
    
    # 图3: 平均置信度
    ax3 = plt.subplot(2, 3, 3)
    avg_conf = [np.mean(c) for c in history['confidence']]
    ax3.plot(avg_conf, '^-', color='green', linewidth=2, markersize=6)
    ax3.set_xlabel('迭代次数')
    ax3.set_ylabel('平均置信度')
    ax3.set_title('预测置信度变化')
    ax3.grid(True, alpha=0.3)
    
    # 图4: 置信度分布
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(confidence, bins=30, edgecolor='black', alpha=0.7)
    ax4.axvline(x=0.7, color='r', linestyle='--', label='阈值')
    ax4.set_xlabel('置信度')
    ax4.set_ylabel('样本数')
    ax4.set_title('最终置信度分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 图5: 类别分布
    ax5 = plt.subplot(2, 3, 5)
    unique_pred, counts = np.unique(predictions, return_counts=True)
    ax5.bar(unique_pred, counts, color='skyblue', edgecolor='black')
    ax5.set_xlabel('类别')
    ax5.set_ylabel('样本数')
    ax5.set_title('预测类别分布')
    ax5.grid(True, alpha=0.3)
    
    # 图6: 总损失
    if 'loss' in history and len(history['loss']) > 0:
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(history['loss'], 'd-', color='purple', linewidth=2, markersize=6)
        ax6.set_xlabel('迭代次数')
        ax6.set_ylabel('总损失')
        ax6.set_title('总损失变化')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('utl_training_diagnostics.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    return fig

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
    
    print("="*60)
    print("无约束迁移学习系统")
    print("="*60)
    
    # Step 1: 创建数据加载器
    loader = DataLoader(MODEL_PATH, SCALER_PATH, ENCODER_PATH)
    
    # Step 2: 加载数据
    X_source, y_source, X_target, target_files = loader.load_data(
        SOURCE_DATA_PATH, TARGET_DATA_PATH
    )
    
    if X_source is None or X_target is None:
        print("❌ 数据加载失败，程序退出")
        return None, None
    
    # Step 3: 创建迁移学习器
    transfer_learner = UnconstrainedTransferLearning(
        base_model=loader.xgb_model,
        label_encoder=loader.label_encoder
    )
    
    # Step 4: 执行迁移学习
    predictions, confidence = transfer_learner.iterative_optimization(
        X_source, y_source, X_target,
        max_iter=50,
        confidence_threshold=0.7
    )
    
    # Step 5: 生成结果
    print("\n" + "="*60)
    print("预测结果汇总")
    print("="*60)
    
    # 反向编码标签
    try:
        if hasattr(loader.label_encoder, 'inverse_transform'):
            predicted_labels = loader.label_encoder.inverse_transform(predictions)
        else:
            predicted_labels = [f"Class_{i}" for i in predictions]
    except:
        predicted_labels = [f"Class_{i}" for i in predictions]
    
    # 按文件统计
    results_df = pd.DataFrame({
        'file': target_files,
        'prediction': predicted_labels,
        'confidence': confidence
    })
    
    # 按文件聚合
    file_predictions = {}
    unique_files = np.unique(target_files)
    
    for file in unique_files:
        file_data = results_df[results_df['file'] == file]
        # 使用众数作为最终预测
        pred_mode = file_data['prediction'].mode()[0] if len(file_data['prediction'].mode()) > 0 else file_data['prediction'].iloc[0]
        avg_conf = file_data['confidence'].mean()
        
        file_predictions[file] = {
            'prediction': pred_mode,
            'confidence': avg_conf,
            'n_samples': len(file_data)
        }
    
    # 输出结果
    print("\n最终预测结果（按文件）:")
    print("-"*60)
    print(f"{'文件':<10} {'预测标签':<15} {'置信度':<10} {'可靠性':<10} {'样本数':<10}")
    print("-"*60)
    
    for file in sorted(file_predictions.keys()):
        result = file_predictions[file]
        reliability = '高' if result['confidence'] > 0.8 else '中' if result['confidence'] > 0.6 else '低'
        print(f"{file:<10} {result['prediction']:<15} {result['confidence']:.3f}      {reliability:<10} {result['n_samples']:<10}")
    
    # Step 6: 可视化
    print("\n生成可视化...")
    plot_results(X_source, y_source, X_target, predictions, confidence, transfer_learner.history)
    
    # Step 7: 保存结果
    output_df = pd.DataFrame(file_predictions).T
    output_df.to_csv('unconstrained_transfer_predictions.csv')
    print("\n✓ 预测结果已保存到: unconstrained_transfer_predictions.csv")
    
    results_df.to_csv('unconstrained_transfer_detailed.csv', index=False)
    print("✓ 详细结果已保存到: unconstrained_transfer_detailed.csv")
    
    return file_predictions, transfer_learner.history

# 运行主程序
if __name__ == "__main__":
    predictions, history = main()
    
    if predictions is not None and history is not None:
        print("\n" + "="*60)
        print("迁移学习统计")
        print("="*60)
        
        if len(history['mmd']) > 0:
            print(f"初始MMD距离: {history['mmd'][0]:.4f}")
            print(f"最终MMD距离: {history['mmd'][-1]:.4f}")
            print(f"MMD减少比例: {(history['mmd'][0] - history['mmd'][-1])/history['mmd'][0]*100:.1f}%")
        
        if 'label_change_rate' in history and len(history['label_change_rate']) > 0:
            print(f"最终标签变化率: {history['label_change_rate'][-1]:.4f}")
        
        if len(history['confidence']) > 0:
            print(f"最终平均置信度: {np.mean(history['confidence'][-1]):.3f}")