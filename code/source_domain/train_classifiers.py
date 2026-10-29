# -*- coding: utf-8 -*-
# 路径约定：仓库目录只读，运行产物统一写入 work/（可用环境变量 BD_ROOT 换盘）。
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import WORK, DATA, PROCESSED, MODEL_DIR, use_workdir  # noqa: E402

use_workdir()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持更好的显示
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

# 读取数据
df1_path = os.path.join(WORK, 'df2_pca_features.csv')
data = pd.read_csv(df1_path, encoding='utf-8-sig')

print("="*60)
print("Label1 标签分布分析")
print("="*60)

# 基本数据信息
print(f"\n=== 数据基本信息 ===")
print(f"数据形状: {data.shape}")
print(f"总样本数: {len(data)}")

# 检查label1列的基本情况
print(f"\n=== Label1列信息 ===")
print(f"Label1唯一值数量: {data['label1'].nunique()}")
print(f"是否有缺失值: {data['label1'].isnull().sum()}")

# 计算各标签的分布
label_counts = data['label1'].value_counts()
label_proportions = data['label1'].value_counts(normalize=True) * 100

print(f"\n=== Label1分布统计 ===")
print("标签分布:")
for label, count in label_counts.items():
    percentage = label_proportions[label]
    print(f"  {label}: {count:,} 样本 ({percentage:.2f}%)")

# 可视化1: 条形图显示各标签数量
print(f"\n正在生成可视化图表...")
plt.figure(figsize=(14, 8))

bars = plt.bar(range(len(label_counts)), label_counts.values, 
               color=plt.cm.Set3(np.linspace(0, 1, len(label_counts))), 
               alpha=0.8, edgecolor='black', linewidth=0.5)

plt.title('Label1 Distribution - Sample Counts', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Label Categories', fontsize=14)
plt.ylabel('Number of Samples', fontsize=14)

# 设置x轴标签
plt.xticks(range(len(label_counts)), label_counts.index, rotation=45, ha='right')

# 在柱子上添加数值标签
for i, (bar, count) in enumerate(zip(bars, label_counts.values)):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 可视化2: 水平条形图（当标签名较长时更清楚）
plt.figure(figsize=(12, 8))

y_pos = np.arange(len(label_counts))
bars = plt.barh(y_pos, label_counts.values, 
                color=plt.cm.viridis(np.linspace(0, 1, len(label_counts))), 
                alpha=0.8, edgecolor='black', linewidth=0.5)

plt.title('Label1 Distribution - Horizontal View', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Number of Samples', fontsize=14)
plt.ylabel('Label Categories', fontsize=14)

# 设置y轴标签
plt.yticks(y_pos, label_counts.index)

# 在柱子上添加数值和百分比标签
for i, (bar, count, pct) in enumerate(zip(bars, label_counts.values, label_proportions.values)):
    plt.text(bar.get_width() + max(label_counts.values)*0.01, bar.get_y() + bar.get_height()/2.,
             f'{count:,} ({pct:.1f}%)', ha='left', va='center', fontsize=10, fontweight='bold')

plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# 可视化3: 饼图显示比例分布
plt.figure(figsize=(12, 10))

# 如果类别太多，只显示前10个，其余合并为"Others"
if len(label_counts) > 10:
    top_labels = label_counts.head(10)
    others_count = label_counts.tail(len(label_counts)-10).sum()
    
    # 创建新的数据用于饼图
    pie_labels = list(top_labels.index) + ['Others']
    pie_values = list(top_labels.values) + [others_count]
    pie_colors = plt.cm.Set3(np.linspace(0, 1, len(pie_values)))
    
    print(f"注意: 由于类别较多，饼图只显示前10个类别，其余归为'Others'")
else:
    pie_labels = label_counts.index
    pie_values = label_counts.values
    pie_colors = plt.cm.Set3(np.linspace(0, 1, len(pie_values)))

# 绘制饼图
wedges, texts, autotexts = plt.pie(pie_values, labels=pie_labels, autopct='%1.1f%%',
                                  colors=pie_colors, startangle=90, 
                                  textprops={'fontsize': 10})

plt.title('Label1 Distribution - Proportion View', fontsize=16, fontweight='bold', pad=20)
plt.axis('equal')
plt.tight_layout()
plt.show()

# 可视化4: 数据分布的详细统计图
if len(label_counts) <= 20:  # 只在类别不太多时显示
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 上图：样本数量
    bars1 = ax1.bar(range(len(label_counts)), label_counts.values, 
                    color=plt.cm.coolwarm(np.linspace(0, 1, len(label_counts))), 
                    alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax1.set_title('Sample Counts by Label', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.set_xticks(range(len(label_counts)))
    ax1.set_xticklabels(label_counts.index, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, count in zip(bars1, label_counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    # 下图：百分比
    bars2 = ax2.bar(range(len(label_proportions)), label_proportions.values, 
                    color=plt.cm.plasma(np.linspace(0, 1, len(label_proportions))), 
                    alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_title('Percentage Distribution by Label', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Label Categories', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_xticks(range(len(label_proportions)))
    ax2.set_xticklabels(label_proportions.index, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加百分比标签
    for bar, pct in zip(bars2, label_proportions.values):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

# 数据不平衡检查
print(f"\n=== 数据平衡性分析 ===")
max_count = label_counts.max()
min_count = label_counts.min()
imbalance_ratio = max_count / min_count

print(f"最多样本的类别: {label_counts.idxmax()} ({max_count:,} 样本)")
print(f"最少样本的类别: {label_counts.idxmin()} ({min_count:,} 样本)")
print(f"不平衡比例: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 10:
    print("⚠️  数据严重不平衡！建议考虑以下处理方法:")
    print("   • 使用SMOTE等过采样技术")
    print("   • 使用类权重平衡")
    print("   • 使用分层抽样")
elif imbalance_ratio > 3:
    print("⚠️  数据存在一定不平衡，建议关注模型性能")
else:
    print("✅ 数据分布相对平衡")

# 保存统计结果
distribution_summary = pd.DataFrame({
    'Label': label_counts.index,
    'Count': label_counts.values,
    'Percentage': label_proportions.values
})

try:
    distribution_summary.to_csv('label1_distribution_summary.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 标签分布统计已保存到: label1_distribution_summary.csv")
except Exception as e:
    print(f"保存统计结果时出错: {e}")

print(f"\n{'='*60}")
print("🎉 Label1标签分布分析完成！")
print(f"总类别数: {len(label_counts)}")
print(f"总样本数: {len(data):,}")
print(f"平均每类样本数: {len(data)/len(label_counts):.1f}")
print(f"{'='*60}")
####处理样本不平衡问题
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib显示
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

# 读取数据
df1_path = os.path.join(WORK, 'df2_pca_features.csv')
data = pd.read_csv(df1_path, encoding='utf-8-sig')

print("="*60)
print("手动实现过采样技术处理不平衡数据")
print("="*60)

# 步骤1：数据准备和原始分布分析
print(f"\n=== 步骤1：数据准备 ===")
print(f"原始数据形状: {data.shape}")

# 检查数据完整性
print(f"label1缺失值: {data['label1'].isnull().sum()}")
if data['label1'].isnull().sum() > 0:
    print("发现缺失值，将删除含有缺失值的行")
    data = data.dropna(subset=['label1'])
    print(f"清理后数据形状: {data.shape}")

# 分离特征和标签
feature_columns = [col for col in data.columns if col.startswith('PC')]
X = data[feature_columns].copy()
y = data['label1'].copy()

print(f"特征矩阵形状: {X.shape}")
print(f"标签向量长度: {len(y)}")

# 原始分布统计
original_distribution = Counter(y)
print(f"\n=== 原始数据分布 ===")
print("类别分布:")
for label, count in original_distribution.most_common():
    percentage = (count / len(y)) * 100
    print(f"  {label}: {count:,} 样本 ({percentage:.2f}%)")

# 计算不平衡比例
max_count = max(original_distribution.values())
min_count = min(original_distribution.values())
imbalance_ratio = max_count / min_count
print(f"\n不平衡比例: {imbalance_ratio:.2f}:1")

# 手动实现简单的SMOTE算法
def simple_smote(X, y, k_neighbors=5, random_state=42):
    """
    简单的SMOTE实现
    """
    np.random.seed(random_state)
    
    # 统计各类别样本数
    class_counts = Counter(y)
    max_count = max(class_counts.values())
    
    # 存储新生成的样本
    X_new = []
    y_new = []
    
    for class_label, count in class_counts.items():
        if count < max_count:
            # 需要生成的新样本数
            n_samples_needed = max_count - count
            
            # 获取当前类别的所有样本
            class_mask = y == class_label
            class_samples = X[class_mask].values
            
            if len(class_samples) < k_neighbors:
                k_neighbors_actual = len(class_samples) - 1
            else:
                k_neighbors_actual = k_neighbors
            
            if k_neighbors_actual > 0:
                # 使用KNN找到邻居
                nbrs = NearestNeighbors(n_neighbors=k_neighbors_actual + 1).fit(class_samples)
                
                for _ in range(n_samples_needed):
                    # 随机选择一个样本
                    sample_idx = np.random.randint(0, len(class_samples))
                    sample = class_samples[sample_idx]
                    
                    # 找到最近邻
                    distances, indices = nbrs.kneighbors([sample])
                    
                    # 排除自己，选择一个邻居
                    neighbor_idx = np.random.choice(indices[0][1:])
                    neighbor = class_samples[neighbor_idx]
                    
                    # 在样本和邻居之间生成新样本
                    diff = neighbor - sample
                    gap = np.random.random()
                    new_sample = sample + gap * diff
                    
                    X_new.append(new_sample)
                    y_new.append(class_label)
            else:
                # 如果样本太少，直接复制现有样本
                for _ in range(n_samples_needed):
                    sample_idx = np.random.randint(0, len(class_samples))
                    X_new.append(class_samples[sample_idx])
                    y_new.append(class_label)
    
    if X_new:
        X_balanced = np.vstack([X.values, np.array(X_new)])
        y_balanced = np.concatenate([y.values, y_new])
    else:
        X_balanced = X.values
        y_balanced = y.values
    
    return X_balanced, y_balanced

# 简单的随机过采样方法
def random_oversampling(X, y, random_state=42):
    """
    随机过采样：复制少数类样本
    """
    np.random.seed(random_state)
    
    # 统计各类别样本数
    class_counts = Counter(y)
    max_count = max(class_counts.values())
    
    X_new = []
    y_new = []
    
    # 添加原始样本
    X_new.extend(X.values)
    y_new.extend(y.values)
    
    for class_label, count in class_counts.items():
        if count < max_count:
            # 需要生成的新样本数
            n_samples_needed = max_count - count
            
            # 获取当前类别的所有样本
            class_mask = y == class_label
            class_samples = X[class_mask].values
            
            # 随机选择并复制样本
            random_indices = np.random.choice(len(class_samples), 
                                            size=n_samples_needed, 
                                            replace=True)
            
            for idx in random_indices:
                X_new.append(class_samples[idx])
                y_new.append(class_label)
    
    return np.array(X_new), np.array(y_new)

# 步骤2：应用过采样方法
print(f"\n=== 步骤2：应用过采样方法 ===")

# 方法选择
sampling_methods = {
    'Random Oversampling': random_oversampling,
    'Simple SMOTE': simple_smote
}

results = {}

for method_name, method_func in sampling_methods.items():
    try:
        print(f"\n正在应用 {method_name}...")
        X_resampled, y_resampled = method_func(X, y)
        
        # 统计结果
        resampled_distribution = Counter(y_resampled)
        
        results[method_name] = {
            'X': X_resampled,
            'y': y_resampled,
            'distribution': resampled_distribution,
            'shape': X_resampled.shape
        }
        
        print(f"✅ {method_name} 完成")
        print(f"   数据形状: {X_resampled.shape}")
        print(f"   新增样本: {len(y_resampled) - len(y):,}")
        
    except Exception as e:
        print(f"❌ {method_name} 失败: {e}")

# 选择最佳方法（这里选择Simple SMOTE，如果失败则用Random Oversampling）
if 'Simple SMOTE' in results:
    best_method = 'Simple SMOTE'
elif 'Random Oversampling' in results:
    best_method = 'Random Oversampling'
else:
    print("❌ 所有过采样方法都失败了！")
    exit()

X_final = results[best_method]['X']
y_final = results[best_method]['y']
final_distribution = results[best_method]['distribution']

print(f"\n=== 选择的方法: {best_method} ===")
print(f"最终数据形状: {X_final.shape}")
print(f"过采样后分布:")
for label, count in final_distribution.most_common():
    percentage = (count / len(y_final)) * 100
    print(f"  {label}: {count:,} 样本 ({percentage:.2f}%)")

# 步骤3：创建对比可视化
print(f"\n=== 步骤3：创建对比可视化 ===")

# 准备可视化数据
original_labels = list(original_distribution.keys())
original_counts = list(original_distribution.values())
final_counts = [final_distribution[label] for label in original_labels]

# 可视化1：过采样前后对比条形图
plt.figure(figsize=(15, 8))

x = np.arange(len(original_labels))
width = 0.35

bars1 = plt.bar(x - width/2, original_counts, width, label='Original', 
                color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = plt.bar(x + width/2, final_counts, width, label=f'After {best_method}', 
                color='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5)

plt.title(f'Sample Distribution: Before vs After {best_method}', fontsize=16, fontweight='bold')
plt.xlabel('Label Categories', fontsize=14)
plt.ylabel('Number of Samples', fontsize=14)
plt.xticks(x, original_labels, rotation=45, ha='right')
plt.legend(fontsize=12)

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 可视化2：增长倍数图
plt.figure(figsize=(12, 8))

growth_ratios = [final_counts[i] / original_counts[i] for i in range(len(original_labels))]
colors = ['red' if ratio > 1 else 'green' for ratio in growth_ratios]

bars = plt.bar(original_labels, growth_ratios, color=colors, alpha=0.7, 
               edgecolor='black', linewidth=0.5)

plt.axhline(y=1, color='black', linestyle='--', alpha=0.8, label='No Change')
plt.title('Sample Growth Ratio by Category', fontsize=16, fontweight='bold')
plt.xlabel('Label Categories', fontsize=14)
plt.ylabel('Growth Ratio (After/Before)', fontsize=14)
plt.xticks(rotation=45, ha='right')

# 添加比例标签
for bar, ratio in zip(bars, growth_ratios):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{ratio:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# 可视化3：数据分布饼图对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 原始分布饼图
colors1 = plt.cm.Set3(np.linspace(0, 1, len(original_labels)))
wedges1, texts1, autotexts1 = ax1.pie(original_counts, labels=original_labels, 
                                      autopct='%1.1f%%', colors=colors1, startangle=90)
ax1.set_title('Original Distribution', fontsize=14, fontweight='bold')

# 过采样后分布饼图
wedges2, texts2, autotexts2 = ax2.pie(final_counts, labels=original_labels, 
                                      autopct='%1.1f%%', colors=colors1, startangle=90)
ax2.set_title(f'After {best_method} Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# 步骤4：创建最终数据集
print(f"\n=== 步骤4：创建最终数据集 ===")

# 创建包含所有列的新数据框
df_balanced = pd.DataFrame(X_final, columns=feature_columns)
df_balanced['label1'] = y_final

# 如果原数据中有label2，需要处理
if 'label2' in data.columns:
    print("处理label2列...")
    # 为新增的样本分配label2值
    original_indices = list(range(len(data)))
    label2_values = []
    
    current_idx = 0
    for label in y_final:
        if current_idx < len(data):
            # 原始样本，直接使用原始label2
            label2_values.append(data.iloc[current_idx]['label2'])
            current_idx += 1
        else:
            # 新增样本，从同类别中随机选择label2
            same_label_mask = data['label1'] == label
            if same_label_mask.any():
                possible_label2 = data[same_label_mask]['label2'].values
                selected_label2 = np.random.choice(possible_label2)
                label2_values.append(selected_label2)
            else:
                # 如果没有同类别，使用最常见的label2
                label2_values.append(data['label2'].mode()[0])
    
    df_balanced['label2'] = label2_values

print(f"最终平衡数据集形状: {df_balanced.shape}")
print(f"最终数据集预览:")
print(df_balanced.head())

#df_balanced.to_csv('数据平衡后的数据集.csv', index=False, encoding='utf-8-sig')

# 步骤5：保存结果
print(f"\n=== 步骤5：保存平衡后的数据 ===")

try:
    # 保存为CSV
    output_csv = 'balanced_feature_set.csv'
    df_balanced.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 平衡后数据已保存为: {output_csv}")
    
    # 保存为Excel
    output_excel = 'balanced_feature_set.xlsx'
    df_balanced.to_excel(output_excel, index=False, engine='openpyxl')
    print(f"✅ 平衡后数据已保存为: {output_excel}")
    
    # 保存统计报告
    comparison_report = pd.DataFrame({
        'Label': original_labels,
        'Original_Count': original_counts,
        'After_Oversampling_Count': final_counts,
        'Growth_Ratio': growth_ratios,
        'Added_Samples': [final_counts[i] - original_counts[i] for i in range(len(original_labels))]
    })
    
    comparison_report.to_csv('oversampling_comparison_report.csv', index=False, encoding='utf-8-sig')
    print(f"✅ 过采样处理报告已保存为: oversampling_comparison_report.csv")
    
except Exception as e:
    print(f"❌ 保存文件时出错: {e}")

# 最终统计
print(f"\n{'='*60}")
print(f"🎉 {best_method}过采样处理完成！")
print(f"{'='*60}")
print(f"📊 处理总结:")
print(f"   • 使用方法: {best_method}")
print(f"   • 原始样本数: {len(data):,}")
print(f"   • 平衡后样本数: {len(df_balanced):,}")
print(f"   • 新增样本数: {len(df_balanced) - len(data):,}")
print(f"   • 类别数量: {len(original_labels)}")
print(f"   • 原始不平衡比例: {imbalance_ratio:.2f}:1")
print(f"   • 平衡后比例: 1:1 (完全平衡)")
print(f"\n📁 输出文件:")
print(f"   • 平衡数据集: {output_csv}")
print(f"   • Excel版本: {output_excel}")
print(f"   • 处理报告: oversampling_comparison_report.csv")
print(f"\n💡 后续建议:")
print(f"   • 使用平衡后的数据进行模型训练")
print(f"   • 建议使用交叉验证评估模型性能")
print(f"   • 注意过采样可能带来的过拟合风险")
print(f"{'='*60}")

# 将最终数据赋值给全局变量
globals()['df_balanced'] = df_balanced
print(f"✅ 平衡后的数据已保存到变量: df_balanced")
print(f"   可以直接使用 df_balanced 进行后续建模")
##########模型训练过程
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关库
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           classification_report, confusion_matrix, roc_curve, auc,
                           roc_auc_score, precision_recall_curve)

# 机器学习模型
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# 模型保存相关库
import pickle
import joblib
import json
from datetime import datetime

# 设置matplotlib显示
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

print("="*80)
print("机器学习模型训练与评估")
print("="*80)

# 步骤1：数据加载和准备
print(f"\n=== 步骤1：数据加载和准备 ===")

# 加载平衡后的数据（如果存在df_balanced变量，使用它，否则从文件读取）
try:
    if 'df_balanced' in globals():
        data = df_balanced.copy()
        print("✅ 从变量df_balanced加载数据")
    else:
        # 尝试从文件读取
        data_files = ['balanced_feature_set.csv', 'balanced_feature_set_smote.csv', 'df2_pca_features.csv']
        data = None
        for file_path in data_files:
            try:
                data = pd.read_csv(file_path, encoding='utf-8-sig')
                print(f"✅ 从文件{file_path}加载数据")
                break
            except:
                continue
        
        if data is None:
            print("❌ 无法找到数据文件，请确保已生成数据")
            exit()
            
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    exit()

print(f"数据形状: {data.shape}")
print(f"列名: {list(data.columns)}")

# 检查标签分布
if 'label1' in data.columns:
    label_distribution = Counter(data['label1'])
    print(f"\nlabel1分布: {dict(label_distribution)}")
else:
    print("❌ 未找到label1列")
    exit()

# 分离特征和标签
feature_columns = [col for col in data.columns if col.startswith('PC')]
if not feature_columns:
    print("❌ 未找到主成分特征列")
    exit()

X = data[feature_columns].copy()
y = data['label1'].copy()

print(f"特征矩阵形状: {X.shape}")
print(f"标签向量长度: {len(y)}")
print(f"类别数量: {len(np.unique(y))}")

# 标签编码
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
class_names = label_encoder.classes_

print(f"类别标签: {class_names}")

# 步骤2：数据预处理和划分
print(f"\n=== 步骤2：数据预处理和划分 ===")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 数据划分（7:3）
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"训练集形状: {X_train.shape}")
print(f"测试集形状: {X_test.shape}")
print(f"训练集标签分布: {Counter(y_train)}")
print(f"测试集标签分布: {Counter(y_test)}")

# 步骤3：定义模型
print(f"\n=== 步骤3：定义模型 ===")

models = {
    'XGBoost': XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss' if len(class_names) > 2 else 'logloss'
    ),
    'Logistic Regression': LogisticRegression(
        random_state=42,
        max_iter=1000,
        multi_class='ovr' if len(class_names) > 2 else 'auto'
    ),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(
        kernel='rbf',
        C=1.0,
        random_state=42,
        probability=True  # 需要概率输出用于ROC曲线
    ),
    'BP Neural Network': MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
}

print(f"定义了{len(models)}个模型")

# 步骤4：模型训练和评估
print(f"\n=== 步骤4：模型训练和评估 ===")

results = {}
trained_models = {}

for model_name, model in models.items():
    print(f"\n--- 训练 {model_name} ---")
    
    try:
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # 计算评估指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 存储结果
        results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'model': model
        }
        
        trained_models[model_name] = model
        
        print(f"✅ {model_name} 训练完成")
        print(f"   准确率: {accuracy:.4f}")
        print(f"   精确率: {precision:.4f}")
        print(f"   召回率: {recall:.4f}")
        print(f"   F1分数: {f1:.4f}")
        
    except Exception as e:
        print(f"❌ {model_name} 训练失败: {e}")

# 步骤5：结果汇总表
print(f"\n=== 步骤5：模型性能汇总 ===")

# 创建结果汇总表
performance_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[model]['accuracy'] for model in results.keys()],
    'Precision': [results[model]['precision'] for model in results.keys()],
    'Recall': [results[model]['recall'] for model in results.keys()],
    'F1-Score': [results[model]['f1_score'] for model in results.keys()]
})

print("\n模型性能对比：")
print(performance_df.to_string(index=False, float_format='%.4f'))

# 找出最佳模型
best_model_name = performance_df.loc[performance_df['Accuracy'].idxmax(), 'Model']
best_model = results[best_model_name]['model']
best_accuracy = performance_df['Accuracy'].max()

print(f"\n🏆 最佳模型: {best_model_name} (准确率: {best_accuracy:.4f})")

# 步骤6：可视化结果
print(f"\n=== 步骤6：创建可视化图表 ===")

# 可视化1：模型性能对比柱状图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    ax = axes[i//2, i%2]
    bars = ax.bar(performance_df['Model'], performance_df[metric], color=color, alpha=0.8)
    ax.set_title(f'{metric} Comparison', fontweight='bold')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# 可视化2：混淆矩阵热力图（为每个模型）
print("生成混淆矩阵热力图...")

n_models = len(results)
fig, axes = plt.subplots((n_models + 1) // 2, 2, figsize=(12, 4 * ((n_models + 1) // 2)))
if n_models == 1:
    axes = [axes]
elif n_models <= 2:
    axes = axes.reshape(1, -1)

for idx, (model_name, result) in enumerate(results.items()):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col] if n_models > 2 else axes[col] if n_models > 1 else axes[0]
    
    cm = confusion_matrix(y_test, result['y_pred'])
    
    # 创建热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(f'{model_name} - Confusion Matrix', fontweight='bold')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

# 隐藏多余的子图
if n_models % 2 == 1 and n_models > 1:
    axes[-1, -1].set_visible(False)

plt.tight_layout()
plt.show()

# 可视化3：ROC曲线
print("生成ROC曲线...")

# 为多分类问题创建ROC曲线
plt.figure(figsize=(15, 10))

colors = ['blue', 'red', 'green', 'orange', 'purple']
n_classes = len(class_names)

for model_idx, (model_name, result) in enumerate(results.items()):
    y_pred_proba = result['y_pred_proba']
    
    if n_classes == 2:
        # 二分类情况
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[model_idx % len(colors)], lw=2,
                label=f'{model_name} (AUC = {roc_auc:.3f})')
    else:
        # 多分类情况 - 计算宏平均AUC
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        # 计算每个类别的ROC
        fpr_dict = {}
        tpr_dict = {}
        roc_auc_dict = {}
        
        for i in range(n_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        
        # 计算宏平均
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= n_classes
        
        mean_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, color=colors[model_idx % len(colors)], lw=2,
                label=f'{model_name} (Macro AUC = {mean_auc:.3f})')

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', alpha=0.8)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 步骤7：详细分类报告
print(f"\n=== 步骤7：详细分类报告 ===")

for model_name, result in results.items():
    print(f"\n--- {model_name} 详细分类报告 ---")
    print(classification_report(y_test, result['y_pred'], 
                              target_names=class_names, digits=4))

# 步骤8：保存结果
print(f"\n=== 步骤8：保存结果 ===")

try:
    # 保存性能对比表
    performance_df.to_csv('model_performance_comparison.csv', index=False, encoding='utf-8-sig')
    print("✅ 模型性能对比表已保存: model_performance_comparison.csv")
    
    # 保存详细结果
    detailed_results = []
    for model_name, result in results.items():
        for i, class_name in enumerate(class_names):
            class_mask = (y_test == i)
            if class_mask.any():
                class_precision = precision_score(y_test, result['y_pred'], labels=[i], average=None)[0]
                class_recall = recall_score(y_test, result['y_pred'], labels=[i], average=None)[0]
                class_f1 = f1_score(y_test, result['y_pred'], labels=[i], average=None)[0]
                
                detailed_results.append({
                    'Model': model_name,
                    'Class': class_name,
                    'Precision': class_precision,
                    'Recall': class_recall,
                    'F1-Score': class_f1
                })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv('detailed_class_performance.csv', index=False, encoding='utf-8-sig')
    print("✅ 详细类别性能已保存: detailed_class_performance.csv")
    
except Exception as e:
    print(f"❌ 保存结果时出错: {e}")

# 步骤9：保存最优模型和相关参数
print(f"\n=== 步骤9：保存最优模型和相关参数 ===")

try:
    # 创建模型保存文件夹
    import os
    model_save_dir = 'saved_models'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
        print(f"创建模型保存文件夹: {model_save_dir}")
    
    # 保存最优模型
    best_model_path = os.path.join(model_save_dir, f'best_model_{best_model_name.lower().replace(" ", "_")}.pkl')
    joblib.dump(best_model, best_model_path)
    print(f"✅ 最优模型已保存: {best_model_path}")
    
    # 保存数据预处理器
    scaler_path = os.path.join(model_save_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"✅ 数据标准化器已保存: {scaler_path}")
    
    # 保存标签编码器
    label_encoder_path = os.path.join(model_save_dir, 'label_encoder.pkl')
    joblib.dump(label_encoder, label_encoder_path)
    print(f"✅ 标签编码器已保存: {label_encoder_path}")
    
    # 保存所有训练好的模型
    all_models_path = os.path.join(model_save_dir, 'all_trained_models.pkl')
    joblib.dump(trained_models, all_models_path)
    print(f"✅ 所有训练模型已保存: {all_models_path}")
    
    # 保存模型元数据和配置信息
    model_metadata = {
        'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'best_model': best_model_name,
        'best_accuracy': float(best_accuracy),
        'model_performance': performance_df.to_dict('records'),
        'feature_columns': feature_columns,
        'class_names': class_names.tolist(),
        'training_data_shape': X_train.shape,
        'test_data_shape': X_test.shape,
        'model_parameters': {
            'test_size': 0.3,
            'random_state': 42,
            'stratify': True
        },
        'model_configs': {}
    }
    
    # 保存每个模型的参数
    for model_name, model in trained_models.items():
        try:
            model_metadata['model_configs'][model_name] = model.get_params()
        except:
            model_metadata['model_configs'][model_name] = str(model)
    
    metadata_path = os.path.join(model_save_dir, 'model_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(model_metadata, f, ensure_ascii=False, indent=2, default=str)
    print(f"✅ 模型元数据已保存: {metadata_path}")
    
    # 保存模型调用示例代码
    example_code = f'''
# 模型加载和使用示例代码
import joblib
import pandas as pd
import numpy as np

# 1. 加载保存的模型和预处理器
best_model = joblib.load('{best_model_path}')
scaler = joblib.load('{scaler_path}')
label_encoder = joblib.load('{label_encoder_path}')

# 2. 特征列名
feature_columns = {feature_columns}

# 3. 类别标签
class_names = {class_names.tolist()}

# 4. 预测函数
def predict_with_best_model(new_data):
    """
    使用最佳模型进行预测
    
    参数:
    new_data: pandas DataFrame, 包含需要预测的数据
    
    返回:
    predictions: 预测结果（原始标签）
    probabilities: 预测概率
    """
    # 确保包含所有必需的特征列
    X_new = new_data[feature_columns]
    
    # 数据标准化
    X_new_scaled = scaler.transform(X_new)
    
    # 预测
    y_pred_encoded = best_model.predict(X_new_scaled)
    y_pred_proba = best_model.predict_proba(X_new_scaled)
    
    # 转换回原始标签
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    return y_pred, y_pred_proba

# 5. 使用示例
# new_data = pd.read_csv('your_new_data.csv')
# predictions, probabilities = predict_with_best_model(new_data)
# print("预测结果:", predictions)
# print("预测概率:", probabilities)

# 6. 单个样本预测示例
def predict_single_sample(sample_dict):
    """
    预测单个样本
    
    参数:
    sample_dict: dict, 包含特征值的字典，如 {{'PC1': 0.1, 'PC2': 0.2, ...}}
    """
    sample_df = pd.DataFrame([sample_dict])
    prediction, probability = predict_with_best_model(sample_df)
    return prediction[0], probability[0]

print("最佳模型: {best_model_name}")
print("模型准确率: {best_accuracy:.4f}")
print("训练完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
'''
    
    example_code_path = os.path.join(model_save_dir, 'model_usage_example.py')
    with open(example_code_path, 'w', encoding='utf-8') as f:
        f.write(example_code)
    print(f"✅ 模型使用示例已保存: {example_code_path}")
    
except Exception as e:
    print(f"❌ 保存模型时出错: {e}")

# 最终总结
print(f"\n{'='*80}")
print("🎉 机器学习模型训练与评估完成！")
print(f"{'='*80}")
print(f"📊 训练总结:")
print(f"   • 训练样本数: {len(X_train):,}")
print(f"   • 测试样本数: {len(X_test):,}")
print(f"   • 特征数量: {X_train.shape[1]}")
print(f"   • 类别数量: {len(class_names)}")
print(f"   • 成功训练模型数: {len(results)}")

print(f"\n🏆 模型排名 (按准确率):")
performance_sorted = performance_df.sort_values('Accuracy', ascending=False)
for idx, row in performance_sorted.iterrows():
    print(f"   {idx+1}. {row['Model']}: {row['Accuracy']:.4f}")

print(f"\n📁 输出文件:")
print(f"   • model_performance_comparison.csv")
print(f"   • detailed_class_performance.csv")
print(f"   • {model_save_dir}/: 模型保存文件夹")
print(f"     - best_model_*.pkl: 最佳模型")
print(f"     - scaler.pkl: 数据标准化器")
print(f"     - label_encoder.pkl: 标签编码器")
print(f"     - all_trained_models.pkl: 所有训练模型")
print(f"     - model_metadata.json: 模型元数据")
print(f"     - model_usage_example.py: 使用示例代码")

print(f"\n💡 建议:")
print(f"   • 最佳模型: {best_model_name}")
print(f"   • 可直接使用保存的模型进行预测")
print(f"   • 查看 model_usage_example.py 了解如何调用模型")
print(f"   • 模型元数据包含完整的训练配置信息")
print(f"{'='*80}")

# 将结果保存到全局变量
globals()['model_results'] = results
globals()['performance_summary'] = performance_df
globals()['best_model_saved'] = best_model
globals()['scaler_saved'] = scaler
globals()['label_encoder_saved'] = label_encoder

print(f"✅ 结果已保存到变量:")
print(f"   • model_results: 所有模型训练结果")
print(f"   • performance_summary: 性能汇总表")
print(f"   • best_model_saved: 最佳训练模型")
print(f"   • scaler_saved: 数据标准化器")
print(f"   • label_encoder_saved: 标签编码器")