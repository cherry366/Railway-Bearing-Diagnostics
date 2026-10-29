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

# 读取df1.csv
df1_path = os.path.join(WORK, 'windows/windows_summary_complete.csv')
df1 = pd.read_csv(df1_path, encoding='utf-8-sig')

# 方法1：统计每列的缺失值数量和百分比
print("=== 缺失值统计 ===")
missing_stats = pd.DataFrame({
    '缺失值数量': df1.isnull().sum(),
    '缺失值百分比': (df1.isnull().sum() / len(df1)) * 100
})

# 只显示有缺失值的列
missing_stats = missing_stats[missing_stats['缺失值数量'] > 0]
missing_stats = missing_stats.sort_values('缺失值数量', ascending=False)

print(missing_stats)
print(f"\n总共有 {len(missing_stats)} 列存在缺失值")

# 方法2：查看数据基本信息
print("\n=== 数据基本信息 ===")
print(f"数据集形状: {df1.shape}")
print(f"总记录数: {len(df1)}")
df1.info()

# 方法3：统计每行的缺失值情况
print("\n=== 每行缺失值统计 ===")
row_missing = df1.isnull().sum(axis=1)
print(f"完全没有缺失值的行数: {(row_missing == 0).sum()}")
print(f"有缺失值的行数: {(row_missing > 0).sum()}")
print(f"最大缺失值数量的行: {row_missing.max()} 个缺失值")

# 方法4：查看缺失值分布
print("\n=== 缺失值分布 ===")
missing_counts = row_missing.value_counts().sort_index()
for missing_count, row_count in missing_counts.items():
    if missing_count > 0:
        print(f"缺失 {missing_count} 个值的行数: {row_count}")

# 方法5：如果你想看具体哪些行有缺失值
print("\n=== 缺失值行索引示例（前10个）===")
rows_with_missing = df1[df1.isnull().any(axis=1)].index[:10].tolist()
if rows_with_missing:
    print(f"有缺失值的行索引（前10个）: {rows_with_missing}")
else:
    print("没有发现缺失值")


# 方法1：统计每列的缺失值数量和百分比
print("=== 缺失值统计 ===")
missing_stats = pd.DataFrame({
    '缺失值数量': df1.isnull().sum(),
    '缺失值百分比': (df1.isnull().sum() / len(df1)) * 100
})

# 只显示有缺失值的列
missing_stats = missing_stats[missing_stats['缺失值数量'] > 0]
missing_stats = missing_stats.sort_values('缺失值数量', ascending=False)

print(missing_stats)
print(f"\n总共有 {len(missing_stats)} 列存在缺失值")

# 方法2：查看数据基本信息
print("\n=== 数据基本信息 ===")
print(f"数据集形状: {df1.shape}")
print(f"总记录数: {len(df1)}")
df1.info()

# 方法3：统计每行的缺失值情况
print("\n=== 每行缺失值统计 ===")
row_missing = df1.isnull().sum(axis=1)
print(f"完全没有缺失值的行数: {(row_missing == 0).sum()}")
print(f"有缺失值的行数: {(row_missing > 0).sum()}")
print(f"最大缺失值数量的行: {row_missing.max()} 个缺失值")

# 方法4：查看缺失值分布
print("\n=== 缺失值分布 ===")
missing_counts = row_missing.value_counts().sort_index()
for missing_count, row_count in missing_counts.items():
    if missing_count > 0:
        print(f"缺失 {missing_count} 个值的行数: {row_count}")

# 方法5：如果你想看具体哪些行有缺失值
print("\n=== 缺失值行索引示例（前10个）===")
rows_with_missing = df1[df1.isnull().any(axis=1)].index[:10].tolist()
if rows_with_missing:
    print(f"有缺失值的行索引（前10个）: {rows_with_missing}")
else:
    print("没有发现缺失值")

# 可选：保存缺失值统计到文件
# missing_stats.to_csv('missing_values_report.csv', encoding='utf-8-sig')
# print("缺失值报告已保存到 missing_values_report.csv")

print("\n" + "="*50)
print("开始数据预处理...")

# 1. 用0填充指定列的缺失值
columns_to_fill_zero = ['BSF', 'BSF_period', 'FTF', 'FTF_period', 
                       'BPFI', 'BPFI_period', 'BPFO', 'BPFO_period', 
                       'Rotation_period']

print(f"\n=== 步骤1：用0填充指定列的缺失值 ===")
for col in columns_to_fill_zero:
    if col in df1.columns:
        missing_before = df1[col].isnull().sum()
        df1[col] = df1[col].fillna(0)
        print(f"{col}: 填充了 {missing_before} 个缺失值")
    else:
        print(f"警告: 列 '{col}' 不存在于数据中")

# 2. 对position、Load列进行独热编码（缺失值作为一类）
print(f"\n=== 步骤2：对position和Load列进行独热编码 ===")
categorical_columns = ['position', 'Load']

for col in categorical_columns:
    if col in df1.columns:
        print(f"\n处理列: {col}")
        
        # 查看缺失值情况
        missing_count = df1[col].isnull().sum()
        print(f"缺失值数量: {missing_count}")
        
        # 将缺失值填充为'Missing'，作为一个独立类别
        df1[col] = df1[col].fillna('Missing')
        
        # 查看唯一值
        unique_values = df1[col].unique()
        print(f"唯一值: {unique_values}")
        print(f"类别数量: {len(unique_values)}")
        
        # 进行独热编码
        dummies = pd.get_dummies(df1[col], prefix=col, dummy_na=False)
        
        # 删除原列
        df1 = df1.drop(columns=[col])
        
        # 添加独热编码列
        df1 = pd.concat([df1, dummies], axis=1)
        
        print(f"生成的独热编码列: {list(dummies.columns)}")
    else:
        print(f"警告: 列 '{col}' 不存在于数据中")

# 3. 检查处理后的缺失值情况
print(f"\n=== 步骤3：检查处理后的数据 ===")
remaining_missing = df1.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]

if len(remaining_missing) > 0:
    print("剩余缺失值:")
    print(remaining_missing)
else:
    print("✅ 所有缺失值已处理完成！")

print(f"\n处理后数据形状: {df1.shape}")
print(f"列数变化: {df1.shape[1] - len(df1.columns) if 'original_columns' in locals() else '无法计算'}")

# 显示前几行数据以确认处理结果
print(f"\n=== 处理后数据预览 ===")
print(df1.head())

# 4. 数据类型检查和优化
print(f"\n=== 步骤4：数据类型检查 ===")
print("各列数据类型:")
dtype_info = df1.dtypes.value_counts()
print(dtype_info)

# 检查独热编码列是否为0/1
categorical_cols = [col for col in df1.columns if col.startswith(('position_', 'Load_'))]
if categorical_cols:
    print(f"\n独热编码列检查（前5列示例）:")
    for col in categorical_cols[:5]:
        unique_vals = df1[col].unique()
        print(f"{col}: {unique_vals}")

# 5. 保存处理后的数据（可选）
print(f"\n=== 步骤5：保存处理后的数据 ===")
# 生成输出文件路径
output_path = os.path.join(WORK, 'windows/windows_summary_processed.csv')
try:
    df1.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 处理后的数据已保存到: {output_path}")
except Exception as e:
    print(f"保存失败: {e}")
    print("你可以手动指定保存路径")

print(f"\n🎉 数据预处理完成！")
print(f"最终数据形状: {df1.shape}")
print(f"最终列名（前10个）: {list(df1.columns[:10])}")
if len(df1.columns) > 10:
    print(f"... 还有 {len(df1.columns) - 10} 列")
    
print(f"\n" + "="*50)
print("开始数据后处理...")

# 步骤6：将False替换成0，True替换成1
print(f"\n=== 步骤6：布尔值转换 ===")
# 检查是否有布尔类型的列
bool_columns = df1.select_dtypes(include=[bool]).columns.tolist()
if bool_columns:
    print(f"发现布尔类型列: {bool_columns}")
    for col in bool_columns:
        true_count = df1[col].sum()
        false_count = len(df1) - true_count
        print(f"{col}: True={true_count}, False={false_count}")
    
    # 转换布尔值
    df1 = df1.replace({True: 1, False: 0})
    print("✅ 布尔值转换完成")
else:
    print("未发现布尔类型列，检查是否有字符串形式的True/False")
    
# 检查字符串形式的True/False
string_bool_replacements = 0
for col in df1.columns:
    if df1[col].dtype == 'object':
        if df1[col].isin(['True', 'False']).any():
            print(f"在列 {col} 中发现字符串形式的True/False")
            df1[col] = df1[col].replace({'True': 1, 'False': 0})
            string_bool_replacements += 1

if string_bool_replacements > 0:
    print(f"✅ 转换了 {string_bool_replacements} 列的字符串布尔值")
else:
    print("未发现字符串形式的True/False")

# 步骤7：删除指定列
print(f"\n=== 步骤7：删除指定列 ===")
columns_to_drop = ['frequency', 'Data length', 'RPM', 'Fault type', 'size', 
                  'file_index', 'file_name', 'window_index', 'window_size_used', 
                  'step_size_used', 'window_length','Source']

existing_columns_to_drop = []
missing_columns = []

for col in columns_to_drop:
    if col in df1.columns:
        existing_columns_to_drop.append(col)
    else:
        missing_columns.append(col)

if existing_columns_to_drop:
    print(f"准备删除的列: {existing_columns_to_drop}")
    df2 = df1.drop(columns=existing_columns_to_drop)
    print(f"✅ 成功删除 {len(existing_columns_to_drop)} 列")
else:
    df2 = df1.copy()
    print("没有需要删除的列")

if missing_columns:
    print(f"警告: 以下列在数据中不存在: {missing_columns}")

print(f"\n=== 最终结果 ===")
print(f"原始df1形状: {df1.shape}")
print(f"最终df2形状: {df2.shape}")
print(f"删除了 {df1.shape[1] - df2.shape[1]} 列")

# 检查最终数据类型
print(f"\ndf2数据类型分布:")
print(df2.dtypes.value_counts())

# 检查是否还有对象类型的列
object_columns = df2.select_dtypes(include=['object']).columns.tolist()
if object_columns:
    print(f"\n剩余的对象类型列: {object_columns}")
    for col in object_columns[:3]:  # 显示前3列的示例值
        unique_vals = df2[col].unique()[:5]  # 前5个唯一值
        print(f"{col} 示例值: {unique_vals}")

print(f"\ndf2列名（前15个）: {list(df2.columns[:15])}")
if len(df2.columns) > 15:
    print(f"... 还有 {len(df2.columns) - 15} 列")

# 显示df2的前几行
print(f"\n=== df2数据预览 ===")
print(df2.head(3))

print(f"\n🎊 所有数据处理完成！df2已准备就绪！")
#########进行主成分分析


# 设置matplotlib支持中文显示
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

print("="*60)
print("开始主成分分析 (PCA)")
print("="*60)

# 步骤1：准备数据
print("\n=== 步骤1：数据准备 ===")

# 分离特征列和标签列
excluded_columns = ['label1', 'label2']
feature_columns = [col for col in df2.columns if col not in excluded_columns]

# 检查被排除的列是否存在
existing_excluded = [col for col in excluded_columns if col in df2.columns]
missing_excluded = [col for col in excluded_columns if col not in df2.columns]

print(f"数据总形状: {df2.shape}")
print(f"特征列数量: {len(feature_columns)}")
print(f"存在的标签列: {existing_excluded}")
if missing_excluded:
    print(f"警告: 以下标签列不存在: {missing_excluded}")

# 提取特征数据
X = df2[feature_columns].copy()
print(f"特征矩阵形状: {X.shape}")

# 检查特征数据的基本信息
print(f"特征数据类型分布:")
print(X.dtypes.value_counts())

# 检查是否有非数值列
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    print(f"警告: 发现非数值列: {non_numeric_cols}")
    print("将尝试转换为数值类型...")
    for col in non_numeric_cols:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            print(f"成功转换 {col} 为数值类型")
        except:
            print(f"无法转换 {col}，将删除此列")
            X = X.drop(columns=[col])
            feature_columns.remove(col)

# 检查缺失值
missing_features = X.isnull().sum()
if missing_features.sum() > 0:
    print(f"发现 {missing_features.sum()} 个缺失值，将用均值填充")
    X = X.fillna(X.mean())

print(f"最终用于PCA的特征矩阵形状: {X.shape}")

# 步骤2：数据标准化和PCA
print(f"\n=== 步骤2：执行PCA分析 ===")

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ 数据标准化完成")

# 执行PCA - 先用所有成分来分析
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

# 计算各主成分的方差贡献度
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print(f"原始特征维度: {X.shape[1]}")
print(f"前10个主成分的方差贡献度: {explained_variance_ratio[:10]}")
print(f"前10个主成分的累积方差贡献度: {cumulative_variance_ratio[:10]}")

# 找到累积贡献度达到95%和99%的主成分数量
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
n_components_99 = np.argmax(cumulative_variance_ratio >= 0.99) + 1

print(f"达到95%方差解释的主成分数量: {n_components_95}")
print(f"达到99%方差解释的主成分数量: {n_components_99}")

# 步骤3：创建可视化图表（分别显示）
print(f"\n=== 步骤3：创建可视化图表 ===")

# 图1：前20个主成分的方差贡献度
print("正在生成图1：前20个主成分的方差贡献度...")
plt.figure(figsize=(12, 6))
n_show = min(20, len(explained_variance_ratio))
bars = plt.bar(range(1, n_show + 1), explained_variance_ratio[:n_show], 
               color='skyblue', alpha=0.8)
plt.title('Top 20 Principal Components - Variance Explained', fontsize=14, fontweight='bold')
plt.xlabel('Principal Component Number', fontsize=12)
plt.ylabel('Explained Variance Ratio', fontsize=12)
plt.xticks(range(1, n_show + 1, 2))
plt.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, bar in enumerate(bars[:10]):  # 只在前10个柱子上添加标签
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# 图2：累积方差贡献度
print("正在生成图2：累积方差贡献度曲线...")
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 
         'bo-', linewidth=2, markersize=4, color='navy')
plt.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Variance')
plt.axhline(y=0.99, color='orange', linestyle='--', linewidth=2, label='99% Variance')
plt.axvline(x=n_components_95, color='red', linestyle='--', alpha=0.7, linewidth=1)
plt.axvline(x=n_components_99, color='orange', linestyle='--', alpha=0.7, linewidth=1)

# 添加文本标注
plt.text(n_components_95, 0.5, f'95% at PC{n_components_95}', 
         rotation=90, ha='right', va='bottom', color='red', fontweight='bold')
plt.text(n_components_99, 0.5, f'99% at PC{n_components_99}', 
         rotation=90, ha='right', va='bottom', color='orange', fontweight='bold')

plt.title('Cumulative Explained Variance Ratio', fontsize=14, fontweight='bold')
plt.xlabel('Number of Principal Components', fontsize=12)
plt.ylabel('Cumulative Explained Variance Ratio', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 图3：主成分散点图（前两个主成分）
print("正在生成图3：前两个主成分散点图...")
plt.figure(figsize=(12, 8))

if existing_excluded:
    # 如果有标签，用不同颜色表示
    label_col = existing_excluded[0]  # 使用第一个可用的标签列
    unique_labels = df2[label_col].unique()[:10]  # 最多显示10个类别
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = df2[label_col] == label
        plt.scatter(X_pca_full[mask, 0], X_pca_full[mask, 1], 
                   c=[colors[i]], label=f'{str(label)[:15]}', alpha=0.7, s=30)
    
    plt.title(f'First Two Principal Components Scatter Plot\n(Colored by {label_col})', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
else:
    plt.scatter(X_pca_full[:, 0], X_pca_full[:, 1], alpha=0.6, s=30, color='steelblue')
    plt.title('First Two Principal Components Scatter Plot', fontsize=14, fontweight='bold')

plt.xlabel(f'PC1 (Explained Variance: {explained_variance_ratio[0]:.3f})', fontsize=12)
plt.ylabel(f'PC2 (Explained Variance: {explained_variance_ratio[1]:.3f})', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 图4：特征载荷图（前两个主成分）
print("正在生成图4：特征载荷图...")
plt.figure(figsize=(12, 8))

# 获取前两个主成分的载荷
loadings = pca_full.components_[:2].T
feature_names = feature_columns

# 只显示载荷绝对值最大的前30个特征
loading_magnitude = np.sqrt(loadings[:, 0]**2 + loadings[:, 1]**2)
top_features_idx = np.argsort(loading_magnitude)[-30:]

plt.scatter(loadings[top_features_idx, 0], loadings[top_features_idx, 1], 
           alpha=0.7, s=50, color='darkred')

# 添加特征标签（只显示前15个最重要的）
top_15_idx = np.argsort(loading_magnitude)[-15:]
for i in top_15_idx:
    plt.annotate(str(feature_names[i])[:20], 
                (loadings[i, 0], loadings[i, 1]), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.xlabel('PC1 Loading', fontsize=12)
plt.ylabel('PC2 Loading', fontsize=12)
plt.title('Feature Loading Plot (First Two Principal Components)', fontsize=14, fontweight='bold')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 图5：不同主成分数量下的累积方差解释度（详细版）
print("正在生成图5：详细的累积方差解释度...")
plt.figure(figsize=(12, 6))
components_range = range(1, min(51, X.shape[1] + 1))  # 最多显示50个主成分
cumvar_range = cumulative_variance_ratio[:len(components_range)]

plt.plot(components_range, cumvar_range, 'g-', linewidth=3, marker='o', markersize=4)
plt.fill_between(components_range, cumvar_range, alpha=0.3, color='green')

# 添加重要的阈值线
plt.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Threshold')
plt.axhline(y=0.99, color='orange', linestyle='--', linewidth=2, label='99% Threshold')
plt.axhline(y=0.90, color='blue', linestyle=':', linewidth=2, label='90% Threshold')

plt.title('Number of Components vs Cumulative Explained Variance', fontsize=14, fontweight='bold')
plt.xlabel('Number of Principal Components', fontsize=12)
plt.ylabel('Cumulative Explained Variance Ratio', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# 图6：各主成分的特征重要性热力图
print("正在生成图6：主成分-特征载荷热力图...")
plt.figure(figsize=(15, 10))

# 显示前10个主成分和重要性最高的前25个特征
n_components_show = min(10, pca_full.n_components_)
n_features_show = min(25, len(feature_columns))

# 选择载荷绝对值最大的特征
total_loading = np.abs(pca_full.components_[:n_components_show]).sum(axis=0)
top_feature_idx = np.argsort(total_loading)[-n_features_show:]

heatmap_data = pca_full.components_[:n_components_show][:, top_feature_idx]
feature_names_show = [str(feature_columns[i])[:20] for i in top_feature_idx]

im = plt.imshow(heatmap_data, cmap='RdBu_r', aspect='auto')
plt.colorbar(im, label='Feature Loading', shrink=0.8)

plt.title('Principal Component - Feature Loading Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Principal Components', fontsize=12)

# 设置刻度
plt.xticks(range(len(feature_names_show)), feature_names_show, rotation=45, ha='right')
plt.yticks(range(n_components_show), [f'PC{i+1}' for i in range(n_components_show)])

plt.tight_layout()
plt.show()

print("所有可视化图表生成完成！")
# 步骤4：生成PCA转换后的数据
print(f"\n=== 步骤4：特征选择建议 ===")

# 建议使用95%方差解释的主成分数量
recommended_components = n_components_95
print(f"建议使用的主成分数量: {recommended_components} (解释95%方差)")

# 创建最终的PCA模型
pca_final = PCA(n_components=recommended_components)
X_pca_final = pca_final.fit_transform(X_scaled)

# 创建PCA特征数据框
pca_feature_names = [f'PC{i+1}' for i in range(recommended_components)]
df_pca_features = pd.DataFrame(X_pca_final, columns=pca_feature_names, index=df2.index)

# 添加原始的标签列
for col in existing_excluded:
    df_pca_features[col] = df2[col]

print(f"PCA转换后的数据形状: {df_pca_features.shape}")
print(f"降维效果: {X.shape[1]} → {recommended_components} 特征")
print(f"保留的方差比例: {cumulative_variance_ratio[recommended_components-1]:.4f}")

# 显示转换后数据的预览
print(f"\n=== PCA转换后的数据预览 ===")
print(df_pca_features.head())

print(f"\n{'='*60}")
print("🎉 主成分分析完成！")
print(f"原始特征数: {len(feature_columns)}")
print(f"推荐主成分数: {recommended_components}")
print(f"方差解释比例: {cumulative_variance_ratio[recommended_components-1]:.4f}")
print(f"数据降维比例: {recommended_components/len(feature_columns):.4f}")
print(f"{'='*60}")

# 保存PCA结果
df_pca_features.to_csv(os.path.join(WORK, 'df2_pca_features.csv'), index=False, encoding='utf-8-sig')
print("PCA特征已保存到", os.path.join(WORK, "df2_pca_features.csv"))

