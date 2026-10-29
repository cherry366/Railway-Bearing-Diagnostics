# -*- coding: utf-8 -*-
# 路径约定：仓库目录只读，运行产物统一写入 work/（可用环境变量 BD_ROOT 换盘）。
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import WORK, DATA, PROCESSED, MODEL_DIR, use_workdir  # noqa: E402

use_workdir()

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib支持中文显示
import matplotlib
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

def calculate_time_domain_features(data):
    """
    计算时域特征
    """
    features = {}
    
    # 基础统计量
    features['mean'] = np.mean(data)  # 均值
    features['std'] = np.std(data)  # 标准差
    features['var'] = np.var(data)  # 方差
    features['rms'] = np.sqrt(np.mean(data**2))  # 均方根
    features['peak'] = np.max(np.abs(data))  # 峰值
    features['p2p'] = np.max(data) - np.min(data)  # 峰峰值
    features['mean_abs'] = np.mean(np.abs(data))  # 平均绝对值
    
    # 无量纲指标
    if features['mean_abs'] != 0:
        features['shape_factor'] = features['rms'] / features['mean_abs']  # 波形系数
    else:
        features['shape_factor'] = 0
    
    if features['rms'] != 0:
        features['crest_factor'] = features['peak'] / features['rms']  # 峰值系数
    else:
        features['crest_factor'] = 0
    
    if features['mean_abs'] != 0:
        features['impulse_factor'] = features['peak'] / features['mean_abs']  # 脉冲指标
    else:
        features['impulse_factor'] = 0
    
    # 裕度指标
    root_mean = np.mean(np.sqrt(np.abs(data)))**2
    if root_mean != 0:
        features['margin_factor'] = features['peak'] / root_mean  # 裕度指标
    else:
        features['margin_factor'] = 0
    
    # 高阶统计量
    features['kurtosis'] = stats.kurtosis(data)  # 峭度
    features['skewness'] = stats.skew(data)  # 偏度
    
    return features

def calculate_frequency_domain_features(data, fs=12000):
    """
    计算频域特征
    fs: 采样频率，默认12000Hz
    """
    features = {}
    
    # 计算FFT
    n = len(data)
    fft_data = np.fft.fft(data)
    fft_abs = np.abs(fft_data[:n//2])
    freqs = np.fft.fftfreq(n, 1/fs)[:n//2]
    
    # 归一化幅值
    if np.sum(fft_abs) != 0:
        fft_norm = fft_abs / np.sum(fft_abs)
    else:
        fft_norm = fft_abs
    
    # 频谱质心
    if np.sum(fft_norm) != 0:
        features['spec_centroid'] = np.sum(freqs * fft_norm) / np.sum(fft_norm)
    else:
        features['spec_centroid'] = 0
    
    # 频谱带宽
    if features['spec_centroid'] != 0 and np.sum(fft_norm) != 0:
        features['spec_bandwidth'] = np.sqrt(
            np.sum(((freqs - features['spec_centroid'])**2) * fft_norm) / np.sum(fft_norm)
        )
    else:
        features['spec_bandwidth'] = 0
    
    # 频谱偏度
    if features['spec_bandwidth'] != 0 and np.sum(fft_norm) != 0:
        features['spec_skewness'] = (
            np.sum(((freqs - features['spec_centroid'])**3) * fft_norm) / 
            (np.sum(fft_norm) * features['spec_bandwidth']**3)
        )
    else:
        features['spec_skewness'] = 0
    
    # 频谱峭度
    if features['spec_bandwidth'] != 0 and np.sum(fft_norm) != 0:
        features['spec_kurtosis'] = (
            np.sum(((freqs - features['spec_centroid'])**4) * fft_norm) / 
            (np.sum(fft_norm) * features['spec_bandwidth']**4)
        )
    else:
        features['spec_kurtosis'] = 0
    
    # 频谱熵
    fft_norm_nonzero = fft_norm[fft_norm > 0]
    if len(fft_norm_nonzero) > 0:
        features['spec_entropy'] = -np.sum(fft_norm_nonzero * np.log2(fft_norm_nonzero))
    else:
        features['spec_entropy'] = 0
    
    # 带通能量（可以计算多个频带）
    # 这里计算一个示例频带的能量
    band_low = 1000  # Hz
    band_high = 3000  # Hz
    band_indices = np.where((freqs >= band_low) & (freqs <= band_high))[0]
    if len(band_indices) > 0:
        features['band_energy'] = np.sum(fft_abs[band_indices]**2)
    else:
        features['band_energy'] = 0
    
    return features

def calculate_time_frequency_features(data, fs=12000):
    """
    计算时频域特征（简化版）
    """
    features = {}
    
    # 时频熵（使用短时傅里叶变换的简化版）
    # 将信号分成若干段，计算每段的熵，然后求平均
    n_segments = min(10, len(data)//10)
    if n_segments > 1:
        segment_length = len(data) // n_segments
        entropies = []
        for i in range(n_segments):
            segment = data[i*segment_length:(i+1)*segment_length]
            if len(segment) > 0:
                # 计算段的功率谱
                fft_seg = np.abs(np.fft.fft(segment))
                fft_norm = fft_seg / np.sum(fft_seg) if np.sum(fft_seg) != 0 else fft_seg
                fft_norm_nonzero = fft_norm[fft_norm > 0]
                if len(fft_norm_nonzero) > 0:
                    entropy = -np.sum(fft_norm_nonzero * np.log2(fft_norm_nonzero))
                    entropies.append(entropy)
        features['tf_entropy'] = np.mean(entropies) if entropies else 0
    else:
        features['tf_entropy'] = 0
    
    # 平均频率（简化计算）
    fft_data = np.fft.fft(data)
    fft_abs = np.abs(fft_data[:len(data)//2])
    if np.sum(fft_abs) != 0:
        freqs = np.fft.fftfreq(len(data), 1/fs)[:len(data)//2]
        features['tf_mean_freq'] = np.sum(freqs * fft_abs) / np.sum(fft_abs)
    else:
        features['tf_mean_freq'] = 0
    
    # 小波能量（简化版，使用标准差代替实际小波变换）
    features['wavelet_energy'] = np.std(data)  # 简化的小波能量
    features['wavelet_energy_ratio'] = features['wavelet_energy'] / (np.sum(data**2) + 1e-10)
    
    return features

def sliding_window_sampling(data, window_size=10000, step_size=1000):
    """
    对数据进行滑动窗口采样
    """
    windows = []
    data_length = len(data)
    
    window_size = int(window_size)
    step_size = int(step_size)
    
    if window_size > data_length:
        print(f"    警告：窗口大小({window_size})超过数据长度({data_length})")
        return []
    
    start = 0
    while start + window_size <= data_length:
        window = data[start:start + window_size]
        windows.append(window)
        start += step_size
    
    return windows

def process_target_domain_data():
    """
    处理目标域数据的主函数
    """
    # 数据路径设置
    data_folder = os.path.join(DATA, 'target')
    output_folder = os.path.join(WORK, 'target_windows')
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")
    
    # 获取所有xlsx文件
    data_files = [f for f in os.listdir(data_folder) if f.endswith('.xlsx')]
    data_files.sort()
    
    print(f"找到 {len(data_files)} 个目标域数据文件")
    print(f"文件列表: {data_files}")
    
    # 存储所有窗口数据
    all_windows_data = []
    successful_count = 0
    failed_count = 0
    
    print("\n" + "="*60)
    print("开始目标域数据划窗采样处理...")
    
    for file_idx, data_file in enumerate(data_files):
        print(f"\n处理第 {file_idx+1} 个文件:")
        
        try:
            data_path = os.path.join(data_folder, data_file)
            print(f"  文件: {data_file}")
            
            # 读取Excel文件
            data_df = pd.read_excel(data_path)
            print(f"  数据形状: {data_df.shape}")
            print(f"  列名: {list(data_df.columns)}")
            
            # 假设数据在第一列，如果列名不同请调整
            if len(data_df.columns) > 0:
                # 获取第一列数据，无论列名是什么
                data_column = data_df.iloc[:, 0]
                data_values = data_column.dropna().values
                
                print(f"  数据长度: {len(data_values)}")
                
                if len(data_values) == 0:
                    print(f"  ✗ 错误：数据为空")
                    failed_count += 1
                    continue
                
                # 执行划窗采样
                windows = sliding_window_sampling(data_values, window_size=10000, step_size=1000)
                print(f"  采样得到 {len(windows)} 个窗口")
                
                if len(windows) == 0:
                    print(f"  ✗ 警告：没有生成任何窗口")
                    failed_count += 1
                    continue
                
                # 为每个窗口计算特征
                for win_idx, window in enumerate(windows):
                    # 创建窗口记录
                    window_info = {
                        'file_name': data_file,
                        'file_label': data_file.split('.')[0],  # 使用文件名作为标签
                        'file_index': file_idx,
                        'window_index': win_idx,
                        'window_size': 10000,
                        'step_size': 1000,
                        'window_length': len(window)
                    }
                    
                    # 计算时域特征
                    time_features = calculate_time_domain_features(window)
                    for key, value in time_features.items():
                        window_info[f'td_{key}'] = value
                    
                    # 计算频域特征
                    freq_features = calculate_frequency_domain_features(window)
                    for key, value in freq_features.items():
                        window_info[f'fd_{key}'] = value
                    
                    # 计算时频域特征
                    tf_features = calculate_time_frequency_features(window)
                    for key, value in tf_features.items():
                        window_info[f'tf_{key}'] = value
                    
                    all_windows_data.append(window_info)
                
                successful_count += 1
                print(f"  ✅ 处理成功")
                
            else:
                print(f"  ✗ 错误：文件中没有数据列")
                failed_count += 1
                continue
                
        except Exception as e:
            print(f"  ✗ 处理出错: {str(e)}")
            failed_count += 1
    
    # 保存汇总结果
    print("\n" + "="*60)
    print("保存汇总结果...")
    
    if all_windows_data:
        summary_df = pd.DataFrame(all_windows_data)
        summary_path = os.path.join(output_folder, 'target_domain_windows_summary.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        print(f"✅ 目标域窗口汇总信息已保存到: {summary_path}")
        print(f"总窗口数: {len(all_windows_data)}")
        print(f"特征总数: {len(summary_df.columns)}")
        
        # 显示特征统计
        print(f"\n特征列统计:")
        td_features = [col for col in summary_df.columns if col.startswith('td_')]
        fd_features = [col for col in summary_df.columns if col.startswith('fd_')]
        tf_features = [col for col in summary_df.columns if col.startswith('tf_')]
        
        print(f"时域特征数量: {len(td_features)}")
        print(f"频域特征数量: {len(fd_features)}")
        print(f"时频域特征数量: {len(tf_features)}")
        print(f"总特征数量: {len(td_features) + len(fd_features) + len(tf_features)}")
        
        print(f"\n时域特征列表: {td_features}")
        print(f"频域特征列表: {fd_features}")
        print(f"时频域特征列表: {tf_features}")
        
        print(f"\n文件统计:")
        file_stats = summary_df['file_label'].value_counts()
        print(file_stats)
        
        return summary_df, output_folder
    else:
        print("✗ 没有成功处理任何数据")
        return None, output_folder

def perform_pca_analysis(df, output_folder, n_components=13):
    """
    对特征数据进行PCA降维分析
    """
    print("\n" + "="*60)
    print("开始主成分分析 (PCA)")
    print("="*60)
    
    # 步骤1：准备数据
    print("\n=== 步骤1：数据准备 ===")
    
    # 分离特征列和标识列
    meta_columns = ['file_name', 'file_label', 'file_index', 'window_index', 
                   'window_size', 'step_size', 'window_length']
    feature_columns = [col for col in df.columns if col.startswith(('td_', 'fd_', 'tf_'))]
    
    print(f"数据总形状: {df.shape}")
    print(f"特征列数量: {len(feature_columns)}")
    print(f"元数据列数量: {len(meta_columns)}")
    
    # 提取特征数据
    X = df[feature_columns].copy()
    print(f"特征矩阵形状: {X.shape}")
    
    # 检查缺失值
    missing_features = X.isnull().sum()
    if missing_features.sum() > 0:
        print(f"发现 {missing_features.sum()} 个缺失值，将用均值填充")
        X = X.fillna(X.mean())
    
    # 步骤2：数据标准化和PCA
    print(f"\n=== 步骤2：执行PCA分析 ===")
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✅ 数据标准化完成")
    
    # 执行完整PCA分析
    pca_full = PCA()
    X_pca_full = pca_full.fit_transform(X_scaled)
    
    # 计算各主成分的方差贡献度
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print(f"原始特征维度: {X.shape[1]}")
    print(f"前10个主成分的方差贡献度: {explained_variance_ratio[:10]}")
    print(f"前10个主成分的累积方差贡献度: {cumulative_variance_ratio[:10]}")
    
    # 找到累积贡献度达到不同阈值的主成分数量
    n_components_90 = np.argmax(cumulative_variance_ratio >= 0.90) + 1
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    n_components_99 = np.argmax(cumulative_variance_ratio >= 0.99) + 1
    
    print(f"达到90%方差解释的主成分数量: {n_components_90}")
    print(f"达到95%方差解释的主成分数量: {n_components_95}")
    print(f"达到99%方差解释的主成分数量: {n_components_99}")
    print(f"指定降维到 {n_components} 维的方差解释度: {cumulative_variance_ratio[n_components-1]:.4f}")
    
    # 执行指定维度的PCA
    pca_final = PCA(n_components=n_components)
    X_pca_final = pca_final.fit_transform(X_scaled)
    
    # 创建PCA结果数据框
    pca_feature_names = [f'PC{i+1}' for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca_final, columns=pca_feature_names, index=df.index)
    
    # 添加元数据列
    for col in meta_columns:
        if col in df.columns:
            df_pca[col] = df[col]
    
    print(f"PCA转换后的数据形状: {df_pca.shape}")
    print(f"降维效果: {X.shape[1]} → {n_components} 特征")
    print(f"保留的方差比例: {cumulative_variance_ratio[n_components-1]:.4f}")
    
    # 保存PCA结果
    pca_path = os.path.join(output_folder, f'target_domain_pca_{n_components}d.csv')
    df_pca.to_csv(pca_path, index=False, encoding='utf-8-sig')
    print(f"✅ PCA结果已保存到: {pca_path}")
    
    return df_pca, pca_full, scaler, feature_columns

def create_visualizations(df_original, df_pca, pca_model, output_folder):
    """
    创建可视化图表
    """
    print("\n" + "="*60)
    print("开始创建可视化图表")
    print("="*60)
    
    # 获取PCA信息
    explained_variance_ratio = pca_model.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # 图1：前20个主成分的方差贡献度
    print("正在生成图1：前20个主成分的方差贡献度...")
    plt.figure(figsize=(14, 6))
    n_show = min(20, len(explained_variance_ratio))
    bars = plt.bar(range(1, n_show + 1), explained_variance_ratio[:n_show], 
                   color='skyblue', alpha=0.8, edgecolor='navy')
    plt.title('Top 20 Principal Components - Variance Explained\n(Target Domain Data)', 
              fontsize=14, fontweight='bold')
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
    plt.savefig(os.path.join(output_folder, 'pca_variance_explained.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 图2：累积方差贡献度
    print("正在生成图2：累积方差贡献度曲线...")
    plt.figure(figsize=(14, 6))
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 
             'bo-', linewidth=2, markersize=4, color='navy')
    
    # 添加重要阈值线
    plt.axhline(y=0.90, color='green', linestyle=':', linewidth=2, label='90% Variance')
    plt.axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Variance')
    plt.axhline(y=0.99, color='orange', linestyle='--', linewidth=2, label='99% Variance')
    
    # 添加13维的标记
    if len(cumulative_variance_ratio) >= 13:
        plt.axvline(x=13, color='purple', linestyle='-', linewidth=2, alpha=0.7, label='13 Components')
        plt.text(13, 0.5, f'PC13\n{cumulative_variance_ratio[12]:.3f}', 
                 ha='center', va='center', color='purple', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.title('Cumulative Explained Variance Ratio\n(Target Domain Data)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Number of Principal Components', fontsize=12)
    plt.ylabel('Cumulative Explained Variance Ratio', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pca_cumulative_variance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 图3：前三个主成分的3D散点图
    print("正在生成图3：前三个主成分3D散点图...")
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 按文件标签着色
    unique_labels = df_pca['file_label'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = df_pca['file_label'] == label
        ax.scatter(df_pca.loc[mask, 'PC1'], 
                  df_pca.loc[mask, 'PC2'], 
                  df_pca.loc[mask, 'PC3'],
                  c=[colors[i]], label=f'File {label}', alpha=0.7, s=30)
    
    ax.set_xlabel(f'PC1 (Variance: {explained_variance_ratio[0]:.3f})', fontsize=12)
    ax.set_ylabel(f'PC2 (Variance: {explained_variance_ratio[1]:.3f})', fontsize=12)
    ax.set_zlabel(f'PC3 (Variance: {explained_variance_ratio[2]:.3f})', fontsize=12)
    ax.set_title('First Three Principal Components 3D Scatter Plot\n(Target Domain Data)', 
                 fontsize=14, fontweight='bold')
    
    # 调整图例位置
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pca_3d_scatter.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 图4：前两个主成分的散点图矩阵
    print("正在生成图4：前两个主成分散点图...")
    plt.figure(figsize=(12, 8))
    
    for i, label in enumerate(unique_labels):
        mask = df_pca['file_label'] == label
        plt.scatter(df_pca.loc[mask, 'PC1'], 
                   df_pca.loc[mask, 'PC2'],
                   c=[colors[i]], label=f'File {label}', alpha=0.7, s=40)
    
    plt.xlabel(f'PC1 (Explained Variance: {explained_variance_ratio[0]:.3f})', fontsize=12)
    plt.ylabel(f'PC2 (Explained Variance: {explained_variance_ratio[1]:.3f})', fontsize=12)
    plt.title('First Two Principal Components Scatter Plot\n(Target Domain Data)', 
              fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pca_2d_scatter.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 图5：各文件的主成分分布箱线图
    print("正在生成图5：各文件主成分分布箱线图...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    pcs_to_plot = ['PC1', 'PC2', 'PC3', 'PC4']
    for i, pc in enumerate(pcs_to_plot):
        ax = axes[i//2, i%2]
        df_pca.boxplot(column=pc, by='file_label', ax=ax)
        ax.set_title(f'{pc} Distribution by File\n(Variance: {explained_variance_ratio[i]:.3f})')
        ax.set_xlabel('File Label')
        ax.set_ylabel(f'{pc} Value')
        plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.suptitle('Principal Components Distribution by File\n(Target Domain Data)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pca_boxplots.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 图6：特征重要性热力图（载荷矩阵）
    print("正在生成图6：主成分载荷热力图...")
    feature_columns = [col for col in df_original.columns if col.startswith(('td_', 'fd_', 'tf_'))]
    
    plt.figure(figsize=(20, 12))
    
    # 显示前13个主成分的载荷
    heatmap_data = pca_model.components_[:13]  # 13个主成分
    
    # 创建更清晰的特征名称
    feature_names_clean = []
    for col in feature_columns:
        if col.startswith('td_'):
            feature_names_clean.append(f"TD_{col[3:]}")
        elif col.startswith('fd_'):
            feature_names_clean.append(f"FD_{col[3:]}")
        elif col.startswith('tf_'):
            feature_names_clean.append(f"TF_{col[3:]}")
    
    im = plt.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, label='Feature Loading', shrink=0.6)
    
    plt.title('Principal Component - Feature Loading Heatmap\n(Target Domain Data - All Features)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Principal Components', fontsize=12)
    
    # 设置刻度
    plt.xticks(range(len(feature_names_clean)), feature_names_clean, rotation=90, ha='center')
    plt.yticks(range(13), [f'PC{i+1}' for i in range(13)])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'pca_loading_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ 所有可视化图表已生成并保存！")

def main():
    """
    主函数
    """
    print("="*80)
    print("目标域数据划窗采样与降维处理")
    print("="*80)
    
    # 步骤1：处理目标域数据
    print("\n步骤1：处理目标域数据...")
    df_summary, output_folder = process_target_domain_data()
    
    if df_summary is None:
        print("❌ 数据处理失败，程序终止")
        return
    
    print(f"\n✅ 数据处理完成！")
    print(f"成功处理数据形状: {df_summary.shape}")
    
    # 步骤2：执行PCA降维
    print(f"\n步骤2：执行PCA降维到13维...")
    df_pca, pca_model, scaler, feature_columns = perform_pca_analysis(df_summary, output_folder, n_components=13)
    
    # 步骤3：创建可视化
    print(f"\n步骤3：创建可视化图表...")
    create_visualizations(df_summary, df_pca, pca_model, output_folder)
    
    # 最终总结
    print("\n" + "="*80)
    print("🎉 所有处理完成！")
    print("="*80)
    print(f"原始窗口数据: {df_summary.shape}")
    print(f"PCA降维后数据: {df_pca.shape}")
    print(f"特征维度: {len(feature_columns)} → 13")
    print(f"特征类型分布:")
    td_count = len([col for col in feature_columns if col.startswith('td_')])
    fd_count = len([col for col in feature_columns if col.startswith('fd_')])
    tf_count = len([col for col in feature_columns if col.startswith('tf_')])
    print(f"  时域特征: {td_count}")
    print(f"  频域特征: {fd_count}")
    print(f"  时频域特征: {tf_count}")
    print(f"输出文件夹: {output_folder}")
    
    # 显示各文件的窗口统计
    print(f"\n文件处理统计:")
    file_stats = df_summary['file_label'].value_counts().sort_index()
    for file_label, count in file_stats.items():
        print(f"  文件 {file_label}: {count} 个窗口")
    
    print(f"\n生成的文件:")
    print(f"  1. target_domain_windows_summary.csv - 原始特征数据")
    print(f"  2. target_domain_pca_13d.csv - PCA降维后数据")
    print(f"  3. 多个可视化图表PNG文件")

if __name__ == "__main__":
    main()
