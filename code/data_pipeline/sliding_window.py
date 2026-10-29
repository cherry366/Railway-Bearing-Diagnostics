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
import random

# 读取df1.csv
df1_path = os.path.join(WORK, 'df1.csv')
df1 = pd.read_csv(df1_path, encoding='utf-8-sig')

import pandas as pd
import numpy as np
import os
import random
from scipy import stats
from scipy import signal

# 读取df1.csv
df1_path = os.path.join(WORK, 'df1.csv')
df1 = pd.read_csv(df1_path, encoding='utf-8-sig')

# 数据表格目录
data_folder = os.path.join(WORK, 'signals_ready')

# 创建保存划窗结果的文件夹
output_folder = os.path.join(WORK, 'windows')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"创建输出文件夹: {output_folder}")

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

def calculate_time_frequency_features(data):
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
        freqs = np.fft.fftfreq(len(data), 1/12000)[:len(data)//2]
        features['tf_mean_freq'] = np.sum(freqs * fft_abs) / np.sum(fft_abs)
    else:
        features['tf_mean_freq'] = 0
    
    # 小波能量（简化版，使用标准差代替实际小波变换）
    features['wavelet_energy'] = np.std(data)  # 简化的小波能量
    features['wavelet_energy_ratio'] = features['wavelet_energy'] / (np.sum(data**2) + 1e-10)
    
    return features

def sliding_window_sampling(data, window_size, step_size):
    """
    对数据进行滑动窗口采样
    """
    windows = []
    data_length = len(data)
    
    window_size = int(window_size)
    step_size = int(step_size)
    
    if window_size > data_length:
        print(f"    警告：窗口大小({window_size})超过数据长度({data_length})，调整为数据长度")
        window_size = data_length
    
    start = 0
    while start + window_size <= data_length:
        window = data[start:start + window_size]
        windows.append(window)
        start += step_size
    
    return windows

def calculate_window_params(row):
    """
    根据Fault type计算窗口大小和步长
    """
    fault_type = row['Fault type']
    
    if pd.isna(fault_type):
        window_size = random.uniform(0.01, 0.09) * 100000
        step_size = random.uniform(0.001, 0.009) * 100000
        print(f"    Fault type=NaN, 使用随机值: 窗口={window_size:.0f}, 步长={step_size:.0f}")
        
    elif fault_type == 'B':
        window_size = row['FTF_period'] * 100000 if pd.notna(row['FTF_period']) else 5000
        step_size = row['BSF_period'] * 100000 if pd.notna(row['BSF_period']) else 500
        print(f"    Fault type=B: 窗口={window_size:.0f}, 步长={step_size:.0f}")
        
    elif fault_type == 'IR':
        window_size = row['Rotation_period'] * 100000 if pd.notna(row['Rotation_period']) else 5000
        step_size = row['BPFI_period'] * 100000 if pd.notna(row['BPFI_period']) else 500
        print(f"    Fault type=IR: 窗口={window_size:.0f}, 步长={step_size:.0f}")
        
    elif fault_type == 'OR':
        window_size = row['Rotation_period'] * 100000 if pd.notna(row['Rotation_period']) else 5000
        step_size = row['BPFO_period'] * 100000 if pd.notna(row['BPFO_period']) else 500
        print(f"    Fault type=OR: 窗口={window_size:.0f}, 步长={step_size:.0f}")
        
    else:
        print(f"    警告：未知的Fault type '{fault_type}'，使用默认值")
        window_size = 0.05 * 100000
        step_size = 0.005 * 100000
    
    return int(window_size), int(step_size)

# 获取数据文件列表
data_files = [f for f in os.listdir(data_folder) 
              if f.endswith(('.csv', '.xlsx', '.xls'))]
data_files.sort()

print(f"找到 {len(data_files)} 个数据文件")
print(f"df1.csv有 {len(df1)} 行数据")

# 主处理流程
print("\n" + "="*60)
print("开始划窗采样处理...")

successful_count = 0
failed_count = 0
all_windows_data = []

for idx in range(min(len(df1), len(data_files))):
    print(f"\n处理第 {idx+1} 个文件:")
    
    try:
        df1_row = df1.iloc[idx]
        data_file = data_files[idx]
        data_path = os.path.join(data_folder, data_file)
        
        print(f"  文件: {data_file}")
        print(f"  Fault type: {df1_row['Fault type']}")
        
        # 读取数据文件
        if data_path.endswith('.csv'):
            data_df = pd.read_csv(data_path, encoding='utf-8-sig')
        else:
            data_df = pd.read_excel(data_path)
        
        if 'mean_time' not in data_df.columns:
            print(f"  ✗ 错误：文件中没有mean_time列")
            failed_count += 1
            continue
        
        mean_time_data = data_df['mean_time'].dropna().values
        print(f"  mean_time数据长度: {len(mean_time_data)}")
        
        window_size, step_size = calculate_window_params(df1_row)
        windows = sliding_window_sampling(mean_time_data, window_size, step_size)
        print(f"  采样得到 {len(windows)} 个窗口")
        
        if len(windows) == 0:
            print(f"  ✗ 警告：没有生成任何窗口")
            failed_count += 1
            continue
        
        # 为每个窗口创建完整的特征记录
        for win_idx, window in enumerate(windows):
            # 首先复制df1的所有原始特征
            window_info = df1_row.to_dict()
            
            # 添加窗口元信息
            window_info.update({
                'file_index': idx,
                'file_name': data_file,
                'window_index': win_idx,
                'window_size_used': window_size,
                'step_size_used': step_size,
                'window_length': len(window)
            })
            
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
        
    except Exception as e:
        print(f"  ✗ 处理出错: {str(e)}")
        failed_count += 1

# 保存完整的windows_summary
print("\n" + "="*60)
print("保存汇总结果...")

summary_df = pd.DataFrame(all_windows_data)
summary_path = os.path.join(output_folder, 'windows_summary_complete.csv')
summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')

print(f"完整窗口汇总信息已保存到: windows_summary_complete.csv")
print(f"总窗口数: {len(all_windows_data)}")
print(f"特征总数: {len(summary_df.columns)}")

print("\n" + "="*60)
print("处理完成！")
print(f"成功处理: {successful_count} 个文件")
print(f"处理失败: {failed_count} 个文件")
print(f"总共生成: {len(all_windows_data)} 个窗口")

# 显示特征列表
print("\n生成的特征列包括:")
print("1. df1原始特征:")
df1_columns = df1.columns.tolist()
print(f"   {df1_columns}")

print("\n2. 时域特征 (td_前缀):")
td_features = [col for col in summary_df.columns if col.startswith('td_')]
print(f"   {td_features}")

print("\n3. 频域特征 (fd_前缀):")
fd_features = [col for col in summary_df.columns if col.startswith('fd_')]
print(f"   {fd_features}")

print("\n4. 时频域特征 (tf_前缀):")
tf_features = [col for col in summary_df.columns if col.startswith('tf_')]
print(f"   {tf_features}")

print(f"\n输出文件: {summary_path}")