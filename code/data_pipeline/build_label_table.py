# -*- coding: utf-8 -*-
# 路径约定：仓库目录只读，运行产物统一写入 work/（可用环境变量 BD_ROOT 换盘）。
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import WORK, DATA, RAW, PROCESSED, MODEL_DIR, use_workdir  # noqa: E402

use_workdir()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 读取CSV文件
file_path = os.path.join(RAW, 'e1.csv')
df = pd.read_csv(file_path, encoding='utf-8-sig')

df['Source'] = df['Source'].apply(
    lambda p: os.path.join(DATA, 'source',
                            str(p).split('源域数据集\\', 1)[-1].replace('\\', os.sep))
    if '源域数据集\\' in str(p) else p)

print(f"原始数据形状: {df.shape}")
print(f"原始列名: {df.columns.tolist()}\n")

# 初始化新列
df['Fault type'] = None
df['size'] = None
df['Load'] = None
df['position'] = None

# 遍历每一行，从Source列提取信息
for idx, row in df.iterrows():
    source = row['Source']
    
    if pd.notna(source):  # 确保Source不为空
        # 1. 提取Fault type (B、IR、OR、N)
        fault_type = None
        # 查找路径中的故障类型
        if '\\B\\' in source or '/B/' in source:
            fault_type = 'B'
        elif '\\IR\\' in source or '/IR/' in source:
            fault_type = 'IR'
        elif '\\OR\\' in source or '/OR/' in source:
            fault_type = 'OR'
        elif '\\N\\' in source or '/N/' in source:
            fault_type = 'N'
        
        # 2. 提取size (0007->0.007, 0014->0.014, 0021->0.021, 0028->0.028)
        size = None
        # 查找四位数字的尺寸信息
        size_patterns = ['0007', '0014', '0021', '0028']
        for pattern in size_patterns:
            if pattern in source:
                # 将字符串转换为数字并除以1000
                size = int(pattern) / 1000
                break
        
        # 3. 提取Load (从文件名中的_0, _1, _2等提取)
        load = None
        # 获取文件名（不含扩展名）
        filename = os.path.basename(source)
        filename_without_ext = os.path.splitext(filename)[0]
        # 使用正则表达式查找_数字的模式
        load_match = re.search(r'_(\d+)$', filename_without_ext)
        if load_match:
            load = int(load_match.group(1))
        
        # 4. 提取position (只在OR目录下有：Centered、Opposite、Orthogonal)
        position = None
        if fault_type == 'OR':  # 只在OR类型中查找position
            if 'Centered' in source:
                position = 'Centered'
            elif 'Opposite' in source:
                position = 'Opposite'
            elif 'Orthogonal' in source:
                position = 'Orthogonal'
        
        # 将提取的信息赋值给对应的行
        df.at[idx, 'Fault type'] = fault_type
        df.at[idx, 'size'] = size
        df.at[idx, 'Load'] = load
        df.at[idx, 'position'] = position
        
        # 打印前几行的提取结果以便检查
        if idx < 5:
            print(f"第 {idx+1} 行:")
            print(f"  Source: {source}")
            print(f"  Fault type: {fault_type}")
            print(f"  size: {size}")
            print(f"  Load: {load}")
            print(f"  position: {position}\n")

# 保存更新后的数据
output_path = os.path.join(WORK, 'extracted_data_with_features.csv')
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print("=" * 50)
print(f"处理完成！")
print(f"新文件已保存到: {output_path}")
print(f"\n新增列的统计信息:")
print(f"Fault type分布:\n{df['Fault type'].value_counts()}")
print(f"\nsize分布:\n{df['size'].value_counts()}")
print(f"\nLoad分布:\n{df['Load'].value_counts()}")
print(f"\nposition分布:\n{df['position'].value_counts()}")

# 显示前5行完整数据
print("\n前5行数据（包含新列）:")
# 选择关键列显示
display_cols = ['Source', 'Fault type', 'size', 'Load', 'position', 'frequency', 'Data length']
print(df[display_cols].head())

# 假设df已经存在并且已经添加了新列
# 创建df1 - 去掉FE_time、DE_time、BA_time三列
columns_to_drop = ['FE_time', 'DE_time', 'BA_time']
df1 = df.drop(columns=columns_to_drop, errors='ignore')

# 查看结果
print(f"df1包含的列: {df1.columns.tolist()}")
print(f"df1的形状: {df1.shape}")
print("\ndf1前5行:")
print(df1.head())

# 新增：从Source列提取RPM值并替换RPM列中的NaN值
print("\n" + "="*50)
print("开始处理RPM列的NaN值...")

# 检查RPM列是否存在
if 'RPM' in df1.columns:
    print(f"RPM列存在，当前NaN值数量: {df1['RPM'].isna().sum()}")
    
    # 遍历每一行，从Source列提取RPM值
    for idx, row in df1.iterrows():
        source = row['Source']
        current_rpm = row['RPM']
        
        # 如果RPM列为NaN且Source不为空，则尝试提取RPM值
        if pd.isna(current_rpm) and pd.notna(source):
            # 使用正则表达式提取rpm前面的数字
            # 匹配模式：数字+rpm，例如1797rpm
            rpm_match = re.search(r'(\d+)rpm', source, re.IGNORECASE)
            if rpm_match:
                extracted_rpm = int(rpm_match.group(1))
                df1.at[idx, 'RPM'] = extracted_rpm
                
                # 打印前几个提取结果以便检查
                if idx < 10:
                    print(f"第 {idx+1} 行: 从 '{source}' 提取RPM值: {extracted_rpm}")
    
    print(f"\nRPM列处理完成！")
    print(f"处理后NaN值数量: {df1['RPM'].isna().sum()}")
    print(f"RPM值分布:\n{df1['RPM'].value_counts().sort_index()}")
    
    # 显示包含RPM列的前几行数据
    print("\n包含RPM列的前5行数据:")
    rpm_display_cols = ['Source', 'RPM', 'Fault type', 'size', 'Load', 'position']
    print(df1[rpm_display_cols].head())
    
else:
    print("警告：df1中没有找到RPM列！")
    print(f"可用的列: {df1.columns.tolist()}")

# 新增：用众数填充size列的缺失值
print("\n" + "="*50)
print("开始处理size列的缺失值...")

# 检查size列是否存在
if 'size' in df1.columns:
    print(f"size列存在，当前缺失值数量: {df1['size'].isna().sum()}")
    
    # 计算size列的众数（排除NaN值）
    size_mode = df1['size'].mode()
    if len(size_mode) > 0:
        mode_value = size_mode.iloc[0]  # 取第一个众数
        print(f"size列的众数: {mode_value}")
        
        # 用众数填充缺失值
        df1['size'].fillna(mode_value, inplace=True)
        
        print(f"size列填充完成！")
        print(f"填充后缺失值数量: {df1['size'].isna().sum()}")
        print(f"size值分布:\n{df1['size'].value_counts().sort_index()}")
        
        # 显示包含size列的前几行数据
        print("\n包含size列的前5行数据:")
        size_display_cols = ['Source', 'size', 'Fault type', 'Load', 'position', 'RPM']
        print(df1[size_display_cols].head())
    else:
        print("警告：size列没有有效的众数值！")
else:
    print("警告：df1中没有找到size列！")
    print(f"可用的列: {df1.columns.tolist()}")

# 新增：生成label1和label2列
print("\n" + "="*50)
print("开始生成label1和label2列...")

# 初始化新列
df1['label1'] = None
df1['label2'] = None

# 遍历每一行，生成label1和label2
for idx, row in df1.iterrows():
    source = row['Source']
    fault_type = row['Fault type']
    size = row['size']
    
    if pd.notna(source):  # 确保Source不为空
        # 判断数据来源类型
        if 'DE' in source:
            data_type = 'DE'
        elif 'FE' in source:
            data_type = 'FE'
        elif 'Normal' in source:
            data_type = 'Normal'
        else:
            data_type = None
        
        # 生成label1和label2
        if data_type in ['DE', 'FE'] and pd.notna(fault_type):
            # DE或FE类型数据：label1 = 数据类型-故障类型，label2 = 数据类型-故障类型-尺寸
            label1 = f"{data_type}-{fault_type}"
            if pd.notna(size):
                label2 = f"{data_type}-{fault_type}-{size}"
            else:
                label2 = f"{data_type}-{fault_type}"
        elif data_type == 'Normal':
            # Normal类型数据：label1和label2都是N
            label1 = 'N'
            label2 = 'N'
        else:
            # 其他情况
            label1 = None
            label2 = None
        
        # 将生成的标签赋值给对应的行
        df1.at[idx, 'label1'] = label1
        df1.at[idx, 'label2'] = label2
        
        # 打印前几行的生成结果以便检查
        if idx < 10:
            print(f"第 {idx+1} 行:")
            print(f"  Source: {source}")
            print(f"  Fault type: {fault_type}")
            print(f"  size: {size}")
            print(f"  label1: {label1}")
            print(f"  label2: {label2}")
            print()

print(f"label1和label2列生成完成！")
print(f"label1分布:\n{df1['label1'].value_counts()}")
print(f"\nlabel2分布:\n{df1['label2'].value_counts()}")

# 显示包含新标签列的前几行数据
print("\n包含label列的前5行数据:")
label_display_cols = ['Source', 'Fault type', 'size', 'label1', 'label2', 'Load', 'position', 'RPM']
print(df1[label_display_cols].head())

# 创建保存处理后数据的文件夹
processed_folder = os.path.join(WORK, 'signals_clean')
if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)
    print(f"创建文件夹: {processed_folder}")

def remove_outliers_and_fill(data):
    """
    使用箱线图方法剔除异常值，并用前后值填充
    """
    if len(data) == 0:
        return data
    
    # 计算Q1, Q3和IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    # 定义异常值的界限
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # 标记异常值位置
    outliers_mask = (data < lower_bound) | (data > upper_bound)
    
    # 复制数据以避免修改原始数据
    cleaned_data = data.copy()
    
    # 用前后值的平均值填充异常值
    for i in range(len(cleaned_data)):
        if outliers_mask.iloc[i]:
            # 找到前一个非异常值
            prev_val = None
            for j in range(i-1, -1, -1):
                if not outliers_mask.iloc[j]:
                    prev_val = cleaned_data.iloc[j]
                    break
            
            # 找到后一个非异常值
            next_val = None
            for j in range(i+1, len(cleaned_data)):
                if not outliers_mask.iloc[j]:
                    next_val = cleaned_data.iloc[j]
                    break
            
            # 用前后值的平均值填充，如果只有一个值则使用该值
            if prev_val is not None and next_val is not None:
                cleaned_data.iloc[i] = (prev_val + next_val) / 2
            elif prev_val is not None:
                cleaned_data.iloc[i] = prev_val
            elif next_val is not None:
                cleaned_data.iloc[i] = next_val
            # 如果前后都没有非异常值，保持原值
    
    return cleaned_data, outliers_mask

def denoise_series(series, window_size: int = 5):
    """
    简单去噪：中心移动平均。window_size 应为正奇数，最小为3。
    """
    if window_size < 3:
        window_size = 3
    if window_size % 2 == 0:
        window_size += 1
    return series.rolling(window=window_size, center=True, min_periods=1).mean()

def detrend_series(series):
    """
    线性去趋势：拟合一阶直线并相减。
    """
    if len(series) == 0:
        return series
    x = np.arange(len(series))
    try:
        slope, intercept = np.polyfit(x, series.values.astype(float), 1)
        trend = slope * x + intercept
        detrended = series.values.astype(float) - trend
        return pd.Series(detrended, index=series.index)
    except Exception:
        return series

def process_excel_file(source_path, save_index):
    """
    处理单个Excel文件：剔除异常值、去噪、去趋势
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(source_path):
            print(f"文件不存在: {source_path}")
            return False, source_path
        
        # 读取Excel文件的所有工作表
        excel_file = pd.ExcelFile(source_path)
        sheet_names = excel_file.sheet_names
        
        # 查找包含FE_time、DE_time、BA_time的工作表
        fe_sheet = None
        de_sheet = None
        ba_sheet = None
        
        for sheet_name in sheet_names:
            sheet_upper = sheet_name.upper()
            if ('FE_TIME' in sheet_upper or '_FE_' in sheet_upper or 
                'EF_TIME' in sheet_upper or '_EF_' in sheet_upper):
                fe_sheet = sheet_name
            elif 'DE_TIME' in sheet_upper or '_DE_' in sheet_upper:
                de_sheet = sheet_name
            elif 'BA_TIME' in sheet_upper or '_BA_' in sheet_upper:
                ba_sheet = sheet_name
        
        # 准备数据和标签
        cleaned_data_dict = {}
        
        # 读取FE_time数据
        if fe_sheet:
            df_fe = pd.read_excel(source_path, sheet_name=fe_sheet)
            if len(df_fe.columns) > 0:
                fe_data = df_fe.iloc[:, 0].dropna()
                if len(fe_data) > 0:
                    # 异常值处理
                    cleaned_fe, _ = remove_outliers_and_fill(fe_data)
                    # 去噪
                    denoised_fe = denoise_series(cleaned_fe, window_size=5)
                    # 去趋势
                    detrended_fe = detrend_series(denoised_fe)
                    
                    cleaned_data_dict[fe_sheet] = pd.DataFrame(detrended_fe)
        
        # 读取DE_time数据
        if de_sheet:
            df_de = pd.read_excel(source_path, sheet_name=de_sheet)
            if len(df_de.columns) > 0:
                de_data = df_de.iloc[:, 0].dropna()
                if len(de_data) > 0:
                    # 异常值处理
                    cleaned_de, _ = remove_outliers_and_fill(de_data)
                    # 去噪
                    denoised_de = denoise_series(cleaned_de, window_size=5)
                    # 去趋势
                    detrended_de = detrend_series(denoised_de)
                    
                    cleaned_data_dict[de_sheet] = pd.DataFrame(detrended_de)
        
        # 读取BA_time数据
        if ba_sheet:
            df_ba = pd.read_excel(source_path, sheet_name=ba_sheet)
            if len(df_ba.columns) > 0:
                ba_data = df_ba.iloc[:, 0].dropna()
                if len(ba_data) > 0:
                    # 异常值处理
                    cleaned_ba, _ = remove_outliers_and_fill(ba_data)
                    # 去噪
                    denoised_ba = denoise_series(cleaned_ba, window_size=5)
                    # 去趋势
                    detrended_ba = detrend_series(denoised_ba)
                    
                    cleaned_data_dict[ba_sheet] = pd.DataFrame(detrended_ba)
        
        # 如果有数据，保存处理后的Excel文件
        if len(cleaned_data_dict) > 0:
            # 构建新的文件名
            # 从完整路径中提取关键部分
            path_parts = source_path.split(os.sep)
            
            # 查找"源域数据集"或"目标域数据集"的位置
            dataset_keywords = ["源域数据集", "目标域数据集"]
            start_index = -1
            
            for i, part in enumerate(path_parts):
                if any(keyword in part for keyword in dataset_keywords):
                    start_index = i + 1  # 从数据集目录的下一级开始
                    break
            
            # 如果没有找到数据集关键字，从最后4个部分开始
            if start_index == -1:
                start_index = max(0, len(path_parts) - 4)
            
            # 提取相关路径部分
            relevant_parts = path_parts[start_index:]
            
            # 移除文件扩展名
            if relevant_parts and '.' in relevant_parts[-1]:
                relevant_parts[-1] = relevant_parts[-1].split('.')[0]
            
            # 构建新文件名
            new_filename = "-".join(relevant_parts) + ".xlsx"
            
            new_file_path = os.path.join(processed_folder, new_filename)
            
            # 写入处理后的数据到新的Excel文件
            with pd.ExcelWriter(new_file_path, engine='openpyxl') as writer:
                # 先写入处理后的数据
                for sheet_name, df in cleaned_data_dict.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # 复制其他工作表（如RPM等）
                for sheet_name in sheet_names:
                    if sheet_name not in cleaned_data_dict:
                        df_other = pd.read_excel(source_path, sheet_name=sheet_name)
                        df_other.to_excel(writer, sheet_name=sheet_name, index=False)
            
            print(f"  处理后数据已保存: {new_filename}")
            return True, None
            
        else:
            print(f"  未找到FE_time、DE_time或BA_time数据")
            return False, source_path
            
    except Exception as e:
        print(f"  处理文件时出错: {str(e)}")
        return False, source_path

# 主处理流程
print("开始处理所有样本文件...")
print("=" * 60)

# 存储处理失败的文件路径
failed_files = []

# 获取所有源文件路径
source_paths = df1['Source'].tolist()

for idx, source_path in enumerate(source_paths):
    if pd.notna(source_path):
        print(f"\n处理第 {idx+1}/{len(source_paths)} 个文件:")
        print(f"  路径: {source_path}")
        
        # 处理文件
        success, failed_path = process_excel_file(source_path, idx+1)
        if not success:
            failed_files.append(failed_path)
            print(f"  处理失败，已添加到失败列表")
        
        # 每处理10个文件显示一次进度
        if (idx + 1) % 10 == 0:
            print(f"\n进度: {idx+1}/{len(source_paths)} 完成")
            print(f"失败文件数: {len(failed_files)}")

print("\n" + "=" * 60)
print("处理完成！")
print(f"总文件数: {len(source_paths)}")
print(f"处理失败: {len(failed_files)}")
print(f"处理后数据保存位置: {processed_folder}")

# 保存失败文件列表
if failed_files:
    failed_df = pd.DataFrame({'Failed Files': failed_files})
    failed_csv_path = os.path.join(processed_folder, 'failed_files.csv')
    failed_df.to_csv(failed_csv_path, index=False, encoding='utf-8-sig')
    print(f"\n失败文件列表已保存到: {failed_csv_path}")
    
    print("\n前10个失败文件:")
    for i, file_path in enumerate(failed_files[:10]):
        print(f"  {i+1}. {file_path}")
else:
    print("\n所有文件处理成功！")
#####计算轴承故障特征频率
# 定义轴承参数
bearing_params = {
    'DE': {'d': 0.3126, 'D': 1.537},  # SKF6205轴承参数
    'FE': {'d': 0.2656, 'D': 1.122}   # SKF6203轴承参数
}

# 定义滚动体数量
Nd = 9

def calculate_bearing_frequencies(row):
    """
    根据文件路径和故障类型计算特征频率
    """
    source_path = row['Source']
    fault_type = row['Fault type']
    
    # 初始化结果为None
    bpfo, bpfi, bsf = None, None, None
    
    # 提取轴承类型（DE或FE）
    bearing_type = None
    if 'DE' in source_path:
        bearing_type = 'DE'
    elif 'FE' in source_path:
        bearing_type = 'FE'
    
    # 如果有轴承类型且不是正常数据，则计算特征频率
    if bearing_type and fault_type != 'None':
        # 获取轴承参数
        d = bearing_params[bearing_type]['d']
        D = bearing_params[bearing_type]['D']
        
        # 获取转速RPM（假设df1中有RPM列）
        rpm = row['RPM']  # 如果列名不同请修改
        
        # 计算转频 fr = n/60
        fr = rpm / 60
        
        # 根据故障类型计算相应的特征频率
        if fault_type == 'OR':  # 外圈故障
            bpfo = fr * (Nd / 2) * (1 - d/D)
        elif fault_type == 'IR':  # 内圈故障
            bpfi = fr * (Nd / 2) * (1 + d/D)
        elif fault_type == 'B':  # 滚动体故障
            bsf = fr * (D / d) * (1 - (d/D)**2)
    
    return bpfo, bpfi, bsf

# 应用函数计算三个特征频率
df1[['BPFO', 'BPFI', 'BSF']] = df1.apply(
    lambda row: pd.Series(calculate_bearing_frequencies(row)), 
    axis=1
)

# 显示结果验证
print("添加特征频率后的数据前几行：")
print(df1[['Source', 'Fault type', 'RPM', 'BPFO', 'BPFI', 'BSF']].head())

# 统计各特征频率的非空数量
print(f"\nBPFO非空值数量: {df1['BPFO'].notna().sum()}")
print(f"BPFI非空值数量: {df1['BPFI'].notna().sum()}")
print(f"BSF非空值数量: {df1['BSF'].notna().sum()}")

# 修正后的周期特征计算函数
def calculate_fault_periods(row):
    """
    根据故障类型计算相应的故障周期和特征
    """
    source_path = row['Source']
    fault_type = row['Fault type']
    
    # 初始化所有结果为None
    bpfo_period, bpfi_period, bsf_period, rotation_period, ftf, ftf_period = None, None, None, None, None, None
    
    # 提取轴承类型（DE或FE）
    bearing_type = None
    if 'DE' in source_path:
        bearing_type = 'DE'
    elif 'FE' in source_path:
        bearing_type = 'FE'
    
    # 如果有轴承类型且不是正常数据，则计算相应的特征
    if bearing_type and fault_type != 'N':
        # 获取轴承参数
        d = bearing_params[bearing_type]['d']
        D = bearing_params[bearing_type]['D']
        
        # 获取转速RPM
        rpm = row['RPM']
        
        # 计算转频 fr = n/60
        fr = rpm / 60
        
        # 所有故障类型都计算旋转周期（轴承旋转周期）
        rotation_period = 1 / fr if fr > 0 else None
        
        # 根据故障类型计算相应的特征
        if fault_type == 'OR':  # 外圈故障
            # 只计算外圈故障周期
            if pd.notna(row['BPFO']) and row['BPFO'] > 0:
                bpfo_period = 1 / row['BPFO']
        
        elif fault_type == 'IR':  # 内圈故障
            # 只计算内圈故障周期
            if pd.notna(row['BPFI']) and row['BPFI'] > 0:
                bpfi_period = 1 / row['BPFI']
        
        elif fault_type == 'B':  # 滚动体故障
            # 计算滚动体故障周期和FTF相关特征
            if pd.notna(row['BSF']) and row['BSF'] > 0:
                bsf_period = 1 / row['BSF']
            
            # 计算FTF（滚动体公转频率）和FTF周期
            ftf = 0.5 * fr * (1 - d/D)
            ftf_period = 1 / ftf if ftf > 0 else None
    
    return bpfo_period, bpfi_period, bsf_period, rotation_period, ftf, ftf_period

# 应用函数计算周期特征
df1[['BPFO_period', 'BPFI_period', 'BSF_period', 'Rotation_period', 'FTF', 'FTF_period']] = df1.apply(
    lambda row: pd.Series(calculate_fault_periods(row)), 
    axis=1
)

# 显示结果验证
print("修正后的周期特征数据前几行：")
print(df1[['Fault type', 'RPM', 'BPFO', 'BPFO_period', 'BPFI', 'BPFI_period', 
           'BSF', 'BSF_period', 'Rotation_period', 'FTF', 'FTF_period']].head(10))

# 按故障类型统计特征
print(f"\n各故障类型的特征统计：")

fault_types = ['OR', 'IR', 'B', 'N']
for fault_type in fault_types:
    subset = df1[df1['Fault type'] == fault_type]
    print(f"\n{fault_type}类型数据（共{len(subset)}个）：")
    
    if fault_type == 'OR':
        bpfo_count = subset['BPFO_period'].notna().sum()
        print(f"  BPFO_period非空值: {bpfo_count}个")
        if bpfo_count > 0:
            print(f"  BPFO_period范围: {subset['BPFO_period'].min():.6f} - {subset['BPFO_period'].max():.6f} 秒")
    
    elif fault_type == 'IR':
        bpfi_count = subset['BPFI_period'].notna().sum()
        print(f"  BPFI_period非空值: {bpfi_count}个")
        if bpfi_count > 0:
            print(f"  BPFI_period范围: {subset['BPFI_period'].min():.6f} - {subset['BPFI_period'].max():.6f} 秒")
    
    elif fault_type == 'B':
        bsf_count = subset['BSF_period'].notna().sum()
        ftf_count = subset['FTF'].notna().sum()
        print(f"  BSF_period非空值: {bsf_count}个")
        print(f"  FTF非空值: {ftf_count}个")
        if bsf_count > 0:
            print(f"  BSF_period范围: {subset['BSF_period'].min():.6f} - {subset['BSF_period'].max():.6f} 秒")
        if ftf_count > 0:
            print(f"  FTF范围: {subset['FTF'].min():.6f} - {subset['FTF'].max():.6f} Hz")
            print(f"  FTF_period范围: {subset['FTF_period'].min():.6f} - {subset['FTF_period'].max():.6f} 秒")
    
    # 所有故障类型（除了正常）都有旋转周期
    if fault_type != 'N':
        rotation_count = subset['Rotation_period'].notna().sum()
        print(f"  Rotation_period非空值: {rotation_count}个")
        if rotation_count > 0:
            print(f"  旋转周期范围: {subset['Rotation_period'].min():.6f} - {subset['Rotation_period'].max():.6f} 秒")

# 检查正常数据是否所有周期特征都为None
normal_data = df1[df1['Fault type'] == 'N']
print(f"\n正常数据检查：")
print(f"正常数据数量: {len(normal_data)}")
print(f"正常数据的BPFO_period全部为None: {normal_data['BPFO_period'].isna().all()}")
print(f"正常数据的BPFI_period全部为None: {normal_data['BPFI_period'].isna().all()}")
print(f"正常数据的BSF_period全部为None: {normal_data['BSF_period'].isna().all()}")
print(f"正常数据的Rotation_period全部为None: {normal_data['Rotation_period'].isna().all()}")
print(f"正常数据的FTF全部为None: {normal_data['FTF'].isna().all()}")

# 导出df1到CSV文件
output_path = os.path.join(WORK, 'df1.csv')
df1.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"数据已成功导出到: {output_path}")
print(f"数据形状: {df1.shape}")
print(f"列名: {list(df1.columns)}")