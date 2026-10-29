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
import os
import re
from pathlib import Path
import warnings
import os
import pandas as pd
import glob
warnings.filterwarnings('ignore')

# 读取CSV文件
file_path = os.path.join(WORK, 'df1.csv')
df1 = pd.read_csv(file_path, encoding='utf-8-sig')


# 创建新的输出文件夹
output_folder = os.path.join(WORK, 'signals_ready')
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"创建文件夹: {output_folder}")

# 输入文件夹路径
input_folder = os.path.join(WORK, 'signals_clean')

def process_single_excel(file_path, output_path):
    """
    处理单个Excel文件，将多个工作表整合到一个表格中
    """
    try:
        # 读取Excel文件的所有工作表
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        print(f"  工作表列表: {sheet_names}")
        
        # 初始化结果DataFrame
        result_df = pd.DataFrame()
        
        # 查找并读取FE_time、DE_time、BA_time/BE_time的工作表
        fe_data = None
        de_data = None
        ba_data = None
        
        for sheet_name in sheet_names:
            sheet_upper = sheet_name.upper()
            
            # 读取工作表数据
            df_sheet = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # 获取第一列数据（假设数据在第一列）
            if len(df_sheet.columns) > 0:
                data = df_sheet.iloc[:, 0]
                
                # 判断是哪种类型的数据
                if 'FE_TIME' in sheet_upper or '_FE_' in sheet_upper or 'EF_TIME' in sheet_upper:
                    fe_data = data
                    print(f"    找到FE_time数据: {sheet_name} (长度: {len(data)})")
                elif 'DE_TIME' in sheet_upper or '_DE_' in sheet_upper:
                    de_data = data
                    print(f"    找到DE_time数据: {sheet_name} (长度: {len(data)})")
                elif 'BA_TIME' in sheet_upper or '_BA_' in sheet_upper or 'BE_TIME' in sheet_upper:
                    ba_data = data
                    ba_type = 'BA_time' if 'BA' in sheet_upper else 'BE_time'
                    print(f"    找到{ba_type}数据: {sheet_name} (长度: {len(data)})")
        
        # 构建结果DataFrame
        # 找出最大长度
        max_length = 0
        if fe_data is not None:
            max_length = max(max_length, len(fe_data))
        if de_data is not None:
            max_length = max(max_length, len(de_data))
        if ba_data is not None:
            max_length = max(max_length, len(ba_data))
        
        # 根据存在的数据创建列
        if fe_data is not None:
            # 如果数据长度不足，用NaN填充
            fe_series = pd.Series(fe_data.values)
            if len(fe_series) < max_length:
                fe_series = fe_series.reindex(range(max_length))
            result_df['FE_time'] = fe_series
        
        if de_data is not None:
            de_series = pd.Series(de_data.values)
            if len(de_series) < max_length:
                de_series = de_series.reindex(range(max_length))
            result_df['DE_time'] = de_series
        
        if ba_data is not None:
            ba_series = pd.Series(ba_data.values)
            if len(ba_series) < max_length:
                ba_series = ba_series.reindex(range(max_length))
            # 统一使用BA_time作为列名
            result_df['BA_time'] = ba_series
        
        # 如果result_df不为空，保存到新文件
        if not result_df.empty:
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  ✓ 整合完成，保存到: {os.path.basename(output_path)}")
            print(f"    最终表格形状: {result_df.shape}")
            print(f"    包含列: {result_df.columns.tolist()}")
            return True
        else:
            print(f"  ✗ 未找到任何时间序列数据")
            return False
            
    except Exception as e:
        print(f"  ✗ 处理文件时出错: {str(e)}")
        return False

# 主处理流程
print("开始处理所有Excel文件...")
print("=" * 60)

# 获取输入文件夹中的所有Excel文件
excel_files = [f for f in os.listdir(input_folder) 
               if f.endswith(('.xlsx', '.xls'))]

print(f"找到 {len(excel_files)} 个Excel文件\n")

successful_count = 0
failed_count = 0
failed_files = []

# 处理每个Excel文件
for idx, filename in enumerate(excel_files):
    print(f"\n处理第 {idx+1}/{len(excel_files)} 个文件:")
    print(f"  文件名: {filename}")
    
    # 构建完整路径
    input_path = os.path.join(input_folder, filename)
    
    # 输出文件名保持不变，但改为CSV格式
    output_filename = filename.replace('.xlsx', '.csv').replace('.xls', '.csv')
    output_path = os.path.join(output_folder, output_filename)
    
    # 处理文件
    if process_single_excel(input_path, output_path):
        successful_count += 1
    else:
        failed_count += 1
        failed_files.append(filename)
    
    # 每处理10个文件显示一次进度
    if (idx + 1) % 10 == 0:
        print(f"\n进度汇总: {idx+1}/{len(excel_files)} 完成")
        print(f"成功: {successful_count}, 失败: {failed_count}")

print("\n" + "=" * 60)
print("处理完成！")
print(f"总文件数: {len(excel_files)}")
print(f"成功处理: {successful_count}")
print(f"处理失败: {failed_count}")

if failed_files:
    print(f"\n失败文件列表:")
    for f in failed_files[:10]:  # 只显示前10个
        print(f"  - {f}")
    if len(failed_files) > 10:
        print(f"  ... 还有 {len(failed_files)-10} 个文件")

print(f"\n输出文件夹: {output_folder}")

# 显示处理结果示例
if successful_count > 0:
    print("\n处理成功的文件示例:")
    output_files = os.listdir(output_folder)[:5]
    for output_file in output_files:
        file_path = os.path.join(output_folder, output_file)
        df_sample = pd.read_csv(file_path)
        print(f"\n  文件: {output_file}")
        print(f"  形状: {df_sample.shape}")
        print(f"  列: {df_sample.columns.tolist()}")
        print(f"  前3行数据:")
        print(df_sample.head(3))
####删除所有BA_time列
# 设置工作目录
work_folder = os.path.join(WORK, 'signals_ready')

def remove_ba_time_column(file_path):
    """
    从单个文件中删除BA_time列
    """
    try:
        # 判断文件类型并读取
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            print(f"  不支持的文件格式: {file_path}")
            return False
        
        print(f"  原始列: {df.columns.tolist()}")
        print(f"  原始形状: {df.shape}")
        
        # 检查是否存在BA_time列
        if 'BA_time' in df.columns:
            # 删除BA_time列
            df = df.drop(columns=['BA_time'])
            print(f"  ✓ 已删除BA_time列")
            
            # 保存更新后的文件（覆盖原文件）
            if file_path.endswith('.csv'):
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            else:
                with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='整合数据', index=False)
            
            print(f"  新列: {df.columns.tolist()}")
            print(f"  新形状: {df.shape}")
            return True
        else:
            print(f"  ⚠ 文件中不存在BA_time列，跳过")
            return False
            
    except Exception as e:
        print(f"  ✗ 处理文件时出错: {str(e)}")
        return False

# 主处理流程
print("开始删除BA_time列...")
print("=" * 60)

# 获取目录下的所有数据文件
data_files = []
for file in os.listdir(work_folder):
    if file.endswith(('.csv', '.xlsx', '.xls')):
        data_files.append(file)

print(f"找到 {len(data_files)} 个数据文件\n")

# 统计变量
files_with_ba_time = 0
files_without_ba_time = 0
failed_files = []

# 处理每个文件
for idx, filename in enumerate(data_files):
    print(f"\n处理第 {idx+1}/{len(data_files)} 个文件:")
    print(f"  文件名: {filename}")
    
    # 构建完整路径
    file_path = os.path.join(work_folder, filename)
    
    # 处理文件
    result = remove_ba_time_column(file_path)
    if result is True:
        files_with_ba_time += 1
    elif result is False:
        files_without_ba_time += 1
    else:
        failed_files.append(filename)
    
    # 每处理10个文件显示一次进度
    if (idx + 1) % 10 == 0:
        print(f"\n进度汇总: {idx+1}/{len(data_files)} 完成")
        print(f"已删除BA_time: {files_with_ba_time}, 无BA_time: {files_without_ba_time}, 失败: {len(failed_files)}")

print("\n" + "=" * 60)
print("处理完成！")
print(f"总文件数: {len(data_files)}")
print(f"成功删除BA_time列的文件: {files_with_ba_time}")
print(f"不包含BA_time列的文件: {files_without_ba_time}")
print(f"处理失败的文件: {len(failed_files)}")

if failed_files:
    print(f"\n失败文件列表:")
    for f in failed_files[:10]:
        print(f"  - {f}")
    if len(failed_files) > 10:
        print(f"  ... 还有 {len(failed_files)-10} 个文件")

# 验证结果 - 检查几个文件确认BA_time列已被删除
print("\n" + "=" * 60)
print("验证处理结果...")

# 随机选择3个文件进行验证
sample_count = min(3, len(data_files))
sample_files = data_files[:sample_count]

for filename in sample_files:
    print(f"\n验证文件: {filename}")
    file_path = os.path.join(work_folder, filename)
    
    try:
        # 读取文件
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        else:
            df = pd.read_excel(file_path)
        
        print(f"  当前列: {df.columns.tolist()}")
        
        if 'BA_time' in df.columns:
            print(f"  ✗ 警告：BA_time列仍然存在！")
        else:
            print(f"  ✓ 确认：BA_time列已成功删除")
            
    except Exception as e:
        print(f"  验证时出错: {str(e)}")

print("\n所有操作完成！")


####对文件夹里面的所有数据表格里面的列求平均得到mean_time
# 设置工作目录
work_folder = os.path.join(WORK, 'signals_ready')

def add_mean_time_column(file_path):
    """
    对单个文件添加mean_time列
    """
    try:
        # 判断文件类型并读取
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            print(f"  不支持的文件格式: {file_path}")
            return False
        
        print(f"  原始表格形状: {df.shape}")
        print(f"  原始列: {df.columns.tolist()}")
        
        # 计算每行的平均值（忽略NaN）
        # 只对时间列进行平均（FE_time, DE_time, BA_time）
        time_columns = [col for col in df.columns if 'time' in col.lower()]
        
        if len(time_columns) == 0:
            print(f"  ✗ 未找到时间列")
            return False
        
        print(f"  找到时间列: {time_columns}")
        
        # 计算mean_time - 使用nanmean忽略NaN值
        df['mean_time'] = df[time_columns].mean(axis=1, skipna=True)
        
        # 统计mean_time的信息
        valid_count = df['mean_time'].notna().sum()
        print(f"  mean_time计算完成:")
        print(f"    有效值数量: {valid_count}")
        print(f"    平均值: {df['mean_time'].mean():.4f}")
        print(f"    标准差: {df['mean_time'].std():.4f}")
        print(f"    最小值: {df['mean_time'].min():.4f}")
        print(f"    最大值: {df['mean_time'].max():.4f}")
        
        # 保存更新后的文件（覆盖原文件）
        if file_path.endswith('.csv'):
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
        else:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='整合数据', index=False)
        
        print(f"  ✓ 文件已更新，新形状: {df.shape}")
        return True
        
    except Exception as e:
        print(f"  ✗ 处理文件时出错: {str(e)}")
        return False

# 主处理流程
print("开始添加mean_time列...")
print("=" * 60)

# 获取目录下的所有数据文件
data_files = []
for file in os.listdir(work_folder):
    if file.endswith(('.csv', '.xlsx', '.xls')):
        data_files.append(file)

print(f"找到 {len(data_files)} 个数据文件\n")

successful_count = 0
failed_count = 0
failed_files = []

# 处理每个文件
for idx, filename in enumerate(data_files):
    print(f"\n处理第 {idx+1}/{len(data_files)} 个文件:")
    print(f"  文件名: {filename}")
    
    # 构建完整路径
    file_path = os.path.join(work_folder, filename)
    
    # 处理文件
    if add_mean_time_column(file_path):
        successful_count += 1
    else:
        failed_count += 1
        failed_files.append(filename)
    
    # 每处理10个文件显示一次进度
    if (idx + 1) % 10 == 0:
        print(f"\n进度汇总: {idx+1}/{len(data_files)} 完成")
        print(f"成功: {successful_count}, 失败: {failed_count}")

print("\n" + "=" * 60)
print("处理完成！")
print(f"总文件数: {len(data_files)}")
print(f"成功处理: {successful_count}")
print(f"处理失败: {failed_count}")

if failed_files:
    print(f"\n失败文件列表:")
    for f in failed_files[:10]:
        print(f"  - {f}")
    if len(failed_files) > 10:
        print(f"  ... 还有 {len(failed_files)-10} 个文件")

# 验证结果 - 随机检查几个文件
print("\n" + "=" * 60)
print("验证处理结果...")

# 随机选择3个成功的文件进行验证
if successful_count > 0:
    sample_files = [f for f in data_files if f not in failed_files][:3]
    
    for filename in sample_files:
        print(f"\n样本文件: {filename}")
        file_path = os.path.join(work_folder, filename)
        
        # 读取文件
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8-sig')
        else:
            df = pd.read_excel(file_path)
        
        print(f"  表格形状: {df.shape}")
        print(f"  列名: {df.columns.tolist()}")
        
        # 显示前5行数据
        print(f"  前5行数据:")
        print(df.head())
        
        # 验证mean_time计算
        if 'mean_time' in df.columns:
            print(f"\n  mean_time列验证:")
            # 手动计算第一行的平均值进行验证
            time_columns = [col for col in df.columns if 'time' in col.lower() and col != 'mean_time']
            if len(time_columns) > 0 and len(df) > 0:
                first_row_values = df.iloc[0][time_columns]
                manual_mean = first_row_values.mean(skipna=True)
                calculated_mean = df.iloc[0]['mean_time']
                print(f"    第一行手动计算平均值: {manual_mean:.4f}")
                print(f"    第一行mean_time值: {calculated_mean:.4f}")
                if np.isclose(manual_mean, calculated_mean, rtol=1e-5):
                    print(f"    ✓ 验证通过")
                else:
                    print(f"    ✗ 验证失败，值不匹配")