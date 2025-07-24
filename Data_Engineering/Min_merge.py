import numpy as np
import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('Cryptoquant_BTCUSDT_Min_funding_All_Ex_20316.516.csv')

# 每 30 行一个组 (eg.30min = 30)
df['Group'] = (df.index // 30) + 1

#把数据有的column 用该数据的算法更改max,min,first,last， column的标题一定要一致
# 获取每组的 high 列最大值
high_max = df.groupby('Group')['high'].max().reset_index()

# 获取每组的 low 列最小值
low_min = df.groupby('Group')['low'].min().reset_index()

# 获取每组的 open 列第一个值
open_first = df.groupby('Group')['open'].first().reset_index()

# 获取每 close 列最后一个值
close_last = df.groupby('Group')['Close'].last().reset_index()

#获取 date 列最后一个值
Date_last  = df.groupby('Group')['Date'].first().reset_index()

funding_rates_last = df.groupby('Group')['funding_rates'].first().reset_index()
# 合并结果  ， 如果有 其他列，以此类推
result = pd.merge(Date_last, open_first, on='Group')
result = pd.merge(result, high_max, on='Group')
result = pd.merge(result, low_min, on='Group')
result = pd.merge(result, close_last, on='Group')
result = pd.merge(result, funding_rates_last, on='Group')

# 重命名列
result = result.rename(columns={
    'high': 'high_max', 
    'low': 'low_min', 
    'open': 'open_first', 
    'Close': 'close_last' ,
    'Date' : 'Date_last',
    'funding_rates' : 'funding_rate_last'
})

# 移除 Group 列
result = result.drop(columns=['Group'])
print(result)

#输出
result.to_csv('fundingrate2.csv')