import pandas as pd

# 读取两个 CSV 文件
df = pd.read_csv('../../CG_BTC.csv')
df1 = pd.read_csv('../../CG_BTC_H_USDT_openInterest.csv.csv')


# 确保 'datetime' 列是日期时间格式
df['datetime'] = pd.to_datetime(df['datetime'],format='%d/%m/%Y %H:%M')
df1['datetime'] = pd.to_datetime(df1['datetime'])

# 找到公共的列
common_columns = df.columns.intersection(df1.columns).tolist()

# 移除 'datetime' 列
common_columns.remove('datetime')

# 删除 df1 中的公共列
df1.drop(columns=common_columns, inplace=True)

# 合并两个 DataFrame
merged_df = pd.merge(df, df1, how='outer', on='datetime')

# 填充 NaN 值为 0
merged_df.fillna(0, inplace=True)

# 保存合并后的 DataFrame 到新的 CSV 文件
merged_df.to_csv('premium.CryptoQuant_BTC_Hour_Binance_TakerBuySellStats.csv', index=False)

print('Merged DataFrame:')
print(merged_df)