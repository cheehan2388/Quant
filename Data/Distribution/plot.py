import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#df = pd.read_csv('../../../Binance_Merge_OI_Taker.csv')
df = pd.read_csv('../../../Binance_Merge_AGGdepth_OI_Taker.csv')
#df = pd.read_csv('../../../CG_BTC_H_All_openInterest.csv')
df
#process

df['ask-bid-usd'] = df['bidsUsd'] -df['askssUsd']

df['mean']          = df['ask-bid-usd'].mean()
df['std']    = df['ask-bid-usd'].std()
df['z']      = ( df['ask-bid-usd'] - df['mean'])/df['std']
#df = df.head(200)


#plot 資料
plt.figure("z")
plt.plot(df.index, df['z'], marker='o', label='ask-bid-usd')  # 使用折線圖
plt.xlabel("Index")  # X 軸標籤
plt.ylabel("Log Open Interest")  # Y 軸標籤
plt.title("zPlot")  # 圖形標題
plt.legend()  # 顯示圖例
plt.grid(True)  # 添加網格
plt.show()