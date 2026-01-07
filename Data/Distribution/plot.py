import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 讀取資料
df = pd.read_csv('../factor/BTC_merged_trades_open.csv')

# 假設你有一欄是價格欄位：例如 'price' 或 'close'
column_name = 'gen_8_elite_35'
price_col = 'gen_8_elite_35'  # 請改成你的價格欄名稱
if price_col not in df.columns:
    raise ValueError(f"請確認你的 DataFrame 有價格欄位：'{price_col}'")
model = 'min_max_scaling'
if model == "zscore":
    df["mean"] = df[column_name].rolling(window=260).mean()
    df["std"] = df[column_name].rolling(window=260).std()
    df["model_data"] = (df[column_name] - df["mean"]) / df["std"]
# min-max-scaling (-1 to 1)
elif model == "min_max_scaling":
    # write the formula
    # x_scaled = 2 * (x - x_min) / (x_max - x_min) - 1
    df["rolling_min"] = df[column_name].rolling(260).min()
    df["rolling_max"] = df[column_name].rolling(260).max()
    df["model_data"] = (
        2
        * (df[column_name] - df["rolling_min"])
        / (df["rolling_max"] - df["rolling_min"])
        - 1.0
    )
# df['Change'] = df[price_col].pct_change()
# # 計算 log return（或 simple return）
# df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))
df = df.dropna()  # 去掉 NA

# 畫出 log return 的分布圖（Histogram + KDE）
plt.figure(figsize=(10, 6))
sns.histplot(df['model_data'], bins=50, kde=True, stat="density", label="Log Return KDE")
# 常態分布線
# mu, std = df['mean'] , df['std'] 
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = stats.norm.pdf(x, mu, std)
# plt.plot(x, p, 'r', linewidth=2, label='Normal PDF')

# plt.title('Distribution of Log Returns')
# plt.xlabel('Log Return')
# plt.ylabel('Density')
# plt.legend()
# plt.grid(True)
# plt.show()
plt.figure("z")
plt.plot(df.index, df[column_name], marker='o', label='ask-bid-usd')  # 使用折線圖
plt.xlabel("Index")  # X 軸標籤
plt.ylabel("Log Open Interest")  # Y 軸標籤
plt.title("zPlot")  # 圖形標題
plt.legend()  # 顯示圖例
plt.grid(True)  # 添加網格
plt.show()
# 正態性檢定（Shapiro-Wilk）
shapiro_test = stats.shapiro(df[column_name])
print("Shapiro-Wilk Test 結果：")
print(f"  W = {shapiro_test.statistic:.4f}")
print(f"  p-value = {shapiro_test.pvalue:.4f}")
if shapiro_test.pvalue < 0.05:
    print("  ❌ 拒絕 H0：log return 顯著不服從常態分布")
else:
    print("  ✅ 無法拒絕 H0：log return 可能是常態分布")
