import pandas as pd
import json
from datetime import datetime , timedelta ,timezone
import numpy as np 
import logging
import requests
import csv
df = pd.DataFrame()
data = []


# API_Library = 'v1/klines'
# ticker = 'BTCUSDT'
# time_interval = '1h'

# start_time = datetime(2020,1,1)
# end_time = datetime(2025,1,10)
# while start_time < end_time:
#     #print(start_time)    
#     start_time2 = int(start_time.timestamp()*1000) #'https://fapi.binance.com/fapi/'
#     url = 'https://fapi.binance.com/fapi/' + str(API_Library) + '?symbol=' + str(ticker) + '&interval=' + str(time_interval) +'&limit=1500&startTime=' + str(start_time2)
#     req = requests.get(url)
#     req = json.loads(req.content.decode())
#     data.append(req)
#     start_time = start_time + timedelta(minutes=1500)

# df =pd.DataFrame(data)
# print(df)
# c_rows =[]
# for _, row in df.iterrows():
#         c_row =[]
#         for cell in row:
                
#             c_row.extend(cell if cell is not None else [np.nan,np.nan,np.nan] )
#         c_rows.append(c_row)

# s_rows =[row[i:i + 12] for row in c_rows for i in  range(0,len(row),12)]

# new_df = pd.DataFrame(s_rows)

# new_df[0] = pd.to_datetime(new_df[0],unit="ms")
# new_df[6] = pd.to_datetime(new_df[6],unit="ms")

# new_df.columns =['Time','Open','High','Low','Close','Ignore','Close Time','Ignore','Ignore','Ignore','Ignore','Ignore']
# new_df = new_df.set_index('Time')

# print('DataFrame with concatenated arrays:')
# print(new_df)
# new_df.to_csv('btcoin24h.csv')
import pandas as pd
import requests
from datetime import datetime, timedelta

API_ENDPOINT = 'https://fapi.binance.com/fapi/v1/premiumIndexKlines' #'https://fapi.binance.com/fapi/v1/klines'

symbol       = 'BTCUSDT'
interval     = '5m'
limit        = 1500

start_time = datetime(2020, 1, 2,tzinfo= timezone.utc)
end_time   = datetime(2025, 7, 7,tzinfo =timezone.utc)

all_klines = []

while start_time < end_time:
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit,
        'startTime': int(start_time.timestamp() * 1000)
    }
    resp = requests.get(API_ENDPOINT, params=params)
    data = resp.json()
    # 如果返回的是错误字典，就打印并跳出
    if isinstance(data, dict) and data.get('code') is not None:
        logging.error(f"Binance error at {start_time}: {data}")
        break
    # 正常情况下，data 是一个 list of lists
    all_klines.extend(data)
    # 每次向后移动 limit 根 K 线对应的时长：limit * 1h
    start_time += timedelta(min=limit)

# 把所有 kline 串成 DataFrame，一次定义好列名
cols = [
    'Open time', 'Open', 'High', 'Low', 'Close', 'Ignor',
    'Close time', 'Ignore', 'Ignore',
    'Ignore', 'Ignore', 'Ignore'
]
# cols = [
#     'Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
#     'Close time', 'Quote asset volume', 'Number of trades',
#     'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
# ]
df = pd.DataFrame(all_klines, columns=cols)

# 转换时间戳列
df['Open time']  = pd.to_datetime(df['Open time'], unit='ms')
df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')

# 设索引、保存 CSV
df = df.set_index('Open time')
df.to_csv('./../Data/Binance_1Hour_SOLUSD_T.csv')

print(df.head())
