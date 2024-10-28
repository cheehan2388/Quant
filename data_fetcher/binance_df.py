import pandas as pd
import json
from datetime import datetime , timedelta
import numpy as np 
import logging
import requests
import csv
df = pd.DataFrame()
data = []


API_Library = 'v1/klines'
ticker = 'BTCUSDT'
time_interval = '1d'

start_time = datetime(2023,6,1)
end_time = datetime(2024,1,1)
while start_time < end_time:
    #print(start_time)    
    start_time2 = int(start_time.timestamp()*1000) #'https://fapi.binance.com/fapi/'
    url = 'https://fapi.binance.com/fapi/' + str(API_Library) + '?symbol=' + str(ticker) + '&interval=' + str(time_interval) +'&limit=1500&startTime=' + str(start_time2)
    req = requests.get(url)
    req = json.loads(req.content.decode())
    data.append(req)
    start_time = start_time + timedelta(minutes=1500)

df =pd.DataFrame(data)
print(df)
c_rows =[]
for _, row in df.iterrows():
        c_row =[]
        for cell in row:
                
            c_row.extend(cell if cell is not None else [np.nan,np.nan,np.nan] )
        c_rows.append(c_row)

s_rows =[row[i:i + 12] for row in c_rows for i in  range(0,len(row),12)]

new_df = pd.DataFrame(s_rows)

new_df[0] = pd.to_datetime(new_df[0],unit="ms")
new_df[6] = pd.to_datetime(new_df[6],unit="ms")

new_df.columns =['Time','Open','High','Low','Close','Ignore','Close Time','Ignore','Ignore','Ignore','Ignore','Ignore']
new_df = new_df.set_index('Time')

print('DataFrame with concatenated arrays:')
print(new_df)
new_df.to_csv('btcoin24h.csv')
        