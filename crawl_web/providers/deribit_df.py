import requests
import pandas as pd
import json
from datetime import timezone , datetime  

instrument = 'BTC-30AUG24-30000-P' #currency-ddmyy-Strike-((call or put))

start_time = datetime(2024, 6, 1).replace(tzinfo=timezone.utc)
end_time   = datetime(2024, 7, 1 ).replace(tzinfo=timezone.utc)
start_stamp = int(start_time.timestamp()*1000)
end_stamp   = int(end_time.timestamp()*1000)
params     = {

    "instrument_name" : instrument,
    "start_timestamp" : start_stamp,
    "end_timestamp"   : end_stamp,
    "resolution"      : '60' # 单位分钟

    }

res = requests.get('https://www.deribit.com/api/v2/public/get_tradingview_chart_data',  params = params)#/public/get_tradingview_chart_data'改这边就好


dt = json.loads(res.content.decode()) #让json响应转换为python structure
result = dt['result']

dataframe = pd.DataFrame(result)
dataframe['Date'] = pd.to_datetime(dataframe['ticks'],unit = 'ms') 
dataframe.to_csv('loption.try.csv')


