import pandas as pd
import glob
import os
from functools import reduce

def load_and_merge_on_datetime(folder_path):
    # 抓全部 CSV
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

    df_list = []
    for f in csv_files:
        df = pd.read_csv(f)

        # 保留 datetime + 其他你要的欄（先刪不需要的）
        df = df.drop(columns=['start_time', 'end_time'], errors='ignore')

        # 確保 datetime 可用來 merge
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])

        # 可選：改欄位名稱加檔名前綴避免衝突（若各檔有相同欄名）
        file_prefix = os.path.splitext(os.path.basename(f))[0]
        df = df.rename(columns={col: f"{file_prefix}_{col}" for col in df.columns if col != 'datetime'})

        df_list.append(df)

    # 根據 datetime 逐一合併
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='datetime', how='inner'), df_list)

    # 最後依 datetime 排序
    df_merged = df_merged.sort_values('datetime').reset_index(drop=True)

    return df_merged

# 使用範例
folder = './All_data'
df_result = load_and_merge_on_datetime(folder)

# 儲存成合併檔案
df_result.to_csv('merged_by_datetime.csv', index=False)
print("已儲存 merged_by_datetime.csv")
# 使用方式)
