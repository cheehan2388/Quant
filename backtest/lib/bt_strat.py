import numpy as np
import pandas as pd

def positions_trend(signal, thr):
    raw = np.where(signal > thr, 1,
           np.where(signal < -thr, -1, 0))
    pos = pd.Series(raw, index=signal.index)
    return pos.replace(0, np.nan).ffill().fillna(0)
#離開你的thresh的時候關掉
def positions_trend_close(signal, thr):
    raw = np.where(signal > thr, 1,
           np.where(signal < -thr, -1, 0))
    return pd.Series(raw, index=signal.index)

def positions_trend_zero(signal, thr):

    
    # Step-1: 產生「開倉」訊號（raw）；0 表示暫時不動
    raw = np.where(signal >  thr, 1,
          np.where(signal < -thr, -1,  0))

    pos = np.zeros_like(raw)                 # 最終倉位陣列

    idx = np.flatnonzero(raw)                # raw ≠ 0 的索引
    if idx.size == 0:
        return pd.Series(pos,index= signal.index)                          # 整條都沒開倉

    # Step-2: 把連續索引分段（>1 表示斷裂）
    splits  = np.where(abs(np.diff(idx)) > 1)[0] + 1
    blocks  = np.split(idx, splits)          # 每個 block 就是一筆 trade [5,6] [9,10,11]

    # Step-3: 逐筆 block 決定平倉點
    for block in blocks:
        start      = block[0] # [2,3][0] 是 2 
        direction  = raw[start]              # +1(多) / -1(空) [1,2] 如果idx 是1那他就是多開始

        # 找第一個「穿回 0」的時間；若沒找到就持倉到最後
        if direction ==  1:
            tail = np.where(signal[start+1:] <= 0)[0]
        else:
            tail = np.where(signal[start+1:] >= 0)[0]

        close_idx  = tail[0] + start + 1 if tail.size else len(signal)
        pos[start:close_idx] = direction

    return pd.Series(pos,index = signal.index)

#反轉時
def positions_mr(signal, thr):
    raw = np.where(signal > thr, -1,
           np.where(signal < -thr, 1, 0))
    pos = pd.Series(raw, index=signal.index)
    return pos.replace(0, np.nan).ffill().fillna(0)

#離開thresh
def positions_mr_close(signal, thr):
    raw = np.where(signal > thr, -1,
           np.where(signal < -thr, 1, 0))
    return pd.Series(raw, index=signal.index)

def positions_mr_mean_zero(signal: np.ndarray, thr: float) -> np.ndarray:

    # Step-1: 產生「開倉」訊號（raw）；0 表示暫時不動
    raw = np.where(signal >  thr, -1,
          np.where(signal < -thr,  1,  0))

    pos = np.zeros_like(raw)                 # 最終倉位陣列

    idx = np.flatnonzero(raw)                # raw ≠ 0 的索引
    if idx.size == 0:
        return pd.Series(pos, index=signal.index)                           # 整條都沒開倉

    # Step-2: 把連續索引分段（>1 表示斷裂）
    splits  = np.where(np.diff(idx) > 1)[0] + 1
    blocks  = np.split(idx, splits)          # 每個 block 就是一筆 trade [5,6] [9,10,11]

    # Step-3: 逐筆 block 決定平倉點
    for block in blocks:
        start      = block[0] # [2,3][0] 是 2 
        direction  = raw[start]              # +1(多) / -1(空) [1,2] 如果idx 是1那他就是多開始

        # 找第一個「穿回 0」的時間；若沒找到就持倉到最後
        if direction ==  1:
            tail = np.where(signal[start+1:] >= 0)[0]
        else:
            tail = np.where(signal[start+1:] <= 0)[0]

        close_idx  = tail[0] + start + 1 if tail.size else len(signal)
        pos[start:close_idx] = direction

    return pd.Series(pos,index = signal.index)