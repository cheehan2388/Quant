import numpy as np
import pandas as pd
import talib as tb

def positions_trend(signal, thr):
    raw = np.where(signal > thr, 1,
           np.where(signal < -thr, -1, 0))
    pos = pd.Series(raw, index=signal.index)
    return pos.replace(0, np.nan).ffill().fillna(0) #如果開始沒有 signal 就放 零， 如前面是做空下一個會繼續forward fill -1 
                                                    #    一直到反轉， 做法是先拿 raw 的然後，forward fill 有0 的地方用有前面的signal
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
        return pos                           # 整條都沒開倉

    # Step-2: 把連續索引分段（>1 表示斷裂）
    splits  = np.where(abs(np.diff(idx)) > 1)[0] + 1 #如[1,2,3,4,5,9,10] 這樣表是9就是就是新的倉。+1 因爲diff  是今天如果是5用diff 那他會與6 做減法，，但其實是下一個時間點的open 前才會開倉所以加一。 
    blocks  = np.split(idx, splits)          # 每個 block 就是一筆 trade [5,6] [9,10,11]

    # Step-3: 逐筆 block 決定平倉點
    for block in blocks:
        start      = block[0] # [2,3][0] 是 2 
        direction  = raw[start]              # +1(多) / -1(空) [1,2] 如果idx 是1那他就是多開始

        # 找第一個「穿回 0」的時間；若沒找到就持倉到最後
        if direction ==  1:
            tail = np.where(signal[start+1:] <= 0)[0]#看那個block 第一個是開多后看接下來的值有沒有到關倉條件，後面的[0]就是選擇第一個變成零的inde
        else:                                        #tail就是offse 因爲原序列[0,1,2]如果start = 0 +1 就是從原序的1起始變成次序的[0]
            tail = np.where(signal[start+1:] >= 0)[0]

        close_idx  = tail[0] + start + 1 if tail.size else len(signal)
        pos[start:close_idx] = direction     # 填入倉位 關倉前面都是填入你第一的direction, 館藏后維持原本np.zer 的零，到下次的open

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
        return pos                           # 整條都沒開倉

    # Step-2: 把連續索引分段（>1 表示斷裂）
    splits  = np.where(np.diff(idx) > 1)[0] + 1 #如[1,2,3,4,5,9,10] 這樣表9 就是新的倉。+1 因爲diff  是[1,2,3,6] 3的diff他會標識3是發生改變的 但其實是下一個時間點的open 前才會開倉所以加一。 
    blocks  = np.split(idx, splits)          # 每個 block 就是一筆 trade [5,6] [9,10,11]

    # Step-3: 逐筆 block 決定平倉點
    for block in blocks:
        start      = block[0] # [2,3][0] 是 2 
        direction  = raw[start]              # +1(多) / -1(空) [1,2] 如果idx 是1那他就是多開始

        # 找第一個「穿回 0」的時間；若沒找到就持倉到最後
        if direction ==  1:
            tail = np.where(signal[start+1:] >= 0)[0]#看那個block 第一個是開多后看接下來的值有沒有到關倉條件，後面的[0]就是選擇第一個變成零的inde
        else:                                        #tail就是offse 因爲原序列[0,1,2]如果start = 0 +1 就是從原序的1起始變成次序的[0]
            tail = np.where(signal[start+1:] <= 0)[0]

        close_idx  = tail[0] + start + 1 if tail.size else len(signal)
        pos[start:close_idx] = direction     # 填入倉位 關倉前面都是填入你第一的direction, 館藏后維持原本np.zer 的零，到下次的open

    return pd.Series(pos,index = signal.index)