import contextlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from modules.plotter2 import plot_bid_ask_curve
from orderbooklib  import process_order_book_data
from modules.data_processor import DataProcessor, DataCleaner
import pandas as pd
import numpy as np

def update_order_book(ob,group_data):
    for _,_,_,_,_,side, price, size in group_data.values :
        book_side = ob.asks if side == 'ask'else ob.bids
        opp_side  = ob.bids if side == 'ask'else ob.asks

        if size > 0:
            book_side[price] = size
        else:
            with contextlib.suppress(KeyError):
                del book_side[price]
                del opp_side[price]
        
def process_order_book_data(df,start=None, end= None, groupby_col = 'timestamp') :
    ob = OrderBook()

    if start :
        df = df[start:end]

    timestamp_order_book_pairs = []

    for timestamp, group_data in tqdm (df.groupby(groupby_col)):
        update_order_book(ob, group_data)
        timestamp_order_book_pairs.append((timestamp, ob.to_dict()))
    
    return pd.DataFrame(timestamp_order_book_pairs,columns=[groupby_col,'order_book'])

df_raw = pd.read_csv("")
trades_df = pd.read_csv("")

df = process_order_book_data(df_raw[:150000])
processor = DataProcessor(df,trades_df)
merged_data = processor.process_data()
mmdf2 = cleaned_data.copy()

def calculate_orderbook_imbalance(order_book):
    if not order_book['ask'] or not order_book['bid']:
        return None
    #Cal Mid Price
    bid_prices = list(order_book['bid'].keys())
    ask_prices = list(order_book['ask'].keys())
    max_bid_price = max(bid_prices) if bid_prices else 0
    min_ask_price = min(ask_prices ) if ask_prices else float('inf')
    mid_price = (max_bid_price + min_ask_price) / 2

#Cal total bid & ask quantities within the price range
    total_bid = 0
    total_ask = 0

    for price, qty in order_book ['bid'].items():
        if mid_price * 0.99<= price <= mid_price * 1.01:
            total_bid += qty
    for price, qty in order_book['ask'].items():
        if mid_price * 0.99 <= price <= mid_price *1.01:
             total_ask += qty
#Cal imbalance
    imbalance = (total_bid - total_ask)/(total_bid + total_ask)
    
                                     



