import contextlib
from tqdm import tqdm
from order_book import OrderBook
import pandas as pd
import numpy as np

# def update_order_book(ob, group_data):
#     for _, _, _, _, _, _, price, size in group_data.values:
#         if size > 0:
#             ob[price] = size
#         else:
#             with contextlib.suppress(KeyError):
#                 del ob[price]
    
# def process_order_bbook_data(df, start = None,end =None):
#     ob = OrderBook()

#     if start:
#         df =df[start:end]

#     bid_df = df[df['side'] == 'bid']
#     ask_df = df[df['side'] == 'ask']

#     timestamp_order_book_pairs =[]

#     for timestamp, group_data in tqdm(bid_df.groupby('timestamp')):
#         update_order_book(ob.bids, group_data)
#         timestamp_order_book_pairs.append((timestamp, ob.to_dict()))

#     for timestamp, group_data in tqdm(bid_df.groupby('timestamp')):
#         update_order_book(ob.asks, group_data)
#         timestamp_order_book_pairs.append((timestamp, ob.to_dict()))
    
#     return pd.DataFrame(timestamp_order_book_pairs, columns =['timestamp','order_book'])

def update_order_book(ob,group_data): 

    for _, _, _, _, _, side, price, size in group_data.values:
        book_side = ob.asks if side == 'ask' else ob.bids
        opp_side = ob.bids  if side == 'ask'else ob.asks

        if size > 0:
            book_side[price] = size
             
        else:
            with contextlib.suppress(KeyError): 
                del book_side
                [price]
                del opp_side [price]

def process_order_book_data(df, start = None, end = None,groupby_col = 'timestamp'):
    ob =OrderBook()

    if start:
        df =df[start:end]

    timestamp_order_book_pairs = []

    for timestamp, group_data in tqdm(df.groupby(groupby_col)):
        update_order_book(ob,group_data)
        timestamp_order_book_pairs.append((timestamp, ob.to_dict()))

    return pd.DataFrame(timestamp_order_book_pairs,columns = [groupby_col,"order_book"])



