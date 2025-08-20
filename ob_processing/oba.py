import plotly.express as px
import pandas as pd


def visual_data(df, depth):
    mid_prices = []
    data = []
   

    for idx, row in df.iterrows():
        
        timestamp = idx  

        order_book = row['order_book']
        
        
        bid = order_book.get('bid', {})
        ask = order_book.get('ask', {})

        if not isinstance(bid, dict) or not isinstance(ask, dict):
           
            continue

        
        bid_depth = dict(sorted(bid.items(), key=lambda item: item[0], reverse=True)[:depth])
        ask_depth = dict(sorted(ask.items(), key=lambda item: item[0])[:depth])

       
        best_bid = max(bid_depth.keys()) if bid_depth else None
        best_ask = min(ask_depth.keys()) if ask_depth else None

       
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
        else:
            spread = None
        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2
            mid_prices.append(mid_price) 
        else:
            mid_prices.append(None)  

       
        for price, qty in bid_depth.items():
            data.append({
                'timestamp': timestamp,
                'price': price,
                'quantity': qty,
                'type': 'bid',
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'color_value': qty  
            })

       
        for price, qty in ask_depth.items():
            data.append({
                'timestamp': timestamp,
                'price': price,
                'quantity': qty,
                'type': 'ask',
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'color_value': -qty  # 使用负数的数量作为颜色值
            })

    
    dfa = pd.DataFrame(data)

    
    fig = px.scatter(
        dfa,
        x='timestamp',
        y='price',
        color='color_value', 
        color_continuous_scale=[
            (0, "lightgreen"),  
            (0.5, "green"),     
            (0.5, "lightcoral"),
            (1, "red")          
        ],
                hover_data={
            'quantity': True,      
            'best_bid': True,     
            'best_ask': True,     
            'spread': True,      
            'price': True,        
        },  
        title='Order Book Visualization'
    )

 
    # 显示图表
    fig.show()