import plotly.express as px
import pandas as pd


def visual_data(df, depth):
    mid_prices = []
    data = []
   

    for idx, row in df.iterrows():
        # 使用 DataFrame 的索引作为 x 轴的值
        timestamp = idx  # 0, 1, 2, ...

        order_book = row['order_book']
        
        # 获取 bid 和 ask，确保它们是字典格式
        bid = order_book.get('bid', {})
        ask = order_book.get('ask', {})

        if not isinstance(bid, dict) or not isinstance(ask, dict):
            # 如果 bid 或 ask 不是字典，跳过该行数据
            continue

        # 按价格排序并截取前 depth 个条目
        bid_depth = dict(sorted(bid.items(), key=lambda item: item[0], reverse=True)[:depth])
        ask_depth = dict(sorted(ask.items(), key=lambda item: item[0])[:depth])

        # 计算 best bid 和 best ask
        best_bid = max(bid_depth.keys()) if bid_depth else None
        best_ask = min(ask_depth.keys()) if ask_depth else None

        # 计算 mid price (中间价格)，如果 best_bid 和 best_ask 都存在
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
        else:
            spread = None
        if best_bid is not None and best_ask is not None:
            mid_price = (best_bid + best_ask) / 2
            mid_prices.append(mid_price)  # 直接添加浮点数 mid_price
        else:
            mid_prices.append(None)  # 如果没有有效的 mid_price，添加 None

        # 遍历每个 bid 的价格和数量
        for price, qty in bid_depth.items():
            data.append({
                'timestamp': timestamp,
                'price': price,
                'quantity': qty,
                'type': 'bid',
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'color_value': qty  # 使用数量作为颜色值
            })

        # 遍历每个 ask 的价格和数量
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

    # 创建 DataFrame
    dfa = pd.DataFrame(data)

    # 创建图表：颜色表示订单的强度，大小保持固定
    fig = px.scatter(
        dfa,
        x='timestamp',
        y='price',
        color='color_value',  # 根据数量决定颜色深浅
        color_continuous_scale=[
            (0, "lightgreen"),  # 买单数量小，颜色浅绿
            (0.5, "green"),     # 买单数量大，颜色深绿
            (0.5, "lightcoral"),# 卖单数量小，颜色浅红
            (1, "red")          # 卖单数量大，颜色深红
        ],
                hover_data={
            'quantity': True,      # 显示数量
            'best_bid': True,      # 显示 best bid
            'best_ask': True,      # 显示 best ask
            'spread': True,        # 显示 spread
            'price': True,         # 显示当前价格
        },  # 悬停时显示的内容
        title='Order Book Visualization'
    )

 
    # 显示图表
    fig.show()