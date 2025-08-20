from datetime import UTC, datetime, timedelta, timezone
from cybotrade.strategy import Strategy as BaseStrategy
from cybotrade.models import (
    Exchange,
    OrderSide,
    RuntimeConfig,
    RuntimeMode,
    OrderParams,
    Symbol,
    OrderStatus,
)
import numpy as np
import asyncio
import logging
import colorlog
from logging.handlers import TimedRotatingFileHandler
from cybotrade.permutation import Permutation
import pandas as pd
from pybit.unified_trading import HTTP
import requests
import os

import pytz
import util

runtime_mode = RuntimeMode.Live

CYBOTRADE_ORDER_QUANTITY = os.getenv("CYBOTRADE_ORDER_QUANTITY")

pybit_mode = False
if runtime_mode == RuntimeMode.LiveTestnet:
    pybit_mode = True
place_order_exchange = Exchange.BitgetLinear

class Strategy(BaseStrategy):
    # Indicator params
    leverage = 1.5
    multiplier = 0.35
    rolling_window = 170
    candle_interval = 900
    fetch_candle_interval = "15min"
    order_pool = []
    qty_precision = 3
    price_precision = 1

    min_qty = float(CYBOTRADE_ORDER_QUANTITY) ##

    pair = Symbol(base="BTC", quote="USDT")
    bot_id = "btc_bitget_spot_prep_pct_chg_threshold_15m"
    total_pnl = 0.0
    entry_time = datetime.now(pytz.timezone("UTC"))
    replace_entry_order_count = 0
    replace_tp_order_count = 0
    cancel_entry_order_count = 0
    replace_limit_max_time_in_min = 5
    replace_interval = 5

    def __init__(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = TimedRotatingFileHandler(
            "y_bitget_spot_prep_pct_chg_15m.log", when="h", backupCount=30
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "rolling_window":
            self.rolling_window = int(value)
        elif identifier == "multiplier":
            self.multiplier = float(value)
        else:
            logging.error(f"Could not set {identifier}, not found")

    def fetch_bitget_spot_candle(self, start_time, end_time, interval, symbol, gap):
        logging.info(
            f"fetching spot candles from {util.convert_ms_to_datetime(start_time)} to {util.convert_ms_to_datetime(end_time)}"
        )
        pi_start_time = start_time
        pi_end_time = end_time
        price_data = []
        base_url = "https://api.bitget.com/api/v2/spot/market/candles"
        params = {
            "symbol": symbol,
            "granularity": interval,
            "limit": 1000,
            "endTime": int(pi_end_time),
        }
        response = requests.get(base_url, params=params)
        candles = response.json()
        price_data = candles["data"]
        second_fetch_end_time = float(price_data[0][0])
        if pi_start_time < second_fetch_end_time:
            while second_fetch_end_time > pi_start_time:
                base_url = "https://api.bitget.com/api/v2/spot/market/history-candles"
                params = {
                    "symbol": symbol,
                    "granularity": interval,
                    "limit": 200,
                    "endTime": int(second_fetch_end_time),
                }
                response = requests.get(base_url, params=params)
                candles = response.json()
                price_data = candles["data"] + price_data
                second_fetch_end_time = second_fetch_end_time - gap * 1000 * 200

        price_data = pd.DataFrame(price_data)
        price_data.columns = [
            "time",
            "open",
            "high",
            "low",
            "close_spot",
            "volume",
            "volume_usd",
            "volume_usdt",
        ]
        price_data["time"] = price_data["time"].astype(float)
        price_data = price_data[price_data["time"] >= start_time]
        price_data = price_data.drop_duplicates(subset=["time"])
        price_data["time"] = pd.to_datetime(price_data["time"], unit="ms")
        price_df = price_data[["time", "open", "high", "low", "close_spot", "volume"]]
        logging.info(
            f"Done fetching spot candles from {price_df['time'].iloc[0]} to {price_df['time'].iloc[-1]}"
        )
        return price_df

    async def on_active_order_interval(self, strategy, active_orders):
        self.order_pool = await util.check_open_order(
            strategy=strategy,
            exchange=place_order_exchange,
            pair=self.pair,
            order_pool=self.order_pool,
        )
        limit_order_handler = await util.limit_order_handler(
            active_orders=active_orders,
            order_pool=self.order_pool,
            strategy=strategy,
            place_order_exchange=place_order_exchange,
            pair=self.pair,
            replace_entry_order_count=self.replace_entry_order_count,
            replace_tp_order_count=self.replace_tp_order_count,
            replace_limit_max_time=self.replace_limit_max_time_in_min * 60,
            replace_order_interval_in_sec=self.replace_interval,
            cancel_entry_order_count=self.cancel_entry_order_count,
        )
        self.order_pool = limit_order_handler[0]
        self.replace_entry_order_count = limit_order_handler[1]
        self.replace_tp_order_count = limit_order_handler[2]
        self.cancel_entry_order_count = limit_order_handler[3]

    async def on_order_update(self, strategy, update):
        if len(self.order_pool) != 0:
            for limit in self.order_pool:
                if limit[0] == update.client_order_id:
                    logging.info(f"Latest order update: {update}")
                    if update.status == OrderStatus.Created:
                        limit[4] = True
                        logging.info(
                            f"Limit order: {update.client_order_id} is {update.status}"
                        )
                    elif (
                        update.status == OrderStatus.Filled
                        or update.status == OrderStatus.PartiallyFilledCancelled
                    ):
                        try:
                            position = await strategy.position(
                                symbol=self.pair, exchange=place_order_exchange
                            )
                        except Exception as e:
                            logging.error(f"Failed to fetch position: {e}")
                        logging.info(
                            f"Latest position: {util.get_position_info(position,datetime.now(pytz.timezone("UTC")))}"
                        )

                        self.order_pool.remove(limit)
                        logging.info(
                            f"Removed {limit} from order_pool due to order {update.status}, current order_pool: {self.order_pool}"
                        )
                    elif (
                        update.status == OrderStatus.Cancelled
                        or update.status == OrderStatus.Rejected
                    ):
                        if limit[4] == False:
                            if limit[2] == True:
                                try:
                                    best_bid_ask = await util.get_order_book(
                                        strategy=strategy,
                                        exchange=place_order_exchange,
                                        pair=self.pair,
                                    )
                                    if (
                                        best_bid_ask[0] != 0.0
                                        and best_bid_ask[1] != 0.0
                                    ):
                                        if update.side == OrderSide.Buy:
                                            price = best_bid_ask[0]
                                        else:
                                            price = best_bid_ask[1]
                                        order_resp = await strategy.order(
                                            params=OrderParams(
                                                limit=price,
                                                side=update.side,
                                                quantity=update.remain_size,
                                                symbol=self.pair,
                                                exchange=place_order_exchange,
                                                is_hedge_mode=False,
                                                is_post_only=True,
                                            )
                                        )
                                        self.order_pool.append(
                                            [
                                                order_resp.client_order_id, # 0
                                                datetime.now(timezone.utc),# 1
                                                True,# 3
                                                False,# 4
                                                False,# 5
                                                price,
                                                datetime.now(timezone.utc)
                                                + timedelta(
                                                    minutes=self.replace_limit_max_time_in_min
                                                ),
                                            ]
                                        )
                                        logging.info(
                                            f"Inserted a replace {update.side} limit order with client_order_id: {order_resp.client_order_id} into order_pool"
                                        )
                                        logging.info(
                                            f"Placed a replace {update.side} order with qty {update.remain_size} when price: {price} due to order get rejected at time {datetime.now(timezone.utc)}"
                                        )
                                except Exception as e:
                                    logging.error(
                                        f"Failed to cancel and replace order: {e}"
                                    )
                        self.order_pool.remove(limit)
                        logging.info(
                            f"Removed {limit} from order_pool due to order {update.status}, current order_pool: {self.order_pool}"
                        )

    async def on_candle_closed(self, strategy, topic, symbol):
        candles = self.data_map[topic] # 拿 candle 
        start_time = np.array(list(map(lambda c: float(c["start_time"]), candles)))
        close = np.array(list(map(lambda c: float(c["close"]), candles)))
        logging.info(
            f"close: {close[-1]} at {util.convert_ms_to_datetime(start_time[-1])}"
        )
        if len(close) < self.rolling_window * 3:
            logging.info("Not enough candles to calculate")
            return
        fetch_end_time = start_time[-1]
        fetch_start_time = start_time[0]
        logging.debug(
            f"fetch_start_time: {util.convert_ms_to_datetime(start_time[0])}, fetch_end_time {util.convert_ms_to_datetime(start_time[-1])}"
        )
        try:
            bitget_spot_df = self.fetch_bitget_spot_candle(
                start_time=fetch_start_time,
                end_time=fetch_end_time,
                interval=self.fetch_candle_interval,
                symbol="BTCUSDT",
                gap=self.candle_interval,
            )
        except Exception as e:
            logging.error(f"Failed to fetch bitget spot candles: {e}")
        close_spot = bitget_spot_df["close_spot"].values
        close_spot = close_spot[-len(close) :]
        logging.debug(f"close_spot: {len(close_spot)}, close_prep: {len(close)}")
        spread = []
        for i in range(len(close_spot)):
            spread.append(float(close[i]) / float(close_spot[i]) - 1.0) # contract/close - 1
        sma = util.get_rolling_mean(spread, self.rolling_window)
        if sma[-1] > 0.0:
            upper_sma_threshold = sma[-1] * (1.0 + self.multiplier)
            lower_sma_threshold = sma[-1] * (1.0 - self.multiplier)
        else:
            upper_sma_threshold = sma[-1] * (1.0 - self.multiplier)
            lower_sma_threshold = sma[-1] * (1.0 + self.multiplier)
        try:
            position = await strategy.position(
                symbol=symbol, exchange=place_order_exchange
            )
        except Exception as e:
            logging.error(f"Failed to fetch position: {e}")
        try:
            wallet_balance = await strategy.get_current_available_balance(  #查钱包余额
                exchange=place_order_exchange, symbol=symbol  
            )
        except Exception as e:
            logging.error(f"Failed to fetch wallet balance: {e}")
        logging.info(
            f"current total_pnl: {self.total_pnl}, current position: {util.get_position_info(position, self.entry_time)} , replace_entry_order_count: {self.replace_entry_order_count}, replace_tp_order_count: {self.replace_tp_order_count}, cancel_entry_order_count: {self.cancel_entry_order_count}, order_pool_length: {len(self.order_pool)}, bitget close_spot: {close_spot[-1]}, close_prep: {close[-1]}, spread: {spread[-1]}, sma: {sma[-1]}, sma_with_multiplier_up: {upper_sma_threshold}, sma_with_multiplier_down: {lower_sma_threshold}, close: {close[-1]} at {util.convert_ms_to_datetime(start_time[-1])}"
        )
        if (
            spread[-1] > upper_sma_threshold #大于 upper band
            and position.short.quantity == 0.0
            and len(self.order_pool) == 0
        ):
            best_bid_ask = await util.get_order_book(
                strategy=strategy, exchange=place_order_exchange, pair=self.pair
            )
            if best_bid_ask[0] != 0.0 and best_bid_ask[1] != 0.0: #best_bid_ask[0] 0 是 b 1是ask
                if position.long.quantity != 0.0:
                    try:
                        order_resp = await strategy.order(
                            params=OrderParams(
                                limit=best_bid_ask[1],
                                side=OrderSide.Sell,
                                quantity=abs(position.long.quantity),
                                symbol=symbol,
                                exchange=place_order_exchange,
                                is_hedge_mode=False,
                                is_post_only=True,
                            )
                        )
                        self.order_pool.append(
                            [
                                order_resp.client_order_id,
                                datetime.now(timezone.utc),
                                True,
                                False,
                                False,
                                best_bid_ask[1],
                                datetime.now(timezone.utc)
                                + timedelta(minutes=self.replace_limit_max_time_in_min),
                            ]
                        )
                        logging.info(
                            f"Inserted a close long limit order with client_order_id: {order_resp.client_order_id} into order_pool"
                        )
                        logging.info(
                            f"Placed a close long order with qty {position.long.quantity} when price: {best_bid_ask[1]}, close_spot: {close_spot[-1]}, close_prep: {close[-1]}, spread: {spread[-1]}, sma: {sma[-1]}, sma_with_multiplier_up: {upper_sma_threshold}, sma_with_multiplier_down: {lower_sma_threshold} at time {util.convert_ms_to_datetime(start_time[-1])}"
                        )

                    except Exception as e:
                        logging.error(f"Failed to close entire position: {e}")
                qty = util.get_qty_with_percentage(            # price 用两个spread 除2
                    (best_bid_ask[0] + best_bid_ask[1]) / 2.0,
                    self.qty_precision,
                    self.leverage,
                    self.min_qty,
                    wallet_balance,
                )
                try:
                    order_resp = await strategy.order(
                        params=OrderParams(
                            limit=best_bid_ask[1],
                            side=OrderSide.Sell,
                            quantity=qty,
                            symbol=symbol,
                            exchange=place_order_exchange,
                            is_hedge_mode=False,
                            is_post_only=True,
                        )
                    )
                    self.order_pool.append(
                        [
                            order_resp.client_order_id,
                            datetime.now(timezone.utc),
                            False,
                            False,
                            False,
                            best_bid_ask[1],
                            datetime.now(timezone.utc)
                            + timedelta(minutes=self.replace_limit_max_time_in_min),
                        ]
                    )
                    logging.info(
                        f"Inserted a sell limit order with client_order_id: {order_resp.client_order_id} into order_pool"
                    )
                    logging.info(
                        f"Placed a sell order with qty {qty} when price: {best_bid_ask[1]}, close_spot: {close_spot[-1]}, close_prep: {close[-1]}, spread: {spread[-1]}, sma: {sma[-1]}, sma_with_multiplier_up: {upper_sma_threshold}, sma_with_multiplier_down: {lower_sma_threshold} at time {util.convert_ms_to_datetime(start_time[-1])}"
                    )

                except Exception as e:
                    logging.error(f"Failed to place sell limit order: {e}")
        elif (
            spread[-1] < lower_sma_threshold
            and position.long.quantity == 0.0
            and len(self.order_pool) == 0
        ):
            best_bid_ask = await util.get_order_book(
                strategy=strategy, exchange=place_order_exchange, pair=self.pair
            )
            if best_bid_ask[0] != 0.0 and best_bid_ask[1] != 0.0:
                if position.short.quantity != 0.0:
                    try:
                        order_resp = await strategy.order(
                            params=OrderParams(
                                limit=best_bid_ask[0],
                                side=OrderSide.Buy,
                                quantity=abs(position.short.quantity),
                                symbol=symbol,
                                exchange=place_order_exchange,
                                is_hedge_mode=False,
                                is_post_only=True,
                            )
                        )
                        self.order_pool.append(
                            [
                                order_resp.client_order_id,
                                datetime.now(timezone.utc),
                                True,
                                False,
                                False,
                                best_bid_ask[0],
                                datetime.now(timezone.utc)
                                + timedelta(minutes=self.replace_limit_max_time_in_min),
                            ]
                        )
                        logging.info(
                            f"Inserted a close short limit order with client_order_id: {order_resp.client_order_id} into order_pool"
                        )
                        logging.info(
                            f"Placed a close short order with qty {position.short.quantity} when price: {best_bid_ask[0]}, close_spot: {close_spot[-1]}, close_prep: {close[-1]}, spread: {spread[-1]}, sma: {sma[-1]}, sma_with_multiplier_up: {upper_sma_threshold}, sma_with_multiplier_down: {lower_sma_threshold} at time {util.convert_ms_to_datetime(start_time[-1])}"
                        )

                    except Exception as e:
                        logging.error(f"Failed to close entire position: {e}")
                qty = util.get_qty_with_percentage(
                    (best_bid_ask[0] + best_bid_ask[1]) / 2.0,
                    self.qty_precision,
                    self.leverage,
                    self.min_qty,
                    wallet_balance,
                )
                try:
                    order_resp = await strategy.order(
                        params=OrderParams(
                            limit=best_bid_ask[0],
                            side=OrderSide.Buy,
                            quantity=qty,
                            symbol=symbol,
                            exchange=place_order_exchange,
                            is_hedge_mode=False,
                            is_post_only=True,
                        )
                    )
                    self.order_pool.append(
                        [
                            order_resp.client_order_id,
                            datetime.now(timezone.utc),
                            False,
                            False,
                            False,
                            best_bid_ask[0],
                            datetime.now(timezone.utc)
                            + timedelta(minutes=self.replace_limit_max_time_in_min),
                        ]
                    )
                    logging.info(
                        f"Inserted a buy limit order with client_order_id: {order_resp.client_order_id} into order_pool"
                    )
                    logging.info(
                        f"Placed a buy order with qty {qty} when price: {best_bid_ask[0]}, close_spot: {close_spot[-1]}, close_prep: {close[-1]}, spread: {spread[-1]}, sma: {sma[-1]}, sma_with_multiplier_up: {upper_sma_threshold}, sma_with_multiplier_down: {lower_sma_threshold} at time {util.convert_ms_to_datetime(start_time[-1])}"
                    )

                except Exception as e:
                    logging.error(f"Failed to place buy limit order: {e}")


config = RuntimeConfig(
    mode=runtime_mode,
    datasource_topics=[],
    active_order_interval=60,
    initial_capital=1000000.0,
    candle_topics=["candles-15m-BTC/USDT-bitget"],
    start_time=datetime(2023, 6, 1, 0, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 2, 1, 0, 0, 0, tzinfo=timezone.utc),
    api_key="dummy",
    api_secret="dummy",
    data_count=2100,
    exchange_keys="./credentials.json",
)

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters["rolling_window"] = [690]
hyper_parameters["multiplier"] = [0.55]


async def start_backtest():
    await permutation.run(hyper_parameters, Strategy)


asyncio.run(start_backtest())
