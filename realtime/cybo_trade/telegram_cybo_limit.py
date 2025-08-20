from datetime import UTC, datetime, timedelta, timezone
from cybotrade.strategy import Strategy as BaseStrategy
from cybotrade.models import (
    Exchange,
    OrderSide,
    RuntimeConfig,
    RuntimeMode,
    Position,
    PositionData,
    Symbol,
    OrderStatus,
    OrderParams,
)
import numpy as np
import asyncio
import logging
import colorlog
from logging.handlers import TimedRotatingFileHandler
from cybotrade.permutation import Permutation
import pytz
import requests
import util
import os

LONG = "Long"
SHORT = "Short"
runtime_mode = RuntimeMode.LiveTestnet
place_order_exchange = Exchange.BybitLinear

CYBOTRADE_API_KEY = os.getenv(
    "MVrhHV9ACjDbxFM43Eswu0AOPPJVvt4U9Q4K0ydzg4H3Z2AC"  # os.getenv("CYBOTRADE_API_KEY")
)
CYBOTRADE_API_SECRET = os.getenv("PNBHpucevGxMZvu5MV308GZJkNvbuLCWmS8TZsyGlm5xstS2e3bBB6GN3DNezzO7sAz1coVi")  # os.getenv("CYBOTRADE_API_SECRET")
CYBOTRADE_ORDER_QUANTITY = os.getenv("0.003")  # os.getenv("CYBOTRADE_ORDER_QUANTITY")
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""
endpoint1 = "cryptoquant|1h|eth/market-data/coinbase-premium-index?window=hour"
endpoint2 = (
    "cryptoquant|1h|eth/market-data/taker-buy-sell-stats?window=hour&exchange=bybit"
)

datasource_topics = [endpoint1, endpoint2]
processed_data = set()


def send_notification(message: str, chat_id: str, token: str):
    url = f"https://api.telegram.org/bot{token}/sendMessage?chat_id={chat_id}&text={message}&parse_mode=HTML"
    try:
        response = requests.get(url)
        response_json = response.json()
        logging.info(f"Telegram response: {response_json}")
        if not response_json.get("ok"):
            logging.error(f"Failed to send message: {response_json}")
        return response_json
    except requests.RequestException as e:
        logging.error(f"Request to Telegram API failed: {e}")
        return None


class Strategy(BaseStrategy):
    # Indicator params
    leverage = 1.5 
    multiplier = 1.6
    rolling_window = 27
    order_pool = []
    qty_precision = 3
    price_precision = 1
    min_qty = CYBOTRADE_ORDER_QUANTITY
  
    pair = Symbol(base="ETH", quote="USDT")
    endpoint1_data = []
    endpoint2_data = []
    processed_data = set()
    bot_id = "[btc_coinbase_premium_index_gap_threshold_1h]"
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
            "y_coinbase_pi_gap_1h_threshold.log", when="h", backupCount=30
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(Strategy.LOG_FORMAT))
        super().__init__(log_level=logging.INFO, handlers=[handler, file_handler])
        mess = "<b>Initializing bot</b> " + str(self.bot_id + " <b>gogo</b>")
        send_notification(
            message=mess,
            chat_id=TELEGRAM_CHAT_ID,
            token=TELEGRAM_TOKEN,
        )


    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "rolling_window":
            self.rolling_window = int(value)
        elif identifier == "multiplier":
            self.multiplier = float(value)
        else:
            logging.error(f"Could not set {identifier}, not found")

    async def on_active_order_interval(self, strategy, active_orders):
        id = self.order_pool
        self.order_pool = await util.check_open_order(
            strategy=strategy,
            exchange=place_order_exchange,
            pair=self.pair,
            order_pool=self.order_pool,
        )
        if len(self.order_pool) == 0:
            for limit in id:
                try:
                        mess = (
                                    f"{self.bot_id} bot <u>Remove Limit Order id {limit[0]}</u>\n"
                                    f"<b>time</b>: {datetime.now(timezone.utc)}"
                                )
                        send_notification(
                                    message=mess,
                                    chat_id=TELEGRAM_CHAT_ID,
                                    token=TELEGRAM_TOKEN,
                                )
                except Exception as e:
                                    logging.error(f"Failed to place open buy market order: {e}") 
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


                        if (limit[2] == True):
                            try:

                        
                                mess = (
                                    f"{self.bot_id} bot <u>Exit Position filled</u>\n"
                                    f"<b>quantity</b>: {update.remain_size}\n"
                                    f"<b>limitprice</b>: {limit[5]}\n"
                                    f"<b>time</b>: {datetime.now(timezone.utc)}"
                                )
                                send_notification(
                                    message=mess,
                                    chat_id=TELEGRAM_CHAT_ID,
                                    token=TELEGRAM_TOKEN,
                                )
                            except Exception as e:
                                logging.error(f"Failed to place open buy market order: {e}") 
                        else :
                                try:

                        
                                    mess = (
                                        f"{self.bot_id} bot <u>Entry Position filled</u>\n"
                                        f"<b>quantity</b>: {update.remain_size}\n"
                                        f"<b>limitprice</b>: {limit[5]}\n"
                                        f"<b>time</b>: {datetime.now(timezone.utc)}"
                                    )
                                    send_notification(
                                        message=mess,
                                        chat_id=TELEGRAM_CHAT_ID,
                                        token=TELEGRAM_TOKEN,
                                    )
                                except Exception as e:
                                    logging.error(f"Failed to place open buy market order: {e}") 
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
                                        try:

                                    
                                            mess = (
                                                f"{self.bot_id} bot <u>reopened a Exit LIMIT position</u>\n"
                                                f"<b>quantity</b>: {update.remain_size}\n"
                                                f"<b>limitprice</b>: {price}\n"
                                                f"<b>time</b>: {datetime.now(timezone.utc)}"
                                            )
                                            send_notification(
                                                message=mess,
                                                chat_id=TELEGRAM_CHAT_ID,
                                                token=TELEGRAM_TOKEN,
                                            )
                                        except Exception as e:
                                            logging.error(f"Failed to place open buy market order: {e}")                                
                                        self.order_pool.append(
                                            [
                                                order_resp.client_order_id, 
                                                datetime.now(timezone.utc),
                                                True,
                                                False,
                                                False,
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

    async def on_datasource_interval(self, strategy, topic, data_list):

        if len(self.endpoint1_data) == 0:  
            self.endpoint1_data = super().data_map[endpoint1]

        if len(self.endpoint2_data) == 0:
            logging.info("shuld be here once")
            self.endpoint2_data = super().data_map[endpoint2]

        if topic in self.processed_data:
            logging.warn(f"{topic} came back twice! unexpected, hence dropping.")
            self.processed_data.clear()
        else:
            self.processed_data.add(topic)
            logging.warn(f"{topic} done fetching.")

        if len(self.processed_data) == len(datasource_topics):
            coinbase_Premium = np.array(
                list(
                    map(
                        lambda c: float(c["coinbase_premium_index"]),
                        self.endpoint1_data,
                    )
                )
            )
            logging.warn(f"coinbase_Premium done fetching.")
            takerb = np.array(
                list(map(lambda c: float(c["taker_buy_ratio"]), self.endpoint2_data))
            )
            logging.warn(f"taker_buy_ratio done fetching.")
            start_time = np.array(
                list(map(lambda c: float(c["start_time"]), self.endpoint1_data))
            )
            # After completion
            self.processed_data.clear()

            # zscore calculation
            ratio = coinbase_Premium / takerb
            ratio = ratio[-self.rolling_window :]
            sma = np.mean(ratio)
            std = np.std(ratio, ddof=1)
            zscore = (ratio[-1] - sma) / std
            try:
                position = await strategy.position( 
                    exchange=place_order_exchange,symbol=self.pair
                )
            except Exception as e:
                logging.error(f"Failed to fetch position: {e}")
            logging.info(
                f"Latest position: {util.get_position_info(position,datetime.now(pytz.timezone("UTC")))}"
            )
            try:
                wallet_balance = await strategy.get_current_available_balance(  
                    exchange=place_order_exchange, symbol=self.pair  
                )
            except Exception as e:
                logging.error(f"Failed to fetch wallet balance: {e}")

            if position.long.quantity == 0.0 and zscore >= self.multiplier  and len(self.order_pool) == 0:
                best_bid_ask = await util.get_order_book(
                strategy=strategy, exchange=place_order_exchange, pair=self.pair)
                if best_bid_ask[0] != 0.0 and best_bid_ask[1] != 0.0: 

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
                                symbol=self.pair,
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
                            f"Inserted a long limit order with client_order_id: {order_resp.client_order_id} into order_pool"
                        )
                    except Exception as e:
                        logging.error(f"Failed to close entire position: {e}")
                    
                    try:

                
                        mess = (
                            f"{self.bot_id} bot <u>opened a LONG position</u>\n"
                            f"<b>quantity</b>: {qty}\n"
                            f"<b>limitprice</b>: {(best_bid_ask[0] + best_bid_ask[1]) / 2.0}\n"
                            f"<b>ratio</b>: {ratio}\n"
                            f"<b>zscore</b>: {round(zscore, 2)}\n"
                            f"<b>time</b>: {util.convert_ms_to_datetime(start_time[-1])}"
                        )
                        send_notification(
                            message=mess,
                            chat_id=TELEGRAM_CHAT_ID,
                            token=TELEGRAM_TOKEN,
                        )
                    except Exception as e:
                        logging.error(f"Failed to place open buy market order: {e}")
            elif  zscore <= -self.multiplier and  position.long.quantity != 0.0 and  len(self.order_pool) == 0:
                best_bid_ask = await util.get_order_book(
                                strategy=strategy, exchange=place_order_exchange, pair=self.pair
                                )
                if best_bid_ask[0] != 0.0 and best_bid_ask[1] != 0.0: #best_bid_ask[0] 0 是 b 1是ask
                    
                    try:
                        await strategy.open(
                            limit=best_bid_ask[0],
                            side=OrderSide.Sell,
                            quantity=position.long.quantity,
                            take_profit=None,
                            stop_loss=None,
                            symbol=self.pair,
                            exchange=Exchange.BybitLinear,
                            is_hedge_mode=False,
                            is_post_only=False,
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

                    except Exception as e:
                        logging.error(f"Failed to close entire position: {e}")

                        mess = (
                            f"{self.bot_id} bot <u>closed a LONG position</u>\n"
                            f"<b>quantity</b>: {abs(position.long.quantity)}\n"
                            f"<b>limitprice</b>: {(best_bid_ask[0] + best_bid_ask[1]) / 2.0}\n"
                            f"<b>ratio</b>: {ratio}\n"
                            f"<b>zscore</b>: {round(zscore, 2)}\n"
                            f"<b>time</b>: {util.convert_ms_to_datetime(start_time[-1])}"
                            )
                    try:
                        send_notification(
                            message=mess,
                            chat_id=TELEGRAM_CHAT_ID,
                            token=TELEGRAM_TOKEN,
                        )
                    except Exception as e:
                        logging.error(f"Failed to place close long market order: {e}")

config = RuntimeConfig(
    mode=runtime_mode,
    datasource_topics=[endpoint1, endpoint2],
    active_order_interval=60,
    initial_capital=1000000.0,
    candle_topics=["candles-1h-ETH/USDT-bybit"],
    start_time=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
    api_key=CYBOTRADE_API_KEY,
    api_secret=CYBOTRADE_API_SECRET,
    data_count=1000,
    exchange_keys="./credentials.json",
)

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters["rolling_window"] = [27]  # , 80, 110, 140
hyper_parameters["multiplier"] = [1.6]


async def start_backtest():
    await permutation.run(hyper_parameters, Strategy)


asyncio.run(start_backtest())

