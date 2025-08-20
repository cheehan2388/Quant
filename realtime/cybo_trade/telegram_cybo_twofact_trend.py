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
)
import numpy as np
import asyncio
import logging
import colorlog
from logging.handlers import TimedRotatingFileHandler
from cybotrade.permutation import Permutation
import pytz
import requests
import sys
import os
import  util

LONG = "Long"
SHORT = "Short"
runtime_mode = RuntimeMode.LiveTestnet


CYBOTRADE_API_KEY = (
    "1PrnrxujzPkaSbCqK7a5x4czOqG7OX3gMHqalvcVWmYB6aXu"  # os.getenv("CYBOTRADE_API_KEY")
)
CYBOTRADE_API_SECRET = "72ehcnPKjSktBEYOoLCEadOrZMcW1WAELscRgEMZCcc0imIKkg5EzZK156ZO0lEtXaiOw5hW"  # os.getenv("CYBOTRADE_API_SECRET")
CYBOTRADE_ORDER_QUANTITY = 0.1 #os.getenv("CYBOTRADE_ORDER_QUANTITY")
TELEGRAM_TOKEN = '7139584207:AAHXcyeEc59ECivlESUAE6S2SJPBDheayfw'
TELEGRAM_CHAT_ID = '-1002241763127'
endpoint1 = "coinglass|futures/open-interest/aggregated-history?symbol=BTC&interval=1m"
endpoint2 = (
    'coinglass|futures/funding-rate/history?exchange=Binance&symbol=BTCUSDT&interval=1m')

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

    multiplier = 0.6
    rolling_window = 5
    qty = CYBOTRADE_ORDER_QUANTITY
    
    pair = Symbol(base="BTC", quote="USDT")
    endpoint1_data = []
    endpoint2_data = []
    processed_data = set()
    
    long_short_data = PositionData(quantity=0.0, avg_price=0.0)
    time = int(datetime.now().timestamp())
    position = Position(pair, long_short_data, long_short_data,updated_time=time)
    bot_id = "[btc_funding_openinterest_rati_threshold_1h]"
    total_pnl = 0.0
    entry_time = datetime.now(pytz.timezone("UTC"))

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
        logging.info(
            f"Position after init: {util.get_position_info(self.position, self.entry_time)}"
        )

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "rolling_window":
            self.rolling_window = int(value)
        elif identifier == "multiplier":
            self.multiplier = float(value)
        else:
            logging.error(f"Could not set {identifier}, not found")

    async def on_datasource_interval(self, strategy, topic, data_list):

        if len(self.endpoint1_data) == 0:  # 初始化， 之后固定关系。每次拿一个 ep 所以用len = 0 确定 拿ep2 的时候不会出差错
            #logging.info("shuld be here once")
            self.endpoint1_data = super().data_map[endpoint1]

        if len(self.endpoint2_data) == 0:
            #logging.info("shuld be here once")
            self.endpoint2_data = super().data_map[endpoint2]

        if topic in self.processed_data:
            #logging.warn(f"{topic} came back twice! unexpected, hence dropping.")
            self.processed_data.clear()
        else:
            self.processed_data.add(topic)
            logging.warn(f"{topic} done fetching.")
        time =int(datetime.now().timestamp())

        if len(self.processed_data) == len(datasource_topics):
            o_i = np.array(
                list(
                    map(
                        lambda c: float(c["close"]),
                        self.endpoint1_data,
                    )
                )
            )
            funding_close = np.array(
                list(map(lambda c: float(c["close"]), self.endpoint2_data))
            )
            start_time = np.array(
                list(map(lambda c: float(c["start_time"]), self.endpoint1_data))
            )
            # After completion
            self.processed_data.clear()
            ln_o_i = np.log(o_i)
            ln_fr = np.log(funding_close)
            # zscore calculation
            ratio = ln_o_i * ln_fr
            ratio = ratio[-self.rolling_window :]
            sma = np.mean(o_i)
            std = np.std(o_i, ddof=1)
            zscore = (o_i[-1] - sma) / std
            
            current_price = await strategy.get_current_price(
                symbol=self.pair, exchange=Exchange.BybitLinear
            )
            logging.info(
                f"current total_pnl: {self.total_pnl}, current position: {util.get_position_info(self.position, self.entry_time)}, current_price: {current_price} at {util.convert_ms_to_datetime(start_time[-1])}"
            )
            qty = self.qty
            if self.position.long.quantity == 0.0 and self.position.short.quantity == 0 and zscore >= self.multiplier:
                
                try:
                    await strategy.open(
                        side=OrderSide.Buy,
                        quantity=qty,
                        take_profit=None,
                        stop_loss=None,
                        symbol=self.pair,
                        exchange=Exchange.BybitLinear,
                        is_hedge_mode=False,
                        is_post_only=False,
                    )
                    logging.info(
                        f"Placed a 1st order (Buy) with qty {qty} when current_price: {current_price}, ratio: {ratio[-1]}, sma: {sma}at {util.convert_ms_to_datetime(start_time[-1])}"
                    )
                    self.position = Position(
                        self.pair,
                        PositionData(quantity=qty, avg_price=current_price),
                        PositionData(quantity=0.0, avg_price=0.0),
                        updated_time = time
                    )
                    self.entry_time = util.convert_ms_to_datetime(start_time[-1])
                    mess = (
                        f"{self.bot_id} bot <u>opened a LONG position</u>\n"
                        f"<b>quantity</b>: {qty}\n"
                        f"<b>current_price</b>: {current_price}\n"
                        f"<b>ratio</b>: {ratio}\n"
                        f"<b>sma</b>: {round(sma, 2)}\n"
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
            elif  zscore <= -self.multiplier and  self.position.long.quantity != 0.0:
                
                    try:
                        await strategy.open(
                            side=OrderSide.Sell,
                            quantity=2*self.position.long.quantity,
                            take_profit=None,
                            stop_loss=None,
                            symbol=self.pair,
                            exchange=Exchange.BybitLinear,
                            is_hedge_mode=False,
                            is_post_only=False,
                        )
                        pnl = (current_price - self.position.long.avg_price) * abs(
                            self.position.long.quantity
                        )
                        self.total_pnl += pnl
                        logging.info(
                            f"Closed a buy position with qty {self.position.long.quantity} and start a Short position with qty {2*self.position.long.quantity}, pnl: {pnl}, total_pnl: {self.total_pnl} when current_price: {current_price}, ratio: {ratio[-1]}, sma: {sma}, sma_with_multiplier at {util.convert_ms_to_datetime(start_time[-1])}"
                        )
                        
                        self.position = Position(
                            self.pair,
                            PositionData(quantity = 0.0, avg_price = 0.0),
                            PositionData(quantity = qty, avg_price = current_price),
                            updated_time = time
                        )
                        mess = (
                            f"{self.bot_id} bot <u>closed a LONG position</u>\n"
                            f"<b>quantity</b>: {abs(2*self.position.long.quantity)}\n"
                            f"<b>pnl</b>: {pnl}\n"
                            f"<b>total_pnl</b>: {self.total_pnl}\n"
                            f"<b>current_price</b>: {current_price}\n"
                            f"<b>ratio</b>: {ratio}\n"
                            f"<b>sma</b>: {round(sma, 2)}\n"
                            f"<b>zscore</b>: {round(zscore, 2)}\n"
                            f"<b>time</b>: {util.convert_ms_to_datetime(start_time[-1])}"
                            f"<b>time</b>: {util.convert_ms_to_datetime(start_time[-1])}"
                        )
                        send_notification(
                            message=mess,
                            chat_id=TELEGRAM_CHAT_ID,
                            token=TELEGRAM_TOKEN,
                        )
                    except Exception as e:
                        logging.error(f"Failed to place close long market order: {e}")
            elif zscore < - self.multiplier and self.position.short.quantity == 0 and self.position.long.quantity == 0:
                
                    try:
                        await strategy.open(
                            side=OrderSide.Sell,
                            quantity= qty,
                            take_profit=None,
                            stop_loss=None,
                            symbol=self.pair,
                            exchange=Exchange.BybitLinear,
                            is_hedge_mode=False,
                            is_post_only=False,
                        )
                        pnl = (current_price - self.position.long.avg_price) * abs(
                            self.position.long.quantity
                        )
                        self.total_pnl += pnl
                        logging.info(
                            f"Open short position with qty {qty},  when current_price: {current_price}, ratio: {ratio[-1]}, sma: {sma}, sma_with_multiplier at {util.convert_ms_to_datetime(start_time[-1])}"
                        )
                        
                        self.position = Position(
                            self.pair,
                            PositionData(quantity = 0.0, avg_price = 0.0),
                            PositionData(quantity = qty, avg_price = current_price),
                            updated_time = time
                        )
                        mess = (
                            f"{self.bot_id} bot <u>start a Short position</u>\n"
                            f"<b>quantity</b>: {abs(qty)}\n"
                            f"<b>current_price</b>: {current_price}\n"
                            f"<b>ratio</b>: {ratio}\n"
                            f"<b>sma</b>: {round(sma, 2)}\n"
                            f"<b>zscore</b>: {round(zscore, 2)}\n"
                            f"<b>time</b>: {util.convert_ms_to_datetime(start_time[-1])}"
                        )
                        send_notification(
                            message=mess,
                            chat_id=TELEGRAM_CHAT_ID,
                            token=TELEGRAM_TOKEN,
                        )
                    except Exception as e:
                        logging.error(f"Failed to place short market order: {e}")

            elif self.position.short.quantity != 0.0 and zscore >= self.multiplier:
                
                try:
                    await strategy.open(
                        side=OrderSide.Buy,
                        quantity=2*self.position.short.quantity,
                        take_profit=None,
                        stop_loss=None,
                        symbol=self.pair,
                        exchange=Exchange.BybitLinear,
                        is_hedge_mode=False,
                        is_post_only=False,
                    )
                    logging.info(
                        f"Close Short Placed a buy order with qty {2*self.position.short.quantity} when current_price: {current_price}, ratio: {ratio[-1]}, sma: {sma}at {util.convert_ms_to_datetime(start_time[-1])}"
                    )
                    self.position = Position(
                        self.pair,
                        PositionData(quantity=qty, avg_price=current_price),
                        PositionData(quantity=0.0, avg_price=0.0),
                        updated_time = time
                    )
                    self.entry_time = util.convert_ms_to_datetime(start_time[-1])
                    mess = (
                        f"{self.bot_id} bot <u>opened a LONG position</u>\n"
                        f"<b>quantity</b>: {qty}\n"
                        f"<b>current_price</b>: {current_price}\n"
                        f"<b>ratio</b>: {ratio}\n"
                        f"<b>sma</b>: {round(sma, 2)}\n"
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
                
                


    async def on_shutdown(self):
        logging.info("Trigger shutdown function")
        if self.position.long.quantity > 0 or self.position.short.quantity > 0:
            logging.info(f"Bot get shutdown, position: {self.position}")


config = RuntimeConfig(
    mode=runtime_mode,
    datasource_topics=[endpoint1, endpoint2],
    active_order_interval=3600,
    initial_capital=1000000.0,
    candle_topics=["bybit-linear|candle?interval=1m&symbol=BTCUSDT"],
    start_time=datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2024, 5, 1, 0, 0, 0, tzinfo=timezone.utc),
    api_key=CYBOTRADE_API_KEY,
    api_secret=CYBOTRADE_API_SECRET,
    data_count=500,
    exchange_keys="../key/credentials.json",
)

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters["rolling_window"] = [6]  # , 80, 110, 140
hyper_parameters["multiplier"] = [0.1]


async def start_backtest():
    await permutation.run(hyper_parameters, Strategy)


asyncio.run(start_backtest())
