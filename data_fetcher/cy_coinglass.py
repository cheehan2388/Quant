from datetime import datetime, timezone
from typing import Dict, List
from cybotrade.strategy import Strategy as BaseStrategy
from cybotrade.runtime import StrategyTrader
from cybotrade.models import (
    RuntimeConfig,
    RuntimeMode,
)
from cybotrade.permutation import Permutation
import numpy as np
import asyncio
import logging
import colorlog
import pandas as pd

class Strategy(BaseStrategy):
    datasource_data = []
    candle_data = []
    start_time = datetime.utcnow()

    def _init_(self):
        handler = colorlog.StreamHandler()
        handler.setFormatter(
            colorlog.ColoredFormatter(f"%(log_color)s{Strategy.LOG_FORMAT}")
        )
        file_handler = logging.FileHandler("cryptoquant-exchange-netflow-log-extraction.log")
        file_handler.setLevel(logging.INFO)
        super()._init_(log_level=logging.INFO, handlers=[handler, file_handler])

    async def set_param(self, identifier, value):
        logging.info(f"Setting {identifier} to {value}")
        if identifier == "sma":
            self.sma_length = int(value)
        elif identifier == "z_score":
            self.z_score_threshold = float(value)
        else:
            logging.error(f"Could not set {identifier}, not found")

    async def on_candle_closed(self, strategy, topic, symbol):
        # Retrieve list of candle data for corresponding symbol that candle closed.
        logging.info("candle data {}".format(super().data_map[topic][-1]))
        self.candle_data.append(super().data_map[topic][-1])
            
    async def on_datasource_interval(self, strategy: StrategyTrader, topic: str, data_list):
        logging.info("datasource data {}".format(super().data_map[topic][-1]))
        self.datasource_data.append(super().data_map[topic][-1])

    async def on_backtest_complete(self, strategy: StrategyTrader):
        df = pd.DataFrame(self.datasource_data)
        df.to_csv("./../Data/CG_BTC_hour_open-interestl.csv") #Day Hour 10m 1h 24h

        time_taken = datetime.utcnow() - self.start_time
        print("Total time taken: ", time_taken)

config = RuntimeConfig(
    mode=RuntimeMode.Backtest,
    datasource_topics=["coinglass|futures/price/history?exchange=Binance&symbol=BTCUSDT&interval=1h"],
    candle_topics=[],
    active_order_interval=1,
    initial_capital=10_000.0,
    exchange_keys="./asdfasd.json",
    start_time=datetime(2024, 1, 11, 0, 0, 0, tzinfo=timezone.utc),
    end_time=datetime(2025, 7, 14, 0, 0, 0, tzinfo=timezone.utc),
    data_count=100,
    api_key="1PrnrxujzPkaSbCqK7a5x4czOqG7OX3gMHqalvcVWmYB6aXu",
    api_secret="72ehcnPKjSktBEYOoLCEadOrZMcW1WAELscRgEMZCcc0imIKkg5EzZK156ZO0lEtXaiOw5hW",
)

permutation = Permutation(config)
hyper_parameters = {}
hyper_parameters["sma"] = np.arange(10, 20, 30)

async def start():
    await permutation.run(hyper_parameters, Strategy)

asyncio.run(start())