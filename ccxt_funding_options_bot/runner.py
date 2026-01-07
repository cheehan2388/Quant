import os
from typing import Dict

from .config import StrategyConfig, default_coin_map, load_exchange_credentials
from .exchanges import ExchangeWrapper
from .strategy import run_loop
from .utils import env_bool, get_logger


logger = get_logger("runner")


def run_for_coin(coin_key: str) -> None:
    coins = default_coin_map()
    if coin_key not in coins:
        raise KeyError(f"Unknown coin_key {coin_key}. Available: {sorted(coins.keys())}")
    coin = coins[coin_key]

    creds_map = load_exchange_credentials()
    creds = creds_map.get(coin.exchange_id)
    if not creds:
        raise RuntimeError(f"No credentials configured for exchange {coin.exchange_id}")

    strat = StrategyConfig(
        dry_run=env_bool("DRY_RUN", True),
        poll_seconds=int(os.getenv("POLL_SECONDS", "60")),
        order_time_in_force=os.getenv("TIF", "GTC"),
    )

    logger.info("Starting %s on %s (dry_run=%s)", coin_key, coin.exchange_id, strat.dry_run)
    ex = ExchangeWrapper(
        exchange_id=creds.exchange_id,
        api_key=creds.api_key,
        secret=creds.secret,
        password=creds.password,
        testnet=creds.testnet,
        dry_run=strat.dry_run,
    )
    run_loop(ex, coin, strat)


if __name__ == "__main__":
    run_for_coin(os.getenv("COIN", "BTC"))


