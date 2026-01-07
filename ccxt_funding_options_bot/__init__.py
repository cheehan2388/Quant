"""
Funding + options-hedged strategy toolkit built on ccxt.

This package provides a skeleton for a funding capture strategy on perpetual
swaps, hedged to near-delta-neutral using listed options. It includes basic
utilities, exchange wrappers, Greeks, risk, and orchestration logic.

Note: Exchange option support varies. Configure symbols and exchanges per coin
in `config.py` before live use.
"""

__all__ = [
    "config",
    "exchanges",
    "greeks",
]


