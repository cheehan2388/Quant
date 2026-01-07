Smart Volatility Arbitrage (VRP) — Paper-Driven Strategy

Overview

- Objective: Harvest the variance risk premium (VRP) by taking delta-hedged option exposure (variance-swap proxy) when 30d implied variance is rich vs a robust forecast of 30d realized variance, with tail hedges and regime filters.
- Core idea: Trade the difference between implied and expected realized variance: VRP_t = IV30_t^2 − E[RV30_{t→t+30}].

Key References

- Bollerslev, Tauchen, Zhou (2009) — Expected Stock Returns and Variance Risk Premia.
- Demeterfi, Derman, Kamal, Zou (1999) — A Guide to Volatility and Variance Swaps.
- Corsi (2009) — A Simple Approximate Long-Memory Model of Realized Volatility (HAR-RV).
- Andersen, Bollerslev, Diebold (2003) — Realized volatility measurement (high-frequency RV).
- Carr and Wu (2009) — Variance Risk Premia and option-implied variance.

Data Requirements

- Underlying OHLCV (e.g., BTCUSDT 1h or 1d). Columns: time, Open, High, Low, Close.
- 30d ATM implied volatility (annualized) or variance (e.g., Deribit DVOL 30d). Column: iv30 (annualized vol) or iv30_var (annualized variance).

Pipeline

1) Forecast RV30 using HAR-RV on daily realized variance (rv_d), with weekly (rv_w) and monthly (rv_m) components.
2) Compute IV30 annualized variance from IV30 vol: iv30_var = (iv30)².
3) VRP = iv30_var − forecast_rv30_annualized.
4) Signals: short variance when VRP above upper quantile and regime is benign; long variance when below lower quantile or during stressed regimes.
5) Filters: jump filter (bipower variation vs RV), regime filter (vol-of-vol and IV slope), event blackout window.
6) Sizing: volatility-target the variance exposure; cap gross vega; add tail hedges with OTM wings if trading options.
7) PnL proxy: daily constant-maturity variance-swap approximation using next-day squared returns versus prior strike.

Quick Start

- Put IV and price CSVs under `Data/` (or anywhere) with the required columns.
- Run the example in `vrp_smart.py` (see __main__), pointing to your files. The script prints key metrics and saves a CSV with time series.

Notes

- The included implementation is data-agnostic and uses a simple variance-swap PnL proxy for constant-maturity exposure. For production, plug in your option chain to synthesize IV30 and use exact variance swap MTM.
- You can also supply your own realized volatility forecast (e.g., your RL model) in place of HAR-RV.

### Volatility arbitrage blueprint (paper-backed)

This folder is intended for a robust, implementable volatility arbitrage strategy informed by leading academic evidence. The core edge is the variance risk premium (VRP): the difference between implied variance and expected realized variance. Harvesting a positive VRP via delta-hedged short options (or variance swaps) is one of the most consistently documented option premia.

### Core strategy: Weekly ATM VRP with risk-managed exposure

- **Signal (VRP)**: For a 7D horizon, compute
  - **Implied variance** from ATM weekly options (e.g., Deribit BTC options, mark IV squared and annualized, scaled to horizon).
  - **Expected realized variance** using a high-accuracy volatility forecast. Prefer a realized-volatility model such as **HAR-RV** (Corsi, 2009), optionally augmented by **Realized GARCH** or jump-robust RV.
  - **VRP** = implied_var − expected_realized_var. When VRP > threshold, short volatility; when < −threshold, long volatility; else flat.

- **Trade construction**: Enter a delta-hedged, near-ATM weekly **straddle** (or replicate a corridor variance swap). Maintain delta neutrality with discrete hedging; size initial vega exposure proportional to signal strength.

- **Risk management (smart scaling)**:
  - **Volatility-managed sizing** (Moreira & Muir, 2017): scale notional by 1 / trailing volatility to stabilize risk through regimes.
  - **Term-structure / regime filter**: avoid short-vol in backwardation and during high vol-of-vol states; favor long-vol in stress.
  - **Jump/event filter**: suppress short-vol around scheduled events with jump risk.
  - **Drawdown controls**: hard stops, forced de-leveraging, and max daily gamma loss caps given hedging cadence and costs.

### Practical implementation notes

- **Data**
  - Options: Deribit public API for BTC weekly options (use `mark_iv` around ATM). Alternatively, DVOL index for IV proxy.
  - Underlying: 1m–1h BTC prices to compute realized variance; higher frequency improves accuracy.

- **Forecasting**
  - Compute realized volatility using 5m/1h returns with microstructure noise adjustments if needed.
  - Fit **HAR-RV** on daily RV with weekly and monthly lags; optionally combine with Realized GARCH; use model averaging.

- **Execution**
  - Choose the nearest standard weekly expiry (~7D). Select strikes bracketing spot for ATM straddle.
  - Hedge delta discretely (e.g., every N minutes or at price grid steps) balancing gamma PnL vs. transaction costs.
  - Re-evaluate signals daily; roll to next week on expiry.

### Why this is “smart”

- **VRP is priced** across assets (equities, FX, rates) and has been observed in crypto options.
- **Forecasting matters**: Replacing realized variance with a forecast (HAR/Realized GARCH) improves signal purity.
- **Risk-managed exposure** reduces left-tail from short-vol while preserving carry through adaptive sizing and filters.

### Key references (starting points)

- Corsi (2009). A Simple Approximate Long-Memory Model of Realized Volatility (HAR-RV).
- Bollerslev, Tauchen, Zhou (2009). Expected Stock Returns and Variance Risk Premia.
- Bakshi & Kapadia (2003). Delta-Hedged Gains and the Negative Market Volatility Risk Premium.
- Moreira & Muir (2017). Volatility-Managed Portfolios.
- Carr & Wu (2009). Variance Risk Premiums.
- Crypto options/Deribit practitioner notes on DVOL and BTC VRP evidence.

### Minimal module plan (to add here)

- `deribit_api.py`: fetch weekly ATM IV and mid quotes from Deribit public endpoints.
- `rv.py`: compute realized variance from high-frequency BTC data.
- `har_model.py`: fit/forecast HAR-RV.
- `strategy_vrp.py`: generate signals, size positions, and simulate delta-hedged PnL.
- `backtest.py`: run historical backtests with transaction costs and hedging frequency controls.

If you want, I can scaffold these modules next and wire a basic backtest using your `1hbtc.csv` as the underlying and Deribit IV as the implied input.


