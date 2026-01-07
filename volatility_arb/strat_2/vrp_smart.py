import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class VRPConfig:
    rv_horiz_days: int = 30
    ann_factor: float = 365.0
    upper_q: float = 0.7
    lower_q: float = 0.3
    vol_target: float = 0.15  # annualized target vol of PnL
    max_gross: float = 1.0    # cap on |exposure|
    blackout_days: int = 2    # event blackout after large jumps
    jump_z: float = 4.0       # jump filter threshold on returns z-score


def compute_realized_variance(returns: pd.Series, window: int) -> pd.Series:
    rv = (returns.rolling(window).apply(lambda x: np.sum(np.square(x)), raw=True))
    return rv.fillna(method="bfill").fillna(0.0)


def realized_volatility_annualized(returns: pd.Series, ann_factor: float) -> pd.Series:
    return returns.rolling(1).std().fillna(0.0) * np.sqrt(ann_factor)


def compute_har_rv_daily(daily_rv: pd.Series) -> pd.Series:
    # HAR on RV (Corsi, 2009) using simple OLS each day on expanding window
    # y_t = a + b_d*rv_d_{t-1} + b_w*rv_w_{t-1} + b_m*rv_m_{t-1} + eps
    df = pd.DataFrame({"rv_d": daily_rv})
    df["rv_w"] = df["rv_d"].rolling(5).mean()
    df["rv_m"] = df["rv_d"].rolling(22).mean()
    df = df.dropna()

    y = df["rv_d"].shift(-1)
    X = pd.DataFrame({
        "const": 1.0,
        "rv_d": df["rv_d"],
        "rv_w": df["rv_w"],
        "rv_m": df["rv_m"],
    })
    mask = ~(y.isna())
    y = y.loc[mask]
    X = X.loc[mask]

    # expanding OLS coefficients via recursive least squares approximation
    beta = np.zeros((X.shape[1],))
    pred = pd.Series(index=y.index, dtype=float)
    P = np.eye(X.shape[1]) * 1e6
    lam = 1.0  # no forgetting
    for t, idx in enumerate(X.index):
        x_t = X.loc[idx].values.reshape(-1, 1)
        y_t = float(y.loc[idx])
        # RLS update
        P_x = P @ x_t
        gain = P_x / (lam + (x_t.T @ P_x))[0, 0]
        err = y_t - float(beta @ x_t.flatten())
        beta = beta + gain * err
        P = (P - np.outer(gain, x_t.flatten()) @ P) / lam
        pred.loc[idx] = float(beta @ x_t.flatten())

    pred = pred.clip(lower=0.0)
    # align to original index (prediction for next day)
    out = pd.Series(index=daily_rv.index, dtype=float)
    out.loc[pred.index] = pred.values
    return out


def variance_swap_pnl_proxy(daily_sq_ret: pd.Series, strike_var: pd.Series) -> pd.Series:
    # Approximate MTM of a 1-day forward-start variance swap: realized var minus strike
    pnl = daily_sq_ret - strike_var
    return pnl.fillna(0.0)


def regime_filter(iv30_var: pd.Series, rv30_forecast: pd.Series, returns: pd.Series, cfg: VRPConfig) -> pd.Series:
    # Jump filter using return z-scores
    ret_std = returns.rolling(60).std().replace(0, np.nan)
    z = returns / ret_std
    jump = z.abs() > cfg.jump_z

    # Blackout window after jumps
    blackout = jump.rolling(cfg.blackout_days).max().fillna(0).astype(bool)

    # Vol-of-vol regime: elevated when IV variance changes rapidly
    iv_var_chg = iv30_var.pct_change().abs().rolling(5).mean()
    high_vov = iv_var_chg > iv_var_chg.quantile(0.9)

    # Filter out stressed regimes for short-variance; allow only long-variance then
    allow = ~(blackout | high_vov)
    return allow.reindex(rv30_forecast.index).fillna(False)


def build_vrp_strategy(df_price_iv: pd.DataFrame, cfg: Optional[VRPConfig] = None) -> Tuple[pd.DataFrame, dict]:
    cfg = cfg or VRPConfig()
    df = df_price_iv.copy()
    assert {"Close"}.issubset(df.columns), "df must include Close"
    assert ({"iv30", "iv30_var"} & set(df.columns)), "df must include iv30 (annualized vol) or iv30_var (annualized variance)"

    # Build iv30 variance
    if "iv30_var" in df.columns:
        iv30_var = df["iv30_var"].astype(float)
    else:
        iv30 = df["iv30"].astype(float)
        iv30_var = (iv30 / 100.0) ** 2 if iv30.max() > 5 else (iv30 ** 2)

    # Compute daily returns and daily realized variance (sum of intraday not available → use squared daily return)
    close = df["Close"].astype(float)
    daily_ret = close.pct_change().fillna(0.0)
    daily_sq = daily_ret.pow(2)

    # Rolling realized variance for horizon
    rv_h = compute_realized_variance(daily_ret, window=cfg.rv_horiz_days)
    # Annualize RV horizon by scaling (approximate)
    rv_h_ann = (rv_h / cfg.rv_horiz_days) * cfg.ann_factor

    # HAR-RV forecast of 1-day RV; convert to 30d horizon by scaling
    # First convert daily RV (per-day variance) as rv_d = daily_sq * ann_factor
    rv_d = daily_sq * cfg.ann_factor
    har_pred_d = compute_har_rv_daily(rv_d)
    # Forecast 30d variance approx: sum of next 30 daily forecasts (proxy via scaling)
    rv30_forecast_ann = har_pred_d.rolling(cfg.rv_horiz_days).mean()  # smooth to stabilize

    # VRP
    vrp = (iv30_var - rv30_forecast_ann).clip(lower=-np.inf, upper=np.inf)

    # Quantile thresholds
    upper = vrp.quantile(cfg.upper_q)
    lower = vrp.quantile(cfg.lower_q)

    # Regime filter
    allow_short = regime_filter(iv30_var, rv30_forecast_ann, daily_ret, cfg)

    # Signals: short variance when VRP rich and allowed; long variance when deeply cheap or stressed
    signal = pd.Series(0.0, index=df.index)
    signal[(vrp > upper) & allow_short] = -1.0
    signal[vrp < lower] = 1.0

    # Vol targeting sizing on proxy PnL volatility
    # Use rolling std of daily pnl proxy under unit notional to scale exposure
    unit_pnl = variance_swap_pnl_proxy(daily_sq * cfg.ann_factor, iv30_var)
    pnl_std = unit_pnl.rolling(60).std().replace(0, np.nan)
    target_notional = (cfg.vol_target / (pnl_std * np.sqrt(cfg.ann_factor))).clip(upper=cfg.max_gross)
    notional = (signal * target_notional).fillna(0.0)

    # Final PnL
    pnl = notional * unit_pnl

    out = pd.DataFrame({
        "close": close,
        "iv30_var": iv30_var,
        "rv30_forecast_ann": rv30_forecast_ann,
        "vrp": vrp,
        "signal": signal,
        "notional": notional,
        "unit_pnl": unit_pnl,
        "pnl": pnl,
        "equity": pnl.cumsum(),
    })

    # Metrics
    pnl_clean = pnl.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    sr = float(pnl_clean.mean() / (pnl_clean.std() + 1e-12) * np.sqrt(cfg.ann_factor)) if pnl_clean.std() > 0 else 0.0
    mdd = float((out["equity"] - out["equity"].cummax()).min())
    ann = float(pnl_clean.mean() * cfg.ann_factor)
    stats = {"sharpe": sr, "annual_return": ann, "max_drawdown": mdd}
    return out, stats


def load_price_iv_csv(price_csv: str, iv_csv: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(price_csv)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])  # optional
        df = df.sort_values("time")
    if iv_csv:
        iv = pd.read_csv(iv_csv)
        if "time" in iv.columns:
            iv["time"] = pd.to_datetime(iv["time"])  # optional
            iv = iv.sort_values("time")
        df = pd.merge_asof(df, iv, on="time")
    if "time" in df.columns:
        df = df.set_index("time")
    return df


if __name__ == "__main__":
    # Example usage (replace paths)
    price_csv = os.environ.get("VRP_PRICE_CSV", "")
    iv_csv = os.environ.get("VRP_IV_CSV", "")
    if price_csv and os.path.exists(price_csv):
        df = load_price_iv_csv(price_csv, iv_csv or None)
        ts, stats = build_vrp_strategy(df)
        print({k: round(v, 4) for k, v in stats.items()})
        out_dir = os.path.dirname(price_csv) or "."
        out_path = os.path.join(out_dir, "vrp_timeseries.csv")
        # Save with datetime index if available
        if isinstance(ts.index, pd.DatetimeIndex):
            ts.to_csv(out_path, index_label="datetime")
        else:
            ts.to_csv(out_path, index=False)
        # Also export a minimal factor CSV for Backtest/backtest integration
        factor_path = os.path.join(out_dir, "vrp_factor_for_backtester.csv")
        factor_df = ts[["close", "vrp"]].copy()
        if isinstance(ts.index, pd.DatetimeIndex):
            factor_df = factor_df.rename_axis("datetime").reset_index()
        factor_df.to_csv(factor_path, index=False)
        print(f"Saved {out_path}")
        print(f"Saved {factor_path}")
    else:
        print("Set VRP_PRICE_CSV to your price CSV (with Close, optional time). Optionally set VRP_IV_CSV (with iv30 or iv30_var).")


