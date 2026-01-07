from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import config, utils


@dataclass
class VrpSignalConfig:
    vega_cap: float = config.DEFAULT_VEGA_CAP
    scale_k: float = 1.0
    vol_of_vol_window: int = 30
    suppress_short_vol_on_extreme_skew: bool = True


def compute_vol_of_vol(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).std().replace(0.0, config.SMALL_EPS)


def build_vrp_signal(
    forecast_ann_vol: pd.Series,
    implied_ann_vol: pd.Series,
    skew_indicator: Optional[pd.Series] = None,
    cfg: VrpSignalConfig = VrpSignalConfig(),
) -> pd.DataFrame:
    f_vol, iv = utils.safe_align(forecast_ann_vol, implied_ann_vol)
    vrp = (iv - f_vol).rename("vrp")
    volvol = compute_vol_of_vol(iv, window=cfg.vol_of_vol_window).reindex(vrp.index)
    raw_vega_target = (cfg.scale_k * vrp / volvol).clip(lower=-cfg.vega_cap, upper=cfg.vega_cap)

    if cfg.suppress_short_vol_on_extreme_skew and skew_indicator is not None:
        _, skew = utils.safe_align(vrp, skew_indicator)
        # Simple rule: when skew is in most negative decile, suppress short vol (set negative targets to 0)
        threshold = skew.rolling(window=252, min_periods=50).quantile(0.1)
        suppress_mask = (skew <= threshold).reindex(raw_vega_target.index).fillna(False)
        raw_vega_target = raw_vega_target.where(~(suppress_mask & (raw_vega_target < 0.0)), 0.0)

    out = pd.DataFrame({
        "forecast_vol": f_vol,
        "implied_vol": iv,
        "vrp": vrp,
        "vol_of_vol": volvol,
        "vega_target": raw_vega_target,
    })
    return out


