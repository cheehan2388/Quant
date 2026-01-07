from dataclasses import dataclass
from typing import Optional

from .config import RISK


@dataclass
class GammaCheckResult:
    ok: bool
    gamma_usd_per_1pct: float
    limit: float
    message: str


def check_gamma_bound(gamma: float, spot: float) -> GammaCheckResult:
    """Approximate gamma exposure in USD per 1% move: gamma * spot^2 * 1%.

    gamma is per underlying unit. This is a simplified, conservative metric.
    """
    gamma_usd_per_1pct = gamma * (spot ** 2) * 0.01
    ok = gamma_usd_per_1pct <= RISK.max_gamma_usd_per_1pct
    return GammaCheckResult(
        ok=ok,
        gamma_usd_per_1pct=gamma_usd_per_1pct,
        limit=RISK.max_gamma_usd_per_1pct,
        message=(
            "Gamma within bound" if ok else f"Gamma risk too high: {gamma_usd_per_1pct:.2f} > {RISK.max_gamma_usd_per_1pct:.2f}"
        ),
    )


def iv_jump_allowed(current_iv_bps_jump: Optional[float]) -> bool:
    if current_iv_bps_jump is None:
        return True
    return current_iv_bps_jump <= RISK.max_iv_jump_bps


def skew_shift_allowed(current_skew_bps_shift: Optional[float]) -> bool:
    if current_skew_bps_shift is None:
        return True
    return current_skew_bps_shift <= RISK.max_skew_shift_bps


