import math
from dataclasses import dataclass


@dataclass
class BlackScholesInputs:
    spot: float
    strike: float
    vol: float  # annualized volatility (decimal, e.g., 0.8 for 80%)
    t_years: float
    rate: float = 0.0


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1(inputs: BlackScholesInputs) -> float:
    s, k, v, t, r = inputs.spot, inputs.strike, max(inputs.vol, 1e-8), max(inputs.t_years, 1e-8), inputs.rate
    return (math.log(s / k) + (r + 0.5 * v * v) * t) / (v * math.sqrt(t))


def call_delta(inputs: BlackScholesInputs) -> float:
    return _norm_cdf(_d1(inputs))


def put_delta(inputs: BlackScholesInputs) -> float:
    return call_delta(inputs) - 1.0


def gamma(inputs: BlackScholesInputs) -> float:
    d1 = _d1(inputs)
    v = max(inputs.vol, 1e-8)
    t = max(inputs.t_years, 1e-8)
    return _norm_pdf(d1) / (inputs.spot * v * math.sqrt(t))


