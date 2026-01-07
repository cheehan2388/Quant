import numpy as np
import pandas as pd
from typing import Tuple


def prepare_har_features(daily_rv: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Create HAR-RV features from daily realized variance series.

    HAR-RV: RV_{t+1} = c + b_d * RV_t + b_w * RV_{t:t-4} + b_m * RV_{t:t-21} + e
    We use simple averages for weekly and monthly components.
    Input daily_rv should be a Series indexed by date with annualized variance.
    """
    rv = daily_rv.copy()
    rv = rv.sort_index()

    rv_d = rv.shift(0)
    rv_w = rv.rolling(5).mean().shift(0)
    rv_m = rv.rolling(22).mean().shift(0)

    X = pd.DataFrame({
        'const': 1.0,
        'rv_d': rv_d,
        'rv_w': rv_w,
        'rv_m': rv_m,
    }).dropna()
    # predict next-day RV
    y = rv.shift(-1).loc[X.index]
    X = X.loc[y.index]
    return X, y


def fit_ols(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Fit OLS using normal equations with ridge regularization fallback."""
    X_mat = X.to_numpy(dtype=float)
    y_vec = y.to_numpy(dtype=float)
    # Add tiny ridge for numerical stability
    ridge = 1e-8 * np.eye(X_mat.shape[1])
    beta = np.linalg.solve(X_mat.T @ X_mat + ridge, X_mat.T @ y_vec)
    return beta


def predict_next(X: pd.DataFrame, beta: np.ndarray) -> pd.Series:
    return pd.Series(X.to_numpy(dtype=float) @ beta, index=X.index)


def rolling_har_forecast(daily_rv: pd.Series, min_samples: int = 250) -> pd.Series:
    """Produce one-step-ahead rolling HAR-RV forecasts.

    For each t, fit on data up to t-1 and predict RV at t using features at t-1.
    Returns a Series aligned to dates where forecast is available.
    """
    rv = daily_rv.sort_index()
    X_all, y_all = prepare_har_features(rv)
    forecasts = []
    forecast_index = []
    for i in range(min_samples, len(X_all)):
        X_train = X_all.iloc[:i]
        y_train = y_all.iloc[:i]
        beta = fit_ols(X_train, y_train)
        x_t = X_all.iloc[i:i+1]
        pred = float(x_t.to_numpy() @ beta)
        forecasts.append(pred)
        forecast_index.append(X_all.index[i])
    return pd.Series(forecasts, index=pd.Index(forecast_index, name=rv.index.name))


