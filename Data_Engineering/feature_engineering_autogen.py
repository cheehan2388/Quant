import os
import json
import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


def load_data(file_path: str = '../Data/All_data/merged/BTC_merged_V.csv') -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    df = df.ffill().bfill()
    return df


def calculate_future_returns(
    df: pd.DataFrame,
    close_dir: str = '../Data/close',
    shifts: List[int] = [-2, -8, -24]
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    future_returns: Dict[str, pd.Series] = {}
    close_path = os.path.join(close_dir, 'Binance_1Hour_BTCUSD_T.csv')
    close_df = pd.read_csv(close_path)
    close_df['datetime'] = pd.to_datetime(close_df['datetime'])
    merged = pd.merge(df, close_df[['datetime', 'Close']], on='datetime', how='left')
    for shift in shifts:
        col = f'future_return_{abs(shift)}'
        merged[col] = merged['Close'].pct_change(periods=abs(shift)).shift(shift)
        future_returns[col] = merged[col]
    for shift in shifts:
        col = f'future_return_{abs(shift)}'
        df[col] = future_returns[col]
    return df, future_returns


def _rolling_percent_rank(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.5
    last = x[-1]
    count_le = np.count_nonzero(x <= last)
    return (count_le - 1) / max(1, x.size - 1)


def _rolling_entropy(x: np.ndarray, bins: int = 10) -> float:
    if x.size < 3:
        return 0.0
    counts, _ = np.histogram(x[~np.isnan(x)], bins=bins)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    h = -(p * np.log2(p)).sum()
    max_h = math.log2(p.size) if p.size > 0 else 1.0
    return float(h / max_h) if max_h > 0 else 0.0


def _rolling_fft_low_energy_ratio(x: np.ndarray, low_bins: int = 3) -> float:
    n = x.size
    if n < 4:
        return 0.0
    y = x - np.nanmean(x)
    spec = np.fft.rfft(y)
    power = (spec.real ** 2 + spec.imag ** 2)
    power[0] = 0.0
    total = power.sum()
    if total <= 0:
        return 0.0
    k = min(low_bins, power.size - 1)
    low_energy = power[1 : 1 + k].sum()
    return float(low_energy / total)


def _rolling_slope(x: np.ndarray) -> float:
    n = x.size
    if n < 2:
        return 0.0
    t = np.arange(n)
    t_mean = t.mean()
    x_mean = np.nanmean(x)
    denom = np.sum((t - t_mean) ** 2)
    if denom == 0:
        return 0.0
    num = np.nansum((t - t_mean) * (x - x_mean))
    return float(num / denom)


def _rolling_autocorr(x: np.ndarray, lag: int) -> float:
    n = x.size
    if n <= lag or lag <= 0:
        return 0.0
    a = x[:-lag]
    b = x[lag:]
    if np.nanstd(a) == 0 or np.nanstd(b) == 0:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    if np.isnan(c):
        return 0.0
    return float(c)


def apply_rolling_features(
    series: pd.Series,
    windows: List[int],
    autocorr_lags: List[int],
    entropy_bins: int,
    fft_low_bins: int
) -> Dict[str, pd.Series]:
    features: Dict[str, pd.Series] = {}
    for w in windows:
        roll = series.rolling(window=w, min_periods=1)
        features[f'{series.name}_rolling_mean_{w}'] = roll.mean()
        features[f'{series.name}_rolling_std_{w}'] = roll.std()
        features[f'{series.name}_rolling_skew_{w}'] = roll.skew()
        features[f'{series.name}_rolling_kurt_{w}'] = roll.kurt()
        features[f'{series.name}_rolling_q25_{w}'] = roll.quantile(0.25)
        features[f'{series.name}_rolling_q75_{w}'] = roll.quantile(0.75)
        features[f'{series.name}_rolling_iqr_{w}'] = features[f'{series.name}_rolling_q75_{w}'] - features[f'{series.name}_rolling_q25_{w}']
        features[f'{series.name}_percent_rank_{w}'] = roll.apply(_rolling_percent_rank, raw=True)
        features[f'{series.name}_trend_slope_{w}'] = roll.apply(_rolling_slope, raw=True)
        features[f'{series.name}_entropy_b{entropy_bins}_{w}'] = roll.apply(lambda x: _rolling_entropy(x, bins=entropy_bins), raw=True)
        features[f'{series.name}_fft_low_energy_ratio_k{fft_low_bins}_{w}'] = roll.apply(lambda x: _rolling_fft_low_energy_ratio(x, low_bins=fft_low_bins), raw=True)
        ewm = series.ewm(span=w, adjust=False)
        features[f'{series.name}_ewm_mean_{w}'] = ewm.mean()
        features[f'{series.name}_ewm_std_{w}'] = ewm.std()
        for lag in autocorr_lags:
            features[f'{series.name}_autocorr_lag{lag}_{w}'] = roll.apply(lambda x, l=lag: _rolling_autocorr(x, l), raw=True)
    return features


def generate_creative_features(
    df: pd.DataFrame,
    windows: List[int] = [12, 24, 52],
    lags: List[int] = [1, 2, 4, 8, 12],
    return_periods: List[int] = [1, 2, 4, 8, 12],
    autocorr_lags: List[int] = [1, 3, 6],
    entropy_bins: int = 10,
    fft_low_bins: int = 3
) -> Tuple[Dict[str, pd.Series], Dict[str, dict]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['datetime'] + [c for c in df.columns if 'future_return' in c]
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    transformed: Dict[str, pd.Series] = {}
    metadata: Dict[str, dict] = {}
    for col in feature_cols:
        s = df[col].astype(float)
        for p in return_periods:
            name = f'{col}_ret_{p}'
            try:
                ser = s.pct_change(p)
                transformed[name] = ser
                metadata[name] = {'base_column': col, 'transformation': f'pct_change_{p}'}
            except Exception:
                pass
        for l in lags:
            name = f'{col}_lag_{l}'
            try:
                ser = s.shift(l)
                transformed[name] = ser
                metadata[name] = {'base_column': col, 'transformation': f'lag_{l}'}
            except Exception:
                pass
        try:
            roll_feats = apply_rolling_features(
                series=s,
                windows=windows,
                autocorr_lags=autocorr_lags,
                entropy_bins=entropy_bins,
                fft_low_bins=fft_low_bins,
            )
            for k, v in roll_feats.items():
                transformed[k] = v
                parts = k.split('_')
                metadata[k] = {
                    'base_column': col,
                    'transformation': '_'.join(parts[len(col.split('_')):])
                }
        except Exception:
            pass
    return transformed, metadata


def calculate_ic(feature_series: pd.Series, target_series: pd.Series) -> float:
    valid = ~(pd.isna(feature_series) | pd.isna(target_series))
    if valid.sum() < 10:
        return 0.0
    a = feature_series[valid]
    b = target_series[valid]
    corr = a.corr(b, method='spearman')
    return float(corr) if not pd.isna(corr) else 0.0


def evaluate_feature_stability(
    feature: pd.Series,
    target: pd.Series,
    folds: int = 5
) -> Dict[str, float]:
    valid = ~(pd.isna(feature) | pd.isna(target))
    idx = feature.index[valid]
    if idx.size < max(50, folds * 10):
        return {'folds': 0, 'mean_ic': 0.0, 'std_ic': 0.0, 'sign_consistency': 0.0}
    indices = idx.values
    splits = np.array_split(indices, folds)
    ic_vals: List[float] = []
    for sp in splits:
        if sp.size < 10:
            continue
        ic = feature.loc[sp].corr(target.loc[sp], method='spearman')
        if pd.isna(ic):
            ic = 0.0
        ic_vals.append(float(ic))
    if not ic_vals:
        return {'folds': 0, 'mean_ic': 0.0, 'std_ic': 0.0, 'sign_consistency': 0.0}
    mean_ic = float(np.mean(ic_vals))
    std_ic = float(np.std(ic_vals))
    sign_consistency = float(np.mean(np.sign(ic_vals) == np.sign(mean_ic)))
    return {'folds': len(ic_vals), 'mean_ic': mean_ic, 'std_ic': std_ic, 'sign_consistency': sign_consistency}


def calculate_ics_for_all_features(
    df: pd.DataFrame,
    transformed: Dict[str, pd.Series],
    metadata: Dict[str, dict],
    shifts: List[int] = [-2, -8, -24],
    stability_folds: int = 5
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    future_cols = [f'future_return_{abs(s)}' for s in shifts]
    for name, series in transformed.items():
        ic_values: Dict[str, float] = {}
        for shift in shifts:
            col = f'future_return_{abs(shift)}'
            if col in df.columns:
                ic = calculate_ic(series, df[col])
            else:
                ic = 0.0
            ic_values[f'shift_{shift}'] = ic
        abs_ic = {k: abs(v) for k, v in ic_values.items()}
        best_key = max(abs_ic.keys(), key=abs_ic.get)
        max_ic = ic_values[best_key]
        best_target = df[best_key.replace('shift_', 'future_return_')]
        stability = evaluate_feature_stability(series, best_target, folds=stability_folds)
        metadata[name] = {
            **metadata.get(name, {}),
            'ic_values': ic_values,
            'max_ic': float(max_ic),
            'best_shift': best_key,
            'abs_max_ic': float(abs(max_ic)),
            'stability': stability,
        }
    return df, metadata


def generate_report(
    feature_metadata: Dict[str, dict],
    output_path: str = 'autogen_feature_report.json',
    top_k: int = 50
) -> dict:
    sorted_items = sorted(feature_metadata.items(), key=lambda x: (abs(x[1]['max_ic']), x[1]['stability'].get('sign_consistency', 0.0)), reverse=True)
    report = {
        'summary': {
            'total_features': len(feature_metadata),
            'top_k': top_k,
            'top_features': [n for n, _ in sorted_items[:top_k]],
        },
        'features': {}
    }
    for name, meta in sorted_items:
        report['features'][name] = {
            'base_column': meta.get('base_column'),
            'transformation': meta.get('transformation'),
            'ic_values': meta.get('ic_values', {}),
            'max_ic': meta.get('max_ic'),
            'abs_max_ic': meta.get('abs_max_ic'),
            'best_shift': meta.get('best_shift'),
            'stability': meta.get('stability', {})
        }
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Report saved to {output_path}')
    return report


def prune_highly_correlated_features(
    df_features: pd.DataFrame,
    ranked_feature_names: List[str],
    corr_threshold: float = 0.98
) -> List[str]:
    selected: List[str] = []
    for name in ranked_feature_names:
        s = df_features[name]
        if s.std(skipna=True) == 0 or s.isna().all():
            continue
        keep = True
        for kept in selected:
            c = s.corr(df_features[kept])
            if pd.isna(c):
                c = 0.0
            if abs(c) >= corr_threshold:
                keep = False
                break
        if keep:
            selected.append(name)
    return selected


def save_selected_features(
    df: pd.DataFrame,
    transformed: Dict[str, pd.Series],
    feature_metadata: Dict[str, dict],
    output_csv: str = 'autogen_features.csv',
    top_k: int = 200,
    corr_threshold: float = 0.98
) -> List[str]:
    items = sorted(feature_metadata.items(), key=lambda x: (abs(x[1]['max_ic']), x[1]['stability'].get('sign_consistency', 0.0)), reverse=True)
    ranked = [n for n, _ in items]
    top = ranked[: max(top_k * 2, 50)]
    feat_df = pd.DataFrame({name: transformed[name] for name in top})
    selected = prune_highly_correlated_features(feat_df, top, corr_threshold=corr_threshold)
    selected = selected[:top_k]
    out_df = pd.concat([df[['datetime']], feat_df[selected]], axis=1)
    for col in [c for c in df.columns if c.startswith('future_return_')]:
        out_df[col] = df[col]
    out_df.to_csv(output_csv, index=False)
    print(f'Selected {len(selected)} features saved to {output_csv}')
    return selected


def main():
    print('=== Creative Auto Feature Engineering Pipeline ===')
    df = load_data()
    print(f'Loaded data: {df.shape}')
    df, _ = calculate_future_returns(df)
    print('Future returns computed')
    print('Generating creative features...')
    transformed, metadata = generate_creative_features(
        df,
        windows=[12, 24, 52],
        lags=[1, 2, 4, 8, 12],
        return_periods=[1, 2, 4, 8, 12],
        autocorr_lags=[1, 3, 6],
        entropy_bins=10,
        fft_low_bins=3,
    )
    print(f'Generated {len(transformed)} features')
    print('Scoring features (IC and stability)...')
    df, metadata = calculate_ics_for_all_features(
        df,
        transformed,
        metadata,
        shifts=[-2, -8, -24],
        stability_folds=5,
    )
    print('Building report...')
    generate_report(metadata, output_path='autogen_feature_report.json', top_k=50)
    print('Saving top features...')
    selected = save_selected_features(
        df,
        transformed,
        metadata,
        output_csv='autogen_features.csv',
        top_k=200,
        corr_threshold=0.98,
    )
    print('Top 20 features:')
    for i, name in enumerate(selected[:20], 1):
        meta = metadata[name]
        print(f"{i:2d}. {name} | max_ic={meta['max_ic']:.5f} | best={meta['best_shift']} | stab={meta['stability'].get('sign_consistency', 0.0):.2f}")
    print('Pipeline completed successfully!')


if __name__ == '__main__':
    main()


