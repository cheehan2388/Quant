import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import  lib .model as md
import lib .bt_strat as bt_strat
#param_zone
import json
import numpy as np

class Int32Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        return super().default(obj)


date_col      = 'datetime'
inter = 'hour' # give an initial inter 
price_global = 'close'
data = 'change_open_taker_ratio'
def delay_test(positions,price, strat, sr_multi):
    
    price = pd.read_csv("./../Data/CG_BTC_hour_open-interest__future_taker_buy_sel.csv")
    pos_shift = positions.shift(1).fillna(0)
    trade_signal = (positions != pos_shift).astype(int)  # 1 = 今天开/平仓，0 = 保持不变

    # 2) 普通 P&L、夏普、DD 等
    ret   = price.pct_change().fillna(0)
    pnl   = pos_shift * ret
    equity = pnl.cumsum()
    dd_pct = equity - equity.cummax()

    # 3) 统计指标：这里保留“笔数”做汇总，但用新的键名 num_trades
    num_trades = int(trade_signal.sum())

    sharpe     = pnl.mean() / pnl.std() * np.sqrt(365 * sr_multi)
    ar         = pnl.mean() * 365
    cr         = pnl.mean() / abs(dd_pct.min())
    trade_per_interval = num_trades/len(ret)
    fitness    = sharpe*np.sqrt(abs(pnl)/ max(0.03,trade_per_interval) )
    sortino    = sortino_ratio(pnl)
    return {
        'pnl':            pnl,
        'equity':         equity,
        'trade_signal':   trade_signal,
        'trades':     num_trades,
        'max_drawdown_pct': dd_pct.min(),
        'anualized_return': ar,
        'sharpe':         sharpe,
        'calmar_ratio':   cr,
        'total_return':   equity.iloc[-1] ,
        'trade_per_interval': trade_per_interval ,
        'fitness' : fitness,
        'sortino' : sortino
    }


def sortino_ratio(returns, rf=0.0, periods_per_year=365):
    excess = returns - rf
    downside_std = excess[excess < 0].std(ddof=0)
    if downside_std == 0:
        return np.inf if excess.mean() > 0 else -np.inf
    return (excess.mean() / downside_std) * np.sqrt(periods_per_year)

# -------------------------
# 1. Data Loader & Splitter
# -------------------------
def load_and_split_data(
    df,
    date_col: str = date_col ,
    backtest_years: int = 1,
    forward_years: int = 1
):

    start = df.index.min()
    back_lens = len(df)*(6/10)
    val_lens = len(df)*(2/10)
            
    # df_back = df.loc[:].copy()
    # # df_val  = df.loc[back_lens:val_lens].copy() 
    # df_fwd  = df.loc[:]
    back_end = start + pd.DateOffset(years=backtest_years)
    forward_end = back_end + pd.DateOffset(years=forward_years)

    
    df_back = df.loc[:back_end].copy()
    df_fwd  = df.loc[back_end:].copy()
    # df_fwd  = df.loc[:forward_end].copy()
    return df_back, df_fwd



def plot_parameter_heatmaps(df, model, strategy, output_dir, interval, data_nam):
    
    fact = data_nam

    sub = df[(df['model']==model) & (df['strategy']==strategy)].copy()
    if sub.empty:
        return

    # Normalize windows to tuple
    sub['win_tup'] = sub['windows'].apply(lambda x: x if isinstance(x, tuple) else (x,))
    # Determine number of window dims
    win_dims = len(sub['win_tup'].iloc[0])
    # Create separate columns for each window dimension
    for i in range(win_dims):
        sub[f'w{i+1}'] = sub['win_tup'].apply(lambda t: t[i])
    # Ensure threshold column

    sub['thr'] = sub['threshold'].astype(float)

    # Collect dimension names
    dims = [f'w{i+1}' for i in range(win_dims)] + ['thr']
    total_dims = len(dims)

    # Helper for plotting a heatmap of dim_x vs dim_y, aggregating over others
    def _plot_heat(dim_x, dim_y, title_suffix, fname_suffix):
        pivot = sub.pivot_table(
            index=dim_y, columns=dim_x, values='train_sharpe', aggfunc='mean'
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot.T,  # transpose so x-axis is dim_x, y-axis is dim_y
            annot=True, fmt=".2f",
            cmap="RdYlGn", center=0,
            linewidths=0.5, linecolor='white',
            cbar_kws={"label": "Train Sharpe"}
        )
        plt.title(f"{model} | {strategy} — Sharpe {title_suffix}")
        plt.xlabel(dim_x)
        plt.ylabel(dim_y)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{fact}_{model}_{strategy}_{interval}_{fname_suffix}.png"))
        plt.close()

    # Depending on number of dims
    if total_dims == 1:
        # Only threshold
        dim = dims[0]
        plt.figure(figsize=(8,6))
        sns.lineplot(x=sub[dim], y=sub['train_sharpe'], marker='o')
        plt.axhline(0, color='gray', linestyle='--')
        plt.title(f"{model} | {strategy} — Sharpe vs {dim}")
        plt.xlabel(dim); plt.ylabel("Train Sharpe")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{fact}_{model}_{strategy}_{interval}_sharpe_vs_{dim}.png"))
        plt.close()
    elif total_dims == 2:
        # Two dims: window & threshold (or two windows)
        dim_x, dim_y = dims
        _plot_heat(dim_x, dim_y, f"{fact}_{dim_x} vs {dim_y}", f"heat_{dim_x}_vs_{dim_y}")
    elif total_dims == 3:
        # Three dims: plot all three pairs
        w1, w2, thr = dims
        # w1 vs w2 (max over thr)
        _plot_heat(w1, w2, f"{fact}_{w1} vs {w2}", f"heat_{w1}_vs_{w2}_max_{thr}")
        # w1 vs thr (max over w2)
        _plot_heat(w1, thr, f"{fact}_{w1} vs {thr}", f"heat_{w1}_vs_{thr}_max_{w2}")
        # w2 vs thr (max over w1)
        _plot_heat(w2, thr, f"{fact}_{w2} vs {thr}", f"heat_{w2}_vs_{thr}_max_{w1}")
    else:
        # More than 3 dims not supported here
        pass

# -------------------------
# 4. Backtester & Metrics
# -------------------------
def backtest(price: pd.Series, positions: pd.Series, sr_multiplier: float, fee : float):

    pos_shift = positions.shift(1).fillna(0)
    trade_signal = (positions - pos_shift).abs().astype(int) # 1 = 今天开/平仓，0 = 保持不变, 因爲是今天高於thres我會在他的clos開positio ,所以是今天不等於明天就是開了一個trad.
    # 2) 普通 P&L、夏普、DD 等
    ret   = price.pct_change().fillna(0)
    pnl = pos_shift * ret - trade_signal * fee
    equity = pnl.cumsum()
    dd_pct = equity - equity.cummax()

    # 3) 统计指标：这里保留“笔数”做汇总，但用新的键名 num_trades
    num_trades = int(trade_signal.sum())

    sharpe     = pnl.mean() / pnl.std() * np.sqrt(365 * sr_multiplier)
    ar         = pnl.mean() * 365
    cr = pnl.mean()* 365 / abs(dd_pct.min())
    trade_per_interval = num_trades/len(ret)
    fitness    = sharpe*np.sqrt(abs(pnl)/ max(0.03,trade_per_interval) )
    sortino    = sortino_ratio(pnl)
    return {
        'pnl':            pnl,
        'equity':         equity,
        'trade_signal':   trade_signal,
        'trades':     num_trades,
        'max_drawdown_pct': dd_pct.min(),
        'anualized_return': ar,
        'sharpe':         sharpe,
        'calmar_ratio':   cr,
        'total_return':   equity.iloc[-1] ,
        'trade_per_interval': trade_per_interval ,
        'fitness' : fitness,
        'sortino' : sortino
    }



def permutation_test(backtested_df, row ,strat_funcs :dict, models_funcs : dict ,column_name : str, num_perm = 500, seed = 20,sr_multiplier = None ): 
    rng = np.random.default_rng(seed) 
    model = row['model']
    strategy = row['strategy']
    window = row['windows']
    threshold = row['threshold']
    sharpe_or = row['full_date_stats']['sharpe']
    
    if isinstance(window,(list, tuple)):
        window_args = tuple(window)
    
    else :
        window_args =(window,)
    
    sig = models_funcs[model](backtested_df[column_name],*window_args)

    pos = strat_funcs[strategy](sig, threshold)
    daily_change = backtested_df[price_global].pct_change().fillna(0)
    pnl_daily = pos.shift(1).fillna(0)*daily_change
    # sharpe_or = pnl_daily.mean() / pnl_daily.std()*np.sqrt(sr_multiplier*365)
    
    n = len(daily_change)
    count = 0
    for _ in range(num_perm):
        # segment into random‐length blocks
        blocks = []
        i = 0
        while i < n:
            L = rng.integers(10, 501)  # block lengths between 10 and 500
            blocks.append(daily_change[i : min(n, i + L)])
            i += L
        rng.shuffle(blocks)
        # recombine shuffled blocks
        shuffled = np.concatenate(blocks)[:n]
        pnl_shuf = pos * shuffled
        shuf_sh = (pnl_shuf.mean() / pnl_shuf.std()) * np.sqrt(365* sr_multiplier)
        if shuf_sh >= sharpe_or:
            count += 1

    p_value = count / num_perm
    return  p_value
    
# -------------------------
# 6. Filters
# -------------------------
def pass_filters(stats, n_periods):
    if stats['sharpe'] < 2.0: return False
    if stats['max_drawdown_pct'] < -50: return False
    if stats['trades'] <= 0.03 * n_periods: return False
    return True

def forward_consistent(train, fwd):
    r = fwd['sharpe'] / train['sharpe']
    return 0.9 <= r <= 1.1

# -------------------------
# 7. Main Pipeline
# -------------------------
def run_pipeline(data_file, output_dir:str,
                 fee:float ,interval:str, column_name : str  ,sr_multiplier = None ,
                 heatmap = True):
    
    df = pd.read_csv(data_file, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df.set_index(date_col, inplace=True)

    fact = column_name
    print("→ Loading data and splitting…")
    os.makedirs(output_dir, exist_ok=True)#創立文件
    
    
    df_back, df_fwd = load_and_split_data(df)
    factor_back = df_back[column_name]
    factor_fwd  = df_fwd[column_name]

    df_full = pd.concat([df_back, df_fwd])
    price_back  = df_back[price_global]
    price_fwd   = df_fwd[price_global]

    strat_funcs = {
        'trend':        bt_strat.positions_trend,
        'trend_close':  bt_strat.positions_trend_close,
        'trend_close_zero': bt_strat.positions_trend_zero,
        'mr':           bt_strat.positions_mr,
        'mr_close':     bt_strat.positions_mr_close,
        'mean_reversion_zero' : bt_strat.positions_mr_mean_zero,
    }
    models_funcs = {
    # "ewma_diff": md.compute_ewma_diff,
    "zscore": md.compute_zscore,
    "min_max_scaling": md.compute_minmax,
    "ma_diff": md.compute_ma_diff,
    # "percentile": md.compute_percentile
    }

    results = []
    all_results =[]
    # --- define ranges once here ---
    # EWMA spans
    fast_spans = np.arange(10, 200, 10)
    slow_spans = np.arange(50, 500, 50)
    # rolling windows
    z_windows   = np.arange(20, 500, 10)
    mm_windows  = np.arange(10, 490, 20)
    ma_short    = np.arange(10, 100, 10)
    ma_long     = np.arange(50, 500, 50)
    # percentile windows
    perc_windows     = np.arange(20, 500, 10)
    perc_thresholds  = [(p, 1-p) for p in [0.1, 0.15, 0.2, 0.25]]
    # thresholds
    thr_trend    = np.arange(0.00, 0.05, 0.01)
    thr_zscore   = np.arange(0.0, 3.0, 0.15)
    thr_mm       = np.arange(0.1, 1.0, 0.1)
    thr_mad      = np.arange(0.00, 0.05, 0.01)

    #alpha_pool
    a_pool =np.arange(0.0, 1,0.05)
    # --- grid search for each model ---
    models = [
    #    ('ewma_diff', md.compute_ewma_diff, [(f,s,alpha) for f in fast_spans for s in slow_spans if f<s for alpha in a_pool], thr_trend),
       ('zscore',    md.compute_zscore,   z_windows,   thr_zscore),
       ('min_max_scaling', md.compute_minmax, mm_windows, thr_mm),
      ('ma_double',  md.compute_ma_double,    [(s,l) for s in ma_short for l in ma_long if s<l], thr_mad),
    
    
    ]
    for name, func, param_list, thr_list in models:
        print(f"→ Scanning {name} ({len(param_list)} param sets)…")
        for param in param_list:
            #bt 與 forward
            
            sig_b = func(factor_back, *param) if isinstance(param, (list,tuple)) else func(factor_back, param)
            sig_f = func(factor_fwd,  *param) if isinstance(param, (list,tuple)) else func(factor_fwd, param)
            sig_all = func(df[column_name],  *param) if isinstance(param, (list,tuple)) else func(df[column_name], param)
            
            for strat, strat_fn in strat_funcs.items(): # mr(strat) : func()(strat_fn)
                
                for thr in thr_list:
                    pos_b = strat_fn(sig_b, thr)
                    stats_b = backtest(price_back, pos_b,sr_multiplier = sr_multiplier, fee = fee)
                    
                    all_results.append({
                        'factor': fact,
                        'model':  name,
                        'strategy': strat,
                        'windows': param,
                        'threshold': thr,
                        'train_sharpe': stats_b['sharpe']
                    })
                    #first_check
                    #if not pass_filters(stats_b, len(price_back)): continue
                    pos_f = strat_fn(sig_f, thr)
                    stats_f = backtest(price_fwd, pos_f,sr_multiplier =sr_multiplier ,fee = fee)
                    #我要做把已經backtest好的，經過我達到的要求后， 自己算他的grad 就是 [1] -[0]/(如果橫軸是rolling1，值軸是rolling2, 如果我先從橫軸走那麽我的gradient就是rolling1 的值)，如果他的上下左右gradient都是很小他就會被選中進入forward.
                    #forward chec
                    #if not forward_consistent(stats_b, stats_f): continue
                    #included forward and backtest 
                    pos_all = strat_fn(sig_all,thr)
                    full_date_stats = backtest(df[price_global],positions= pos_all,sr_multiplier = sr_multiplier,fee = fee) 
                    results.append({
                        'factor':        f'{fact}',
                        'model' :        name,
                        'windows':       param,
                        'strategy':      strat,
                        'threshold':     thr,
                        'train_stats':   stats_b,
                        'forward_stats': stats_f,
                        'full_date_stats': full_date_stats,
                        'positions':     pd.concat([pos_b, pos_f])
                        
                        })
    # 1) Produce one file containing all backtests that passed your filters
    backtests_out = []
    for r in results:
        bt = r['train_stats']
        backtests_out.append({
            "factor_name": fact,
            "backtest_mode": r["strategy"],
            "start_time": df_back.index[0].strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": df_back.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "fees": fee,
            "interval": interval,
            "model": r["model"],
            "window": str(r["windows"]),  # always serialize as string
            "multiplier": float(r["threshold"]),  # threshold = your “multiplier”
            "num_of_trades": int(bt["trades"]),
            "TR": float(bt["total_return"]),
            "MDD": float(bt["max_drawdown_pct"]),
            "SR": float(bt["sharpe"]),
            "trade_per_interval": float(bt["trade_per_interval"])
        })

    with open(os.path.join(output_dir, f'{fact}_{interval}.json'), "w") as f:
        json.dump({"backtests": backtests_out}, f, indent=4, cls=Int32Encoder)

    forward_out = []
    for r in results:
        bt = r['forward_stats']
        forward_out.append({
            "factor_name": fact,
            "backtest_mode": r["strategy"],
            "start_time": df_fwd.index[0].strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": df_fwd.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "fees": fee,
            "interval": interval,
            "model": r["model"],
            "window": str(r["windows"]),  # always serialize as string
            "multiplier": float(r["threshold"]),  # threshold = your “multiplier”
            "num_of_trades": int(bt["trades"]),
            "TR": float(bt["total_return"]),
            "MDD": float(bt["max_drawdown_pct"]),
            "SR": float(bt["sharpe"]),
            "trade_per_interval": float(bt["trade_per_interval"])
        })

    with open(os.path.join(output_dir, f'{fact}_{interval}.json'), "w") as f:
        json.dump({"forward": forward_out}, f, indent=4, cls=Int32Encoder)


    # -- percentile separately -
    print("→ Scanning percentile model…")
    # for w in perc_windows:
    #     sig_b = compute_percentile(factor_back, w)
    #     sig_f = compute_percentile(factor_fwd,  w)
    #     sig_all = compute_percentile(factor_fwd,  w)
    #     for low, high in perc_thresholds:
    #         pos_b = pd.Series(0, index=sig_b.index)
    #         pos_b[sig_b > high] = 1
    #         pos_b[sig_b < low]  = -1
    #         stats_b = backtest(price_back, pos_b)
    #         all_results.append({
    #                     'factor': fact,
    #                     'model':  "percentile",
    #                     'strategy': 'percentile',
    #                     'windows': w,
    #                     'threshold': (low, high),
    #                     'train_sharpe': stats_b['sharpe']
    #                 })
    #         if not pass_filters(stats_b, len(price_back)): continue
    #         pos_f = pd.Series(0, index=sig_f.index)
    #         pos_f[sig_f > high] = 1
    #         pos_f[sig_f < low]  = -1
    #         stats_f = backtest(price_fwd, pos_f)
    #         if not forward_consistent(stats_b, stats_f): continue
    #         pos_all = pd.Series(0, index=sig_all.index)
    #         pos_all[sig_all > high] = 1
    #         pos_all[sig_all < low]  = -1
    #         stats_f = backtest(df[""], pos_all)
    #         results.append({
    #             'factor':        f'{fact}',
    #             'model' :        "percentile",
    #             'windows':       w,
    #             'strategy':      'percentile',
    #             'threshold':     (low, high),
    #             'train_stats':   stats_b,
    #             'forward_stats': stats_f,
    #             'positions':     pd.concat([pos_b, pos_f])
    #         })

    #heatmap_all_resul
    df_all = pd.DataFrame(all_results)
    
    if heatmap == True:

       for f in df_all['model'].unique():
            for s in df_all[df_all['model']==f]['strategy'].unique():
                plot_parameter_heatmaps(df_all, f, s, output_dir, interval, data_nam = data)


    # if nothing passed, bail out 
    if not results:
        print("X  No strategies met the filters. Adjust your parameter ranges or loosening filters.")
        return

    df_res = pd.DataFrame(results)
    df_res['train_sharpe']   = df_res['train_stats'].apply(lambda s: s['sharpe'])
    df_res['forward_sharpe'] = df_res['forward_stats'].apply(lambda s: s['sharpe'])


    equities = []
    #Final state of strategy choosing
    significant_rows = []
    for idx, row in df_res.iterrows():
        factor = row['factor']
        model  = row['model']
        strategy =row["strategy"]
        window = row['windows']
        threshold = row['threshold']
        pos_full = row['positions']


        # 运行 permutation_test
        pval= permutation_test(
            df,
            row,
            models_funcs=models_funcs,
            strat_funcs=strat_funcs,
            column_name= column_name,
            num_perm=100,
            seed=42,
            sr_multiplier=sr_multiplier
        )
        
        # 如果 p-value < 0.05，输出 JSON & CSV
        if pval < 0.05:
            significant_rows.append(idx)
            base = f"{factor}_{strategy}_{model}_{threshold}_{window}_{interval}"
            # Normalize window to tuple
            win_args = tuple(window) if isinstance(window,(list,tuple)) else (window,)

            # 1) Signal on full series
            sig_full = models_funcs[model](df_full[column_name], *win_args)

            # 2) Positions on full series
            pos_full = strat_funcs[strategy](sig_full, threshold)

            # 3) Backtest full series
            bt_full = backtest(df_full[price_global], pos_full,fee = fee, sr_multiplier=sr_multiplier)            
            # delay_test = delay_test(pos_full,)
            # # 写 JSON
            win_json = list(win_args)	
            ts, fs, all_stats = row['train_stats'], row['forward_stats'], row['full_date_stats']
            with open(f"{output_dir}/final_{base}_backtest.json",'w') as f:
                json.dump({'factor':factor,'model' : model,'windows': win_json, 'strategy':strategy,'threshold':threshold,
                        'sharpe':ts['sharpe'],'mdd_pct':ts['max_drawdown_pct'],'trades':ts['trades'],'anualized_return' :ts['anualized_return'] ,
            'sharpe': ts['sharpe'],'calmar_ratio' : ts['calmar_ratio'] ,'p_value':[pval], 'trade_per_interva':ts['trade_per_interval']}, f,cls= Int32Encoder, indent=2)
            with open(f"{output_dir}/{base}_forward_test.json",'w') as f:
                json.dump({'factor':factor, 'model' : model, 'windows':win_json,'strategy':strategy,'threshold':threshold,
                        'sharpe':fs['sharpe'],'mdd_pct':fs['max_drawdown_pct'],'trades':fs['trades'] ,'anualized_return' :fs['anualized_return'] ,
            'sharpe': fs['sharpe'],'calmar_ratio' : fs['calmar_ratio'],'p_value':[pval], 'trade_per_interva':fs['trade_per_interval']}, f,cls=  Int32Encoder, indent=2)
            
         
         
                        # 先重算 signal & positions
            df_csv = pd.DataFrame({
                'time':         df_full.index,
                'price':        df_full['close'].values,
                'factor':  df_full[column_name].values,
                'position':     pos_full.values,
                'pnl':          bt_full['pnl'].values,
                'trades':       bt_full['trade_signal'].values
            })
            df_csv.to_csv(os.path.join(output_dir, f"{base}_perm_trades.csv"), index=False)

            # --------- Plot Close Price with Trade Markers and Equity Curve ---------
            fig, ax1 = plt.subplots(figsize=(18, 8))
            # Plot close price
            ax1.plot(df_full.index, df_full[price_global], color="black", label="Close Price")

            # Identify trade entry points
            entry_long = (pos_full == 1) & (pos_full.shift(1).fillna(0) != 1)
            entry_short = (pos_full == -1) & (pos_full.shift(1).fillna(0) != -1)

            # Plot long (green triangle up) and short (red triangle down) markers
            ax1.scatter(df_full.index[entry_long], df_full.loc[entry_long, price_global],
                        marker="^", color="green", s=80, label="Long Entry")
            ax1.scatter(df_full.index[entry_short], df_full.loc[entry_short, price_global],
                        marker="v", color="red", s=80, label="Short Entry")

            ax1.set_ylabel("Close Price")

            # Plot equity curve on secondary y-axis
            ax2 = ax1.twinx()
            ax2.plot(bt_full['equity'].index, bt_full['equity'].values,
                     color="blue", label="Equity Curve", alpha=0.6)
            ax2.set_ylabel("Equity")

            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

            ax1.set_title(f"Price & Equity Curve — {base}")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f"{base}_price_equity.png"))
            plt.close(fig)
            # -----------------------------------------------------------------------

            eq_backtest = ts['equity']
            eq_forw   = fs['equity']
            eq = bt_full['equity']

            # eq = row['full_date_stats']['equity']
            name = f"{row['model']}_{row['strategy']}_{row['windows']},{row['threshold']}"
            equities.append((name, eq_backtest, eq_forw)) 

    #equity curve
    plt.figure(figsize=(30,11))
    for name, eq_b, eq_f in equities:
        # backtest in solid line
        plt.plot(eq_b.index, eq_b.values,
                label=f"{name} (back)", linewidth=1.5)
        # forward‐test in dashed line
        plt.plot(eq_f.index, eq_f.values,
                label=f"{name} (fwd)", linestyle="--", linewidth=1.5)

    plt.axvline(df_back.index[-1], color="gray", linestyle=":", label="split")
    plt.legend(loc="upper left")
    plt.title("Back vs Forward Equity Curves for Permutation-Significant Strategies")
    plt.xlabel("Datetime")
    plt.ylabel("Equity")
   
    plt.savefig(os.path.join(output_dir, "all_equity_curves.png"))
    plt.close()


    print(f'Progress complete.')

if __name__ == '__main__':
    DATA_FILE = '../Data/factor/Binance_BTCUSDT_perpetual_1H_takerbuysellvolume_facto.csv'
    run_pipeline(DATA_FILE, output_dir='results', interval='1h', column_name='change_open_taker_ratio',sr_multiplier = 24, fee =0.0006)
