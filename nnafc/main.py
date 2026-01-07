from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import logging

from .config import TrainingConfig
from .data import CrossSectionalBatcher, load_symbol_csvs, compute_forward_return
from .models import build_model
from .train_pk import train_prior_knowledge, magnitude_prune
from .train_rankic import train_rank_ic
from .generate_factors import generate_factors_over_split
from .evaluate import compute_daily_spearman_ic, ic_summary


def parse_args() -> TrainingConfig:
	parser = argparse.ArgumentParser(description="NNAFC Pipeline")
	parser.add_argument("--data_dir", type=str, default="Data/close")
	parser.add_argument("--symbols", type=str, default="", help="Comma-separated symbols like BTCUSDT,ETHUSDT")
	parser.add_argument("--lookback", type=int, default=30)
	parser.add_argument("--horizon", type=int, default=24)
	parser.add_argument("--model", type=str, default="lstm")
	parser.add_argument("--epochs_pk", type=int, default=5)
	parser.add_argument("--epochs_rank", type=int, default=10)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--indicator", type=str, default="sma")
	parser.add_argument("--indicator_params", type=str, default="")
	parser.add_argument("--rank_p", type=float, default=1.83)
	parser.add_argument("--prune_pct", type=float, default=0.0)
	parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--output_dir", type=str, default="nnafc_outputs")
	parser.add_argument("--no_tqdm", action="store_true")
	parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
	parser.add_argument("--train_start", type=str, default="")
	parser.add_argument("--train_end", type=str, default="")
	parser.add_argument("--val_start", type=str, default="")
	parser.add_argument("--val_end", type=str, default="")
	parser.add_argument("--test_start", type=str, default="")
	parser.add_argument("--test_end", type=str, default="")
	args = parser.parse_args()

	def parse_list(s: str):
		return [x.strip() for x in s.split(",") if x.strip()] if s else None

	def parse_dict(s: str):
		# format: key1=val1,key2=val2
		d = {}
		for kv in [x for x in s.split(",") if x]:
			if "=" in kv:
				k, v = kv.split("=", 1)
				try:
					v = float(v) if "." in v or v.isdigit() else int(v)
				except Exception:
					pass
				d[k.strip()] = v
		return d or None

	cfg = TrainingConfig(
		data_dir=args.data_dir,
		symbols=parse_list(args.symbols),
		lookback=args.lookback,
		horizon=args.horizon,
		model_name=args.model,
		learning_rate=args.lr,
		epochs_pk=args.epochs_pk,
		epochs_rank=args.epochs_rank,
		indicator=args.indicator,
		indicator_params=parse_dict(args.indicator_params),
		rank_p=args.rank_p,
		prune_pct=args.prune_pct,
		device=args.device,
		output_dir=args.output_dir,
		use_tqdm=not args.no_tqdm,
		log_level=args.log_level,
		train_start=args.train_start or None,
		train_end=args.train_end or None,
		val_start=args.val_start or None,
		val_end=args.val_end or None,
		test_start=args.test_start or None,
		test_end=args.test_end or None,
	)
	return cfg


def main():
	cfg = parse_args()
	logging.basicConfig(level=getattr(logging, cfg.log_level, logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
	Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

	# Load data
	symbol_to_df = load_symbol_csvs(cfg.data_dir, cfg.symbols)
	if not symbol_to_df:
		raise SystemExit("No symbols loaded from data_dir.")

	# Build batcher for splitting and normalization
	# If no explicit dates, use 60% train, 20% val, 20% test on the common timeline
	# We derive ranges after creating a temporary batcher to get the timeline
	from .data import CrossSectionalBatcher as CSB

	tmp_batcher = CSB(symbol_to_df, cfg.lookback, cfg.horizon, (None, None), (None, None), (None, None))
	full_ts = tmp_batcher.timeline
	if cfg.train_start is None:
		n = len(full_ts)
		tr_end = full_ts[int(n * 0.6)] if n > 0 else None
		va_end = full_ts[int(n * 0.8)] if n > 0 else None
		train_range = (None, tr_end)
		val_range = (tr_end, va_end)
		test_range = (va_end, None)
	else:
		train_range = (cfg.train_start, cfg.train_end)
		val_range = (cfg.val_start, cfg.val_end)
		test_range = (cfg.test_start, cfg.test_end)

	batcher = CrossSectionalBatcher(symbol_to_df, cfg.lookback, cfg.horizon, train_range, val_range, test_range)

	# Compute forward log-returns into dfs for stage-2 use
	for s, df in symbol_to_df.items():
		symbol_to_df[s] = df.copy()
		symbol_to_df[s]["forward_return"] = np.log(df["close"].shift(-cfg.horizon) / df["close"]) 

	# Build model
	# Determine number of features from batcher
	num_features = len(batcher.feature_columns)

	model = build_model(
		cfg.model_name,
		num_features=num_features,
		lookback=cfg.lookback,
		hidden_sizes=cfg.hidden_sizes,
		lstm_hidden=cfg.lstm_hidden,
		lstm_layers=cfg.lstm_layers,
		cnn_channels=cfg.cnn_channels,
		dropout=cfg.dropout,
	)

	# Stage 1: PK pretraining
	model = train_prior_knowledge(
		model,
		symbol_to_df,
		cfg.lookback,
		cfg.indicator,
		cfg.indicator_params,
		list(batcher.train_ts),
		batcher.normalizer,
		batcher.feature_columns,
		device=cfg.device,
		lr=cfg.learning_rate,
		epochs=cfg.epochs_pk,
	)

	mask = magnitude_prune(model, cfg.prune_pct) if cfg.prune_pct > 0 else None

	# Stage 2: RankIC fine-tuning
	model = train_rank_ic(
		model,
		symbol_to_df,
		cfg.lookback,
		list(batcher.train_ts),
		batcher.normalizer,
		batcher.feature_columns,
		device=cfg.device,
		lr=cfg.learning_rate,
		epochs=cfg.epochs_rank,
		p=cfg.rank_p,
		mask=mask,
	)

	# Generate factors on test split
	factors_test = generate_factors_over_split(
		model,
		symbol_to_df,
		cfg.lookback,
		list(batcher.test_ts),
		batcher.normalizer,
		batcher.feature_columns,
		device=cfg.device,
	)

	# Prepare returns on the same index for evaluation
	rows = []
	for s, df in symbol_to_df.items():
		w = df[["forward_return"]].rename(columns={"forward_return": "value"})
		w = w.loc[factors_test.index.get_level_values(0).unique().min(): factors_test.index.get_level_values(0).unique().max()]
		w["symbol"] = s
		w = w.reset_index().rename(columns={"index": "datetime"})
		rows.append(w)
	ret_df = pd.concat(rows, axis=0)
	ret_df = ret_df.set_index(["datetime", "symbol"]).sort_index()

	ic_df = compute_daily_spearman_ic(factors_test, ret_df)
	sum_stats = ic_summary(ic_df["ic"]) if not ic_df.empty else {"mean": np.nan, "std": np.nan, "ir": np.nan}

	# Save artifacts
	model_path = os.path.join(cfg.output_dir, f"final_model_{cfg.model_name}.pt")
	torch.save(model.state_dict(), model_path)
	factors_path = os.path.join(cfg.output_dir, "factors_test.parquet")
	ic_path = os.path.join(cfg.output_dir, "ic_test.csv")
	factors_test.to_parquet(factors_path)
	ic_df.to_csv(ic_path)
	logging.info({"IC_mean": sum_stats["mean"], "IC_std": sum_stats["std"], "IC_IR": sum_stats["ir"], "model": model_path, "factors": factors_path, "ic_csv": ic_path})


if __name__ == "__main__":
	main()


