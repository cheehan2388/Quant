from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import logging

from .indicators import compute_pk_indicator
from .models import BaseFactorModel


def build_pk_targets(symbol_to_df: Dict[str, pd.DataFrame], indicator: str, params: Optional[dict]) -> Dict[str, pd.Series]:
	return {s: compute_pk_indicator(indicator, df["close"], params) for s, df in symbol_to_df.items()}


def train_prior_knowledge(
		model: BaseFactorModel,
		symbol_to_df: Dict[str, pd.DataFrame],
		lookback: int,
		indicator: str,
		indicator_params: Optional[dict],
		train_timestamps: List[pd.Timestamp],
		normalizer,
		feature_columns: List[str],
		device: str = "cuda",
		lr: float = 1e-3,
		epochs: int = 10,
):
	model.to(device)
	model.train()
	optim = Adam(model.parameters(), lr=lr)
	crit = nn.MSELoss()

	# Precompute PK targets per symbol
	pk = build_pk_targets(symbol_to_df, indicator, indicator_params)

	best_loss = float("inf")
	best_state = None

	logger = logging.getLogger(__name__)

	for epoch in range(epochs):
		total_loss = 0.0
		count = 0
		iterable = tqdm(train_timestamps, desc=f"[PK] Epoch {epoch+1}/{epochs}") if logger.isEnabledFor(logging.INFO) else train_timestamps
		for t in iterable:
			X_list: List[np.ndarray] = []
			Y_list: List[float] = []
			for s, df in symbol_to_df.items():
				try:
					loc = df.index.get_loc(t)
				except KeyError:
					continue
				start = loc - (lookback - 1)
				if start < 0:
					continue
				# Use fixed feature order
				feat = df.iloc[start : loc + 1][feature_columns].to_numpy().T
				y_val = pk[s].iloc[loc]
				if np.isnan(y_val):
					continue
				X_list.append(feat)
				Y_list.append(float(y_val))
			if not X_list:
				continue
			X = np.stack(X_list).astype(np.float32)
			X = normalizer.transform(X)
			Y = np.asarray(Y_list, dtype=np.float32)[:, None]
			xt = torch.from_numpy(X).to(device=device, dtype=torch.float32)
			yt = torch.from_numpy(Y).to(device=device, dtype=torch.float32)
			optim.zero_grad()
			pred = model(xt)
			loss = crit(pred, yt)
			loss.backward()
			optim.step()
			total_loss += float(loss.item())
			count += 1
		avg = total_loss / max(count, 1)
		if avg < best_loss:
			best_loss = avg
			best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
		logger.info(f"[PK] Epoch {epoch+1}/{epochs} avg MSE: {avg:.6f}")

	# Restore best
	if best_state is not None:
		model.load_state_dict(best_state)

	return model


def magnitude_prune(model: BaseFactorModel, prune_pct: float) -> Dict[str, torch.Tensor]:
	"""Return a mask dict of the same keys as model.state_dict() with 0/1 for prunable weights.

	Only linear and conv weight tensors are considered; biases are not pruned.
	"""
	if prune_pct <= 0:
		return {}

	with torch.no_grad():
		weights: List[torch.Tensor] = []
		keys: List[str] = []
		for name, param in model.named_parameters():
			if not param.requires_grad:
				continue
			if name.endswith(".weight"):
				weights.append(param.detach().abs().flatten())
				keys.append(name)
		if not weights:
			return {}
		all_w = torch.cat(weights)
		k = int(len(all_w) * prune_pct)
		if k <= 0:
			return {}
		threshold = torch.topk(all_w, k, largest=False).values.max()
		mask: Dict[str, torch.Tensor] = {}
		for name, param in model.named_parameters():
			if name.endswith(".weight"):
				mask[name] = (param.detach().abs() > threshold).float()
		return mask


def apply_prune_mask_to_grads(model: BaseFactorModel, mask: Dict[str, torch.Tensor]) -> None:
	for name, param in model.named_parameters():
		if name in mask and param.grad is not None:
			param.grad.mul_(mask[name].to(param.grad.device))


