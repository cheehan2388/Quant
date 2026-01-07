from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from tqdm import tqdm
import logging

from .losses import RankICLoss
from .models import BaseFactorModel
from .train_pk import apply_prune_mask_to_grads


def train_rank_ic(
		model: BaseFactorModel,
		symbol_to_df,
		lookback: int,
		train_timestamps: List[pd.Timestamp],
		normalizer,
		feature_columns: List[str],
		device: str = "cuda",
		lr: float = 1e-3,
		epochs: int = 20,
		p: float = 1.83,
		mask: Optional[Dict[str, torch.Tensor]] = None,
):
	model.to(device)
	model.train()
	optim = Adam(model.parameters(), lr=lr)
	loss_fn = RankICLoss(p=p)

	logger = logging.getLogger(__name__)

	for ei in range(epochs):
		iterable = tqdm(train_timestamps, desc=f"[RankIC] Epoch {ei+1}/{epochs}") if logger.isEnabledFor(logging.INFO) else train_timestamps
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
				feat = df.iloc[start : loc + 1][feature_columns].to_numpy().T
				# y_true will be provided by caller as an added column "forward_return" on df
				y_val = df.iloc[loc].get("forward_return", np.nan)
				if np.isnan(y_val):
					continue
				X_list.append(feat)
				Y_list.append(float(y_val))
			if not X_list:
				continue
			X = np.stack(X_list).astype(np.float32)
			X = normalizer.transform(X)
			Y = np.asarray(Y_list, dtype=np.float32)[:, None]
			xb = torch.from_numpy(X).to(device=device, dtype=torch.float32)
			yb = torch.from_numpy(Y).to(device=device, dtype=torch.float32)
			optim.zero_grad()
			pred = model(xb)
			loss = loss_fn(pred, yb)
			loss.backward()
			if mask:
				apply_prune_mask_to_grads(model, mask)
			optim.step()
		# Could log rolling stats if needed

	return model


