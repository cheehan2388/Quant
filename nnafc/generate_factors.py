from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

from .models import BaseFactorModel
from tqdm import tqdm
import logging


@torch.no_grad()
def generate_factors_over_split(
		model: BaseFactorModel,
		symbol_to_df: Dict[str, pd.DataFrame],
		lookback: int,
		timestamps: Iterable[pd.Timestamp],
		normalizer,
		feature_columns: List[str],
		device: str = "cuda",
) -> pd.DataFrame:
	model.eval()
	model.to(device)
	rows: List[Tuple[pd.Timestamp, str, float]] = []
	logger = logging.getLogger(__name__)
	iterable = tqdm(list(timestamps), desc="[Gen] Factors") if logger.isEnabledFor(logging.INFO) else timestamps
	for t in iterable:
		X_list: List[np.ndarray] = []
		syms: List[str] = []
		for s, df in symbol_to_df.items():
			try:
				loc = df.index.get_loc(t)
			except KeyError:
				continue
			start = loc - (lookback - 1)
			if start < 0:
				continue
			feat = df.iloc[start : loc + 1][feature_columns].to_numpy().T
			X_list.append(feat)
			syms.append(s)
		if not X_list:
			continue
		X = np.stack(X_list).astype(np.float32)
		X = normalizer.transform(X)
		xb = torch.from_numpy(X).to(device=device, dtype=torch.float32)
		pred = model(xb).detach().cpu().numpy().squeeze()
		for s, v in zip(syms, pred):
			rows.append((t, s, float(v)))
	return pd.DataFrame(rows, columns=["datetime", "symbol", "value"]).set_index(["datetime", "symbol"]).sort_index()


