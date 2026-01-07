from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class BaseFactorModel(nn.Module):
	"""Base interface: forward expects input of shape [batch, features, lookback] == [n, F, m].

	All models output a single scalar per sample (factor value) with shape [batch, 1].
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - to be implemented
		raise NotImplementedError



class FCNFactor(BaseFactorModel):
	def __init__(self, num_features: int, lookback: int, hidden_sizes: List[int], dropout: float = 0.1):
		super().__init__()
		input_dim = num_features * lookback
		layers: List[nn.Module] = []
		prev = input_dim
		for hs in hidden_sizes:
			layers.append(nn.Linear(prev, hs))
			layers.append(nn.ReLU())
			if dropout > 0:
				layers.append(nn.Dropout(dropout))
			prev = hs
		layers.append(nn.Linear(prev, 1))
		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, 5, m]
		b = x.shape[0]
		flat = x.reshape(b, -1)
		out = self.net(flat)
		return out



class LSTMFactor(BaseFactorModel):
	def __init__(self, num_features: int, lookback: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.1):
		super().__init__()
		self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
		self.head = nn.Linear(hidden_size, 1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# x: [B, 5, m] -> [B, m, 5]
		x = x.transpose(1, 2)
		out, _ = self.lstm(x)
		last = out[:, -1, :]
		return self.head(last)



class CNN1DFactor(BaseFactorModel):
	def __init__(self, num_features: int, lookback: int, channels: List[int] = [16, 32], dropout: float = 0.1):
		super().__init__()
		c1, c2 = channels[0], channels[1] if len(channels) > 1 else channels[0]
		self.conv = nn.Sequential(
			nn.Conv1d(num_features, c1, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.Conv1d(c1, c2, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.AdaptiveAvgPool1d(1),
		)
		self.head = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
			nn.Linear(c2, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		feat = self.conv(x)
		return self.head(feat)


def build_model(name: str, num_features: int, lookback: int, *, hidden_sizes: Optional[List[int]] = None, lstm_hidden: int = 64, lstm_layers: int = 1, cnn_channels: Optional[List[int]] = None, dropout: float = 0.1) -> BaseFactorModel:
	name = name.lower()
	if name == "fcn":
		return FCNFactor(num_features=num_features, lookback=lookback, hidden_sizes=hidden_sizes or [256, 128, 64], dropout=dropout)
	elif name == "lstm":
		return LSTMFactor(num_features=num_features, lookback=lookback, hidden_size=lstm_hidden, num_layers=lstm_layers, dropout=dropout)
	elif name in ("cnn", "cnn1d"):
		return CNN1DFactor(num_features=num_features, lookback=lookback, channels=cnn_channels or [16, 32], dropout=dropout)
	else:
		raise ValueError(f"Unknown model name: {name}")


