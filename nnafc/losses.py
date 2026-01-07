from __future__ import annotations

import torch
import torch.nn as nn


def sigmoid_kernel(x: torch.Tensor, p: float = 1.83, eps: float = 1e-8) -> torch.Tensor:
	"""g(x) = 1 / (1 + exp(-p * (x - mean(x)) / (2 * std(x)))) computed per batch.

	- x: shape [N, 1] or [N]
	- returns: shape [N, 1]
	"""
	if x.dim() == 1:
		x = x.unsqueeze(1)
	mu = x.mean(dim=0, keepdim=True)
	sigma = x.std(dim=0, keepdim=True).clamp_min(eps)
	z = (x - mu) / (2.0 * sigma)
	return torch.sigmoid(-p * (-z))  # equivalent to 1 / (1 + exp(-p*z))


def pearson_correlation(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
	"""Compute Pearson correlation between two vectors of shape [N, 1]."""
	if a.dim() == 1:
		a = a.unsqueeze(1)
	if b.dim() == 1:
		b = b.unsqueeze(1)
	na = a - a.mean(dim=0, keepdim=True)
	nb = b - b.mean(dim=0, keepdim=True)
	den = (na.std(dim=0, keepdim=True) * nb.std(dim=0, keepdim=True)).clamp_min(eps)
	cov = (na * nb).mean(dim=0, keepdim=True)
	r = cov / den
	return r.squeeze()


class RankICLoss(nn.Module):
	def __init__(self, p: float = 1.83):
		super().__init__()
		self.p = p

	def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
		g_pred = sigmoid_kernel(y_pred, p=self.p)
		g_true = sigmoid_kernel(y_true, p=self.p)
		r = pearson_correlation(g_pred, g_true)
		return -r


