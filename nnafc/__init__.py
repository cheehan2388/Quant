"""Neural Network-based Automatic Factor Construction (NNAFC)

This package implements a modular pipeline for:
- Data preparation and cross-sectional batching
- Model architectures (FCN, LSTM, 1D CNN)
- Stage 1 pre-training with prior-knowledge technical indicators
- Stage 2 fine-tuning with differentiable Rank IC loss
- Factor generation and evaluation utilities

See `nnafc/README.md` and `nnafc/main.py` for quick start.
"""

__all__ = [
	"config",
	"data",
	"indicators",
	"models",
	"losses",
	"train_pk",
	"train_rankic",
	"evaluate",
	"generate_factors",
]


