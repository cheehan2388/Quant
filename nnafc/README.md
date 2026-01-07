## NNAFC (Neural Network-based Automatic Factor Construction)

This package implements an end-to-end pipeline inspired by Fang et al. (2020):

- Module 1: Data preparation, time-series normalization, cross-sectional batching
- Module 2: Model architectures: FCN, LSTM, 1D-CNN
- Module 3: Stage 1 pre-training to mimic prior-knowledge (PK) technical indicators
- Module 4: Stage 2 training with differentiable Rank IC loss and optional pruning masks
- Module 5: Factor generation and evaluation (IC, IR, factor correlation, quintile backtest)

### Quick Start

```bash
python -m nnafc.main \
  --data_dir Data/close \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --lookback 30 \
  --horizon 24 \
  --model lstm \
  --epochs_pk 5 \
  --epochs_rank 10
```

Artifacts are saved under `nnafc_outputs/` by default.


