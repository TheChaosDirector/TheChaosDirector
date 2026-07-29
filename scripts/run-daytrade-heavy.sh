#!/bin/bash
set -e
cd /workspace
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "=== HEAVY SMOKE DAYTRADE TRAIN ==="
python3 -m src.cli train --config configs/daytrade-smoke-heavy.yaml
echo "=== BACKTEST ==="
python3 -m src.cli backtest --config configs/daytrade-smoke-heavy.yaml
echo "=== PAPER ==="
rm -rf artifacts/smoke-heavy/daytrade/paper
python3 -m src.cli paper --config configs/daytrade-smoke-heavy.yaml --days 20
echo "=== REFEREE ==="
python3 -m src.cli referee --config configs/daytrade-smoke-heavy.yaml
echo "HEAVY_SMOKE_DONE"
