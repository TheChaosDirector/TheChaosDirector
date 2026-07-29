#!/bin/bash
set -e
cd /workspace
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "===== SMOKE HEAVY ====="
python3 -m src.cli train --config configs/daytrade-smoke-heavy.yaml
python3 -m src.cli backtest --config configs/daytrade-smoke-heavy.yaml
rm -rf artifacts/smoke-heavy/daytrade/paper
python3 -m src.cli paper --config configs/daytrade-smoke-heavy.yaml --days 20
python3 -m src.cli referee --config configs/daytrade-smoke-heavy.yaml
echo "===== INTENSIVE ====="
python3 -m src.cli train --config configs/daytrade-intensive.yaml
python3 -m src.cli backtest --config configs/daytrade-intensive.yaml
rm -rf artifacts/daytrade-intensive/daytrade/paper
python3 -m src.cli paper --config configs/daytrade-intensive.yaml --days 20
python3 -m src.cli referee --config configs/daytrade-intensive.yaml
echo CAMPAIGN2_DONE
