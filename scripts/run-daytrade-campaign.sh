#!/bin/bash
set -e
cd /workspace
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

echo "===== 1) SUPERVISED HEAVY SMOKE ====="
rm -rf artifacts/smoke-heavy/daytrade
python3 -m src.cli train --config configs/daytrade-smoke-heavy.yaml
python3 -m src.cli backtest --config configs/daytrade-smoke-heavy.yaml
rm -rf artifacts/smoke-heavy/daytrade/paper
python3 -m src.cli paper --config configs/daytrade-smoke-heavy.yaml --days 20
python3 -m src.cli referee --config configs/daytrade-smoke-heavy.yaml | tee /tmp/daytrade-smoke-heavy-referee.txt

echo "===== 2) SUPERVISED INTENSIVE (40 liquid names, full lake) ====="
rm -rf artifacts/daytrade-intensive/daytrade
python3 -m src.cli train --config configs/daytrade-intensive.yaml
python3 -m src.cli backtest --config configs/daytrade-intensive.yaml
rm -rf artifacts/daytrade-intensive/daytrade/paper
python3 -m src.cli paper --config configs/daytrade-intensive.yaml --days 20
python3 -m src.cli referee --config configs/daytrade-intensive.yaml | tee /tmp/daytrade-intensive-referee.txt

echo "CAMPAIGN_DONE"
