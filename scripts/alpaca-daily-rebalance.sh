#!/bin/bash
# Local cron/systemd wrapper for daily Alpaca PAPER rebalance.
# Skips cleanly when the US market is closed (holidays / weekends if mis-scheduled).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
# Loads .env automatically via src.cli
exec python3 -m src.cli alpaca-rebalance \
  --config configs/default.yaml \
  --mode portfolio \
  --execute \
  --skip-if-closed \
  "$@"
