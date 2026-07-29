#!/bin/bash
# Example daily Alpaca PAPER rebalance wrapper.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
# Loads .env automatically via src.cli
python3 -m src.cli alpaca-rebalance --config configs/default.yaml --mode portfolio --execute "$@"
