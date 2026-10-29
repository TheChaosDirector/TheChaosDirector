#!/bin/bash
set -e
cd /workspace
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 -m src.cli train --config configs/daytrade-smoke.yaml > /tmp/daytrade-smoke.log 2>&1
echo "EXIT:$?" >> /tmp/daytrade-smoke.log
