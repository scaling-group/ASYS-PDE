#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
exec timeout --signal=TERM --kill-after=10s 420s "${ASYS_BF_PYTHON:-${ASYS_PYTHON:-python3}}" "$(dirname "$0")/evaluate.py"
