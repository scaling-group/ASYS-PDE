#!/usr/bin/env bash
set -euo pipefail
export ASYS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
exec "${ASYS_PYTHON:-python3}" "$(dirname "$0")/bounded.py"
